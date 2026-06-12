"""
Step 2 of Engle's two-step DCC estimator.

Inputs are the standardized residuals z_t from the univariate GARCH step,
arranged as an (T x N) matrix. We then assume:

        Q_t = (1 - a - b) * Q_bar  +  a * z_{t-1} z_{t-1}'  +  b * Q_{t-1}
        R_t = diag(Q_t)^{-1/2} * Q_t * diag(Q_t)^{-1/2}

where Q_bar is the unconditional covariance of the standardized residuals
(the standard correlation-targeting choice). The conditional covariance
matrix of the original returns is then:

        Sigma_t = D_t * R_t * D_t,   D_t = diag(sigma_1,t, ..., sigma_N,t)

We estimate (a, b) by maximizing the Gaussian DCC log-likelihood (the part that
depends on R_t — see Engle 2002):

        L_c(a, b) = -0.5 * sum_t [ log|R_t|  +  z_t' R_t^{-1} z_t  -  z_t' z_t ]

Constraints: a >= 0, b >= 0, a + b < 1 (stationarity).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class DCCResult:
    """Bundle of DCC outputs needed downstream."""

    a: float                          # ARCH-like DCC parameter
    b: float                          # GARCH-like DCC parameter
    Q_bar: np.ndarray                 # unconditional covariance of std. residuals
    R_t: np.ndarray                   # (T, N, N) conditional correlations
    Sigma_t: np.ndarray               # (T, N, N) conditional covariances of returns
    loglik: float                     # log-likelihood at the optimum
    dates: pd.DatetimeIndex           # time index aligned to R_t / Sigma_t
    tickers: tuple[str, ...]          # column ordering


def _dcc_recursion(
    z: np.ndarray, a: float, b: float, Q_bar: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the Q_t / R_t recursion. Returns (Q_t, R_t) each of shape (T, N, N).
    Vectorized as much as is practical; the recursion itself is inherently
    sequential.
    """
    T, N = z.shape
    Q = np.empty((T, N, N))
    R = np.empty((T, N, N))

    # Initialize Q_0 at the long-run target.
    Q_prev = Q_bar.copy()
    one_minus_ab_Qbar = (1.0 - a - b) * Q_bar

    for t in range(T):
        if t == 0:
            Qt = Q_bar.copy()
        else:
            zlag = z[t - 1, :].reshape(-1, 1)
            Qt = one_minus_ab_Qbar + a * (zlag @ zlag.T) + b * Q_prev

        # Convert Q_t -> R_t by rescaling rows/cols by 1 / sqrt(diag(Q_t)).
        d_inv_sqrt = 1.0 / np.sqrt(np.diag(Qt))
        Rt = Qt * np.outer(d_inv_sqrt, d_inv_sqrt)

        Q[t] = Qt
        R[t] = Rt
        Q_prev = Qt

    return Q, R


def _neg_loglik(theta: np.ndarray, z: np.ndarray, Q_bar: np.ndarray) -> float:
    """
    Negative of the DCC composite log-likelihood (the part of the joint
    likelihood that depends on the correlations only).
    """
    a, b = theta
    # Soft barriers so the optimizer never enters the invalid region.
    if a <= 0 or b <= 0 or a + b >= 0.999:
        return 1e10

    _, R = _dcc_recursion(z, a, b, Q_bar)

    # Sum_t [ log|R_t| + z_t' R_t^{-1} z_t - z_t' z_t ]
    # The last term is constant in (a, b) but we keep it for completeness.
    ll = 0.0
    for t in range(z.shape[0]):
        Rt = R[t]
        # Cholesky-based logdet and solve are stabler than direct inversion.
        try:
            L = np.linalg.cholesky(Rt)
        except np.linalg.LinAlgError:
            return 1e10
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        zt = z[t, :]
        solved = np.linalg.solve(L, zt)
        quad = solved @ solved
        ll += logdet + quad - zt @ zt

    return 0.5 * ll


def estimate_dcc(
    std_resid: pd.DataFrame, sigma: pd.DataFrame
) -> DCCResult:
    """
    Estimate the DCC (a, b) by quasi-MLE on the standardized residuals and
    return everything downstream code needs.

    Parameters
    ----------
    std_resid : (T x N) standardized residuals from the univariate GARCH step.
    sigma     : (T x N) conditional standard deviations from the same step.
                Used to rebuild Sigma_t = D_t R_t D_t.
    """
    z = std_resid.to_numpy()
    T, N = z.shape
    tickers = tuple(std_resid.columns)
    dates = std_resid.index

    # Q_bar = unconditional cov of the standardized residuals. With z ~ N(0, I)
    # in theory, Q_bar should be ~ R_bar (the unconditional correlation). We
    # use the sample covariance directly: it is the textbook correlation-
    # targeting choice and reduces the parameter count by N(N-1)/2.
    Q_bar = np.cov(z, rowvar=False, ddof=0)

    # Optimize (a, b) under linear constraints a + b < 1, a >= 0, b >= 0.
    # We use L-BFGS-B with box bounds plus the soft barrier inside the
    # likelihood to enforce a + b < 1.
    x0 = np.array([0.02, 0.95])
    bounds = [(1e-6, 0.5), (1e-6, 0.999)]
    opt = minimize(
        _neg_loglik,
        x0=x0,
        args=(z, Q_bar),
        method="L-BFGS-B",
        bounds=bounds,
    )
    a_hat, b_hat = float(opt.x[0]), float(opt.x[1])
    loglik = -float(opt.fun)

    # Compute R_t and Sigma_t at the estimated parameters.
    _, R_t = _dcc_recursion(z, a_hat, b_hat, Q_bar)

    # Sigma_t = D_t R_t D_t where D_t = diag(sigma_t)
    sigma_arr = sigma.to_numpy()             # (T, N)
    # outer product per time step in a vectorized way
    D_outer = sigma_arr[:, :, None] * sigma_arr[:, None, :]   # (T, N, N)
    Sigma_t = R_t * D_outer

    return DCCResult(
        a=a_hat,
        b=b_hat,
        Q_bar=Q_bar,
        R_t=R_t,
        Sigma_t=Sigma_t,
        loglik=loglik,
        dates=dates,
        tickers=tickers,
    )


def replay_with_params(
    dcc: DCCResult, std_resid: pd.DataFrame, sigma: pd.DataFrame, a: float, b: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Re-run the DCC recursion at user-supplied (a, b), keeping the empirical
    Q_bar from the fitted model. This is what powers the "DCC what-if" sliders
    in the Streamlit dashboard — no refit needed.

    Returns (R_t, Sigma_t).
    """
    z = std_resid.to_numpy()
    _, R_t = _dcc_recursion(z, float(a), float(b), dcc.Q_bar)
    sigma_arr = sigma.to_numpy()
    D_outer = sigma_arr[:, :, None] * sigma_arr[:, None, :]
    return R_t, R_t * D_outer


def pairwise_corr_frame(R_t: np.ndarray, dates: pd.DatetimeIndex, tickers: tuple[str, ...]) -> pd.DataFrame:
    """
    Convert the (T, N, N) correlation cube into a tidy (T x num_pairs)
    DataFrame with columns "A-B" for each upper-triangle pair. Easy to plot.
    """
    T, N, _ = R_t.shape
    cols: dict[str, np.ndarray] = {}
    for i in range(N):
        for j in range(i + 1, N):
            cols[f"{tickers[i]}-{tickers[j]}"] = R_t[:, i, j]
    return pd.DataFrame(cols, index=dates)


if __name__ == "__main__":
    from src.data import load_dataset
    from src.garch import fit_all, collect_sigma, collect_std_resid

    bundle = load_dataset()
    fits = fit_all(bundle.returns)
    sigma = collect_sigma(fits)
    std_resid = collect_std_resid(fits)

    print("Running DCC estimation ...")
    dcc = estimate_dcc(std_resid, sigma)
    print(f"\nEstimated DCC: a = {dcc.a:.4f}, b = {dcc.b:.4f}, a+b = {dcc.a + dcc.b:.4f}")
    print(f"Composite log-likelihood = {dcc.loglik:.3f}")
    print(f"\nQ_bar (unconditional std-resid cov):\n{pd.DataFrame(dcc.Q_bar, index=dcc.tickers, columns=dcc.tickers).round(3)}")

    rho = pairwise_corr_frame(dcc.R_t, dcc.dates, dcc.tickers)
    print("\nConditional correlations — last 5 days:")
    print(rho.tail().round(3))
    print(f"\nConditional correlation range per pair:")
    print(pd.DataFrame({"min": rho.min(), "max": rho.max(), "mean": rho.mean()}).round(3))
