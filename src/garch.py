"""
Step 1 of Engle's two-step DCC: a univariate GARCH(1,1) on each return series.

For each ticker i we fit:

        r_{i,t}   = mu_i  +  eps_{i,t}
        eps_{i,t} = sigma_{i,t} * z_{i,t}        with z_{i,t} ~ N(0, 1)
        sigma^2_{i,t} = omega_i + alpha_i * eps^2_{i,t-1} + beta_i * sigma^2_{i,t-1}

We extract the conditional standard deviation sigma_{i,t} and the standardized
residual z_{i,t}. The standardized residuals are the input to the DCC step.

We use the ``arch`` library: it is the standard Python GARCH implementation,
returns clean diagnostics, and handles the MLE numerics for us.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model


@dataclass(frozen=True)
class UnivariateFit:
    """Container for a single ticker's GARCH(1,1) results."""

    ticker: str
    params: dict[str, float]      # mu, omega, alpha, beta
    sigma: pd.Series              # conditional standard deviation (same index as returns)
    std_resid: pd.Series          # standardized residuals z = eps / sigma
    loglik: float                 # log-likelihood
    aic: float
    bic: float
    persistence: float            # alpha + beta — closeness to a unit root


def fit_one(series: pd.Series) -> UnivariateFit:
    """
    Fit GARCH(1,1) with a constant mean and Gaussian innovations to one series.

    Why GARCH(1,1)?
      - It is the de-facto standard and what Engle (2002) used in the original
        DCC paper. (1,1) captures the vast majority of volatility clustering
        observed in daily equity returns.
      - The assignment asks for "univariate GARCH specifications" without further
        constraint, so GARCH(1,1) is the natural baseline.
    """
    # The arch library is sensitive to scale; our returns are already in percent
    # which is the recommended scale, so we pass them as-is.
    model = arch_model(
        series.dropna(),
        mean="constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="normal",
        rescale=False,
    )
    res = model.fit(disp="off")

    sigma = res.conditional_volatility.copy()
    sigma.index = series.dropna().index
    sigma.name = series.name

    # Standardized residual z_t = (r_t - mu) / sigma_t
    std_resid = (series.dropna() - res.params["mu"]) / sigma
    std_resid.name = series.name

    params = {
        "mu": float(res.params["mu"]),
        "omega": float(res.params["omega"]),
        "alpha": float(res.params["alpha[1]"]),
        "beta": float(res.params["beta[1]"]),
    }

    return UnivariateFit(
        ticker=str(series.name),
        params=params,
        sigma=sigma,
        std_resid=std_resid,
        loglik=float(res.loglikelihood),
        aic=float(res.aic),
        bic=float(res.bic),
        persistence=params["alpha"] + params["beta"],
    )


def fit_all(returns: pd.DataFrame) -> dict[str, UnivariateFit]:
    """Fit GARCH(1,1) for every column of ``returns``."""
    return {ticker: fit_one(returns[ticker]) for ticker in returns.columns}


def collect_sigma(fits: dict[str, UnivariateFit]) -> pd.DataFrame:
    """Stack the conditional standard-deviation series into one DataFrame."""
    return pd.concat({t: f.sigma for t, f in fits.items()}, axis=1)


def collect_std_resid(fits: dict[str, UnivariateFit]) -> pd.DataFrame:
    """Stack standardized residuals — the matrix that feeds the DCC step."""
    return pd.concat({t: f.std_resid for t, f in fits.items()}, axis=1)


def params_table(fits: dict[str, UnivariateFit]) -> pd.DataFrame:
    """Tidy parameter table for the dashboard."""
    rows = []
    for t, f in fits.items():
        row = {"Ticker": t, **f.params,
               "alpha+beta": f.persistence,
               "Log-lik": f.loglik, "AIC": f.aic, "BIC": f.bic}
        rows.append(row)
    return pd.DataFrame(rows).set_index("Ticker")


if __name__ == "__main__":
    from src.data import load_dataset

    bundle = load_dataset()
    fits = fit_all(bundle.returns)
    print("GARCH(1,1) parameters per series:")
    print(params_table(fits).round(4))
    print("\nConditional sigma — last 5 days:")
    print(collect_sigma(fits).tail().round(3))
    print("\nStandardized residuals — last 5 days:")
    print(collect_std_resid(fits).tail().round(3))
    print("\nStandardized-residual summary (should be ~N(0,1) if GARCH fits well):")
    z = collect_std_resid(fits)
    print(pd.DataFrame({"mean": z.mean(), "std": z.std()}).round(3))
