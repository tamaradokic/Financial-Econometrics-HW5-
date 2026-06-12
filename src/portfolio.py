"""
Portfolio construction and return generation for Q6 - Q8.

Three strategies are computed:
  - MVP (DCC):    weights derived from the time-varying conditional covariance
                  Sigma_t estimated by the DCC model.
  - MVP (static): same closed-form MVP weights, but using the *unconditional*
                  sample covariance of returns. This is the benchmark that
                  isolates "what does the dynamic part of DCC actually add?".
  - Equal weight: 1/N at every date. The naive benchmark.

Q6 closed-form minimum-variance weights:

        w_t^* = ( Sigma_t^{-1} * 1 ) / ( 1' Sigma_t^{-1} * 1 )

Q7 portfolio returns: a strict no-look-ahead convention is used. The weights
applied at date t are computed from information available through t-1:

        r_p,t = w_{t-1}' * r_t

The portfolio return on the first date is therefore undefined and dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioResult:
    """Bundle of one strategy's outputs."""

    name: str
    weights: pd.DataFrame                # (T x N), summing to 1 each row
    returns: pd.Series                   # portfolio returns, length T-1
    mean: float                          # arithmetic mean of returns (% per day)
    variance: float                      # sample variance
    std: float                           # sample std (% per day)
    annualized_mean: float               # mean * 252
    annualized_std: float                # std * sqrt(252)
    sharpe: float                        # annualized Sharpe (rf = 0)


def _mvp_weights_from_sigma(Sigma: np.ndarray) -> np.ndarray:
    """
    Closed-form MVP weights from a single covariance matrix.

    Solves   w = Sigma^{-1} 1 / (1' Sigma^{-1} 1).
    Uses np.linalg.solve rather than explicit inversion for numerical stability.
    """
    N = Sigma.shape[0]
    ones = np.ones(N)
    x = np.linalg.solve(Sigma, ones)
    return x / x.sum()


def mvp_weights_dynamic(Sigma_t: np.ndarray, dates: pd.DatetimeIndex, tickers: tuple[str, ...]) -> pd.DataFrame:
    """
    MVP weights from a (T, N, N) sequence of conditional covariance matrices.
    Returns a tidy DataFrame indexed by date, columns = tickers.
    """
    T = Sigma_t.shape[0]
    W = np.empty((T, len(tickers)))
    for t in range(T):
        W[t] = _mvp_weights_from_sigma(Sigma_t[t])
    return pd.DataFrame(W, index=dates, columns=list(tickers))


def mvp_weights_static(returns: pd.DataFrame) -> pd.DataFrame:
    """
    "Static" MVP: weights from the full-sample sample covariance — i.e. the
    weights you would have chosen if you believed the covariance never moved.
    Same weights every day. Acts as a benchmark for the dynamic strategy.
    """
    Sigma = returns.cov().values
    w = _mvp_weights_from_sigma(Sigma)
    return pd.DataFrame(
        np.tile(w, (len(returns), 1)),
        index=returns.index,
        columns=list(returns.columns),
    )


def equal_weights(returns: pd.DataFrame) -> pd.DataFrame:
    """1/N at every date."""
    N = returns.shape[1]
    return pd.DataFrame(
        np.full((len(returns), N), 1.0 / N),
        index=returns.index,
        columns=list(returns.columns),
    )


def custom_fixed_weights(returns: pd.DataFrame, weight_dict: dict[str, float]) -> pd.DataFrame:
    """
    Build a constant-weight portfolio from a user-specified allocation. Weights
    are normalized so they sum to one before being applied. Used by the
    "Build your own portfolio" Streamlit control.
    """
    raw = pd.Series({c: float(weight_dict.get(c, 0.0)) for c in returns.columns})
    total = raw.sum()
    if total <= 0:
        raise ValueError("Custom weights must sum to a strictly positive value before normalization.")
    normalized = raw / total
    return pd.DataFrame(
        np.tile(normalized.values, (len(returns), 1)),
        index=returns.index,
        columns=list(returns.columns),
    )


def portfolio_returns(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """
    Compute portfolio returns using *lagged* weights: r_p,t = w_{t-1}' r_t.

    Lagging is essential — without it, the strategy uses information from
    date t to choose its position at date t, which is a look-ahead bias and
    inflates apparent performance. Many textbook errors live here.
    """
    aligned_weights = weights.shift(1)
    # Element-wise product then row-sum; nan rows (first date) are dropped.
    return (aligned_weights * returns).sum(axis=1).dropna()


def summarize(name: str, weights: pd.DataFrame, returns: pd.DataFrame, rf_annual: float = 0.0) -> PortfolioResult:
    """
    Wrap a (weights, returns) pair into a PortfolioResult with summary stats.

    rf_annual is an annual risk-free rate in percent; the Sharpe is computed
    on annualized excess returns. Defaults to 0 since the assignment does
    not ask for it explicitly.
    """
    r = portfolio_returns(weights, returns)
    mean = float(r.mean())
    var = float(r.var(ddof=1))
    std = float(r.std(ddof=1))
    ann_mean = mean * 252.0
    ann_std = std * np.sqrt(252.0)
    sharpe = (ann_mean - rf_annual) / ann_std if ann_std > 0 else np.nan

    return PortfolioResult(
        name=name,
        weights=weights,
        returns=r,
        mean=mean,
        variance=var,
        std=std,
        annualized_mean=ann_mean,
        annualized_std=ann_std,
        sharpe=float(sharpe),
    )


def compare_strategies(
    Sigma_t: np.ndarray,
    dates: pd.DatetimeIndex,
    tickers: tuple[str, ...],
    returns: pd.DataFrame,
    rf_annual: float = 0.0,
) -> dict[str, PortfolioResult]:
    """Build all three strategies and return them keyed by name."""
    # Align returns to Sigma_t's index in case of any mismatch.
    returns = returns.loc[dates]

    w_dyn = mvp_weights_dynamic(Sigma_t, dates, tickers)
    w_stat = mvp_weights_static(returns)
    w_eq = equal_weights(returns)

    return {
        "MVP (DCC)": summarize("MVP (DCC)", w_dyn, returns, rf_annual),
        "MVP (static)": summarize("MVP (static)", w_stat, returns, rf_annual),
        "Equal-weight": summarize("Equal-weight", w_eq, returns, rf_annual),
    }


if __name__ == "__main__":
    from src.data import load_dataset
    from src.garch import fit_all, collect_sigma, collect_std_resid
    from src.dcc import estimate_dcc

    bundle = load_dataset()
    fits = fit_all(bundle.returns)
    dcc = estimate_dcc(collect_std_resid(fits), collect_sigma(fits))

    results = compare_strategies(dcc.Sigma_t, dcc.dates, dcc.tickers, bundle.returns)

    print(f"{'Strategy':<14} {'Mean (%, daily)':>16} {'Std (%, daily)':>16} {'Variance':>10} {'Ann. Sharpe':>12}")
    for r in results.values():
        print(f"{r.name:<14} {r.mean:>16.4f} {r.std:>16.4f} {r.variance:>10.4f} {r.sharpe:>12.3f}")

    print("\nMVP (DCC) weights — last 5 days:")
    print(results["MVP (DCC)"].weights.tail().round(3))
    print("\nMVP (DCC) weights — range across the sample:")
    w = results["MVP (DCC)"].weights
    print(pd.DataFrame({"min": w.min(), "max": w.max(), "mean": w.mean()}).round(3))
