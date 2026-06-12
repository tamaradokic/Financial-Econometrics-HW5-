"""
Q1 + Q2 helpers: descriptive statistics, unconditional correlation, and the
data inputs for the bivariate scatter matrix.

The summary table answers Q1 in the most informative way possible:
    mean / std / min / max / skewness / excess kurtosis / Jarque-Bera p-value
    / ARCH-LM p-value (motivates GARCH) / Ljung-Box p-value on squared returns.
A small p-value on JB confirms non-normality; a small p-value on ARCH-LM or
Ljung-Box(r^2) confirms volatility clustering — exactly the stylized facts
that make a GARCH model appropriate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch


def descriptive_stats(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Return a per-asset summary table. Rows are tickers, columns are the
    statistics. Heavy-tail and volatility-clustering tests are reported as
    p-values; small p-values (< 0.05) are evidence in favour of the alternative.
    """
    rows: list[dict[str, float]] = []
    for ticker in returns.columns:
        series = returns[ticker].dropna()

        # Basic moments
        mean = series.mean()
        std = series.std(ddof=1)
        minimum = series.min()
        maximum = series.max()
        skew = sps.skew(series, bias=False)
        # statsmodels/scipy "kurtosis" with fisher=True is *excess* kurtosis
        # (normal => 0). Positive values mean fatter tails than the normal.
        ex_kurt = sps.kurtosis(series, fisher=True, bias=False)

        # Jarque-Bera: tests joint hypothesis of zero skew + zero excess kurtosis.
        jb_stat, jb_p = sps.jarque_bera(series)

        # ARCH-LM (Engle): tests for autocorrelation in squared residuals.
        # A small p-value motivates a GARCH specification.
        try:
            arch_p = het_arch(series.values, nlags=10)[1]
        except Exception:
            arch_p = np.nan

        # Ljung-Box on squared returns at lag 10: another volatility-clustering check.
        try:
            lb_p = acorr_ljungbox(series.values ** 2, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
        except Exception:
            lb_p = np.nan

        rows.append(
            {
                "Mean": mean,
                "Std. Dev.": std,
                "Min": minimum,
                "Max": maximum,
                "Skewness": skew,
                "Excess Kurtosis": ex_kurt,
                "JB p-value": jb_p,
                "ARCH-LM(10) p-value": arch_p,
                "Ljung-Box(r^2, 10) p-value": lb_p,
                "N obs": int(len(series)),
            }
        )

    return pd.DataFrame(rows, index=returns.columns)


def unconditional_correlation(returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation over the full sample — the static "one number per pair" view."""
    return returns.corr(method="pearson")


def pairwise_data(returns: pd.DataFrame) -> list[tuple[str, str, pd.Series, pd.Series]]:
    """
    Yield the inputs for the bivariate scatterplots Q2 asks for. Returned in
    upper-triangle order (no duplicates), so for 5 tickers this is 10 pairs.
    """
    cols = list(returns.columns)
    out: list[tuple[str, str, pd.Series, pd.Series]] = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            out.append((a, b, returns[a], returns[b]))
    return out


if __name__ == "__main__":
    from src.data import load_dataset

    bundle = load_dataset()
    print("Descriptive statistics:")
    print(descriptive_stats(bundle.returns).round(4))
    print("\nUnconditional correlation matrix:")
    print(unconditional_correlation(bundle.returns).round(3))
    print(f"\nNumber of unique pairs for Q2 scatterplots: {len(pairwise_data(bundle.returns))}")
