"""
Data layer: download adjusted closing prices and compute percentage log-returns.

The assignment asks for "daily adjusted closing prices ... transformed in
percentage returns". We use yfinance's auto-adjusted close (which accounts for
splits and dividends) and the standard continuously-compounded log-return:

        r_t = 100 * ( log(P_t) - log(P_{t-1}) )

Prices are cached to data/prices.parquet so re-runs are instant and we do not
hammer Yahoo's endpoints during development.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# The fixed sector-diverse basket chosen during scoping:
#   NVDA — semiconductors / AI
#   TSLA — automotive / EV
#   LLY  — pharmaceuticals
#   JPM  — banking / financials
#   XOM  — energy
DEFAULT_TICKERS: tuple[str, ...] = ("NVDA", "TSLA", "LLY", "JPM", "XOM")

# A friendly long name per ticker for chart labels and interpretation text.
TICKER_NAMES: dict[str, str] = {
    "NVDA": "NVIDIA (semiconductors / AI)",
    "TSLA": "Tesla (auto / EV)",
    "LLY": "Eli Lilly (pharma)",
    "JPM": "JPMorgan Chase (financials)",
    "XOM": "ExxonMobil (energy)",
}

# Path to the on-disk cache. Anything in data/ is .gitignored.
_CACHE_DIR = Path(__file__).resolve().parent.parent / "data"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class PriceData:
    """Bundle of everything downstream code needs about the price series."""

    prices: pd.DataFrame      # adjusted close, one column per ticker
    returns: pd.DataFrame     # 100 * log returns, same shape, first row dropped
    tickers: tuple[str, ...]  # canonical column ordering
    start: pd.Timestamp       # actual first available date
    end: pd.Timestamp         # actual last available date


def _cache_path(tickers: tuple[str, ...], start: str, end: str) -> Path:
    """Deterministic filename derived from the request — one parquet per request."""
    key = f"{'-'.join(tickers)}__{start}__{end}.parquet"
    return _CACHE_DIR / key


def download_prices(
    tickers: tuple[str, ...] = DEFAULT_TICKERS,
    years: int = 10,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download adjusted closing prices for ``tickers`` covering the last ``years``
    years up to ``end`` (defaults to today). Result is one column per ticker,
    indexed by trading day.
    """
    end_date = pd.Timestamp(end) if end else pd.Timestamp(date.today())
    start_date = end_date - pd.Timedelta(days=int(years * 365.25))
    start_str, end_str = start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    cache = _cache_path(tickers, start_str, end_str)
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    # auto_adjust=True returns the dividend/split-adjusted close in the "Close" column,
    # which is what we want per the assignment ("adjusted closing prices").
    raw = yf.download(
        list(tickers),
        start=start_str,
        end=end_str,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    # yfinance returns a column-MultiIndex when multiple tickers are requested;
    # pull out the Close level and re-order to match the requested ticker order.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices[list(tickers)].dropna(how="all").sort_index()
    prices.index.name = "Date"

    prices.to_parquet(cache)
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Percentage log-returns: r_t = 100 * (log P_t - log P_{t-1}).
    The first row is NaN by construction and is dropped.
    """
    returns = 100.0 * np.log(prices / prices.shift(1))
    return returns.dropna(how="any")


def load_dataset(
    tickers: tuple[str, ...] = DEFAULT_TICKERS,
    years: int = 10,
    end: str | None = None,
    use_cache: bool = True,
) -> PriceData:
    """
    One-shot helper: download prices, compute returns, return a tidy bundle.
    This is what the Streamlit app calls.
    """
    prices = download_prices(tickers, years=years, end=end, use_cache=use_cache)
    returns = compute_log_returns(prices)
    return PriceData(
        prices=prices,
        returns=returns,
        tickers=tuple(tickers),
        start=pd.Timestamp(returns.index.min()),
        end=pd.Timestamp(returns.index.max()),
    )


def download_riskfree_rate(
    years: int = 10,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.Series:
    """
    Download the 3-Month US Treasury Constant Maturity yield from FRED
    (series ID ``DGS3MO``) over the last ``years`` years up to ``end``.
    The series is returned as the **annualized yield in percent** at daily
    frequency, indexed by trading day.

    This is the standard academic proxy for the risk-free rate when computing
    Sharpe ratios on equity portfolios. The 3-month bill is preferred over
    longer-maturity Treasuries (e.g. the 10-year note) because its price has
    negligible sensitivity to short-horizon yield changes, so it behaves as a
    true risk-free asset over the relevant evaluation window.

    FRED is used in preference to Yahoo's ``^IRX`` ticker because Yahoo's
    short-end-rate endpoints have become unreliable (frequent rate-limit
    errors); FRED's public CSV endpoint is free, key-less, and stable.
    """
    end_date = pd.Timestamp(end) if end else pd.Timestamp(date.today())
    start_date = end_date - pd.Timedelta(days=int(years * 365.25))
    start_str, end_str = start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    cache = _CACHE_DIR / f"riskfree_DGS3MO__{start_str}__{end_str}.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)["rf_annual_pct"]

    # FRED publishes the full history at this URL; we slice to the requested
    # window afterwards. "." denotes missing observations (FRED's convention)
    # and is parsed as NaN by pandas with na_values=["."].
    fred_url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id=DGS3MO&cosd={start_str}&coed={end_str}"
    )
    df = pd.read_csv(fred_url, parse_dates=["observation_date"], na_values=["."])
    # FRED has at times labelled the date column 'DATE' instead of 'observation_date'.
    # Normalize either way.
    date_col = "observation_date" if "observation_date" in df.columns else "DATE"
    df = df.rename(columns={date_col: "Date", "DGS3MO": "rf_annual_pct"})
    rf = (
        df.set_index("Date")["rf_annual_pct"]
        .dropna()
        .sort_index()
    )
    rf.index.name = "Date"

    rf.to_frame().to_parquet(cache)
    return rf


if __name__ == "__main__":
    # Smoke test: download and print a quick summary.
    bundle = load_dataset()
    print(f"Tickers: {bundle.tickers}")
    print(f"Window: {bundle.start.date()} to {bundle.end.date()}")
    print(f"Observations: {len(bundle.returns)}")
    print("\nFirst rows of returns:")
    print(bundle.returns.head().round(3))
    print("\nReturn summary:")
    print(bundle.returns.describe().round(3))
