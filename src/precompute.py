"""
Run this script once locally to generate cached_data/ so Streamlit Cloud
can load the app instantly without downloading or computing anything.

    python -m src.precompute

Outputs to cached_data/ (tracked in git, not gitignored):
    returns.parquet        log-return DataFrame
    prices.parquet         adjusted price DataFrame
    rf.parquet             FRED 3M T-bill series
    garch_sigma.parquet    conditional volatilities
    garch_std_resid.parquet standardized residuals
    garch_params.parquet   GARCH parameter table (for display)
    dcc_R_t.npy            (T, N, N) conditional correlation matrices
    dcc_Sigma_t.npy        (T, N, N) conditional covariance matrices
    dcc_Q_bar.npy          (N, N) long-run Q target
    dcc_meta.json          scalar params + dates + tickers
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "cached_data"
OUT.mkdir(exist_ok=True)

from src.data import download_riskfree_rate, load_dataset
from src.dcc import estimate_dcc
from src.garch import collect_sigma, collect_std_resid, fit_all, params_table

# Fixed end date so the pre-computed snapshot is reproducible and the
# local parquet cache is always hit (avoids Yahoo rate limits on re-runs).
SNAPSHOT_END = "2026-06-12"

print("Downloading prices ...")
bundle = load_dataset(end=SNAPSHOT_END)
bundle.returns.to_parquet(OUT / "returns.parquet")
bundle.prices.to_parquet(OUT / "prices.parquet")
print(f"  {len(bundle.returns):,} obs  {bundle.start.date()} -> {bundle.end.date()}")

print("Downloading 3M T-bill (FRED) ...")
rf = download_riskfree_rate(end=SNAPSHOT_END)
rf.to_frame().to_parquet(OUT / "rf.parquet")

print("Fitting univariate GARCH(1,1) ...")
fits = fit_all(bundle.returns)
sigma = collect_sigma(fits)
std_resid = collect_std_resid(fits)
sigma.to_parquet(OUT / "garch_sigma.parquet")
std_resid.to_parquet(OUT / "garch_std_resid.parquet")
params_table(fits).to_parquet(OUT / "garch_params.parquet")

print("Estimating DCC ...")
dcc = estimate_dcc(std_resid, sigma)
print(f"  a={dcc.a:.4f}  b={dcc.b:.4f}  a+b={dcc.a+dcc.b:.4f}  loglik={dcc.loglik:.2f}")

np.save(OUT / "dcc_R_t.npy", dcc.R_t)
np.save(OUT / "dcc_Sigma_t.npy", dcc.Sigma_t)
np.save(OUT / "dcc_Q_bar.npy", dcc.Q_bar)

meta = {
    "a": dcc.a,
    "b": dcc.b,
    "loglik": dcc.loglik,
    "tickers": list(dcc.tickers),
    "dates": [str(d.date()) for d in dcc.dates],
}
with open(OUT / "dcc_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"Done. All files written to {OUT}/")
