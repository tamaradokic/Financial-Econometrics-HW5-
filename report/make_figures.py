"""
Generate publication-quality PDF figures for the report by running the full
analysis pipeline once and rendering with matplotlib.

Outputs the following files into ``report/figures/``:
    1. fig_sigma.pdf         — conditional volatilities (Q4)
    2. fig_correlations.pdf  — dynamic conditional correlations (Q5)
    3. fig_weights.pdf       — MVP (DCC) weights with MVP (static) reference (Q6)
    4. fig_cumret.pdf        — cumulative return of the strategies (Q7/Q8)
    5. fig_drawdown.pdf      — drawdowns of the strategies (Q7/Q8)
    6. fig_corr_heatmap.pdf  — unconditional correlation matrix (Q1)

Numbers used in the report's tables are also printed to stdout so they can be
hand-copied (or piped into a tex file) for the LaTeX summary tables.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make src importable when running from the report/ folder.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import TICKER_NAMES, download_riskfree_rate, load_dataset
from src.descriptive import descriptive_stats, unconditional_correlation
from src.garch import collect_sigma, collect_std_resid, fit_all, params_table
from src.dcc import estimate_dcc, pairwise_corr_frame
from src.portfolio import compare_strategies, mvp_weights_dynamic, mvp_weights_static


OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Visual style consistent with an academic paper: serif, single neutral palette,
# minimal chartjunk.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

CRISIS_BANDS = [
    ("COVID-19", "2020-02-15", "2020-06-30", "#e74c3c"),
    ("2022 rate shock", "2022-01-01", "2022-12-31", "#f1c40f"),
]


def _add_crisis_bands(ax):
    for label, start, end, color in CRISIS_BANDS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   color=color, alpha=0.08, linewidth=0)


# ---------------------------------------------------------------------------
# Run the full pipeline once
# ---------------------------------------------------------------------------
print("Loading data ...")
bundle = load_dataset()
print(f"  {len(bundle.returns):,} observations, {bundle.start.date()} -> {bundle.end.date()}")

print("Fitting univariate GARCH(1,1) ...")
fits = fit_all(bundle.returns)
sigma = collect_sigma(fits)
std_resid = collect_std_resid(fits)

print("Estimating DCC ...")
dcc = estimate_dcc(std_resid, sigma)
print(f"  a = {dcc.a:.4f}, b = {dcc.b:.4f}, a+b = {dcc.a + dcc.b:.4f}, loglik = {dcc.loglik:.2f}")

rho = pairwise_corr_frame(dcc.R_t, dcc.dates, dcc.tickers)
mvp_w = mvp_weights_dynamic(dcc.Sigma_t, dcc.dates, dcc.tickers)
mvp_static_w = mvp_weights_static(bundle.returns.loc[dcc.dates]).iloc[0]

results = compare_strategies(dcc.Sigma_t, dcc.dates, dcc.tickers, bundle.returns)
rf_series = download_riskfree_rate()
rf_eff = float(rf_series.loc[dcc.dates[0]:dcc.dates[-1]].mean())
print(f"  rf (3M T-bill, sample avg) = {rf_eff:.3f}%")


# ---------------------------------------------------------------------------
# Figure 1 — Conditional volatilities sigma_t per asset (Q4)
# ---------------------------------------------------------------------------
print("Rendering fig_sigma.pdf ...")
fig, axes = plt.subplots(5, 1, figsize=(6.8, 4.6), sharex=True)
for ax, (i, t) in zip(axes, enumerate(sigma.columns)):
    ax.plot(sigma.index, sigma[t], lw=0.8, color=PALETTE[i % len(PALETTE)])
    ax.set_ylabel(f"{t}\n(% / day)", rotation=0, ha="right", va="center", labelpad=18)
    _add_crisis_bands(ax)
axes[0].set_title("Estimated conditional standard deviation $\\sigma_t$ from GARCH(1,1)", pad=4)
axes[-1].set_xlabel("Date")
plt.savefig(OUT / "fig_sigma.pdf")
plt.close()


# ---------------------------------------------------------------------------
# Figure 2 — Dynamic conditional correlations (Q5)
# ---------------------------------------------------------------------------
print("Rendering fig_correlations.pdf ...")
fig, ax = plt.subplots(figsize=(6.8, 3.6))
cmap = plt.get_cmap("tab10")
for j, col in enumerate(rho.columns):
    ax.plot(rho.index, rho[col], lw=0.7, color=cmap(j % 10), label=col, alpha=0.85)
ax.axhline(0, color="grey", ls=":", lw=0.6)
_add_crisis_bands(ax)
ax.set_ylabel(r"Conditional correlation $\rho_{ij,t}$")
ax.set_xlabel("Date")
ax.set_title("DCC dynamic conditional correlations for all 10 pairs", pad=4)
ax.set_ylim(-0.4, 0.85)
ax.legend(ncol=5, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, -0.32), frameon=False)
plt.savefig(OUT / "fig_correlations.pdf")
plt.close()


# ---------------------------------------------------------------------------
# Figure 3 — MVP (DCC) weights with MVP (static) reference (Q6)
# ---------------------------------------------------------------------------
print("Rendering fig_weights.pdf ...")
fig, ax = plt.subplots(figsize=(6.8, 3.4))
for i, col in enumerate(mvp_w.columns):
    color = PALETTE[i % len(PALETTE)]
    ax.plot(mvp_w.index, mvp_w[col], lw=1.0, color=color, label=col)
    ax.axhline(mvp_static_w[col], color=color, ls="--", lw=0.6, alpha=0.6)
ax.axhline(0, color="grey", ls=":", lw=0.6)
_add_crisis_bands(ax)
ax.set_ylabel("Portfolio weight")
ax.set_xlabel("Date")
ax.set_title(r"MVP (DCC) weights $w_t^* = \Sigma_t^{-1}\mathbf{1}/(\mathbf{1}^\top \Sigma_t^{-1}\mathbf{1})$. Dashed = MVP (static).", pad=4)
ax.legend(ncol=5, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.28), frameon=False)
plt.savefig(OUT / "fig_weights.pdf")
plt.close()


# ---------------------------------------------------------------------------
# Figure 4 — Cumulative returns (Q7/Q8)
# ---------------------------------------------------------------------------
print("Rendering fig_cumret.pdf ...")
fig, ax = plt.subplots(figsize=(6.8, 3.0))
for i, (name, res) in enumerate(results.items()):
    wealth = (1.0 + res.returns / 100.0).cumprod()
    ax.plot(wealth.index, wealth, lw=1.4, color=PALETTE[i], label=name)
_add_crisis_bands(ax)
ax.set_ylabel(r"Cumulative growth of \$1")
ax.set_xlabel("Date")
ax.set_title("Cumulative wealth: MVP (DCC), MVP (static), and equal-weight portfolios", pad=4)
ax.legend(loc="upper left", frameon=False)
plt.savefig(OUT / "fig_cumret.pdf")
plt.close()


# ---------------------------------------------------------------------------
# Figure 5 — Drawdowns (Q7/Q8)
# ---------------------------------------------------------------------------
print("Rendering fig_drawdown.pdf ...")
fig, ax = plt.subplots(figsize=(6.8, 2.6))
for i, (name, res) in enumerate(results.items()):
    wealth = (1.0 + res.returns / 100.0).cumprod()
    dd = (wealth / wealth.cummax() - 1.0) * 100.0
    ax.fill_between(dd.index, dd, 0, color=PALETTE[i], alpha=0.35, lw=0)
    ax.plot(dd.index, dd, lw=0.8, color=PALETTE[i], label=name)
_add_crisis_bands(ax)
ax.set_ylabel("Drawdown (%)")
ax.set_xlabel("Date")
ax.set_title("Peak-to-trough drawdown", pad=4)
ax.legend(loc="lower left", frameon=False)
plt.savefig(OUT / "fig_drawdown.pdf")
plt.close()


# ---------------------------------------------------------------------------
# Figure 6 — Unconditional correlation heatmap (Q1)
# ---------------------------------------------------------------------------
print("Rendering fig_corr_heatmap.pdf ...")
corr = unconditional_correlation(bundle.returns)
fig, ax = plt.subplots(figsize=(3.2, 2.8))
im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(len(corr)))
ax.set_yticks(range(len(corr)))
ax.set_xticklabels(corr.columns)
ax.set_yticklabels(corr.index)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                fontsize=7, color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
ax.set_title("Unconditional ρ", pad=4)
plt.colorbar(im, ax=ax, shrink=0.7)
plt.savefig(OUT / "fig_corr_heatmap.pdf")
plt.close()


# ---------------------------------------------------------------------------
# Numbers for the LaTeX tables — written to a JSON for the .tex to consume.
# ---------------------------------------------------------------------------
print("Computing summary numbers ...")
stats = descriptive_stats(bundle.returns)
params = params_table(fits)

rho_range = pd.DataFrame({"min": rho.min(), "max": rho.max(), "mean": rho.mean()})
flip_pairs = rho.columns[(rho.min() < 0) & (rho.max() > 0)].tolist()

# Summary table per strategy
strategy_summary = []
for name, res in results.items():
    ann_ret = res.annualized_mean
    ann_vol = res.annualized_std
    sharpe = (ann_ret - rf_eff) / ann_vol if ann_vol > 0 else float("nan")
    wealth = (1.0 + res.returns / 100.0).cumprod()
    max_dd = float(((wealth / wealth.cummax()) - 1.0).min() * 100.0)
    strategy_summary.append({
        "name": name,
        "mean_daily": res.mean,
        "var_daily": res.variance,
        "std_daily": res.std,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
    })

numbers = {
    "n_obs": int(len(bundle.returns)),
    "start": str(bundle.start.date()),
    "end": str(bundle.end.date()),
    "tickers": list(bundle.tickers),
    "ticker_names": TICKER_NAMES,
    "rf_eff": rf_eff,
    "dcc_a": dcc.a,
    "dcc_b": dcc.b,
    "dcc_persistence": dcc.a + dcc.b,
    "dcc_loglik": dcc.loglik,
    "garch_params": params.round(4).to_dict(orient="index"),
    "uncond_corr": corr.round(3).to_dict(orient="index"),
    "rho_widest_pair": (rho.max() - rho.min()).idxmax(),
    "rho_widest_min": float(rho[(rho.max() - rho.min()).idxmax()].min()),
    "rho_widest_max": float(rho[(rho.max() - rho.min()).idxmax()].max()),
    "n_flip_pairs": len(flip_pairs),
    "mvp_dcc_mean_weights": mvp_w.mean().round(3).to_dict(),
    "mvp_static_weights": mvp_static_w.round(3).to_dict(),
    "strategies": strategy_summary,
    "stats": stats.round(3).to_dict(orient="index"),
}

with open(OUT.parent / "numbers.json", "w") as f:
    json.dump(numbers, f, indent=2, default=str)

print("Done.")
print(f"  Figures in: {OUT}")
print(f"  Numbers   : {OUT.parent / 'numbers.json'}")
