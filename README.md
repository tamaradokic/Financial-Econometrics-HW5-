# Dynamic Conditional Correlations during financial turmoil

**Financial Econometrics — Homework 5**
ESSEC Master in Data Science & Business Analytics, term T3 2025-26
Prof. Jeroen V.K. Rombouts

A multivariate **DCC-GARCH** analysis of five sector-diverse US stocks, with
an interactive Streamlit dashboard for exploration.

> **📄 Two-page write-up:** [`report/report.pdf`](report/report.pdf) — the
> compiled report with all key findings, methodology, figures, and the strategy verdict.
> LaTeX source: [`report/report.tex`](report/report.tex).

## What the project does

We estimate a Dynamic Conditional Correlation (DCC) GARCH model on five
sector-diverse stocks — **NVDA** (semiconductors), **TSLA** (auto),
**LLY** (pharma), **JPM** (financials), **XOM** (energy) — using ~10 years
of daily adjusted closing prices, then use the model's time-varying
covariance matrix to build a minimum-variance portfolio and compare it
against a naive equal-weight benchmark.

The 8 questions in the homework brief are answered in order in the dashboard:

1. Returns plots + descriptive stats + unconditional correlation
2. Bivariate scatterplots
3. DCC-GARCH estimation (Engle's two-step procedure)
4. Conditional volatility (σₜ) per asset
5. Dynamic conditional correlations (ρₜ) per pair — **the centrepiece**
6. Minimum-variance portfolio weights through time
7. MVP portfolio returns (mean, variance)
8. Equal-weight comparison and verdict

## Quick start

```bash
# 1. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. Run the dashboard
streamlit run app.py
```

The first run downloads ~10 years of daily prices from Yahoo Finance
(via `yfinance`), caches them to `data/prices.parquet`, then fits
GARCH(1,1) + DCC. Cached so subsequent runs are instant.

## Interactive controls

The sidebar lets you:

- **Date-range slider** — zoom into sub-periods (e.g. just COVID, just 2022).
- **Crisis-period shading** — overlay coloured bands for COVID-2020 and the
  2022 rate shock on every time-series plot.
- **DCC (a, b) what-if sliders** — override the fitted DCC parameters and
  see in real-time how correlation dynamics reshape. Try setting `b` very
  low to see what "non-persistent" correlations look like.
- **Strategy selector** — toggle which of `MVP (DCC)`, `MVP (static)`, and
  `Equal-weight` to compare in Q7/Q8.

Every chart is followed by a plain-language **"What this shows / Why it
matters"** block so the dashboard is accessible without an econometrics
background.

## Methodology in one paragraph

**Step 1 (univariate).** For each return series we fit a GARCH(1,1) with
constant mean and Gaussian innovations:
σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁. Done with the `arch` library.

**Step 2 (DCC).** Standardized residuals zₜ = εₜ/σₜ feed the DCC recursion
Qₜ = (1−a−b)Q̄ + a·zₜ₋₁zₜ₋₁' + b·Qₜ₋₁, where Q̄ is the sample covariance of
the standardized residuals (correlation targeting). The conditional
correlation matrix Rₜ is obtained by rescaling Qₜ. We estimate (a, b) by
maximizing the composite Gaussian DCC log-likelihood with a bounded
L-BFGS-B optimizer. Coded by hand in NumPy/SciPy.

The full conditional covariance matrix is Σₜ = Dₜ·Rₜ·Dₜ where
Dₜ = diag(σ₁,ₜ, …, σ₅,ₜ).

**Portfolios.** The minimum-variance weights are wₜ = Σₜ⁻¹·1 / (1ᵀΣₜ⁻¹·1).
Portfolio returns use **lagged** weights (rₚ,ₜ = wₜ₋₁ᵀ·rₜ) so there is no
look-ahead bias. The static-MVP benchmark uses the unconditional sample
covariance; the equal-weight benchmark is just 1/N at every date.

## Project layout

```
HW5/
├── Homework5_2026T3.pdf       Original assignment brief
├── app.py                     Streamlit dashboard (one-page scroll)
├── README.md                  This file
├── requirements.txt           Pinned Python dependencies
├── .gitignore
├── data/                      Cached parquet (gitignored)
├── report/
│   ├── report.tex             LaTeX source (2-page, two-column)
│   ├── report.pdf             Compiled PDF deliverable
│   ├── make_figures.py        Generates print-quality PDF figures
│   ├── figures/               Generated figures (PDF)
│   └── numbers.json           Sample stats consumed by the report
└── src/
    ├── data.py                Yahoo download + FRED T-bill + log-returns
    ├── descriptive.py         Q1 stats, unconditional ρ, Q2 pair data
    ├── garch.py               Univariate GARCH(1,1) step (uses `arch`)
    ├── dcc.py                 DCC step 2 (hand-rolled MLE + recursion)
    ├── portfolio.py           MVP, static-MVP, equal-weight (lagged weights!)
    ├── plots.py               Plotly chart helpers + crisis shading
    └── interpretations.py     Plain-language "what/why" generators
```

Every module under `src/` is a pure-function library — no Streamlit
imports — and can be smoke-tested standalone (`python -m src.garch`,
`python -m src.dcc`, etc.).
