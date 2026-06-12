"""
Streamlit dashboard — Financial Econometrics Homework 5.

Run with:    streamlit run app.py

Layout: single long-scroll page, organized Q1 -> Q8 to mirror the homework.
A sidebar holds the interactive controls.

All expensive computations (data download, GARCH fitting, DCC estimation) are
cached so the user can drag sliders without re-fitting anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.data import DEFAULT_TICKERS, TICKER_NAMES, download_riskfree_rate, load_dataset
from src.descriptive import (
    descriptive_stats,
    pairwise_data,
    unconditional_correlation,
)
from src.garch import collect_sigma, collect_std_resid, fit_all, params_table
from src.dcc import estimate_dcc, pairwise_corr_frame, replay_with_params
from src.portfolio import (
    compare_strategies,
    custom_fixed_weights,
    equal_weights,
    mvp_weights_dynamic,
    mvp_weights_static,
    portfolio_returns,
    summarize,
)
from src import plots, interpretations as interp


# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="DCC-GARCH dashboard — HW5",
    page_icon="📈",
    layout="wide",
)


# ----------------------------------------------------------------------
# Cached pipeline — each step memoized so slider-drags are instant
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Downloading prices ...")
def _cached_dataset():
    return load_dataset()


@st.cache_resource(show_spinner="Fitting univariate GARCH(1,1) ...")
def _cached_garch(returns: pd.DataFrame):
    fits = fit_all(returns)
    return fits, collect_sigma(fits), collect_std_resid(fits)


@st.cache_resource(show_spinner="Estimating DCC ...")
def _cached_dcc(_std_resid: pd.DataFrame, _sigma: pd.DataFrame):
    return estimate_dcc(_std_resid, _sigma)


@st.cache_data(show_spinner="Downloading 3-month T-bill yield from FRED ...")
def _cached_riskfree():
    return download_riskfree_rate()


# ----------------------------------------------------------------------
# Helper to print a "What this shows / Why it matters" block
# ----------------------------------------------------------------------
def _interp(what: str, why: str) -> None:
    with st.container(border=True):
        st.markdown(f"**What this shows.** {what}")
        st.markdown(f"**Why it matters.** {why}")


# ----------------------------------------------------------------------
# Sidebar — interactive controls
# ----------------------------------------------------------------------
st.sidebar.title("Controls")

st.sidebar.markdown("### View")
show_crises = st.sidebar.checkbox(
    "Shade crisis periods (COVID-2020, 2022 rate shock)", value=True,
    help="Overlay coloured bands on time-series charts to mark known turmoil windows.",
)

# Load data + run the full pipeline once.
bundle = _cached_dataset()
fits, sigma, std_resid = _cached_garch(bundle.returns)
dcc = _cached_dcc(std_resid, sigma)

st.sidebar.markdown("### Date range (display only)")
all_dates = bundle.returns.index
min_date, max_date = all_dates.min().date(), all_dates.max().date()
date_range = st.sidebar.slider(
    "Zoom into a sub-period",
    min_value=min_date, max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM",
)
start_d, end_d = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])

st.sidebar.markdown("### DCC parameters — what-if sensitivity")
with st.sidebar.expander("Wait — aren't a and b *estimated*?", expanded=False):
    st.markdown(
        f"Yes — the model has already estimated **a = {dcc.a:.3f}** and "
        f"**b = {dcc.b:.3f}** by maximum likelihood. Those are the right values.\n\n"
        "These sliders **do not re-estimate** anything. They plug alternative "
        "(a, b) into the same DCC recursion and replay the correlation series "
        "using the existing GARCH residuals and the same long-run target Q̄.\n\n"
        "The goal is **pedagogical**: build intuition for what each parameter "
        "controls, and run a sensitivity check on whether your conclusions would "
        "change if the parameters were different.\n\n"
        "- **a** = shock sensitivity. Try a = 0.20 to see correlations jolt at every shock.\n"
        "- **b** = persistence. Try b = 0.3 to see slow regimes collapse into jagged noise.\n"
        "- Both near 0 → every correlation pins to its long-run average (no dynamics)."
    )
custom_a = st.sidebar.slider(
    "DCC a (shock sensitivity)", 0.0, 0.30,
    value=float(round(dcc.a, 3)), step=0.005,
    help=f"Estimated value: {dcc.a:.4f}. Changing this does NOT refit the model; it replays the DCC recursion.",
)
custom_b = st.sidebar.slider(
    "DCC b (persistence)", 0.0, 0.999,
    value=float(round(dcc.b, 3)), step=0.005,
    help=f"Estimated value: {dcc.b:.4f}. Changing this does NOT refit the model; it replays the DCC recursion.",
)
if custom_a + custom_b >= 1.0:
    st.sidebar.error("a + b must be < 1 for the model to be stable.")
if st.sidebar.button("Reset to fitted values"):
    st.experimental_rerun()

st.sidebar.markdown("### Portfolio comparison")
show_strategies = st.sidebar.multiselect(
    "Strategies to compare in Q7–Q8",
    options=["MVP (DCC)", "MVP (static)", "Equal-weight"],
    default=["MVP (DCC)", "MVP (static)", "Equal-weight"],
    help=(
        "MVP (DCC) — weights recomputed every day from the DCC time-varying covariance. "
        "MVP (static) — same Markowitz formula but using the unconditional sample covariance "
        "(one set of weights for the whole sample). "
        "Equal-weight — naive 1/N at every date."
    ),
)

st.sidebar.markdown("### Build your own portfolio")
use_custom = st.sidebar.checkbox(
    "Include a custom portfolio in Q7–Q8", value=False,
    help="Specify fixed weights for each asset. They are normalized to sum to 1.",
)
custom_raw_weights: dict[str, float] = {}
if use_custom:
    st.sidebar.caption(
        "Move the sliders to set a fixed allocation. The values are normalized so they "
        "sum to 100%."
    )
    for t in bundle.tickers:
        custom_raw_weights[t] = st.sidebar.slider(
            f"{t} weight", 0.0, 1.0, value=0.20, step=0.01,
            help=f"Raw weight for {t} before normalization.",
        )
    total = sum(custom_raw_weights.values())
    if total > 0:
        normalized_preview = {t: w / total for t, w in custom_raw_weights.items()}
        st.sidebar.caption(
            "**Normalized weights:** "
            + ", ".join(f"{t}={w:.1%}" for t, w in normalized_preview.items())
        )
    else:
        st.sidebar.error("Total weight is zero — set at least one weight above zero.")

st.sidebar.markdown("### Risk-free rate (for Sharpe)")
with st.sidebar.expander("Which risk-free rate should I use?", expanded=False):
    st.markdown(
        "The Sharpe ratio measures **excess return per unit of risk**, defined as "
        "(annualized portfolio return − rf) / annualized portfolio volatility. The "
        "rf used should be the yield on a **truly risk-free** asset over the "
        "investor's horizon.\n\n"
        "The academic convention is the **short Treasury bill** (1-month or "
        "3-month) — *not* the 10-year Treasury bond. A long-dated bond has "
        "**interest-rate risk**: its price fluctuates when yields change, so over "
        "short horizons it is not actually risk-free. The 3-month bill's price is "
        "essentially insensitive to small yield changes over its remaining life and "
        "is the standard proxy.\n\n"
        "Over this sample (2016–2026), the 3-month T-bill averaged ~2.4% but "
        "varied widely (0% during COVID emergency cuts, 5.6% at the 2023 peak), so "
        "setting rf = 0 can materially overstate Sharpe ratios. Switching to the "
        "time-varying series is the most academically rigorous choice."
    )
rf_mode = st.sidebar.radio(
    "Convention",
    [
        "Time-varying — 3M T-bill (FRED DGS3MO)",
        "Constant annual rate",
        "Zero (rf = 0)",
    ],
    index=0,
    help="The chosen rf is used only in the Sharpe ratio column of the Q7/Q8 summary table.",
)
if rf_mode == "Constant annual rate":
    rf_constant = st.sidebar.number_input(
        "Annual rf (%)", min_value=0.0, max_value=15.0, value=2.0, step=0.25,
    )
else:
    rf_constant = None


# ----------------------------------------------------------------------
# Slice helper
# ----------------------------------------------------------------------
def _slice(df_or_series):
    return df_or_series.loc[start_d:end_d]


# ----------------------------------------------------------------------
# Resolve the effective risk-free rate from the sidebar choice.
#
# The Sharpe formula used is the standard:
#     Sharpe = (ann_mean_return - rf_eff) / ann_vol
# where rf_eff is annualized percent. For the time-varying choice, rf_eff is
# the sample mean of the daily rf series over the user's selected date range.
# This is the academically conventional "single number" Sharpe given a
# fluctuating short rate.
# ----------------------------------------------------------------------
if rf_mode.startswith("Zero"):
    rf_effective = 0.0
    rf_label = "rf = 0"
elif rf_mode.startswith("Constant"):
    rf_effective = float(rf_constant) if rf_constant is not None else 0.0
    rf_label = f"rf = {rf_effective:.2f}%"
else:
    rf_series = _cached_riskfree()
    rf_slice = rf_series.loc[start_d:end_d]
    rf_effective = float(rf_slice.mean()) if len(rf_slice) > 0 else 0.0
    rf_label = f"rf = {rf_effective:.2f}% (3M T-bill avg)"


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("Dynamic Conditional Correlations during financial turmoil")
st.markdown(
    "**Financial Econometrics — Homework 5.** This dashboard estimates a "
    "**DCC-GARCH** model on five sector-diverse US stocks and compares the "
    "implied minimum-variance portfolio to a naive equal-weight benchmark."
)
st.markdown(
    f"**Sample:** {bundle.start.date()} → {bundle.end.date()} "
    f"({len(bundle.returns):,} trading days). **Universe:** "
    + ", ".join(f"`{t}` ({TICKER_NAMES[t]})" for t in bundle.tickers)
    + "."
)
st.divider()


# ----------------------------------------------------------------------
# Q1 — Returns, descriptive stats, unconditional correlation
# ----------------------------------------------------------------------
st.header("Q1 — Returns, descriptive statistics, unconditional correlation")

st.subheader("Daily returns")
st.plotly_chart(
    plots.returns_lines(_slice(bundle.returns), show_crises=show_crises),
    use_container_width=True,
)
_interp(*interp.returns_block(_slice(bundle.returns)))

st.subheader("Descriptive statistics")
stats = descriptive_stats(_slice(bundle.returns))
st.dataframe(
    stats.style.format(
        {
            "Mean": "{:.3f}", "Std. Dev.": "{:.3f}", "Min": "{:.2f}",
            "Max": "{:.2f}", "Skewness": "{:.2f}", "Excess Kurtosis": "{:.2f}",
            "JB p-value": "{:.3f}", "ARCH-LM(10) p-value": "{:.3f}",
            "Ljung-Box(r^2, 10) p-value": "{:.3f}", "N obs": "{:,}",
        }
    ),
    use_container_width=True,
)
_interp(*interp.descriptive_stats_block(stats))

st.subheader("Unconditional correlation matrix")
corr = unconditional_correlation(_slice(bundle.returns))
st.plotly_chart(plots.correlation_heatmap(corr), use_container_width=True)
_interp(*interp.unconditional_correlation_block(corr))

st.divider()


# ----------------------------------------------------------------------
# Q2 — Bivariate scatterplots
# ----------------------------------------------------------------------
st.header("Q2 — Bivariate scatterplots")
st.plotly_chart(plots.scatter_matrix(_slice(bundle.returns)), use_container_width=True)
_interp(*interp.scatter_block())
st.divider()


# ----------------------------------------------------------------------
# Q3 — DCC-GARCH estimation
# ----------------------------------------------------------------------
st.header("Q3 — DCC-GARCH estimation")
st.markdown(
    "We follow Engle's two-step procedure: **(1)** fit a univariate "
    "**GARCH(1,1)** to each return series, **(2)** fit the **DCC** dynamics on "
    "the standardized residuals from step 1."
)

with st.expander("Univariate GARCH(1,1) parameters per stock", expanded=False):
    st.dataframe(
        params_table(fits).style.format("{:.4f}"),
        use_container_width=True,
    )
    st.caption(
        "Each model has the same form: r_t = μ + σ_t · z_t with "
        "σ²_t = ω + α · ε²_{t-1} + β · σ²_{t-1}. "
        "The persistence α+β is close to 1 for all stocks, which is typical for daily equity data."
    )

st.subheader("DCC parameters")
col1, col2, col3 = st.columns(3)
col1.metric("DCC a (shock sensitivity)", f"{dcc.a:.4f}")
col2.metric("DCC b (persistence)", f"{dcc.b:.4f}")
col3.metric("a + b (stationarity)", f"{dcc.a + dcc.b:.4f}", help="Must be < 1 for the process to be stable.")
_interp(*interp.dcc_params_block(dcc.a, dcc.b))
st.divider()


# ----------------------------------------------------------------------
# Q4 — Conditional standard deviations
# ----------------------------------------------------------------------
st.header("Q4 — Conditional volatility")
st.plotly_chart(
    plots.sigma_lines(_slice(sigma), show_crises=show_crises),
    use_container_width=True,
)
_interp(*interp.sigma_block(_slice(sigma)))
st.divider()


# ----------------------------------------------------------------------
# Q5 — Conditional correlations (with the what-if sliders)
# ----------------------------------------------------------------------
st.header("Q5 — Dynamic conditional correlations")

st.markdown(
    "Each line is the model-implied correlation between one pair of stocks, "
    "at each point in time."
)

with st.expander("About the sidebar (a, b) sliders — what does it mean to change them?", expanded=False):
    st.markdown(
        f"The DCC parameters a and b are **estimated** by the model — the values shown "
        f"in the metrics above (**a = {dcc.a:.3f}**, **b = {dcc.b:.3f}**) are the maximum-likelihood "
        f"estimates and are the right values to use for analysis.\n\n"
        f"The sliders in the sidebar do **not** re-estimate the model. They replay the same "
        f"DCC recursion at alternative (a, b), keeping the GARCH residuals and the long-run "
        f"target Q̄ fixed. The result you see when you move the slider is a *counterfactual*: "
        f"what the correlation dynamics **would** look like under different parameters.\n\n"
        f"This is useful for **two** reasons:\n\n"
        f"1. **Intuition.** a controls how strongly correlations react to new shocks; b controls "
        f"how persistent correlation regimes are. Watching the plot reshape as you drag b from "
        f"~0.95 down to ~0.3 makes \"persistence\" concrete — slow regimes become jagged noise.\n"
        f"2. **Robustness.** If the conclusions you draw about portfolio behaviour are sensitive "
        f"to small changes in (a, b), that is a warning. If they are stable, you have evidence "
        f"that the analysis is not fragile."
    )

# Reconstruct R_t at the (possibly user-overridden) parameters.
if custom_a + custom_b < 1.0:
    R_user, _ = replay_with_params(dcc, std_resid, sigma, custom_a, custom_b)
    rho_df = pairwise_corr_frame(R_user, dcc.dates, dcc.tickers)
    if abs(custom_a - dcc.a) > 1e-4 or abs(custom_b - dcc.b) > 1e-4:
        st.info(
            f"⚠️ **Counterfactual view:** showing the DCC recursion replayed with "
            f"**a = {custom_a:.3f}, b = {custom_b:.3f}** "
            f"(estimated values: a = {dcc.a:.3f}, b = {dcc.b:.3f}). "
            f"Reset the sliders to compare against the fitted model."
        )
else:
    rho_df = pairwise_corr_frame(dcc.R_t, dcc.dates, dcc.tickers)

st.plotly_chart(
    plots.correlation_lines(_slice(rho_df), show_crises=show_crises),
    use_container_width=True,
)
_interp(*interp.rho_block(_slice(rho_df)))
st.divider()


# ----------------------------------------------------------------------
# Q6 — Minimum-variance portfolio weights
# ----------------------------------------------------------------------
st.header("Q6 — Minimum-variance portfolio weights")

st.markdown(
    "The closed-form minimum-variance allocation is "
    "**wₜ* = Σₜ⁻¹·1 / (1ᵀΣₜ⁻¹·1)**. Substituting the DCC time-varying conditional "
    "covariance matrix Σₜ produces an allocation that is recomputed at each date, "
    "tracking changes in conditional volatility and correlation. Substituting the "
    "unconditional sample covariance Σ̂ yields a single fixed allocation — the classical "
    "Markowitz (1952) MVP."
)

mvp_w = mvp_weights_dynamic(dcc.Sigma_t, dcc.dates, dcc.tickers)
mvp_static_w_df = mvp_weights_static(bundle.returns.loc[dcc.dates])
mvp_static_w_vec = mvp_static_w_df.iloc[0]  # same weights every day; take the first row

st.subheader("Time series of MVP (DCC) weights with MVP (static) reference")
st.plotly_chart(
    plots.weights_lines_with_static_overlay(
        _slice(mvp_w), mvp_static_w_vec, show_crises=show_crises,
        title="MVP (DCC) weights — solid; MVP (static) reference — dashed",
    ),
    use_container_width=True,
)
_interp(*interp.weights_block(_slice(mvp_w)))

st.subheader("Average allocation comparison: MVP (DCC) vs MVP (static) vs Equal-weight")
sliced_mvp_w = _slice(mvp_w)
weights_compare = {
    "MVP (DCC) — average": sliced_mvp_w.mean(),
    "MVP (static)": mvp_static_w_vec,
    "Equal-weight (1/N)": pd.Series({t: 1.0 / len(bundle.tickers) for t in bundle.tickers}),
}
st.plotly_chart(plots.weights_comparison_bar(weights_compare), use_container_width=True)
_interp(*interp.weights_comparison_block(sliced_mvp_w, mvp_static_w_vec))
st.divider()


# ----------------------------------------------------------------------
# Q7 + Q8 — Portfolio comparison
# ----------------------------------------------------------------------
st.header("Q7 & Q8 — Portfolio comparison: MVP vs equal-weight (and static MVP)")

with st.expander("What is the difference between the three strategies?", expanded=True):
    st.markdown(
        "All three strategies hold the same five stocks. The only thing that differs is **how the "
        "weights are chosen**.\n\n"
        "| Strategy | Covariance used | Weights change over time? | What it represents |\n"
        "|---|---|---|---|\n"
        "| **MVP (DCC)** | Time-varying Σₜ from DCC | ✅ Yes — every day | The optimal risk-minimizing portfolio if you take time-varying volatility and correlation seriously |\n"
        "| **MVP (static)** | Unconditional Σ̂ from the full sample | ❌ No — same weights every day | Classical Markowitz (1952) — pretends correlations and vols never change |\n"
        "| **Equal-weight** | None | ❌ No — fixed at 1/N | Naive benchmark; no estimation, no model |\n\n"
        "**Why this comparison matters.** The gap between MVP (DCC) and MVP (static) tells you "
        "**how much the dynamic part of the DCC model is actually contributing** to risk reduction. "
        "If they had the same realized variance, the DCC dynamics would add nothing on top of the "
        "static covariance. If MVP (DCC) achieves a strictly lower variance, you have direct "
        "empirical evidence that modelling time-variation in correlations is worth the effort."
    )

# Build the three benchmark strategies on the full sample, then slice.
results = compare_strategies(dcc.Sigma_t, dcc.dates, dcc.tickers, bundle.returns)

# Filter to user-selected strategies and slice to the user-selected date range.
returns_by_strategy: dict[str, pd.Series] = {
    name: _slice(r.returns) for name, r in results.items() if name in show_strategies
}

# Append the user's custom portfolio (if enabled and well-defined).
custom_normalized: dict[str, float] | None = None
if use_custom and sum(custom_raw_weights.values()) > 0:
    custom_w_df = custom_fixed_weights(bundle.returns.loc[dcc.dates], custom_raw_weights)
    custom_returns_full = portfolio_returns(custom_w_df, bundle.returns.loc[dcc.dates])
    custom_returns = _slice(custom_returns_full)
    custom_normalized = {t: w for t, w in zip(custom_w_df.columns, custom_w_df.iloc[0].values)}
    returns_by_strategy["Custom"] = custom_returns

if not returns_by_strategy:
    st.warning("Select at least one strategy in the sidebar to display.")
else:
    st.subheader("Cumulative growth of \\$1 invested")
    st.plotly_chart(
        plots.cumulative_returns_lines(returns_by_strategy, show_crises=show_crises),
        use_container_width=True,
    )

    st.subheader("Daily portfolio returns")
    st.plotly_chart(
        plots.portfolio_returns_lines(returns_by_strategy, show_crises=show_crises),
        use_container_width=True,
    )

    st.subheader("Peak-to-trough drawdown")
    st.plotly_chart(
        plots.drawdown_lines(returns_by_strategy, show_crises=show_crises),
        use_container_width=True,
    )
    _interp(*interp.drawdown_block({name: plots.drawdown_series(r) for name, r in returns_by_strategy.items()}))

    # Summary table — recomputed on the sliced returns so the user's date range matters.
    sharpe_col = f"Sharpe ({rf_label})"
    rows = []
    for name, r in returns_by_strategy.items():
        mean = float(r.mean())
        var = float(r.var(ddof=1))
        std = float(r.std(ddof=1))
        ann_mean = mean * 252.0
        ann_std = std * np.sqrt(252.0)
        # Sharpe = (annualized return - rf_effective) / annualized vol
        sharpe = (ann_mean - rf_effective) / ann_std if ann_std > 0 else np.nan
        rows.append({
            "Strategy": name,
            "Mean (% / day)": mean,
            "Std (% / day)": std,
            "Variance": var,
            "Ann. Return (%)": ann_mean,
            "Ann. Volatility (%)": ann_std,
            sharpe_col: sharpe,
        })
    summary = pd.DataFrame(rows).set_index("Strategy")
    st.caption(
        f"Sharpe ratio computed against **{rf_label}** "
        f"(set in the sidebar). The annualized risk-free rate is subtracted from the "
        "annualized portfolio return before dividing by the annualized volatility."
    )
    st.dataframe(
        summary.style.format({
            "Mean (% / day)": "{:.4f}", "Std (% / day)": "{:.4f}",
            "Variance": "{:.4f}", "Ann. Return (%)": "{:.2f}",
            "Ann. Volatility (%)": "{:.2f}", sharpe_col: "{:.3f}",
        }),
        use_container_width=True,
    )
    _interp(*interp.strategy_comparison_block(summary))

    # Dedicated commentary on the custom portfolio if it is in the comparison.
    if custom_normalized is not None and "Custom" in returns_by_strategy:
        st.subheader("Diagnostics for the custom portfolio")
        custom_row = summary.loc["Custom"]
        summary_dict = {
            "mean": custom_row["Mean (% / day)"],
            "std": custom_row["Std (% / day)"],
            "sharpe": custom_row[sharpe_col],
        }
        _interp(*interp.custom_portfolio_block(custom_normalized, summary_dict))

st.divider()


# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
st.caption(
    "Two-step DCC-GARCH(1,1), Gaussian innovations, normal-distribution log-likelihood. "
    "Univariate step fit with `arch`; DCC step hand-rolled with NumPy/SciPy. "
    "Portfolio returns use lagged weights (no look-ahead). "
    "Made for ESSEC Financial Econometrics, Prof. J. Rombouts."
)
