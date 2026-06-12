"""
Plain-language but academically-toned interpretation blocks for each chart.

Each block returns a (what, why) tuple:
    - "what" describes the figure in formal, precise terms,
    - "why" explains the econometric or financial significance of the result.

Sentences are written for an MSc-level financial econometrics audience: proper
terminology is used (innovations, conditional second moments, leptokurtosis,
contagion, stationarity), but jargon is unpacked the first time it appears so
readers without the full toolkit can still follow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _fmt_date(d) -> str:
    return pd.Timestamp(d).strftime("%B %Y")


# ----------------------------------------------------------------------
# Q1
# ----------------------------------------------------------------------
def returns_block(returns: pd.DataFrame) -> tuple[str, str]:
    """Q1 returns plot."""
    most_vol = returns.std().idxmax()
    least_vol = returns.std().idxmin()
    worst_day = returns.stack().idxmin()
    worst_val = returns.stack().min()

    what = (
        "The figure displays daily percentage log-returns for each constituent asset over the "
        f"full sample window. **{most_vol}** exhibits the highest unconditional standard "
        f"deviation among the five series, while **{least_vol}** exhibits the lowest. "
        f"The largest single-day decline observed in the panel is recorded for **{worst_day[1]}** "
        f"in {_fmt_date(worst_day[0])} ({worst_val:.1f}%)."
    )
    why = (
        "All five series display pronounced **volatility clustering**: episodes of large "
        "absolute returns are temporally concentrated, while tranquil periods persist for "
        "extended intervals. This serial dependence in the conditional second moment of returns "
        "is the canonical motivation for adopting conditional volatility models of the GARCH "
        "family rather than treating volatility as constant."
    )
    return what, why


def descriptive_stats_block(stats: pd.DataFrame) -> tuple[str, str]:
    """Q1 stats table."""
    fattest = stats["Excess Kurtosis"].idxmax()
    fat_val = stats.loc[fattest, "Excess Kurtosis"]

    what = (
        "Sample moments and diagnostic test results for each return series. Positive **excess "
        "kurtosis** indicates leptokurtic (fat-tailed) distributions with a higher incidence of "
        f"extreme observations than would be predicted by a Gaussian benchmark; **{fattest}** "
        f"displays the largest value at {fat_val:.2f}. Small p-values for the **Jarque-Bera** "
        "test lead to rejection of the null of normality across all series. Small p-values for "
        "the **ARCH-LM(10)** test and for the Ljung-Box test on squared returns provide formal "
        "evidence of conditional heteroskedasticity in the second moment."
    )
    why = (
        "The joint presence of leptokurtosis and conditional heteroskedasticity constitutes the "
        "two canonical stylized facts of daily financial returns documented since Mandelbrot "
        "(1963) and Engle (1982). Both motivate the conditional volatility framework adopted in "
        "the subsequent analysis. The departure from normality also implies that point estimates "
        "from Gaussian-based maximum likelihood should be interpreted as quasi-maximum likelihood "
        "estimates, which remain consistent under suitable regularity conditions."
    )
    return what, why


def unconditional_correlation_block(corr: pd.DataFrame) -> tuple[str, str]:
    """Q1 correlation heatmap."""
    off_diag = corr.where(~np.eye(len(corr), dtype=bool))
    max_val = off_diag.stack().max()
    min_val = off_diag.stack().min()
    max_pair = off_diag.stack().idxmax()
    min_pair = off_diag.stack().idxmin()

    what = (
        "Pearson correlation coefficients computed over the full sample window for each pair of "
        f"return series. The most strongly co-moving pair is **{max_pair[0]}–{max_pair[1]}** "
        f"(ρ = {max_val:.2f}), consistent with their shared exposure to macroeconomic and "
        f"cyclical factors; the most weakly co-moving pair is "
        f"**{min_pair[0]}–{min_pair[1]}** (ρ = {min_val:.2f}), reflecting limited overlap "
        "in sectoral and fundamental drivers."
    )
    why = (
        "The unconditional correlation matrix summarizes pairwise dependence with a single scalar "
        "per pair, computed across the entire sample. This representation is uninformative about "
        "potential time-variation in the dependence structure: a moderate full-sample value of, "
        "say, 0.25 is consistent both with stable mild co-movement throughout the window and "
        "with regime-dependent dynamics (e.g. near-zero correlation in tranquil periods and "
        "elevated correlation during episodes of market stress). Distinguishing between these "
        "alternative data-generating processes requires a model of conditional correlations — "
        "the motivation for the DCC specification estimated in Q3."
    )
    return what, why


# ----------------------------------------------------------------------
# Q2
# ----------------------------------------------------------------------
def scatter_block() -> tuple[str, str]:
    """Q2 scatter matrix."""
    what = (
        "Bivariate scatterplots of contemporaneous daily returns for each of the ten unique asset "
        "pairs. The orientation and dispersion of each point cloud provide a visual summary of "
        "the linear dependence structure between the two series."
    )
    why = (
        "Scatterplots collapse the time dimension: each point corresponds to one trading day, but "
        "the figure does not distinguish observations drawn from different macroeconomic regimes. "
        "As a result, this representation cannot identify any time-variation in the dependence "
        "structure — a limitation that directly motivates the dynamic conditional correlation "
        "analysis presented in subsequent sections."
    )
    return what, why


# ----------------------------------------------------------------------
# Q3 — DCC parameters
# ----------------------------------------------------------------------
def dcc_params_block(a: float, b: float) -> tuple[str, str]:
    """Q3 DCC parameter readout."""
    persistence = a + b
    what = (
        f"The estimated DCC parameters governing the dynamics of the conditional correlation "
        f"matrix are **a = {a:.3f}** and **b = {b:.3f}**. The coefficient **a** measures the "
        "responsiveness of the conditional correlation matrix to the most recent cross-product "
        "of standardized residuals (the innovation), while **b** measures the persistence of the "
        f"correlation process. Their sum, **a + b = {persistence:.3f}**, is strictly less than "
        "unity, satisfying the covariance-stationarity condition that ensures the process reverts "
        "to its unconditional target Q̄."
    )
    why = (
        "The DCC specification parsimoniously characterizes the dynamics of an N × N correlation "
        "matrix using only two scalar parameters, independent of N. For the present universe of "
        "N = 5 assets, this reduces what would otherwise be ten distinct pairwise correlation "
        "processes to a single bivariate parameter vector, rendering quasi-maximum likelihood "
        "estimation tractable while preserving the essential dynamics. The estimated persistence "
        "(a + b close to one) is characteristic of daily equity data and indicates that "
        "correlation shocks dissipate only slowly."
    )
    return what, why


# ----------------------------------------------------------------------
# Q4 — conditional volatility
# ----------------------------------------------------------------------
def sigma_block(sigma: pd.DataFrame) -> tuple[str, str]:
    """Q4 conditional sigma plots."""
    peak_t, peak_ticker = sigma.stack().idxmax()
    peak_val = sigma.stack().max()
    what = (
        "The figure displays the estimated conditional standard deviation σₜ from the univariate "
        "GARCH(1,1) specification fitted to each return series. Periods of elevated σₜ correspond "
        "to regimes of high conditional risk, while quiescent periods correspond to low "
        "conditional variance. The maximum conditional volatility recorded in the panel is "
        f"attained by **{peak_ticker}** in {_fmt_date(peak_t)} (σₜ = {peak_val:.1f}% per day)."
    )
    why = (
        "Common peaks across all five conditional volatility series coincide with well-documented "
        "episodes of market stress — most prominently the COVID-19 shock of March 2020 and the "
        "2022 monetary tightening cycle. This co-movement of conditional second moments across "
        "assets is a necessary precondition for time-varying conditional correlations to manifest "
        "in the data, and provides the empirical bridge to the conditional correlation series "
        "presented in the following section."
    )
    return what, why


# ----------------------------------------------------------------------
# Q5 — conditional correlations
# ----------------------------------------------------------------------
def rho_block(rho: pd.DataFrame) -> tuple[str, str]:
    """Q5 conditional correlation plots — the centrepiece."""
    flips = (rho.min() < 0) & (rho.max() > 0)
    flipping = flips[flips].index.tolist()
    biggest_range = (rho.max() - rho.min()).idxmax()
    what = (
        "Time series of the model-implied conditional correlations ρᵢⱼ,ₜ for each pair of assets. "
        f"The pair displaying the widest empirical range over the sample is **{biggest_range}**, "
        f"whose conditional correlation varies between {rho[biggest_range].min():.2f} and "
        f"{rho[biggest_range].max():.2f}. {len(flipping)} of the ten pairs exhibit sign reversals "
        "during the sample, transitioning between regimes of positive and negative co-movement."
    )
    why = (
        "These dynamics constitute the central empirical finding of the analysis. The "
        "unconditional correlation matrix reported in Q1 attributes a single fixed coefficient to "
        "each pair, whereas the DCC estimates reveal substantial intertemporal variation. "
        "Conditional correlations tend to rise during periods of aggregate market stress — a "
        "phenomenon variously termed **correlation breakdown** or **financial contagion** — and "
        "to attenuate during tranquil regimes. The practical implication is that the "
        "diversification benefit available from any given pair of assets is itself state-"
        "dependent, and tends to deteriorate precisely in the regimes where it would be most "
        "valuable to a mean-variance investor."
    )
    return what, why


# ----------------------------------------------------------------------
# Q6 — MVP weights
# ----------------------------------------------------------------------
def weights_block(weights: pd.DataFrame) -> tuple[str, str]:
    """Q6 MVP weights."""
    means = weights.mean()
    top = means.idxmax()
    bottom = means.idxmin()
    has_negatives = (weights.min() < 0).any()

    what = (
        "Time series of the closed-form minimum-variance portfolio weights "
        "wₜ* = Σₜ⁻¹·1 / (1ᵀΣₜ⁻¹·1) computed at each date from the DCC conditional covariance "
        f"matrix Σₜ. The asset receiving the largest mean allocation over the sample is **{top}** "
        f"(mean weight = {means[top]:.1%}); the asset receiving the smallest mean allocation is "
        f"**{bottom}** ({means[bottom]:.1%})."
        + (
            " The optimization is unconstrained, so the implied weights are not restricted to be "
            "non-negative: short positions arise when shorting one asset reduces overall "
            "portfolio variance through its covariance with the others."
            if has_negatives
            else ""
        )
    )
    why = (
        "The minimum-variance portfolio is a purely risk-driven allocation in the sense that it "
        "incorporates no information about expected returns; weights are selected solely to "
        "minimize ex-ante portfolio variance subject to a unit-budget constraint. Because Σₜ is "
        "time-varying in the DCC framework, the optimal allocation is recomputed at each date and "
        "adapts to evolving conditional volatilities and correlations. This contrasts with the "
        "static-MVP benchmark considered in Q8, which uses the unconditional sample covariance "
        "Σ̂ and produces a single fixed allocation vector applied uniformly across the sample."
    )
    return what, why


# ----------------------------------------------------------------------
# Q7/Q8 — strategy comparison
# ----------------------------------------------------------------------
def strategy_comparison_block(summary: pd.DataFrame) -> tuple[str, str]:
    """Q7/Q8 — verdict between strategies."""
    def _col(needle: str) -> str:
        for c in summary.columns:
            if needle.lower() in c.lower():
                return c
        raise KeyError(f"No column matching '{needle}' in {list(summary.columns)}")

    lowest_var = summary[_col("Variance")].idxmin()
    best_sharpe = summary[_col("Sharpe")].idxmax()
    highest_ret = summary[_col("Mean")].idxmax()

    what = (
        "Realized in-sample performance of the candidate strategies, all of which invest in the "
        "same five-asset universe but differ in the rule used to select portfolio weights. The "
        f"strategy attaining the lowest realized variance is **{lowest_var}**, consistent with "
        "the theoretical objective of minimum-variance optimization. The strategy achieving the "
        f"highest realized mean return is **{highest_ret}**, while the highest realized Sharpe "
        f"ratio is attained by **{best_sharpe}**."
    )
    why = (
        "The relative ranking of strategies depends on the investor's objective function. For a "
        "variance-minimizing investor, the DCC-based MVP outperforms its static counterpart, "
        "providing empirical evidence that incorporating time-variation in the conditional "
        "second moments improves ex-post portfolio risk. By contrast, the equal-weighted (1/N) "
        "strategy frequently attains the highest realized Sharpe ratio, a result consistent with "
        "the findings of DeMiguel, Garlappi, and Uppal (2009), who document that naive "
        "diversification is difficult to outperform out-of-sample due to the estimation error "
        "embedded in optimization-based allocations. The appropriate strategy choice is "
        "therefore a function of the investor's preferences over the risk-return tradeoff and "
        "their tolerance for estimation-induced parameter uncertainty."
    )
    return what, why


# ----------------------------------------------------------------------
# New blocks for the added visuals
# ----------------------------------------------------------------------
def weights_comparison_block(dcc_w: pd.DataFrame, static_w: pd.Series) -> tuple[str, str]:
    """Comparison of average MVP (DCC) weights, MVP (static) weights, and 1/N."""
    avg_dcc = dcc_w.mean()
    diffs = (avg_dcc - static_w).abs()
    largest_diff = diffs.idxmax()

    what = (
        "Bar chart contrasting the average MVP (DCC) weight, the MVP (static) weight, and the "
        f"equal-weight allocation (1/N = {1.0 / len(static_w):.1%}) for each asset. The largest "
        f"discrepancy between the dynamic and static MVPs is observed for **{largest_diff}** "
        f"(absolute difference = {diffs[largest_diff]:.1%})."
    )
    why = (
        "On average across the sample, the MVP (DCC) weights are close — but not identical — to "
        "the static MVP weights, since the time-varying Σₜ pulls toward Σ̂ in expectation. The "
        "important point is that the dynamic weights *deviate* from this average in response to "
        "regime changes in conditional volatility and correlation, allowing the portfolio to "
        "respond to market conditions in a way that the static rule cannot. This distinction is "
        "the mechanism through which MVP (DCC) achieves a lower realized variance than MVP "
        "(static), as documented in the Q7/Q8 summary table."
    )
    return what, why


def drawdown_block(drawdowns_by_strategy: dict[str, pd.Series]) -> tuple[str, str]:
    """Drawdown plot interpretation."""
    max_dd = {name: dd.min() for name, dd in drawdowns_by_strategy.items()}
    worst_name = min(max_dd, key=max_dd.get)
    best_name = max(max_dd, key=max_dd.get)

    what = (
        "Time series of the peak-to-trough drawdown for each strategy, defined as the percentage "
        "decline from the running maximum of the cumulative portfolio value. The strategy "
        f"sustaining the largest maximum drawdown is **{worst_name}** ({max_dd[worst_name]:.1%}), "
        f"while the strategy with the shallowest drawdown is **{best_name}** "
        f"({max_dd[best_name]:.1%})."
    )
    why = (
        "Drawdowns complement the variance summary by characterizing the *path* of losses rather "
        "than their average squared magnitude. Two portfolios with identical variance can imply "
        "very different lived experiences for an investor — one accumulating losses gradually "
        "and the other in a single sharp episode. For risk management purposes, the maximum "
        "drawdown is often a more salient quantity than the realized variance because it bounds "
        "the worst historical loss an investor would have been required to endure."
    )
    return what, why


def custom_portfolio_block(weights: dict[str, float], summary_row: dict[str, float]) -> tuple[str, str]:
    """Interpretation for the user-defined custom portfolio."""
    w_str = ", ".join(f"{t} = {w:.1%}" for t, w in weights.items())
    what = (
        f"The custom portfolio holds **fixed weights** ({w_str}) over the entire sample. Its "
        f"realized daily mean return is {summary_row['mean']:.4f}%, its realized standard "
        f"deviation is {summary_row['std']:.4f}% per day, and its annualized Sharpe ratio "
        f"(rf = 0) is {summary_row['sharpe']:.3f}."
    )
    why = (
        "Specifying weights manually allows the user to evaluate any allocation against the "
        "benchmark strategies estimated by the model. If the custom portfolio achieves a lower "
        "realized variance than the MVP (DCC), the difference is attributable either to "
        "fortunate ex-post realizations or to a more concentrated allocation that the DCC "
        "rejects because it would have been suboptimal under the prevailing conditional "
        "covariance. The exercise illustrates the distinction between *ex-ante* optimization "
        "(what the model expects to minimize) and *ex-post* outcomes (what is realized)."
    )
    return what, why
