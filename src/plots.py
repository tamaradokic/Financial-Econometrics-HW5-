"""
Plotly chart helpers shared across the Streamlit dashboard.

Two design principles:
  - Every time-series plot supports an optional crisis-shading overlay
    (COVID-2020 and the 2022 rate shock), keyed off the sidebar toggles.
  - Every figure uses a consistent colorway and minimal Plotly chrome so
    the dashboard reads cleanly to non-finance audiences.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Crisis windows used by the optional shading overlay. Edit here to add more.
@dataclass(frozen=True)
class CrisisBand:
    name: str
    start: str        # ISO date
    end: str
    color: str        # rgba string for Plotly

CRISIS_BANDS: tuple[CrisisBand, ...] = (
    CrisisBand("COVID-19 crash & recovery", "2020-02-15", "2020-06-30", "rgba(231, 76, 60, 0.12)"),
    CrisisBand("2022 rate shock (inflation / Fed tightening)", "2022-01-01", "2022-12-31", "rgba(241, 196, 15, 0.12)"),
)

# Consistent colorway used across asset / pair plots.
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def _add_crisis_bands(fig: go.Figure, show: bool) -> None:
    """Overlay shaded crisis bands on every x-axis of the figure if requested."""
    if not show:
        return
    for band in CRISIS_BANDS:
        fig.add_vrect(
            x0=band.start, x1=band.end,
            fillcolor=band.color, opacity=1.0,
            layer="below", line_width=0,
            annotation_text=band.name, annotation_position="top left",
            annotation_font_size=10,
        )


def returns_lines(returns: pd.DataFrame, show_crises: bool = False) -> go.Figure:
    """Q1: small-multiples-style returns plot, one row per ticker."""
    n = returns.shape[1]
    fig = make_subplots(
        rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.025,
        subplot_titles=[f"{t} — daily log return (%)" for t in returns.columns],
    )
    for i, t in enumerate(returns.columns):
        fig.add_trace(
            go.Scatter(x=returns.index, y=returns[t], mode="lines",
                       line=dict(width=0.8, color=PALETTE[i % len(PALETTE)]),
                       showlegend=False, name=t),
            row=i + 1, col=1,
        )
    fig.update_layout(height=180 * n, margin=dict(l=40, r=20, t=40, b=30), template="simple_white")
    _add_crisis_bands(fig, show_crises)
    return fig


def correlation_heatmap(corr: pd.DataFrame, title: str = "Unconditional correlation") -> go.Figure:
    """Q1: heatmap of an N x N correlation matrix."""
    fig = px.imshow(
        corr.round(3),
        x=corr.columns, y=corr.index,
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        text_auto=True, aspect="equal", title=title,
    )
    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40), template="simple_white")
    return fig


def scatter_matrix(returns: pd.DataFrame) -> go.Figure:
    """Q2: bivariate scatterplots — Plotly's scatter_matrix on the return data."""
    fig = px.scatter_matrix(
        returns, dimensions=list(returns.columns),
        opacity=0.4,
    )
    fig.update_traces(diagonal_visible=False, showupperhalf=False, marker=dict(size=3))
    fig.update_layout(
        height=720, template="simple_white",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def sigma_lines(sigma: pd.DataFrame, show_crises: bool = False) -> go.Figure:
    """Q4: conditional standard deviation, one panel per ticker."""
    n = sigma.shape[1]
    fig = make_subplots(
        rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.025,
        subplot_titles=[f"{t} — conditional std. dev. σₜ (%)" for t in sigma.columns],
    )
    for i, t in enumerate(sigma.columns):
        fig.add_trace(
            go.Scatter(x=sigma.index, y=sigma[t], mode="lines",
                       line=dict(width=1.4, color=PALETTE[i % len(PALETTE)]),
                       showlegend=False, name=t),
            row=i + 1, col=1,
        )
    fig.update_layout(height=180 * n, margin=dict(l=40, r=20, t=40, b=30), template="simple_white")
    _add_crisis_bands(fig, show_crises)
    return fig


def correlation_lines(rho: pd.DataFrame, show_crises: bool = False, highlight: list[str] | None = None) -> go.Figure:
    """Q5: all pairwise conditional correlations on one chart."""
    fig = go.Figure()
    for i, col in enumerate(rho.columns):
        emphasized = (highlight is None) or (col in highlight)
        fig.add_trace(
            go.Scatter(
                x=rho.index, y=rho[col], mode="lines", name=col,
                line=dict(width=1.5 if emphasized else 0.8,
                          color=PALETTE[i % len(PALETTE)]),
                opacity=1.0 if emphasized else 0.35,
            )
        )
    fig.add_hline(y=0.0, line=dict(width=1, dash="dot", color="grey"))
    fig.update_layout(
        height=520, template="simple_white",
        margin=dict(l=40, r=20, t=40, b=40),
        yaxis_title="Conditional correlation ρₜ",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
    )
    _add_crisis_bands(fig, show_crises)
    return fig


def weights_area(weights: pd.DataFrame, show_crises: bool = False, title: str = "MVP weights") -> go.Figure:
    """Q6: stacked area chart of portfolio weights over time."""
    fig = go.Figure()
    for i, col in enumerate(weights.columns):
        fig.add_trace(
            go.Scatter(
                x=weights.index, y=weights[col], mode="lines", name=col,
                stackgroup="one",
                line=dict(width=0.5, color=PALETTE[i % len(PALETTE)]),
            )
        )
    fig.update_layout(
        height=420, template="simple_white", title=title,
        margin=dict(l=40, r=20, t=50, b=40),
        yaxis_title="Weight",
    )
    _add_crisis_bands(fig, show_crises)
    return fig


def weights_lines(weights: pd.DataFrame, show_crises: bool = False, title: str = "MVP weights") -> go.Figure:
    """Q6 (alternative): line chart of weights — better when weights go negative."""
    fig = go.Figure()
    for i, col in enumerate(weights.columns):
        fig.add_trace(
            go.Scatter(
                x=weights.index, y=weights[col], mode="lines", name=col,
                line=dict(width=1.4, color=PALETTE[i % len(PALETTE)]),
            )
        )
    fig.add_hline(y=0.0, line=dict(width=1, dash="dot", color="grey"))
    fig.update_layout(
        height=420, template="simple_white", title=title,
        margin=dict(l=40, r=20, t=50, b=40),
        yaxis_title="Weight",
        legend=dict(orientation="h", yanchor="bottom", y=-0.18),
    )
    _add_crisis_bands(fig, show_crises)
    return fig


def cumulative_returns_lines(returns_by_strategy: dict[str, pd.Series], show_crises: bool = False) -> go.Figure:
    """Q7/Q8: cumulative growth of $1 invested in each strategy."""
    fig = go.Figure()
    for i, (name, r) in enumerate(returns_by_strategy.items()):
        cum = (1.0 + r / 100.0).cumprod()   # r is in percent
        fig.add_trace(
            go.Scatter(
                x=cum.index, y=cum, mode="lines", name=name,
                line=dict(width=2, color=PALETTE[i % len(PALETTE)]),
            )
        )
    fig.update_layout(
        height=480, template="simple_white",
        margin=dict(l=40, r=20, t=40, b=40),
        yaxis_title="Cumulative growth of $1",
        legend=dict(orientation="h", yanchor="bottom", y=-0.18),
    )
    _add_crisis_bands(fig, show_crises)
    return fig


def portfolio_returns_lines(returns_by_strategy: dict[str, pd.Series], show_crises: bool = False) -> go.Figure:
    """Q7/Q8: raw daily portfolio returns side-by-side."""
    fig = go.Figure()
    for i, (name, r) in enumerate(returns_by_strategy.items()):
        fig.add_trace(
            go.Scatter(
                x=r.index, y=r, mode="lines", name=name,
                line=dict(width=0.7, color=PALETTE[i % len(PALETTE)]),
                opacity=0.85,
            )
        )
    fig.update_layout(
        height=360, template="simple_white",
        margin=dict(l=40, r=20, t=40, b=40),
        yaxis_title="Daily return (%)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.22),
    )
    _add_crisis_bands(fig, show_crises)
    return fig


def weights_lines_with_static_overlay(
    dynamic_weights: pd.DataFrame,
    static_weights: pd.Series,
    show_crises: bool = False,
    title: str = "MVP (DCC) weights with MVP (static) reference",
) -> go.Figure:
    """
    Q6: MVP (DCC) weights as time series, with MVP (static) weights overlaid
    as horizontal dashed reference lines. Highlights how much the dynamic
    optimization deviates from the static benchmark on any given day.
    """
    fig = go.Figure()
    for i, col in enumerate(dynamic_weights.columns):
        color = PALETTE[i % len(PALETTE)]
        # Dynamic series — solid
        fig.add_trace(
            go.Scatter(
                x=dynamic_weights.index, y=dynamic_weights[col],
                mode="lines", name=f"{col} (DCC)",
                line=dict(width=1.4, color=color),
                legendgroup=col,
            )
        )
        # Static benchmark — horizontal dashed
        fig.add_trace(
            go.Scatter(
                x=[dynamic_weights.index[0], dynamic_weights.index[-1]],
                y=[static_weights[col], static_weights[col]],
                mode="lines", name=f"{col} (static)",
                line=dict(width=1.0, color=color, dash="dash"),
                legendgroup=col, showlegend=False, opacity=0.65,
            )
        )
    fig.add_hline(y=0.0, line=dict(width=1, dash="dot", color="grey"))
    fig.update_layout(
        height=460, template="simple_white", title=title,
        margin=dict(l=40, r=20, t=50, b=40),
        yaxis_title="Weight",
        legend=dict(orientation="h", yanchor="bottom", y=-0.22),
    )
    _add_crisis_bands(fig, show_crises)
    return fig


def weights_comparison_bar(weights_by_strategy: dict[str, pd.Series]) -> go.Figure:
    """
    Bar chart comparing per-asset weight across multiple strategies. For dynamic
    strategies pass the *mean* weight across the sample; for static strategies
    pass the single weight vector directly. All values should be on the same
    scale (fractions summing to 1).
    """
    fig = go.Figure()
    for i, (name, w) in enumerate(weights_by_strategy.items()):
        fig.add_trace(
            go.Bar(
                x=list(w.index), y=list(w.values), name=name,
                marker_color=PALETTE[i % len(PALETTE)],
                text=[f"{v:.1%}" for v in w.values],
                textposition="outside",
            )
        )
    fig.add_hline(y=0.0, line=dict(width=1, dash="dot", color="grey"))
    fig.update_layout(
        barmode="group", template="simple_white", height=420,
        margin=dict(l=40, r=20, t=40, b=40),
        yaxis_title="Allocation",
        yaxis_tickformat=".0%",
        legend=dict(orientation="h", yanchor="bottom", y=-0.18),
    )
    return fig


def drawdown_lines(returns_by_strategy: dict[str, pd.Series], show_crises: bool = False) -> go.Figure:
    """
    Peak-to-trough drawdown of cumulative portfolio value, expressed as a
    percentage of the running maximum. Useful complement to the variance summary
    because it characterizes the *path* of losses rather than their squared mean.
    """
    fig = go.Figure()
    for i, (name, r) in enumerate(returns_by_strategy.items()):
        wealth = (1.0 + r / 100.0).cumprod()
        peak = wealth.cummax()
        dd = (wealth / peak - 1.0) * 100.0
        fig.add_trace(
            go.Scatter(
                x=dd.index, y=dd, mode="lines", name=name,
                line=dict(width=1.6, color=PALETTE[i % len(PALETTE)]),
                fill="tozeroy", opacity=0.55,
            )
        )
    fig.update_layout(
        height=380, template="simple_white",
        margin=dict(l=40, r=20, t=40, b=40),
        yaxis_title="Drawdown (%)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.22),
    )
    _add_crisis_bands(fig, show_crises)
    return fig


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Helper returning the drawdown series, used by interpretation generators."""
    wealth = (1.0 + returns / 100.0).cumprod()
    peak = wealth.cummax()
    return (wealth / peak - 1.0)
