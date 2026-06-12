"""
HW5 — Multivariate GARCH / DCC dynamic conditional correlations.

Modules:
    data            Download prices and compute log-returns.
    descriptive     Q1 descriptive stats + Q2 bivariate scatterplots.
    garch           Univariate GARCH(1,1) step of Engle's two-step DCC.
    dcc             DCC second step: (a, b) MLE on standardized residuals.
    portfolio       Minimum-variance, static-MVP, and equal-weight portfolios.
    plots           Plotly chart helpers (crisis shading, etc.).
    interpretations Plain-language "what this shows / why it matters" generators.
"""
