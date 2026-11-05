"""Design tokens, the shared Plotly template and number formatting. Every chart and card uses these.

Base palette = .streamlit/config.toml. Green/amber/red appear ONLY for risk (grades, SHAP direction, flags).
"""
import plotly.graph_objects as go
import plotly.io as pio

NAVY = "#0B1220"
SURFACE = "#131C2E"
BORDER = "#22304A"
TEXT = "#E5EAF3"
MUTED = "#8A98B3"
ACCENT = "#5B8DEF"
ACCENT_SOFT = "#8DB0F5"

RISK_GREEN = "#22C55E"
RISK_AMBER = "#F59E0B"
RISK_RED = "#EF4444"

GRADE_COLORS = {"A": RISK_GREEN, "B": RISK_GREEN, "C": RISK_AMBER, "D": RISK_RED, "E": RISK_RED}
GRADE_LABELS = {"A": "Minimal", "B": "Low", "C": "Moderate", "D": "High", "E": "Severe"}

FONT = "Inter, -apple-system, Segoe UI, Roboto, sans-serif"

pio.templates["credit_desk"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family=FONT, color=TEXT, size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=[ACCENT, ACCENT_SOFT, "#3E5C9A", "#A9B6CF", "#6B7A99"],
        margin=dict(l=8, r=8, t=36, b=8),
        title=dict(font=dict(size=14, color=MUTED), x=0, xanchor="left"),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER, tickcolor=BORDER,
                   tickfont=dict(color=MUTED)),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER, tickcolor=BORDER,
                   tickfont=dict(color=MUTED)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED), orientation="h", y=-0.2),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER, font=dict(family=FONT, color=TEXT)),
    )
)
TEMPLATE = "credit_desk"
CHART_CONFIG = {"displayModeBar": False, "responsive": True}


def fmt_pd(p: float | None) -> str:
    """PD as a percentage with sensible precision: '<0.1%', '2.4%', '43%'."""
    if p is None:
        return "—"
    if p < 0.001:
        return "<0.1%"
    return f"{p:.1%}" if p < 0.1 else f"{p:.0%}"


def fmt_ratio(v: float | None) -> str:
    return "—" if v is None else f"{v:.4f}"


def fmt_pctile(p: float) -> str:
    """Position in the training distribution, e.g. 'P82' = 82nd percentile."""
    return f"P{p:.0f}"
