"""Shared UI building blocks: session state, data loading, cards, badges and the two main charts."""
import json
from html import escape
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from ui import api
from ui.theme import (
    ACCENT,
    BORDER,
    CHART_CONFIG,
    GRADE_COLORS,
    GRADE_LABELS,
    MUTED,
    RISK_GREEN,
    RISK_RED,
    SURFACE,
    TEMPLATE,
    TEXT,
    fmt_pctile,
    fmt_pd,
    fmt_ratio,
)

ROOT = Path(__file__).resolve().parent.parent
HEALTHY, DISTRESSED = "Healthy firm (median of non-bankrupt)", "Distressed firm (median of bankrupt)"


# ---------- data ----------

@st.cache_resource
def load_presets() -> dict:
    with open(ROOT / "ui" / "presets.json") as fh:
        return json.load(fh)


def load_json(relative: str) -> dict | None:
    """A file from the repo (reports/*.json, config/*.json); None if missing or unreadable."""
    try:
        with open(ROOT / relative) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


@st.cache_data(ttl=600, show_spinner=False)
def load_meta() -> dict:
    return api.meta()


def get_meta() -> dict | None:
    try:
        return load_meta()
    except api.ApiError as e:
        st.error(str(e))
        return None


def init_state() -> None:
    defaults = {"token": None, "email": None, "firm_label": DISTRESSED, "result": None, "cf": None,
                "latency": None, "stress": None, "portfolio": None}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("firm", dict(load_presets()[DISTRESSED]))


def inject_css() -> None:
    st.html(f"<style>{(ROOT / 'ui' / 'styles.css').read_text()}</style>")


def grade_for(pd_value: float, grades: list) -> str:
    """Same rule as the API (app/services/risk.py): first grade whose max_pd the PD is strictly below."""
    for g in grades:
        if pd_value < g["max_pd"]:
            return g["grade"]
    return grades[-1]["grade"]


# ---------- small HTML pieces (all dynamic text escaped) ----------

def hero(eyebrow: str, title: str, sub: str) -> None:
    st.html(f'<div class="cr-hero"><div class="cr-hero__eyebrow">{escape(eyebrow)}</div>'
            f'<div class="cr-hero__title">{escape(title)}</div><p class="cr-hero__sub">{escape(sub)}</p></div>')


def cards(items: list) -> None:
    """items: (label, value, sub) tuples rendered as a responsive grid of metric cards."""
    body = "".join(f'<div class="cr-card"><div class="cr-card__label">{escape(label)}</div>'
                   f'<div class="cr-card__value">{escape(value)}</div>'
                   f'<div class="cr-card__sub">{escape(sub)}</div></div>' for label, value, sub in items)
    st.html(f'<div class="cr-grid">{body}</div>')


def grade_badge(grade: str, flagged: bool | None = None) -> str:
    flag = ""
    if flagged is not None:
        flag = ('<span class="cr-flag cr-flag--on">Flagged: above threshold</span>' if flagged
                else '<span class="cr-flag cr-flag--off">Below threshold</span>')
    return (f'<span class="cr-badge" style="--c:{GRADE_COLORS[grade]}"><span class="cr-badge__letter">{escape(grade)}'
            f'</span><span class="cr-badge__text">Grade {escape(grade)} · {GRADE_LABELS[grade]} risk</span></span>{flag}')


def empty_state(title: str, body: str) -> None:
    st.html(f'<div class="cr-empty"><div class="cr-empty__title">{escape(title)}</div>{escape(body)}</div>')


def panel(title: str, body_html: str) -> None:
    st.html(f'<div class="cr-panel"><div class="cr-panel__title">{escape(title)}</div>'
            f'<div class="cr-panel__body">{body_html}</div></div>')


def require_sign_in(action: str) -> bool:
    if st.session_state.get("token"):
        return True
    st.info(f"Sign in from the sidebar to {action}.", icon=":material/lock:")
    return False


# ---------- charts ----------

def pd_gauge(pd_value: float, grades: list, threshold: float) -> go.Figure:
    """Gauge with one equal-width band per grade (PD bands are very uneven: <1% … 30-100%).
    The needle position is interpolated inside its band; the white tick marks the decision threshold."""
    bounds = [0.0] + [g["max_pd"] for g in grades]

    def position(p):
        for i in range(len(grades)):
            if p < bounds[i + 1] or i == len(grades) - 1:
                return i + min(1.0, (p - bounds[i]) / (bounds[i + 1] - bounds[i]))
        return float(len(grades))

    grade = grade_for(pd_value, grades)
    steps = [dict(range=[i, i + 1], color=_alpha(GRADE_COLORS[g["grade"]], 0.16)) for i, g in enumerate(grades)]
    fig = go.Figure(go.Indicator(
        mode="gauge", value=position(pd_value),
        gauge=dict(
            axis=dict(range=[0, len(grades)], tickvals=[i + 0.5 for i in range(len(grades))],
                      ticktext=[g["grade"] for g in grades], tickfont=dict(color=MUTED, size=13), tickwidth=0),
            bar=dict(color=GRADE_COLORS[grade], thickness=0.3), bgcolor=SURFACE, borderwidth=0, steps=steps,
            threshold=dict(line=dict(color=TEXT, width=3), thickness=0.9, value=position(threshold)),
        ),
        domain=dict(x=[0, 1], y=[0.12, 1]),
    ))
    fig.add_annotation(text=f"<b>{fmt_pd(pd_value)}</b>", x=0.5, y=0.16, showarrow=False,
                       font=dict(size=40, color=TEXT))
    fig.add_annotation(text="probability of default · white tick = decision threshold", x=0.5, y=0.0,
                       showarrow=False, font=dict(size=12, color=MUTED))
    fig.update_layout(template=TEMPLATE, height=280, margin=dict(l=24, r=24, t=34, b=10))
    return fig


def shap_waterfall(result: dict) -> go.Figure:
    """Base value → top factors → all others → this firm's score, in log-odds (before calibration)."""
    factors = result["top_factors"]
    labels = [_short(f["feature"]) for f in factors] + ["All other features"]
    values = [f["shap"] for f in factors] + [result["shap_other"]]
    hover = [f"{escape(f['feature'])}<br>input value {fmt_ratio(f['value'])}<br>{f['shap']:+.3f} log-odds"
             for f in factors] + [f"{result['shap_other']:+.3f} log-odds"]
    fig = go.Figure(go.Waterfall(
        orientation="h", base=result["shap_base_value"], y=labels + ["This firm"],
        x=values + [0], measure=["relative"] * len(values) + ["total"],
        text=[f"{v:+.2f}" for v in values] + [f"{result['shap_base_value'] + sum(values):.2f}"],
        textposition="outside", textfont=dict(color=MUTED, size=11), cliponaxis=False,
        hovertext=hover + ["model score (log-odds)"], hoverinfo="text",
        increasing=dict(marker=dict(color=RISK_RED)), decreasing=dict(marker=dict(color=RISK_GREEN)),
        totals=dict(marker=dict(color=ACCENT)), connector=dict(line=dict(color=BORDER, width=1)),
    ))
    fig.update_layout(template=TEMPLATE, height=430, showlegend=False, title="Why this score",
                      margin=dict(l=8, r=48, t=36, b=8),
                      yaxis=dict(autorange="reversed", tickfont=dict(size=11, color=TEXT), automargin=True),
                      xaxis=dict(title=dict(text=f"log-odds, from the average firm ({result['shap_base_value']:.2f})"
                                                 " · red raises risk, green lowers it",
                                            font=dict(size=11, color=MUTED))))
    return fig


def path_to_safety(cf: dict) -> None:
    target = cf["target_grade"]
    if cf.get("already_at_target"):
        panel("Path to safety", f"Already at grade {escape(cf['grade_before'])}, within the target grade {escape(target)}. "
              "No change needed.")
        return
    if not cf["reachable"]:
        panel("Path to safety", f"No combination of up to 3 adjustable ratios, kept within the range seen in the "
              f"training data, brings this firm to grade {escape(target)}. Structural change would be needed.")
        return
    rows = "".join(
        f'<div class="cr-change"><span class="cr-change__name">{escape(_short(c["feature"], 44))}</span>'
        f'<span class="cr-change__move">{fmt_ratio(c["from_value"])} → {fmt_ratio(c["to_value"])}</span>'
        f'<span class="cr-change__pct">{fmt_pctile(c["from_percentile"])} → {fmt_pctile(c["to_percentile"])} '
        f'of firms in the training data</span></div>' for c in cf["changes"])
    summary = (f"PD {fmt_pd(cf['pd_before'])} (grade {escape(cf['grade_before'])}) → <b>{fmt_pd(cf['pd_after'])} "
               f"(grade {escape(cf['grade_after'])})</b> by changing {len(cf['changes'])} ratio"
               f"{'s' if len(cf['changes']) > 1 else ''}:")
    panel(f"Path to safety · target grade {target}", f"{summary}{rows}"
          '<div class="cr-muted" style="margin-top:.4rem">Smallest move found by a bounded search over ratios '
          "management can influence; other ratios held fixed. A starting point for discussion, not advice.</div>")


def plot(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)


def _short(name: str, n: int = 34) -> str:
    return name if len(name) <= n else name[: n - 1] + "…"


def _alpha(hex_color: str, a: float) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{a})"
