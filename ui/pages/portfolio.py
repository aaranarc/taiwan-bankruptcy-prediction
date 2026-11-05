"""Portfolio upload: score a CSV of firms, show the grade mix and the riskiest names, download the results."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui import api
from ui.components import ROOT, cards, empty_state, get_meta, hero, init_state, plot, require_sign_in
from ui.theme import GRADE_COLORS, TEMPLATE, fmt_pd

CHUNK = 250  # rows per API call; the API accepts up to 500 and Vercel caps bodies at 4.5 MB

init_state()
meta = get_meta()
hero("Credit risk desk", "Portfolio upload",
     "Score a whole book at once. Upload a CSV with the model's 95 ratio columns (an optional firm_id column is "
     "kept as the label), or try the 150-firm sample.")

upload = st.file_uploader("Portfolio CSV (max 4 MB)", type="csv")
use_sample = st.button("Use the sample portfolio", icon=":material/dataset:")
if use_sample:
    st.caption("The sample firms come from the training data, so their scores are in-sample (optimistic).")


def score(df: pd.DataFrame) -> pd.DataFrame:
    features = [c for c in df.columns if c != "firm_id"]
    rows = df[features].astype(float).to_dict("records")
    out, bar = [], st.progress(0.0, text="Scoring…")
    for i in range(0, len(rows), CHUNK):
        out += api.predict_batch(rows[i:i + CHUNK], st.session_state.token)["results"]
        bar.progress(min(1.0, (i + CHUNK) / len(rows)), text=f"Scored {min(i + CHUNK, len(rows))} of {len(rows)}")
    bar.empty()
    ids = df["firm_id"] if "firm_id" in df.columns else pd.Series(range(1, len(df) + 1))
    return pd.DataFrame({
        "firm_id": ids.astype(str).to_numpy(),
        "pd": [r["probability"] for r in out],
        "grade": [r["grade"] for r in out],
        "flagged": [r["flagged"] for r in out],
        "top_reasons": ["; ".join(f"{f['feature']} ({'+' if f['shap'] > 0 else '−'})" for f in r["top_factors"] or [])
                        for r in out],
    })


source = upload if upload is not None else (ROOT / "ui" / "sample_portfolio.csv" if use_sample else None)
signature = (upload.name, upload.size) if upload is not None else ("sample" if use_sample else None)
# An uploaded file stays attached across reruns; only score when it is new, not on every widget interaction.
is_new = signature is not None and (use_sample or signature != st.session_state.get("portfolio_sig"))
if is_new and require_sign_in("score a portfolio") and meta:
    st.session_state.portfolio_sig = signature
    try:
        df = pd.read_csv(source)
        df.columns = df.columns.str.strip()
    except (ValueError, UnicodeDecodeError, pd.errors.ParserError):
        st.error("That file isn't a readable CSV.")
        df = None
    if df is not None:
        expected = set(api.features())
        missing = sorted(expected - set(df.columns))
        if missing:
            st.error(f"{len(missing)} required column(s) missing, e.g. {', '.join(missing[:3])}.")
        elif df.empty:
            st.error("The CSV has no rows.")
        else:
            try:
                st.session_state.portfolio = score(df[["firm_id", *sorted(expected)]] if "firm_id" in df.columns
                                                   else df[sorted(expected)])
            except (api.ApiError, ValueError) as e:
                st.error(str(e) if isinstance(e, api.ApiError) else "Every ratio column must be numeric.")

res = st.session_state.portfolio
if res is None:
    empty_state("No portfolio scored yet", "Upload a CSV or use the sample portfolio.")
else:
    cards([
        ("Firms scored", f"{len(res):,}", ""),
        ("Flagged", f"{int(res.flagged.sum()):,}", f"{res.flagged.mean():.0%} of the book"),
        ("Mean PD", fmt_pd(res.pd.mean()), "expected default rate of the book"),
        ("Grades D–E", f"{int(res.grade.isin(['D', 'E']).sum()):,}", "high or severe risk"),
    ])
    left, right = st.columns([5, 7], gap="large")
    with left:
        counts = res.grade.value_counts().reindex(list(GRADE_COLORS), fill_value=0)
        fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=[GRADE_COLORS[g] for g in counts.index],
                               text=counts.values, textposition="outside"))
        fig.update_layout(template=TEMPLATE, height=300, title="Grade distribution", yaxis=dict(showgrid=False,
                          showticklabels=False), xaxis=dict(title=None))
        plot(fig)
    with right:
        st.markdown("**Top 10 riskiest firms** and the three factors behind each score (+ raises risk, − lowers it)")
        top = res.nlargest(10, "pd")
        st.dataframe(top.assign(pd=top.pd.map(fmt_pd)), hide_index=True, use_container_width=True,
                     column_config={"firm_id": "Firm", "pd": "PD", "grade": "Grade", "flagged": "Flagged",
                                    "top_reasons": st.column_config.TextColumn("Top reasons", width="large")})
    st.markdown("**All firms**")
    st.dataframe(res.sort_values("pd", ascending=False), hide_index=True, use_container_width=True,
                 column_config={"firm_id": "Firm", "grade": "Grade", "flagged": "Flagged",
                                "pd": st.column_config.NumberColumn("PD", format="percent"),
                                "top_reasons": st.column_config.TextColumn("Top reasons", width="large")})
    st.download_button("Download scored portfolio (CSV)", res.to_csv(index=False), "scored_portfolio.csv",
                       "text/csv", icon=":material/download:")
