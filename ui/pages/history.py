"""History: the signed-in user's past predictions, read from Supabase through the API."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui import api
from ui.components import cards, empty_state, get_meta, hero, init_state, plot, require_sign_in
from ui.theme import ACCENT, MUTED, TEMPLATE, fmt_pd

init_state()
hero("Credit risk desk", "History", "Every company you have scored, newest first. Stored in Supabase, so it "
     "survives sign-out and redeploys.")

if require_sign_in("see your prediction history"):
    try:
        with st.spinner("Loading history…"):
            rows = api.history(st.session_state.token)
    except api.ApiError as e:
        rows = None
        st.error(str(e), icon=":material/error:")

    if rows == []:
        empty_state("Nothing scored yet", "Scores you run on the Score page appear here.")
    elif rows:
        meta = get_meta()
        records = []
        for row in rows:
            res = row.get("prediction_result") or {}
            records.append({
                "Scored at": pd.to_datetime(row["created_at"]),
                "PD": res.get("probability"),
                # Rows saved before the A-E grades existed carry the old risk_tier instead.
                "Grade": res.get("grade") or res.get("risk_tier") or "—",
                "Flagged": res.get("flagged"),
                "Model": res.get("model_version") or "earlier model",
            })
        df = pd.DataFrame(records).sort_values("Scored at", ascending=False)
        cards([
            ("Companies scored", f"{len(df):,}", ""),
            ("Flagged", f"{int(df.Flagged.fillna(False).sum()):,}", "at or above the threshold"),
            ("Latest PD", fmt_pd(df.PD.iloc[0]), f"grade {df.Grade.iloc[0]}"),
        ])
        trend = df.dropna(subset=["PD"]).sort_values("Scored at")
        if len(trend) > 1:
            fig = go.Figure(go.Scatter(x=trend["Scored at"], y=trend.PD, mode="lines+markers",
                                       line=dict(color=ACCENT), name="PD"))
            if meta:
                fig.add_hline(y=meta["threshold"], line=dict(color=MUTED, dash="dash"),
                              annotation_text="decision threshold", annotation_font_color=MUTED)
            fig.update_layout(template=TEMPLATE, height=280, title="PD of each scored company over time",
                              yaxis=dict(tickformat=".0%", rangemode="tozero"))
            plot(fig)
        st.dataframe(df, hide_index=True, use_container_width=True, column_config={
            "Scored at": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
            "PD": st.column_config.NumberColumn(format="percent"),
        })
