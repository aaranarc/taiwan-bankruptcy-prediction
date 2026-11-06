"""Score a company: grouped inputs + presets on the left; PD gauge, grade, path to safety and SHAP on the right."""
import time
from html import escape

import streamlit as st

from ui import api
from ui.components import (
    DISTRESSED,
    HEALTHY,
    empty_state,
    get_meta,
    grade_badge,
    hero,
    init_state,
    path_to_safety,
    pd_gauge,
    plot,
    require_sign_in,
    shap_waterfall,
)
from ui.inputs import current_firm, load_firm, render_inputs

init_state()
meta = get_meta()
hero("DefaultPulse", "Score a company",
     "Probability of default from a monotone-constrained, calibrated XGBoost model, with the reasons behind it "
     "and the smallest realistic path to a safer grade.")

left, right = st.columns([5, 7], gap="large")

with left:
    st.markdown("**Start from a preset**")
    c1, c2 = st.columns(2)
    if c1.button("Healthy firm", use_container_width=True, icon=":material/verified:"):
        load_firm(HEALTHY)
    if c2.button("Distressed firm", use_container_width=True, icon=":material/warning:"):
        load_firm(DISTRESSED)
    st.caption(f"Editing **{st.session_state.firm_label}**. All other ratios stay at this preset's values. "
               "Inputs use the dataset's 0–1 normalised scale.")
    render_inputs()
    grades = [g["grade"] for g in meta["grades"]] if meta else ["A", "B", "C", "D", "E"]
    default_target = meta["default_target_grade"] if meta else "C"
    target = st.selectbox("Path-to-safety target grade", grades[:-1], index=grades.index(default_target),
                          help="The counterfactual search looks for the smallest change that reaches this grade.")
    run = st.button("Score company", type="primary", use_container_width=True, icon=":material/speed:")

if run and require_sign_in("score a company") and meta:
    firm = current_firm()
    try:
        with right, st.spinner("Scoring… the first request after a quiet period can take a few seconds"):
            start = time.perf_counter()
            result = api.predict(firm, st.session_state.token)
            st.session_state.latency = time.perf_counter() - start
            st.session_state.cf = api.counterfactual(firm, target, st.session_state.token)
        st.session_state.result, st.session_state.result_firm = result, firm
    except api.ApiError as e:
        st.session_state.result = None
        with right:
            st.error(str(e), icon=":material/error:")

with right:
    result = st.session_state.result
    if not result:
        if not run:
            empty_state("No score yet", "Pick a preset or edit the ratios, then press Score company.")
    else:
        st.html(f'<div style="margin-bottom:.3rem">{grade_badge(result["grade"], result["flagged"])}</div>')
        if meta:  # the gauge needs the grade bands from /meta
            plot(pd_gauge(result["probability"], meta["grades"], result["threshold"]))
        source = "served from cache" if result.get("cached") else "computed by the model"
        st.caption(f"API responded in {st.session_state.latency:.2f} s · {source} · decision threshold "
                   f"{result['threshold']:.1%} (missed default = {meta['cost_ratio_fn_to_fp'] if meta else '?'}× "
                   f"a false alarm) · model {escape(result['model_version'])}")
        if st.session_state.get("result_firm") != current_firm():
            st.warning("Inputs changed since this score. Press Score company to update.", icon=":material/sync:")
        if st.session_state.cf:
            path_to_safety(st.session_state.cf)
        if result.get("top_factors"):
            plot(shap_waterfall(result))
            st.caption("SHAP values explain the XGBoost score in log-odds, before calibration. Calibration is a single "
                       "increasing curve, so the order and direction of the reasons also hold for the PD.")
