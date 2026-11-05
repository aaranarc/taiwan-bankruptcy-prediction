"""Stress test: apply shocks to the firm's ratios and compare PD before and after, side by side."""
from html import escape

import pandas as pd
import streamlit as st

from ui import api
from ui.components import cards, empty_state, get_meta, grade_badge, hero, init_state, require_sign_in
from ui.inputs import current_firm
from ui.theme import fmt_pd, fmt_ratio

init_state()
meta = get_meta()
hero("Credit risk desk", "Stress test",
     "Shock the firm you are editing on the Score page and see how its PD and grade move. Each shock touches only "
     "the ratios listed under it.")
st.caption(f"Firm: **{st.session_state.firm_label}** with your edits from the Score page.")

if meta:
    shocks = {}
    cols = st.columns(2)
    for i, (name, spec) in enumerate(meta["stress_shocks"].items()):
        with cols[i % 2]:
            shocks[name] = st.slider(spec["label"], min_value=float(spec["min"]), max_value=float(spec["max"]),
                                     value=float(spec["default"]), step=1.0, key=f"shock::{name}",
                                     help=f"{spec['doc']} Touches: {', '.join(spec['columns'])}.")
    if st.button("Run stress test", type="primary", icon=":material/bolt:") and require_sign_in("run a stress test"):
        try:
            with st.spinner("Applying shocks…"):
                st.session_state.stress = api.stress(current_firm(), shocks, st.session_state.token)
        except api.ApiError as e:
            st.session_state.stress = None
            st.error(str(e), icon=":material/error:")

res = st.session_state.stress
if not res:
    empty_state("No stress test run yet", "Set the shocks and press Run stress test.")
else:
    before, after = st.columns(2, gap="large")
    with before:
        st.markdown("**Before**")
        st.html(grade_badge(res["grade_before"]))
        cards([("PD before", fmt_pd(res["pd_before"]), "current ratios")])
    with after:
        st.markdown("**After shocks**")
        st.html(grade_badge(res["grade_after"]))
        delta = res["pd_after"] - res["pd_before"]
        cards([("PD after", fmt_pd(res["pd_after"]), f"{delta * 100:+.1f} percentage points")])
    if res["changed_features"]:
        st.markdown("**Ratios changed by the shocks**")
        st.dataframe(pd.DataFrame([{"Ratio": c["feature"], "Before": fmt_ratio(c["before"]),
                                    "After": fmt_ratio(c["after"])} for c in res["changed_features"]]),
                     hide_index=True, use_container_width=True)
    if res["clipped_to_training_range"]:
        st.caption("Capped at the range seen in training (the model knows nothing beyond it): "
                   + escape(", ".join(res["clipped_to_training_range"])))
    st.caption("Shocks are approximations on the dataset's pre-normalised ratios, not an accounting model. "
               "Definitions: config/risk_policy.json.")
