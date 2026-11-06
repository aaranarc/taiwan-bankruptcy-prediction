"""Streamlit entrypoint: page config, theme CSS, sign-in sidebar and navigation. UI logic lives in ui/."""
import streamlit as st

from ui.account import sidebar_account
from ui.components import init_state, inject_css

st.set_page_config(page_title="DefaultPulse | Credit-Risk Engine", page_icon=":material/account_balance:", layout="wide")
inject_css()
init_state()
sidebar_account()

st.navigation([
    st.Page("ui/pages/score.py", title="Score a company", icon=":material/speed:", default=True),
    st.Page("ui/pages/portfolio.py", title="Portfolio upload", icon=":material/table_view:"),
    st.Page("ui/pages/stress.py", title="Stress test", icon=":material/bolt:"),
    st.Page("ui/pages/model_card.py", title="Model card", icon=":material/fact_check:"),
    st.Page("ui/pages/history.py", title="History", icon=":material/history:"),
]).run()
