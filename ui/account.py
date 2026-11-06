"""Sidebar: brand, sign in / create account, sign out. The JWT lives only in this session's memory."""
import streamlit as st

from ui import api


def _sign_in(token_response: dict) -> None:
    st.session_state.token = token_response["access_token"]
    st.session_state.email = token_response["email"]


def sidebar_account() -> None:
    with st.sidebar:
        st.html('<div class="cr-hero" style="margin-bottom:.4rem"><div class="cr-hero__eyebrow">Corporate credit risk</div>'
                '<div class="cr-hero__title" style="font-size:1.15rem">DefaultPulse</div></div>')
        if st.session_state.get("token"):
            st.caption(f"Signed in as **{st.session_state.email}**")
            if st.button("Sign out", use_container_width=True):
                st.session_state.token = st.session_state.email = None
                st.rerun()
        else:
            sign_in, create = st.tabs(["Sign in", "Create account"])
            with sign_in, st.form("sign_in", border=False):
                email = st.text_input("Email", key="si_email")
                password = st.text_input("Password", type="password", key="si_password")
                if st.form_submit_button("Sign in", type="primary", use_container_width=True):
                    try:
                        _sign_in(api.login(email, password))
                        st.rerun()
                    except api.ApiError as e:
                        st.error(str(e) if e.status != 401 else "Incorrect email or password.")
            with create, st.form("create", border=False):
                email = st.text_input("Email", key="su_email")
                password = st.text_input("Password", type="password", key="su_password")
                if st.form_submit_button("Create account", use_container_width=True):
                    if len(password) < 8:
                        st.error("Use at least 8 characters.")
                    else:
                        try:
                            _sign_in(api.signup(email, password))
                            st.rerun()
                        except api.ApiError as e:
                            st.error(str(e))
        st.caption("Your sign-in lasts for this browser session only; refreshing the page signs you out.")
