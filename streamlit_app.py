import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import os

# Page Config
st.set_page_config(
    page_title="Taiwan Bankruptcy Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Custom CSS for rich premium aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        font-family: 'Outfit', sans-serif;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(99, 102, 241, 0.25);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
    }
    .main-header h1 {
        color: #f1f5f9;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: #a5b4fc;
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #818cf8;
        margin: 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }

    /* Risk badge */
    .risk-badge {
        display: inline-block;
        padding: 0.6rem 1.8rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.02em;
        margin-top: 0.5rem;
    }

    /* Prediction result card */
    .prediction-result {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid rgba(99, 102, 241, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }

    /* Auth container */
    .auth-container {
        max-width: 450px;
        margin: auto;
        padding: 2.5rem;
        background: #111827;
        border-radius: 16px;
        border: 1px solid rgba(99, 102, 241, 0.2);
        box-shadow: 0 10px 40px rgba(0,0,0,0.5);
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if "token" not in st.session_state:
    st.session_state["token"] = None
if "email" not in st.session_state:
    st.session_state["email"] = None

# Sidebar Navigation & Auth info
with st.sidebar:
    st.markdown("### 📊 Navigation")
    if st.session_state["token"]:
        st.markdown(f"**Logged in as:** `{st.session_state['email']}`")
        page = st.radio(
            "Go to",
            ["🏠 Overview", "🔍 Predict Bankruptcy", "📂 Prediction History"],
        )
        if st.button("🚪 Log Out", use_container_width=True):
            st.session_state["token"] = None
            st.session_state["email"] = None
            st.rerun()
    else:
        st.markdown("**Not Logged In**")
        page = st.radio(
            "Go to",
            ["🏠 Overview", "🔑 Sign In / Sign Up", "🔍 Predict (Guest Mode)"],
        )

# Header
st.markdown(
    '<div class="main-header">'
    '<h1>📊 Corporate Bankruptcy Predictor</h1>'
    '<p>De-coupled Streamlit UI powered by FastAPI, Supabase Database, and Redis caching</p>'
    '</div>',
    unsafe_allow_html=True,
)

# ── PAGE: OVERVIEW ──
if page == "🏠 Overview":
    st.markdown("### Model Performance & Dataset overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><p class="metric-value">95.2%</p><p class="metric-label">ROC-AUC Score</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><p class="metric-value">92.4%</p><p class="metric-label">PR-AUC Score</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><p class="metric-value">6,819</p><p class="metric-label">Dataset Size</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><p class="metric-value">95</p><p class="metric-label">Model Features</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    
    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### About this Predictor")
        st.markdown("""
        This machine learning system predicts the probability of a company going bankrupt based on key financial indicators.
        It uses an **XGBoost Classifier** trained on the Taiwan Bankruptcy Dataset (UCI ML Repository).
        
        **Integration Details:**
        - **Decoupled Frontend:** Streamlit communicates solely via JSON APIs with the backend FastAPI.
        - **Auth & Database:** Hashed user accounts and prediction histories are securely saved in a Supabase Postgres cluster.
        - **High Performance Caching:** Prediction requests with matching features bypass inference models entirely by hitting a Redis Cache instance.
        """)
        
    with right:
        st.markdown("#### Baseline Comparison")
        st.markdown("""
        The system compares predictions against the **Altman Z-Score (1968)**, a classic formula for predicting bankruptcy.
        
        **Altman Zone Boundaries:**
        - **Safe Zone:** Z-Score > 2.99
        - **Grey Zone:** 1.81 < Z-Score ≤ 2.99
        - **Distress Zone:** Z-Score ≤ 1.81
        """)

# ── PAGE: AUTHENTICATION ──
elif page == "🔑 Sign In / Sign Up":
    st.markdown("### User Authentication")
    tab1, tab2 = st.tabs(["🔑 Sign In", "📝 Sign Up"])
    
    with tab1:
        st.markdown("#### Sign In to your account")
        login_email = st.text_input("Email Address", key="login_email")
        login_pwd = st.text_input("Password", type="password", key="login_pwd")
        if st.button("Sign In", type="primary", use_container_width=True):
            try:
                res = requests.post(
                    f"{BACKEND_URL}/auth/login",
                    data={"username": login_email, "password": login_pwd}
                )
                if res.status_code == 200:
                    data = res.json()
                    st.session_state["token"] = data["access_token"]
                    st.session_state["email"] = data["email"]
                    st.success("Successfully logged in!")
                    st.rerun()
                else:
                    st.error(f"Login failed: {res.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")
                
    with tab2:
        st.markdown("#### Create a new account")
        reg_email = st.text_input("Email Address", key="reg_email")
        reg_pwd = st.text_input("Password", type="password", key="reg_pwd")
        reg_pwd_confirm = st.text_input("Confirm Password", type="password", key="reg_pwd_confirm")
        
        if st.button("Sign Up", type="primary", use_container_width=True):
            if reg_pwd != reg_pwd_confirm:
                st.error("Passwords do not match!")
            elif len(reg_pwd) < 6:
                st.error("Password must be at least 6 characters long.")
            else:
                try:
                    res = requests.post(
                        f"{BACKEND_URL}/auth/signup",
                        json={"email": reg_email, "password": reg_pwd}
                    )
                    if res.status_code == 200:
                        data = res.json()
                        st.session_state["token"] = data["access_token"]
                        st.session_state["email"] = data["email"]
                        st.success("Registration successful and logged in!")
                        st.rerun()
                    else:
                        st.error(f"Signup failed: {res.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Error connecting to backend: {e}")

# ── PAGE: PREDICT ──
elif page in ["🔍 Predict Bankruptcy", "🔍 Predict (Guest Mode)"]:
    st.markdown("### Run Bankruptcy Risk Prediction")
    st.caption("Adjust key financial ratios below. Missing inputs will fall back to model-trained defaults.")
    
    # Defaults
    key_features = [
        ('ROA(B) before interest and depreciation after tax', 'Profitability', 0.5),
        ('Persistent EPS in the Last Four Seasons', 'Profitability', 0.5),
        ('Net Income to Total Assets', 'Profitability', 0.5),
        ('Debt ratio %', 'Leverage', 0.1),
        ('Net worth/Assets', 'Leverage', 0.8),
        ('Borrowing dependency', 'Leverage', 0.3),
        ('Current Ratio', 'Liquidity', 1.5),
        ('Working Capital to Total Assets', 'Liquidity', 0.2),
        ('Retained Earnings to Total Assets', 'Profitability', 0.4),
        ('Total Asset Turnover', 'Activity', 0.6),
        ('Cash Flow to Total Assets', 'Cash Flow', 0.1),
        ('Net Value Growth Rate', 'Growth', 0.5),
    ]

    input_values = {}
    cols = st.columns(2)
    for i, (feat, category, default_val) in enumerate(key_features):
        with cols[i % 2]:
            input_values[feat] = st.number_input(
                f"{feat}",
                value=float(default_val),
                format="%.6f",
                help=f"Category: {category}",
                key=f"predict_{feat}"
            )
            
    st.markdown("---")
    if st.button("🔮 Run Risk Prediction", type="primary", use_container_width=True):
        # Fetch expected features to build the full payload
        try:
            feat_res = requests.get(f"{BACKEND_URL}/features")
            if feat_res.status_code == 200:
                all_features = feat_res.json()
            else:
                st.error("Failed to load feature templates from backend.")
                st.stop()
        except Exception as e:
            st.error(f"Failed to communicate with backend: {e}")
            st.stop()

        # Build complete features dictionary (remaining default to 0.5 or custom default)
        payload_features = {}
        for f in all_features:
            if f in input_values:
                payload_features[f] = input_values[f]
            else:
                # default fallback
                payload_features[f] = 0.5
                
        # Send Request to Backend
        headers = {}
        if st.session_state["token"]:
            headers["Authorization"] = f"Bearer {st.session_state['token']}"
            
        try:
            with st.spinner("Processing inference..."):
                pred_res = requests.post(
                    f"{BACKEND_URL}/predict",
                    json={"features": payload_features},
                    headers=headers
                )
                
            if pred_res.status_code == 200:
                res = pred_res.json()
                st.success("Prediction succeeded!")
                
                # Display Cache Status Badge
                if res.get("cached"):
                    st.info("⚡ Fast Response: Loaded directly from Redis Cache")
                else:
                    st.info("⚙️ Model Inference: Computed by XGBoost model")

                # Columns for comparison
                col_ml, col_altman = st.columns(2)
                
                with col_ml:
                    color = res["risk_color"]
                    emoji = res["risk_emoji"]
                    risk = res["risk_tier"]
                    prob = res["probability"]
                    
                    st.markdown(
                        f'<div class="prediction-result" style="text-align:center;">'
                        f'<p style="color:#94a3b8;font-size:0.95rem;margin-bottom:0.5rem;text-transform:uppercase;">XGBoost Predictor Probability</p>'
                        f'<p style="font-size:3.5rem;font-weight:800;color:{color};margin:0;">'
                        f'{prob*100:.2f}%</p>'
                        f'<p class="risk-badge" style="background:{color}20;color:{color};">'
                        f'{emoji} {risk}</p>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    
                with col_altman:
                    if res.get("altman_z_score") is not None:
                        alt_color = res["altman_color"]
                        alt_emoji = res["altman_emoji"]
                        alt_zone = res["altman_zone"]
                        alt_z = res["altman_z_score"]
                        
                        st.markdown(
                            f'<div class="prediction-result" style="text-align:center;">'
                            f'<p style="color:#94a3b8;font-size:0.95rem;margin-bottom:0.5rem;text-transform:uppercase;">Altman Z-Score</p>'
                            f'<p style="font-size:3.5rem;font-weight:800;color:{alt_color};margin:0;">'
                            f'{alt_z:.3f}</p>'
                            f'<p class="risk-badge" style="background:{alt_color}20;color:{alt_color};">'
                            f'{alt_emoji} {alt_zone}</p>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.warning("Altman Z-Score features were not fully supplied.")
                        
                if st.session_state["token"]:
                    st.toast("💾 Prediction saved automatically to your profile in Supabase.")
            else:
                st.error(f"Inference request failed: {pred_res.json().get('detail', 'Unknown error')}")
        except Exception as e:
            st.error(f"Connection issue: {e}")

# ── PAGE: HISTORY ──
elif page == "📂 Prediction History":
    st.markdown("### Prediction History")
    st.caption("All prediction logs loaded dynamically from your user profile on Supabase Postgres.")
    
    headers = {"Authorization": f"Bearer {st.session_state['token']}"}
    try:
        res = requests.get(f"{BACKEND_URL}/predict/history", headers=headers)
        if res.status_code == 200:
            history = res.json()
            if not history:
                st.info("No past predictions logged yet. Run a prediction to see history.")
            else:
                records = []
                for entry in history:
                    res_obj = entry["prediction_result"]
                    records.append({
                        "Date": pd.to_datetime(entry["created_at"]).strftime("%Y-%m-%d %H:%M:%S"),
                        "Risk Tier": res_obj.get("risk_tier", "Unknown"),
                        "Probability": f"{res_obj.get('probability', 0)*100:.2f}%",
                        "Altman Z": f"{res_obj.get('altman_z_score', 0):.3f}" if res_obj.get('altman_z_score') is not None else "N/A",
                        "Altman Zone": res_obj.get("altman_zone", "N/A"),
                    })
                
                df_hist = pd.DataFrame(records)
                st.dataframe(df_hist, use_container_width=True)
                
                # Visual history charts
                st.markdown("#### Trend of Bankruptcy Risk Probability")
                prob_vals = [float(x["prediction_result"].get("probability", 0)) for x in history]
                dates = [pd.to_datetime(x["created_at"]) for x in history]
                
                chart_df = pd.DataFrame({"Date": dates, "Probability (%)": [p*100 for p in prob_vals]})
                fig = px.line(chart_df, x="Date", y="Probability (%)", markers=True, color_discrete_sequence=["#818cf8"])
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#94a3b8', family='Outfit'),
                    yaxis=dict(gridcolor='rgba(99, 102, 241, 0.1)'),
                    xaxis=dict(gridcolor='rgba(99, 102, 241, 0.1)'),
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Failed to load history logs from backend.")
    except Exception as e:
        st.error(f"Error fetching history: {e}")
