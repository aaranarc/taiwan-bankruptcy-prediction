"""
Taiwan Bankruptcy Prediction — Streamlit Application

A premium, interactive dashboard for predicting corporate bankruptcy
using XGBoost + SHAP explanations + Altman Z-Score comparison.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.evaluation import (
    load_model_artifacts,
    preprocess_input,
    predict_bankruptcy,
    get_risk_color,
    get_risk_emoji,
)
from src.altman import (
    compute_altman_z,
    classify_zone,
    get_zone_color,
    get_zone_emoji,
    ALTMAN_FEATURES,
)

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Taiwan Bankruptcy Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(99, 102, 241, 0.2);
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
    }
    .main-header h1 {
        color: #f1f5f9;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.15);
    }
    .metric-value {
        font-size: 2rem;
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
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.02em;
    }

    /* Prediction result */
    .prediction-result {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid rgba(99, 102, 241, 0.15);
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
    }

    /* Section dividers */
    .section-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.3), transparent);
        margin: 2rem 0;
    }

    /* Sidebar styling */
    .stSidebar > div {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ─── Load artifacts (cached) ─────────────────────────────────────────────────

@st.cache_resource
def get_artifacts():
    return load_model_artifacts()


@st.cache_data
def get_dataset():
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'raw', 'data.csv')
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    return df


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📊 Navigation")
    page = st.radio(
        "Go to",
        ["🏠 Overview", "🔍 Predict Bankruptcy", "📊 SHAP Explanations", "📈 Dataset Explorer"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<div style='color: #64748b; font-size: 0.8rem;'>"
        "Built with XGBoost + SHAP<br>"
        "Dataset: UCI Taiwan Economic Journal<br>"
        "Period: 1999–2009"
        "</div>",
        unsafe_allow_html=True,
    )


# ─── Header ──────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="main-header">'
    '<h1>📊 Taiwan Bankruptcy Predictor</h1>'
    '<p>XGBoost-powered corporate bankruptcy prediction with SHAP explanations & Altman Z-Score comparison</p>'
    '</div>',
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    artifacts = get_artifacts()
    df = get_dataset()

    if artifacts['test_metrics'] is None:
        st.error("⚠️ Model artifacts not found. Run `python train_and_save.py` first.")
        st.stop()

    metrics = artifacts['test_metrics']

    # ── Key Metrics Row ──
    st.markdown("### Model Performance")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f'<div class="metric-card">'
            f'<p class="metric-value">{metrics["roc_auc"]:.3f}</p>'
            f'<p class="metric-label">ROC-AUC</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="metric-card">'
            f'<p class="metric-value">{metrics["pr_auc"]:.3f}</p>'
            f'<p class="metric-label">PR-AUC</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="metric-card">'
            f'<p class="metric-value">{df.shape[0]:,}</p>'
            f'<p class="metric-label">Companies</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f'<div class="metric-card">'
            f'<p class="metric-value">{df.shape[1] - 1}</p>'
            f'<p class="metric-label">Features</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Two column layout ──
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### Class Distribution")
        class_counts = df['Bankrupt?'].value_counts().reset_index()
        class_counts.columns = ['Class', 'Count']
        class_counts['Label'] = class_counts['Class'].map({0: 'Healthy', 1: 'Bankrupt'})
        class_counts['Percentage'] = (class_counts['Count'] / class_counts['Count'].sum() * 100).round(2)

        fig = px.bar(
            class_counts,
            x='Label',
            y='Count',
            color='Label',
            color_discrete_map={'Healthy': '#6366f1', 'Bankrupt': '#ef4444'},
            text=class_counts.apply(lambda r: f"{r['Count']:,} ({r['Percentage']}%)", axis=1),
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8', family='Inter'),
            showlegend=False,
            xaxis=dict(title='', showgrid=False),
            yaxis=dict(title='Count', gridcolor='rgba(99, 102, 241, 0.1)'),
            margin=dict(t=20, b=20, l=40, r=20),
            height=350,
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("### About the Model")
        st.markdown("""
        **Algorithm:** XGBoost with `scale_pos_weight` for class imbalance handling

        **Preprocessing pipeline:**
        1. Train/test split (80/20, stratified)
        2. Winsorization of extreme outlier columns (bounds from training data)
        3. RobustScaler normalization

        **Hyperparameter tuning:** GridSearchCV with 3-fold CV optimizing ROC-AUC

        **Comparison baseline:** Altman Z-Score (1968) achieves ~0.92 ROC-AUC — our ML model improves on this with ~0.95 ROC-AUC
        """)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Feature categories ──
    st.markdown("### Feature Categories")
    categories = {
        '💰 Profitability': 17,
        '💧 Liquidity': 11,
        '⚖️ Leverage': 18,
        '🔄 Activity': 14,
        '💵 Cash Flow': 9,
        '📈 Growth': 8,
        '📊 Per Share & Size': 18,
    }

    cols = st.columns(len(categories))
    for i, (cat, count) in enumerate(categories.items()):
        with cols[i]:
            st.metric(cat, count)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT BANKRUPTCY
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Predict Bankruptcy":
    artifacts = get_artifacts()
    df = get_dataset()

    if artifacts['model'] is None:
        st.error("⚠️ Model artifacts not found. Run `python train_and_save.py` first.")
        st.stop()

    model = artifacts['model']
    scaler = artifacts['scaler']
    bounds = artifacts['bounds']
    feature_names = artifacts['feature_names']

    # ── Input mode selection ──
    input_mode = st.radio(
        "Choose input method",
        ["📝 Manual Input (Key Features)", "📁 CSV Upload (Batch)"],
        horizontal=True,
    )

    if input_mode == "📝 Manual Input (Key Features)":
        st.markdown("### Enter Company Financial Ratios")
        st.caption("Adjust the key financial ratios below. Remaining features are set to dataset median values.")

        # Get medians for defaults
        feature_df = df.drop(columns=['Bankrupt?'])
        medians = feature_df.median()

        # Top features to show (based on SHAP importance)
        key_features = [
            ('ROA(B) before interest and depreciation after tax', 'Profitability', 0.0, 1.0),
            ('Persistent EPS in the Last Four Seasons', 'Profitability', 0.0, 1.0),
            ('Net Income to Total Assets', 'Profitability', 0.0, 1.0),
            ('Debt ratio %', 'Leverage', 0.0, 1.0),
            ('Net worth/Assets', 'Leverage', 0.0, 1.0),
            ('Borrowing dependency', 'Leverage', 0.0, 1.0),
            ('Current Ratio', 'Liquidity', 0.0, 10.0),
            ('Working Capital to Total Assets', 'Liquidity', -1.0, 1.0),
            ('Retained Earnings to Total Assets', 'Profitability', 0.0, 1.0),
            ('Total Asset Turnover', 'Activity', 0.0, 1.0),
            ('Cash Flow to Total Assets', 'Cash Flow', -1.0, 1.0),
            ('Net Value Growth Rate', 'Growth', -1.0, 10.0),
            ('Interest Coverage Ratio (Interest expense to EBIT)', 'Leverage', 0.0, 1.0),
            ('Equity to Liability', 'Leverage', 0.0, 1.0),
        ]

        # Create input form
        input_values = {}
        cols = st.columns(2)
        for i, (feat, category, vmin, vmax) in enumerate(key_features):
            with cols[i % 2]:
                default_val = float(medians.get(feat, 0.5))
                input_values[feat] = st.number_input(
                    f"{feat}",
                    value=default_val,
                    format="%.6f",
                    help=f"Category: {category}",
                    key=f"input_{feat}",
                )

        # Build full feature row using medians for non-displayed features
        if st.button("🔮 Predict Bankruptcy Risk", type="primary", use_container_width=True):
            row = medians.copy()
            for feat, val in input_values.items():
                row[feat] = val

            input_df = pd.DataFrame([row])
            input_df = input_df[feature_names]

            # Preprocess and predict
            try:
                processed = preprocess_input(input_df, scaler, bounds, feature_names)
                result = predict_bankruptcy(model, processed)

                prob = result['probability'].iloc[0]
                risk = result['risk_tier'].iloc[0]
                pred = result['prediction'].iloc[0]

                # Altman Z-Score
                altman_z = compute_altman_z(pd.Series(row))
                altman_zone = classify_zone(altman_z)

                # ── Display results ──
                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                st.markdown("### Prediction Results")

                res_col1, res_col2, res_col3 = st.columns(3)

                with res_col1:
                    color = get_risk_color(risk)
                    emoji = get_risk_emoji(risk)
                    st.markdown(
                        f'<div class="prediction-result" style="text-align:center;">'
                        f'<p style="color:#94a3b8;font-size:0.9rem;margin-bottom:0.5rem;">ML PREDICTION</p>'
                        f'<p style="font-size:2.5rem;font-weight:800;color:{color};margin:0;">'
                        f'{prob*100:.1f}%</p>'
                        f'<p class="risk-badge" style="background:{color}20;color:{color};">'
                        f'{emoji} {risk}</p>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                with res_col2:
                    zone_color = get_zone_color(altman_zone)
                    zone_emoji = get_zone_emoji(altman_zone)
                    st.markdown(
                        f'<div class="prediction-result" style="text-align:center;">'
                        f'<p style="color:#94a3b8;font-size:0.9rem;margin-bottom:0.5rem;">ALTMAN Z-SCORE</p>'
                        f'<p style="font-size:2.5rem;font-weight:800;color:{zone_color};margin:0;">'
                        f'{altman_z:.3f}</p>'
                        f'<p class="risk-badge" style="background:{zone_color}20;color:{zone_color};">'
                        f'{zone_emoji} {altman_zone}</p>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                with res_col3:
                    verdict_color = '#ef4444' if pred == 1 else '#10b981'
                    verdict_label = 'BANKRUPT' if pred == 1 else 'HEALTHY'
                    verdict_emoji = '⚠️' if pred == 1 else '✅'
                    st.markdown(
                        f'<div class="prediction-result" style="text-align:center;">'
                        f'<p style="color:#94a3b8;font-size:0.9rem;margin-bottom:0.5rem;">FINAL VERDICT</p>'
                        f'<p style="font-size:2.5rem;font-weight:800;color:{verdict_color};margin:0;">'
                        f'{verdict_emoji}</p>'
                        f'<p class="risk-badge" style="background:{verdict_color}20;color:{verdict_color};">'
                        f'{verdict_label}</p>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # SHAP explanation for this prediction
                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
                st.markdown("### Why This Prediction?")
                try:
                    import shap
                    explainer = shap.TreeExplainer(model)
                    shap_vals = explainer.shap_values(processed)

                    fig_shap, ax = plt.subplots(figsize=(12, 6))
                    shap.summary_plot(
                        shap_vals, processed,
                        plot_type='bar', max_display=15,
                        show=False,
                    )
                    plt.tight_layout()
                    st.pyplot(fig_shap)
                    plt.close(fig_shap)
                except Exception as e:
                    st.warning(f"SHAP explanation unavailable: {e}")

            except Exception as e:
                st.error(f"Prediction error: {e}")

    else:  # CSV Upload
        st.markdown("### Upload Company Data")
        st.caption("Upload a CSV file with the same columns as the training dataset. Multiple rows = batch predictions.")

        uploaded = st.file_uploader("Choose CSV file", type=['csv'])

        if uploaded is not None:
            try:
                upload_df = pd.read_csv(uploaded)
                upload_df.columns = upload_df.columns.str.strip()
                st.success(f"✅ Loaded {upload_df.shape[0]} companies × {upload_df.shape[1]} features")

                # Preview
                with st.expander("📋 Preview uploaded data"):
                    st.dataframe(upload_df.head(10), use_container_width=True)

                if st.button("🔮 Run Batch Prediction", type="primary", use_container_width=True):
                    # Drop target if present
                    has_target = 'Bankrupt?' in upload_df.columns
                    pred_df = upload_df.copy()

                    processed = preprocess_input(pred_df, scaler, bounds, feature_names)
                    results = predict_bankruptcy(model, processed)

                    # Combine results
                    output = upload_df.copy()
                    output['ML_Probability'] = results['probability'].values
                    output['ML_Prediction'] = results['prediction'].values
                    output['Risk_Tier'] = results['risk_tier'].values

                    # Altman Z-Score
                    raw_for_altman = upload_df.copy()
                    if 'Bankrupt?' in raw_for_altman.columns:
                        raw_for_altman = raw_for_altman.drop(columns=['Bankrupt?'])

                    from src.altman import compute_altman_z_batch, ALTMAN_FEATURES
                    altman_cols = list(ALTMAN_FEATURES.values())
                    if all(c in upload_df.columns for c in altman_cols):
                        output['Altman_Z'] = compute_altman_z_batch(upload_df).values
                        output['Altman_Zone'] = output['Altman_Z'].apply(classify_zone)

                    st.markdown("### Results")

                    # Summary metrics
                    n_bankrupt = (results['prediction'] == 1).sum()
                    n_healthy = (results['prediction'] == 0).sum()
                    avg_prob = results['probability'].mean()

                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Predicted Healthy", f"{n_healthy}")
                    mc2.metric("Predicted Bankrupt", f"{n_bankrupt}")
                    mc3.metric("Avg Probability", f"{avg_prob:.2%}")

                    # Show results table
                    display_cols = ['ML_Probability', 'ML_Prediction', 'Risk_Tier']
                    if 'Altman_Z' in output.columns:
                        display_cols += ['Altman_Z', 'Altman_Zone']
                    if has_target:
                        display_cols = ['Bankrupt?'] + display_cols

                    st.dataframe(
                        output[display_cols].style.background_gradient(
                            subset=['ML_Probability'],
                            cmap='RdYlGn_r',
                        ),
                        use_container_width=True,
                    )

                    # Download button
                    csv_out = output.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results CSV",
                        csv_out,
                        "bankruptcy_predictions.csv",
                        "text/csv",
                        use_container_width=True,
                    )

            except Exception as e:
                st.error(f"Error processing file: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SHAP EXPLANATIONS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📊 SHAP Explanations":
    artifacts = get_artifacts()

    if artifacts['shap_values'] is None:
        st.error("⚠️ SHAP values not found. Run `python train_and_save.py` first.")
        st.stop()

    shap_values = artifacts['shap_values']
    x_test = artifacts['x_test']

    st.markdown("### Global Feature Importance (SHAP)")
    st.caption("Which features matter most for bankruptcy prediction across all test companies.")

    # Compute mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.DataFrame({
        'Feature': x_test.columns,
        'Mean |SHAP|': mean_abs_shap,
    }).sort_values('Mean |SHAP|', ascending=True).tail(20)

    fig = px.bar(
        shap_importance,
        x='Mean |SHAP|',
        y='Feature',
        orientation='h',
        color='Mean |SHAP|',
        color_continuous_scale='Viridis',
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8', family='Inter'),
        coloraxis_showscale=False,
        xaxis=dict(title='Mean |SHAP Value|', gridcolor='rgba(99, 102, 241, 0.1)'),
        yaxis=dict(title=''),
        margin=dict(t=20, b=20, l=20, r=20),
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # SHAP Beeswarm Plot
    st.markdown("### SHAP Beeswarm Plot")
    st.caption("Each dot = one company. Color = feature value (red=high, blue=low). Position = impact on prediction.")

    try:
        import shap
        fig_bee, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(shap_values, x_test, max_display=20, show=False)
        plt.tight_layout()
        st.pyplot(fig_bee)
        plt.close(fig_bee)
    except Exception as e:
        st.warning(f"Beeswarm plot unavailable: {e}")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Individual prediction explainer
    st.markdown("### Explain a Specific Prediction")
    st.caption("Select a test-set company to see why it was predicted as bankrupt or healthy.")

    y_test = artifacts['y_test']
    company_idx = st.slider(
        "Select company index",
        0, len(x_test) - 1, 0,
    )

    actual = "Bankrupt" if y_test.iloc[company_idx] == 1 else "Healthy"
    model = artifacts['model']
    prob = model.predict_proba(x_test.iloc[[company_idx]])[:, 1][0]

    ic1, ic2 = st.columns(2)
    ic1.metric("Actual", actual)
    ic2.metric("Predicted Probability", f"{prob:.2%}")

    # Top SHAP features for this company
    company_shap = shap_values[company_idx]
    feature_shap = pd.DataFrame({
        'Feature': x_test.columns,
        'SHAP Value': company_shap,
        'Feature Value': x_test.iloc[company_idx].values,
    }).sort_values('SHAP Value', key=abs, ascending=False).head(15)

    fig_local = go.Figure()
    colors = ['#ef4444' if v > 0 else '#6366f1' for v in feature_shap['SHAP Value']]

    fig_local.add_trace(go.Bar(
        x=feature_shap['SHAP Value'].values[::-1],
        y=feature_shap['Feature'].values[::-1],
        orientation='h',
        marker_color=colors[::-1],
        text=[f"{v:.4f}" for v in feature_shap['SHAP Value'].values[::-1]],
        textposition='outside',
    ))
    fig_local.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8', family='Inter'),
        xaxis=dict(title='SHAP Value (impact on prediction)', gridcolor='rgba(99, 102, 241, 0.1)'),
        yaxis=dict(title=''),
        margin=dict(t=20, b=20, l=20, r=20),
        height=500,
    )
    st.plotly_chart(fig_local, use_container_width=True)
    st.caption("🔴 Red = pushes toward bankruptcy | 🟣 Purple = pushes toward healthy")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DATASET EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Dataset Explorer":
    df = get_dataset()

    st.markdown("### Interactive Dataset Explorer")

    tab1, tab2, tab3 = st.tabs(["📊 Distributions", "🔗 Correlations", "📋 Raw Data"])

    with tab1:
        st.markdown("#### Feature Distribution")
        feature_list = [c for c in df.columns if c != 'Bankrupt?']
        selected_feature = st.selectbox("Select feature", feature_list)

        fig = px.histogram(
            df,
            x=selected_feature,
            color=df['Bankrupt?'].map({0: 'Healthy', 1: 'Bankrupt'}),
            color_discrete_map={'Healthy': '#6366f1', 'Bankrupt': '#ef4444'},
            barmode='overlay',
            opacity=0.7,
            nbins=50,
            labels={'color': 'Class'},
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8', family='Inter'),
            xaxis=dict(gridcolor='rgba(99, 102, 241, 0.1)'),
            yaxis=dict(title='Count', gridcolor='rgba(99, 102, 241, 0.1)'),
            margin=dict(t=20, b=20, l=40, r=20),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Box plot comparison
        st.markdown("#### By Bankruptcy Status")
        fig_box = px.box(
            df,
            x=df['Bankrupt?'].map({0: 'Healthy', 1: 'Bankrupt'}),
            y=selected_feature,
            color=df['Bankrupt?'].map({0: 'Healthy', 1: 'Bankrupt'}),
            color_discrete_map={'Healthy': '#6366f1', 'Bankrupt': '#ef4444'},
        )
        fig_box.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8', family='Inter'),
            showlegend=False,
            xaxis=dict(title=''),
            yaxis=dict(gridcolor='rgba(99, 102, 241, 0.1)'),
            margin=dict(t=20, b=20, l=40, r=20),
            height=400,
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with tab2:
        st.markdown("#### Correlation Heatmap (Top Features)")
        n_features = st.slider("Number of top features", 5, 30, 15)

        target = 'Bankrupt?'
        correlations = df.corr(numeric_only=True)[target].drop(target)
        top_n = correlations.abs().sort_values(ascending=False).head(n_features).index.tolist()
        corr_cols = top_n + [target]
        corr_matrix = df[corr_cols].corr()

        fig_corr = px.imshow(
            corr_matrix,
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            aspect='auto',
            text_auto='.2f',
        )
        fig_corr.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8', family='Inter', size=9),
            margin=dict(t=20, b=20, l=20, r=20),
            height=700,
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab3:
        st.markdown("#### Raw Dataset")

        class_filter = st.radio(
            "Filter by class",
            ["All", "Healthy Only", "Bankrupt Only"],
            horizontal=True,
        )

        filtered_df = df.copy()
        if class_filter == "Healthy Only":
            filtered_df = df[df['Bankrupt?'] == 0]
        elif class_filter == "Bankrupt Only":
            filtered_df = df[df['Bankrupt?'] == 1]

        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=500,
        )

        st.caption(f"Showing {len(filtered_df):,} of {len(df):,} records")

        # Download
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            "📥 Download filtered data",
            csv_data,
            "taiwan_bankruptcy_data.csv",
            "text/csv",
        )
