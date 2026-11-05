"""Model card: everything here is read from reports/metrics.json and config/monotone_constraints.json."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.components import cards, empty_state, hero, load_json, plot
from ui.theme import ACCENT, BORDER, GRADE_COLORS, MUTED, RISK_AMBER, TEMPLATE, fmt_pd

hero("Credit risk desk", "Model card",
     "How the model was evaluated, how well its probabilities are calibrated, and where it should not be used. "
     "Every number on this page is read from reports/metrics.json.")

r = load_json("reports/metrics.json")
constraints = load_json("config/monotone_constraints.json")
if not r:
    empty_state("No evaluation report", "Run python -m training.cv_evaluate to generate reports/metrics.json.")
    st.stop()


def pm(m: dict, digits: int = 3) -> str:
    return f"{m['mean']:.{digits}f} ± {m['std']:.{digits}f}"


cv, dep, thr = r["cv"], r["deployed_model"], r["threshold"]
st.caption(f"{cv['n_repeats']}×{cv['n_splits']} repeated stratified cross-validation · {r['data']['rows']:,} firms, "
           f"{r['data']['bankrupt']} bankrupt ({r['data']['base_rate']:.1%}) · mean ± std over "
           f"{cv['n_splits'] * cv['n_repeats']} folds · report commit {r['git']['commit'][:7]}, {r['generated_at'][:10]}")
cards([
    ("ROC-AUC", f"{dep['roc_auc']['mean']:.3f}", f"± {dep['roc_auc']['std']:.3f} · ranking quality"),
    ("PR-AUC", f"{dep['pr_auc']['mean']:.3f}", f"± {dep['pr_auc']['std']:.3f} · base rate {r['data']['base_rate']:.3f}"),
    ("Brier score", f"{dep['brier']['mean']:.4f}", f"± {dep['brier']['std']:.4f} · calibrated PD, lower is better"),
    ("Decision threshold", fmt_pd(thr["chosen"]), f"missed default = {thr['cost_ratio_fn_to_fp']}× false alarm"),
    ("Recall at threshold", f"{thr['at_chosen']['recall']['mean']:.0%}",
     f"± {thr['at_chosen']['recall']['std']:.0%} · defaults caught"),
    ("Precision at threshold", f"{thr['at_chosen']['precision']['mean']:.0%}",
     f"± {thr['at_chosen']['precision']['std']:.0%} · flags that defaulted"),
])

st.subheader("Model comparison")
names = {"logistic_regression": "Logistic regression", "random_forest": "Random forest",
         "xgboost_unconstrained": "XGBoost, unconstrained", "xgboost_monotone": "XGBoost, monotone"}
rows = [{"Model": "Altman Z (1968), as −Z", "ROC-AUC": pm(r["altman_z_benchmark"]["roc_auc"]),
         "PR-AUC": pm(r["altman_z_benchmark"]["pr_auc"])}]
rows += [{"Model": names[k], "ROC-AUC": pm(m["roc_auc"]), "PR-AUC": pm(m["pr_auc"])} for k, m in r["models"].items()]
rows.append({"Model": f"{dep['description']} (deployed)", "ROC-AUC": pm(dep["roc_auc"]), "PR-AUC": pm(dep["pr_auc"])})
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

left, right = st.columns(2, gap="large")
with left:
    st.subheader("Calibration")
    def nonzero(b):  # a decile with zero observed defaults has no point on a log axis
        pts = [(x, y) for x, y in zip(b["mean_predicted"], b["observed"]) if y > 0]
        return [p[0] for p in pts], [p[1] for p in pts]

    bins = {k: nonzero(v) for k, v in r["calibration"]["reliability_bins"].items()}
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1e-3, 1], y=[1e-3, 1], mode="lines", name="Perfect",
                             line=dict(color=MUTED, dash="dash", width=1)))
    fig.add_trace(go.Scatter(x=bins["uncalibrated_monotone"][0], y=bins["uncalibrated_monotone"][1],
                             mode="lines+markers",
                             name="Before calibration", line=dict(color=RISK_AMBER)))
    fig.add_trace(go.Scatter(x=bins["calibrated"][0], y=bins["calibrated"][1],
                             mode="lines+markers", name=f"Calibrated ({r['calibration']['chosen_method']})",
                             line=dict(color=ACCENT, width=3)))
    fig.update_layout(template=TEMPLATE, height=360, title="Predicted PD vs observed default rate (deciles)",
                      xaxis=dict(type="log", title="mean predicted PD", dtick=1, tickformat=".1%"),
                      yaxis=dict(type="log", title="observed rate", dtick=1, tickformat=".1%"))
    plot(fig)
    st.caption(f"Brier score {pm(dep['brier'], 4)} after calibration vs "
               f"{pm(r['models']['xgboost_monotone']['brier'], 4)} before. Bins with zero defaults can't be drawn "
               "on a log axis. Rule for choosing the method: " + r["calibration"]["rule"] + ".")
with right:
    st.subheader("Risk grades")
    g = pd.DataFrame(r["grades"])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=g.grade, y=g.observed_default_rate, name="Observed default rate",
                         marker_color=[GRADE_COLORS[x] for x in g.grade]))
    fig.add_trace(go.Scatter(x=g.grade, y=g.mean_pd, name="Mean predicted PD", mode="markers",
                             marker=dict(color="#E5EAF3", size=10, symbol="diamond", line=dict(color=BORDER))))
    fig.update_layout(template=TEMPLATE, height=360, title="Default rate rises with grade (out-of-fold)",
                      yaxis=dict(tickformat=".0%"))
    plot(fig)
    st.dataframe(pd.DataFrame({"Grade": g.grade, "PD band": [f"< {m:.0%}" for m in g.max_pd],
                               "Firm-predictions": g.n, "Defaults": g.defaults,
                               "Observed": g.observed_default_rate.map(lambda v: f"{v:.1%}"),
                               "Mean PD": g.mean_pd.map(lambda v: f"{v:.1%}")}),
                 hide_index=True, use_container_width=True)

st.subheader("Monotone constraints")
cost = r["constraint_cost"]
st.markdown(f"{r['monotone_constraints']['n_constrained']} features are constrained so that, for example, more "
            f"leverage can never lower the predicted risk. Measured cost, paired by fold: ROC-AUC "
            f"**{cost['roc_auc']['mean_difference']:+.4f}** (± {cost['roc_auc']['std_difference']:.4f}), PR-AUC "
            f"**{cost['pr_auc']['mean_difference']:+.4f}** (± {cost['pr_auc']['std_difference']:.4f}).")
if constraints:
    st.dataframe(pd.DataFrame([{"Feature": f, "Higher value": "raises risk" if c["direction"] == 1 else "lowers risk",
                                "Why": c["reason"]} for f, c in constraints["constraints"].items()]),
                 hide_index=True, use_container_width=True)
    with st.expander("Deliberately not constrained"):
        for f, why in constraints["excluded_on_purpose"].items():
            st.markdown(f"- **{f}**: {why}")

st.subheader("Known limitations")
st.markdown("""
- **Old, single-market data.** Taiwanese listed companies, 1999–2009, collected from the Taiwan Economic Journal; bankruptcy as defined by Taiwan Stock Exchange business regulations ([UCI dataset 572](https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction)). Accounting rules, rates and industries have changed since; there is no out-of-time or out-of-country test.
- **Not a credit decision tool.** A portfolio demo for learning. PDs are relative to this dataset's base rate, not a regulatory PD.
- **Pre-normalised inputs.** The ratios were already scaled to 0–1 by the data provider, so inputs are not raw accounting numbers, and the stress shocks are approximations.
- **Slight optimism.** XGBoost hyperparameters were tuned in the original notebook on a split that overlaps these CV folds, and the threshold is chosen on the same out-of-fold predictions it is evaluated on.
- **Counterfactuals hold other ratios fixed.** Real changes move several ratios at once.
""")
for c in r["caveats"]:
    st.caption(c)
