"""Rewrite the README metrics section from reports/metrics.json, so no number is typed by hand.

Replaces everything between <!-- metrics:start --> and <!-- metrics:end --> in README.md.
Usage: python -m training.render_readme
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(ROOT, 'README.md')
METRICS = os.path.join(ROOT, 'reports', 'metrics.json')
NAMES = {
    'logistic_regression': 'Logistic Regression (standardised, balanced weights)',
    'random_forest': 'Random Forest (balanced weights)',
    'xgboost_unconstrained': 'XGBoost + `scale_pos_weight`, unconstrained',
    'xgboost_monotone': 'XGBoost + `scale_pos_weight`, monotone constraints',
}


def pm(m, pct=False, digits=3):
    """Format a {'mean', 'std'} summary as 'mean ± std'."""
    if pct:
        return f"{m['mean'] * 100:.1f}% ± {m['std'] * 100:.1f}%"
    return f"{m['mean']:.{digits}f} ± {m['std']:.{digits}f}"


def render(r, deployed):
    cv, d, old, alt = r['cv'], r['data'], r['old_single_split'], r['altman_z_benchmark']
    cal, dep, thr = r['calibration'], r['deployed_model'], r['threshold']
    n = r['nested']
    nm = n['metrics']
    folds = cv['n_splits'] * cv['n_repeats']
    rows = [f"| {NAMES[k]} | {pm(m['roc_auc'])} | {pm(m['pr_auc'])} | {pm(m['brier'], digits=4)} | "
            f"{pm(m['recall'], pct=True)} |" for k, m in r['models'].items()]
    rows.insert(0, f"| Altman Z-Score (1968), scored as −Z | {pm(alt['roc_auc'])} | {pm(alt['pr_auc'])} | — | — |")
    rows.append(f"| {dep['description']} | {pm(dep['roc_auc'])} | {pm(dep['pr_auc'])} | "
                f"{pm(dep['brier'], digits=4)} | — |")
    cost = r['constraint_cost']
    grade_rows = [f"| {g['grade']} | < {g['max_pd']:.0%} | {g['n']:,} | {g['defaults']:,} | "
                  f"{g['mean_predicted_pd']:.1%} | {g['observed_default_rate']:.1%} |" for g in n['grades']]
    methods = ', '.join(f"{k} in {v} of {folds} folds" for k, v in n['choices']['method'].items())
    commit = r['git']['commit'][:7] + (' (uncommitted changes)' if r['git']['dirty'] else '')
    ncommit = n['git']['commit'][:7] + (' (uncommitted changes)' if n['git']['dirty'] else '')
    return '\n'.join([
        f"All {d['rows']:,} firms ({d['bankrupt']} bankrupt, base rate {d['base_rate']:.1%}), "
        f"{cv['n_repeats']}×{cv['n_splits']} repeated stratified cross-validation, mean ± std over {folds} test folds. "
        f"The old README used one 80/20 split with only {old['test_bankrupt']} bankrupt firms in its test set.",
        '',
        '### Headline: nested cross-validation (after)',
        '',
        "Every choice is made inside each outer training fold: hyperparameter search "
        f"({n['inner']['search']}), calibration method ({n['inner']['calibration_rule']}) and the "
        f"{thr['cost_ratio_fn_to_fp']}:1 cost-based threshold. Each outer test fold is scored once. The deployed "
        "model is produced by the same procedure (`training/procedure.py`) run on all data.",
        '',
        '| Deployed procedure, nested CV | ROC-AUC | PR-AUC | Brier |',
        '|-------------------------------|---------|--------|-------|',
        f"| Monotone XGBoost, tuned + calibrated inside each fold | {pm(nm['roc_auc'])} | {pm(nm['pr_auc'])} | "
        f"{pm(nm['brier'], digits=4)} |",
        '',
        '| Decision rule (nested, same outer folds) | Precision | Recall | Firms flagged |',
        '|------------------------------------------|-----------|--------|---------------|',
        f"| Uncalibrated monotone XGBoost at 0.50 | {pm(nm['uncalibrated_precision_at_0_50'], pct=True)} | "
        f"{pm(nm['uncalibrated_recall_at_0_50'], pct=True)} | — |",
        f"| Calibrated model at the fold's cost-based threshold (mean {nm['threshold']['mean']:.3f}) | "
        f"{pm(nm['precision_at_threshold'], pct=True)} | {pm(nm['recall_at_threshold'], pct=True)} | "
        f"{pm(nm['flagged_share'], pct=True)} |",
        '',
        f"Calibration method chosen: {methods}. Deployed model (`reports/deployed_model.json`): "
        f"{deployed['calibration_method']} calibration, threshold {deployed['threshold']:.3f}, "
        f"max_depth {deployed['xgb_params']['max_depth']}, learning_rate {deployed['xgb_params']['learning_rate']}, "
        f"{deployed['xgb_params']['n_estimators']} trees.",
        '',
        '**Risk grades are rank-ordered, not a calibration claim.** Out-of-fold, nested:',
        '',
        '| Grade | PD band | Firm-predictions | Defaults | Mean predicted PD | Observed default rate |',
        '|-------|---------|------------------|----------|-------------------|-----------------------|',
        *grade_rows,
        '',
        f"Observed default rate rises with every grade: **{'yes' if n['grades_rank_ordered'] else 'NO'}**. Counts pool "
        f"all {cv['n_repeats']} repeats, so each firm appears {cv['n_repeats']} times.",
        '',
        '### Before: non-nested comparison',
        '',
        "Hyperparameters fixed from the original notebook (tuned on a split that overlaps these folds), so these rows are "
        "slightly optimistic. Kept for the model comparison and the cost of the monotone constraints.",
        '',
        '| Model | ROC-AUC | PR-AUC | Brier | Recall @ 0.50 (uncalibrated) |',
        '|-------|---------|--------|-------|------------------------------|',
        *rows,
        '',
        f"**Cost of the monotone constraints** ({r['monotone_constraints']['n_constrained']} features, "
        "`config/monotone_constraints.json`), paired by fold: "
        f"ROC-AUC {cost['roc_auc']['mean_difference']:+.4f} (± {cost['roc_auc']['std_difference']:.4f}), "
        f"PR-AUC {cost['pr_auc']['mean_difference']:+.4f} (± {cost['pr_auc']['std_difference']:.4f}). "
        f"Calibration: {cal['chosen_method']}; reliability plot `reports/calibration.png`.",
        '',
        f"> **Altman Z-Score:** {alt['caveat']} Lower Z means more risk, so the score is negated before computing AUC.",
        '>',
        *[f"> **Caveat:** {c}" for c in r['caveats']],
        '',
        f"<sub>Generated by `python -m training.render_readme` from `reports/metrics.json` (non-nested: commit "
        f"`{commit}`, {r['generated_at']}; nested: commit `{ncommit}`, {n['generated_at']}) and "
        f"`reports/deployed_model.json`.</sub>",
    ])


def main():
    with open(METRICS) as fh:
        report = json.load(fh)
    with open(os.path.join(ROOT, 'reports', 'deployed_model.json')) as fh:
        deployed = json.load(fh)
    with open(README) as fh:
        text = fh.read()
    new, n = re.subn(r'(<!-- metrics:start -->\n).*?(<!-- metrics:end -->)',
                     lambda m: m.group(1) + render(report, deployed) + '\n' + m.group(2), text, flags=re.S)
    if n != 1:
        raise SystemExit('README.md must contain exactly one <!-- metrics:start --> ... <!-- metrics:end --> block')
    with open(README, 'w') as fh:
        fh.write(new)
    print('README.md metrics section updated')


if __name__ == '__main__':
    main()
