"""What-if analysis on top of a scoring function: counterfactual "path to safety" and stress tests.

Both take `score(df) -> array of PDs` so they can be tested with a toy model.
`quantiles` is a DataFrame indexed by percentile 0..100 (one column per feature) of the raw training data.
"""
from itertools import combinations, product

import numpy as np
import pandas as pd

from app.services.risk import apply_linked, grade_for, grade_max_pd

PCTS = np.arange(101, dtype=float)


def to_percentile(value: float, q: pd.Series) -> float:
    """Where a value sits in the training distribution (0-100). Ties/flat regions resolve to the middle."""
    lo = np.interp(value, q.to_numpy(), PCTS, left=0.0, right=100.0)
    hi = 100.0 - np.interp(-value, -q.to_numpy()[::-1], PCTS, left=0.0, right=100.0)
    return float((lo + hi) / 2)


def from_percentile(pct: float, q: pd.Series) -> float:
    return float(np.interp(np.clip(pct, 0, 100), PCTS, q.to_numpy()))


def counterfactual(features: dict, score, quantiles: pd.DataFrame, policy: dict, directions: dict,
                   grades: list, target_grade: str) -> dict:
    """Smallest change to at most `max_features_changed` adjustable ratios that brings PD into `target_grade`.

    "Smallest" = fewest ratios first, then the smallest total move measured in training-data percentile points.
    Candidate values only move in the risk-reducing direction and stay within the 1st-99th percentile.
    """
    cfg = policy['counterfactual']
    target_max = grade_max_pd(target_grade, grades)
    base = pd.DataFrame([features])
    pd_before = float(score(base)[0])
    result = {'pd_before': pd_before, 'grade_before': grade_for(pd_before, grades), 'target_grade': target_grade}
    if pd_before < target_max:
        return {**result, 'already_at_target': True, 'reachable': True, 'pd_after': pd_before,
                'grade_after': result['grade_before'], 'changes': []}

    # Candidate values per feature, walking from the current percentile to the safe bound (1st or 99th).
    grids = {}
    cur = {f: to_percentile(features[f], quantiles[f]) for f in cfg['adjustable_features']}
    for f in cfg['adjustable_features']:
        q, cur_pct = quantiles[f], cur[f]
        safe_end = 1.0 if directions[f] == 1 else 99.0  # +1: higher = riskier, so move down
        if (safe_end - cur_pct) * -directions[f] <= 0:  # already at or beyond the safe bound
            continue
        pcts = np.linspace(cur_pct, safe_end, cfg['steps_per_feature'] + 1)[1:]
        grids[f] = [(p, from_percentile(p, q)) for p in pcts]

    for k in range(1, cfg['max_features_changed'] + 1):
        best = None
        for combo in combinations(grids, k):
            options = list(product(*(grids[f] for f in combo)))
            cand = pd.DataFrame([features] * len(options))
            for j, f in enumerate(combo):
                cand[f] = [opt[j][1] for opt in options]
            cand = apply_linked(cand, policy['linked_features'])
            pds = np.asarray(score(cand))
            ok = np.flatnonzero(pds < target_max)
            if not len(ok):
                continue
            moves = np.array([sum(abs(opt[j][0] - cur[f]) for j, f in enumerate(combo)) for opt in options])
            i = ok[np.argmin(moves[ok])]
            if best is None or moves[i] < best[0]:
                best = (moves[i], combo, options[i], float(pds[i]))
        if best:
            _, combo, option, pd_after = best
            changes = [{'feature': f, 'from_value': float(features[f]), 'to_value': option[j][1],
                        'from_percentile': cur[f], 'to_percentile': option[j][0]}
                       for j, f in enumerate(combo)]
            return {**result, 'already_at_target': False, 'reachable': True, 'pd_after': pd_after,
                    'grade_after': grade_for(pd_after, grades), 'changes': changes}
    return {**result, 'already_at_target': False, 'reachable': False, 'pd_after': None, 'grade_after': None,
            'changes': []}


def stress(features: dict, score, quantiles: pd.DataFrame, policy: dict, grades: list, shocks: dict) -> dict:
    """Apply configured shocks to the input ratios and return PD before and after.

    Results are clipped to the min/max seen in training: the model has no information beyond that range.
    """
    cfg = policy['stress_shocks']
    base = pd.DataFrame([features])
    shocked = base.copy()
    for name, amount in shocks.items():
        if name not in cfg:
            raise ValueError(f"Unknown shock {name!r}; expected one of {sorted(cfg)}")
        spec = cfg[name]
        if not spec['min'] <= amount <= spec['max']:
            raise ValueError(f"{name} must be between {spec['min']} and {spec['max']}, got {amount}")
        for col in spec['columns']:
            if spec['kind'] == 'relative':
                shocked[col] = shocked[col] * (1 + amount / 100)
            else:  # percentile_shift, kept within P1-P99: P0/P100 are single extreme outliers, not a "shock"
                current = shocked[col].iloc[0]
                target = np.clip(to_percentile(current, quantiles[col]) + amount, 1.0, 99.0)
                moved = from_percentile(target, quantiles[col])
                # Never move against the shock (a firm already below P1 stays where it is).
                shocked[col] = min(current, moved) if amount < 0 else max(current, moved)
    shocked = apply_linked(shocked, policy['linked_features'])

    touched, clipped = [], []
    for col in shocked.columns:
        lo, hi = quantiles[col].iloc[0], quantiles[col].iloc[-1]
        if shocked[col].iloc[0] < lo or shocked[col].iloc[0] > hi:
            clipped.append(col)
            shocked[col] = shocked[col].clip(lo, hi)
        if not np.isclose(shocked[col].iloc[0], base[col].iloc[0], rtol=0, atol=1e-12):  # float noise isn't a change
            touched.append({'feature': col, 'before': float(base[col].iloc[0]), 'after': float(shocked[col].iloc[0])})

    pd_before, pd_after = (float(v) for v in score(pd.concat([base, shocked], ignore_index=True)))
    return {'pd_before': pd_before, 'grade_before': grade_for(pd_before, grades),
            'pd_after': pd_after, 'grade_after': grade_for(pd_after, grades),
            'shocks': shocks, 'changed_features': touched, 'clipped_to_training_range': clipped}
