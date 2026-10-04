"""Heterogeneous treatment effects and predictive enrichment for trial design.

A T-learner (one XGBoost model per randomized arm) estimates each veteran's
individual benefit from active rTMS over sham. Ranking patients by that estimate
lets us ask a study-design question: if a trial enrolled only the patients the
model predicts will benefit most, how many fewer would need to be randomized?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr, spearmanr
from xgboost import XGBClassifier

from neuromod.models.features import encode

ARM_FEATURES_DROPPED = ["active", "target"]


def n_per_arm_two_proportions(p1: float, p0: float, alpha: float = 0.05,
                              power: float = 0.80) -> float:
    """Sample size per arm for a two-sided two-proportion z-test."""
    if p1 <= p0:
        return float("inf")
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    pbar = (p1 + p0) / 2
    num = (za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p1 * (1 - p1) + p0 * (1 - p0))) ** 2
    return float(np.ceil(num / (p1 - p0) ** 2))


@dataclass
class HTEResult:
    cate_pred: np.ndarray
    cate_true: np.ndarray
    enrichment: pd.DataFrame
    metrics: dict


def _arm_model(seed: int) -> XGBClassifier:
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.04, subsample=0.8,
                         colsample_bytree=0.8, min_child_weight=5, eval_metric="logloss",
                         random_state=seed, n_jobs=2)


def t_learner(df: pd.DataFrame, truth: pd.DataFrame, train_idx: np.ndarray,
              test_idx: np.ndarray, seed: int = 7) -> HTEResult:
    # Patient characteristics only; the arm is what we intervene on. Sessions completed
    # is kept because both arms receive (active or sham) sessions.
    feats = [c for c in encode(df).columns if c.split("=")[0] not in ARM_FEATURES_DROPPED]
    X = encode(df)[feats]
    y = df["responder"].to_numpy()
    a = df["active"].to_numpy()

    tr = np.zeros(len(df), bool)
    tr[train_idx] = True
    m1 = _arm_model(seed).fit(X[tr & (a == 1)], y[tr & (a == 1)])
    m0 = _arm_model(seed).fit(X[tr & (a == 0)], y[tr & (a == 0)])

    Xte = X.iloc[test_idx]
    cate = m1.predict_proba(Xte)[:, 1] - m0.predict_proba(Xte)[:, 1]
    true_cate = truth["true_cate"].to_numpy()[test_idx]
    p1_true = truth["p_response_if_active"].to_numpy()[test_idx]
    p0_true = truth["p_response_if_sham"].to_numpy()[test_idx]

    rows = []
    order = np.argsort(-cate)
    for frac in [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2]:
        sel = order[: max(10, int(round(frac * len(order))))]
        p1, p0 = p1_true[sel].mean(), p0_true[sel].mean()
        n_arm = n_per_arm_two_proportions(p1, p0)
        rows.append({
            "enrolled_fraction": frac,
            "true_response_active": p1,
            "true_response_sham": p0,
            "true_effect": p1 - p0,
            "n_per_arm": n_arm,
            "n_randomized": 2 * n_arm,
            "n_screened": 2 * n_arm / frac,
        })
    enrichment = pd.DataFrame(rows)

    # Oracle ranking: same curve if we could rank by the true effect.
    oracle = np.argsort(-true_cate)
    sel = oracle[: int(round(0.5 * len(oracle)))]
    oracle_n = n_per_arm_two_proportions(p1_true[sel].mean(), p0_true[sel].mean())

    full = enrichment.iloc[0]
    half = enrichment.loc[enrichment.enrolled_fraction == 0.5].iloc[0]
    metrics = {
        "cate_pearson_r": float(pearsonr(cate, true_cate)[0]),
        "cate_spearman_rho": float(spearmanr(cate, true_cate)[0]),
        "ate_true_test": float(true_cate.mean()),
        "ate_pred_test": float(cate.mean()),
        "n_randomized_all_comers": float(full.n_randomized),
        "n_randomized_top50pct": float(half.n_randomized),
        "n_randomized_top50pct_oracle": float(2 * oracle_n),
        "randomized_sample_reduction_top50pct": float(1 - half.n_randomized / full.n_randomized),
    }
    return HTEResult(cate_pred=cate, cate_true=true_cate, enrichment=enrichment, metrics=metrics)
