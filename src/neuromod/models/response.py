"""Acute rTMS response classifier (XGBoost) with a logistic-regression baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from neuromod.models.features import encode

PARAM_SPACE = {
    "n_estimators": randint(150, 600),
    "max_depth": randint(2, 5),
    "learning_rate": loguniform(0.01, 0.15),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.5, 0.5),
    "min_child_weight": randint(1, 10),
    "reg_lambda": loguniform(0.1, 10),
}


@dataclass
class ResponseResult:
    model: XGBClassifier
    baseline: object
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    idx_test: np.ndarray
    p_test: np.ndarray
    p_test_baseline: np.ndarray
    metrics: dict
    best_params: dict


def bootstrap_ci(y, p, metric, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        stats.append(metric(y[idx], p[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def classification_metrics(y, p) -> dict:
    auc = roc_auc_score(y, p)
    lo, hi = bootstrap_ci(y, p, roc_auc_score)
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=8, strategy="quantile")
    return {
        "auroc": float(auc),
        "auroc_95ci": [lo, hi],
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": float(np.mean(np.abs(frac_pos - mean_pred))),
        "prevalence": float(np.mean(y)),
    }


def train_response_model(
    df: pd.DataFrame,
    target: str = "responder",
    test_size: float = 0.25,
    n_iter: int = 20,
    seed: int = 7,
) -> ResponseResult:
    X = encode(df)
    y = df[target].to_numpy()
    idx = np.arange(len(df))
    X_tr, X_te, y_tr, y_te, _, idx_te = train_test_split(
        X, y, idx, test_size=test_size, stratify=y, random_state=seed
    )

    search = RandomizedSearchCV(
        XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                      random_state=seed, n_jobs=2),
        PARAM_SPACE,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=StratifiedKFold(5, shuffle=True, random_state=seed),
        random_state=seed,
        n_jobs=1,
    )
    search.fit(X_tr, y_tr)
    model: XGBClassifier = search.best_estimator_
    p_te = model.predict_proba(X_te)[:, 1]

    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    baseline.fit(X_tr, y_tr)
    p_te_base = baseline.predict_proba(X_te)[:, 1]

    metrics = {
        "xgboost": classification_metrics(y_te, p_te),
        "logistic_regression": classification_metrics(y_te, p_te_base),
        "cv_auroc_xgboost": float(search.best_score_),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
    }
    return ResponseResult(
        model=model,
        baseline=baseline,
        X_train=X_tr,
        X_test=X_te,
        y_train=y_tr,
        y_test=y_te,
        idx_test=idx_te,
        p_test=p_te,
        p_test_baseline=p_te_base,
        metrics=metrics,
        best_params={k: (float(v) if isinstance(v, float) else int(v))
                     for k, v in search.best_params_.items()},
    )
