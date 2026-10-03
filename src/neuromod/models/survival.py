"""Time-to-relapse models with scikit-survival plus an XGBoost Cox model.

All models predict relapse risk from *pre-treatment* information (baseline clinical
features, randomized arm, protocol and adherence). Sustained cessation is the
complement: surviving relapse-free to the end of the 52-week follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.util import Surv
from xgboost import XGBRegressor

from neuromod.models.features import encode

EVAL_TIMES = np.array([4.0, 8.0, 12.0, 26.0, 39.0, 48.0])


class XGBCox:
    """XGBoost with the ``survival:cox`` objective and a Breslow baseline hazard.

    Exposes ``predict`` (log-hazard risk score) and ``predict_survival_function`` so it
    can be scored with the same scikit-survival metrics as the other models.
    """

    def __init__(self, **params):
        defaults = dict(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8,
                        colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
                        random_state=0, n_jobs=2)
        defaults.update(params)
        self.model = XGBRegressor(objective="survival:cox", tree_method="hist", **defaults)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> XGBCox:
        time, event = y["time"], y["event"]
        label = np.where(event, time, -time)  # xgboost convention: negative = censored
        self.model.fit(X, label)
        risk = self.predict(X)
        # Breslow estimator of the cumulative baseline hazard
        order = np.argsort(time)
        t_sorted, e_sorted, r_sorted = time[order], event[order], np.exp(risk[order])
        uniq = np.unique(t_sorted[e_sorted])
        at_risk = np.array([r_sorted[t_sorted >= t].sum() for t in uniq])
        deaths = np.array([np.sum((t_sorted == t) & e_sorted) for t in uniq])
        self.event_times_ = uniq
        self.cum_baseline_hazard_ = np.cumsum(deaths / at_risk)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X, output_margin=True)

    def survival_at(self, X: pd.DataFrame, times: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.event_times_, times, side="right") - 1
        H0 = np.where(idx >= 0, self.cum_baseline_hazard_[np.clip(idx, 0, None)], 0.0)
        return np.exp(-np.exp(self.predict(X))[:, None] * H0[None, :])


@dataclass
class SurvivalResult:
    models: dict
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    idx_test: np.ndarray
    metrics: dict


def make_surv(df: pd.DataFrame) -> np.ndarray:
    return Surv.from_arrays(
        event=df["relapse"].astype(bool).to_numpy(),
        time=df["weeks_to_relapse_or_censor"].to_numpy(),
        name_event="event",
        name_time="time",
    )


def _survival_matrix(model, X, times) -> np.ndarray:
    if isinstance(model, XGBCox):
        return model.survival_at(X, times)
    fns = model.predict_survival_function(X)
    return np.vstack([fn(times) for fn in fns])


def default_models(seed: int = 7) -> dict:
    return {
        "CoxPH (scikit-survival)": make_pipeline(StandardScaler(),
                                                 CoxPHSurvivalAnalysis(alpha=0.1)),
        "Random Survival Forest": RandomSurvivalForest(
            n_estimators=300, min_samples_leaf=15, max_features="sqrt",
            random_state=seed, n_jobs=2),
        "Gradient-Boosted Cox (scikit-survival)": GradientBoostingSurvivalAnalysis(
            n_estimators=300, learning_rate=0.05, max_depth=2, subsample=0.8,
            random_state=seed),
        "XGBoost Cox": XGBCox(random_state=seed),
    }


def train_survival_models(df: pd.DataFrame, test_size: float = 0.25, seed: int = 7,
                          models: dict | None = None) -> SurvivalResult:
    X = encode(df)
    y = make_surv(df)
    idx = np.arange(len(df))
    X_tr, X_te, y_tr, y_te, _, idx_te = train_test_split(
        X, y, idx, test_size=test_size, stratify=y["event"], random_state=seed
    )
    models = models or default_models(seed)
    times = EVAL_TIMES[EVAL_TIMES < y_te["time"].max()]
    tau = float(times.max())
    metrics = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        risk = model.predict(X_te)
        c_h = concordance_index_censored(y_te["event"], y_te["time"], risk)[0]
        c_u = concordance_index_ipcw(y_tr, y_te, risk, tau=tau)[0]
        _, mean_auc = cumulative_dynamic_auc(y_tr, y_te, risk, times)
        surv = _survival_matrix(model, X_te, times)
        ibs = integrated_brier_score(y_tr, y_te, surv, times)
        metrics[name] = {
            "harrell_c": float(c_h),
            "uno_c": float(c_u),
            "mean_time_dependent_auc": float(mean_auc),
            "integrated_brier_score": float(ibs),
        }
    metrics["_meta"] = {
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "event_rate": float(y["event"].mean()),
        "eval_times_weeks": times.tolist(),
    }
    return SurvivalResult(models=models, X_train=X_tr, X_test=X_te, y_train=y_tr,
                          y_test=y_te, idx_test=idx_te, metrics=metrics)
