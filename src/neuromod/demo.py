"""Model bundle behind the interactive demo (``app/streamlit_app.py``).

Trains fast, fixed-hyper-parameter versions of the pipeline models on the
synthetic cohort and exposes per-patient predictions: response probability under
each arm, a conformal call, a SHAP breakdown and a relapse-free survival curve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from neuromod.data.synthetic import BASELINE_FEATURES, simulate_cohort
from neuromod.explain.shap_analysis import grouped_shap
from neuromod.models.clinical_eval import ConformalClassifier
from neuromod.models.features import PRETTY_NAMES, encode
from neuromod.models.survival import XGBCox, make_surv

# Tuned values from the randomized search in `neuromod ml` (reports/metrics.json).
RESPONSE_PARAMS = dict(n_estimators=330, max_depth=3, learning_rate=0.0106, subsample=0.69,
                       colsample_bytree=0.57, min_child_weight=9, reg_lambda=1.36)

DEFAULT_PATIENT = {
    "age": 38, "female": 0, "combat_deployments": 2, "pcl5": 55, "phq9": 14,
    "baseline_craving": 65, "days_used_past30": 16, "years_use": 12, "prior_tx_attempts": 2,
    "tbi_history": 0, "psqi": 12, "on_ssri": 0, "on_mat": 0, "social_support": 4.5,
    "housing_unstable": 0, "rmt": 48, "bmi": 29.0, "education_years": 13,
    "substance": "alcohol", "active": 1, "target": "L-DLPFC 10Hz", "sessions_completed": 18,
}


@dataclass
class PatientReport:
    p_active: float
    p_sham: float
    conformal_label: str
    shap: pd.DataFrame  # feature, value, contribution (log-odds)
    base_value: float
    survival: pd.DataFrame  # week, relapse_free (patient), cohort_median


class DemoModels:
    def __init__(self, n: int = 2000, seed: int = 7):
        cohort = simulate_cohort(n=n, seed=seed)
        self.df = cohort.data
        X = encode(self.df)
        y = self.df["responder"].to_numpy()
        X_tr, X_cal, y_tr, y_cal = train_test_split(X, y, test_size=0.25, stratify=y,
                                                    random_state=seed)
        self.response = XGBClassifier(**RESPONSE_PARAMS, eval_metric="logloss",
                                      random_state=seed, n_jobs=2).fit(X_tr, y_tr)
        self.conformal = ConformalClassifier(alpha=0.1).fit(
            self.response.predict_proba(X_cal)[:, 1], y_cal)
        self.explainer = shap.TreeExplainer(self.response)
        self.relapse = XGBCox(random_state=seed).fit(X, make_surv(self.df))
        self.weeks = np.arange(0, 53, 1.0)
        cohort_surv = self.relapse.survival_at(X, self.weeks)
        self.cohort_median_curve = np.median(cohort_surv, axis=0)
        self.columns = list(X.columns)

    def _row(self, patient: dict) -> pd.DataFrame:
        frame = pd.DataFrame([{**DEFAULT_PATIENT, **patient}])
        frame["active"] = int(frame["target"].iloc[0] != "sham")
        return encode(frame)[self.columns]

    def predict(self, patient: dict) -> PatientReport:
        x = self._row(patient)
        x_sham = self._row({**patient, "target": "sham"})
        p_active = float(self.response.predict_proba(x)[:, 1][0])
        p_sham = float(self.response.predict_proba(x_sham)[:, 1][0])
        p_used = p_active if patient.get("target", "L-DLPFC 10Hz") != "sham" else p_sham
        label = ConformalClassifier.label(self.conformal.predict_sets(np.array([p_used])))[0]

        sv = self.explainer.shap_values(x)
        grouped = grouped_shap(np.asarray(sv), self.columns).iloc[0]
        merged = {**DEFAULT_PATIENT, **patient}
        shap_df = pd.DataFrame({
            "feature": [PRETTY_NAMES.get(f, f) for f in grouped.index],
            "value": [merged.get(f) for f in grouped.index],
            "contribution": grouped.to_numpy(),
        }).sort_values("contribution", key=np.abs, ascending=False)

        surv = self.relapse.survival_at(x, self.weeks)[0]
        survival = pd.DataFrame({"week": self.weeks, "This patient": surv,
                                 "Cohort median": self.cohort_median_curve})
        base = self.explainer.expected_value
        return PatientReport(p_active=p_active, p_sham=p_sham, conformal_label=str(label),
                             shap=shap_df, base_value=float(np.ravel(base)[0]),
                             survival=survival)


FEATURES = BASELINE_FEATURES
