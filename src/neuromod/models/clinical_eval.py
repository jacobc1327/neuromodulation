"""Clinical-grade evaluation of the response model.

* **Decision-curve analysis** - would using the model to decide who starts a
  20-session rTMS course beat treating everyone or no one?
* **Conformal prediction** - distribution-free prediction sets with guaranteed
  coverage, overall and per class (Mondrian), so every patient gets an honest
  "responder / non-responder / uncertain" call.
* **Subgroup audit** - discrimination and calibration by sex, substance, TBI and
  housing status, to catch groups the model serves worse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ----------------------------------------------------------------- decision curves
def net_benefit(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    y, p = np.asarray(y), np.asarray(p)
    n = len(y)
    out = []
    for t in thresholds:
        treat = p >= t
        tp = np.sum(treat & (y == 1)) / n
        fp = np.sum(treat & (y == 0)) / n
        out.append(tp - fp * t / (1 - t))
    return np.array(out)


def decision_curve(y, p, thresholds=None) -> pd.DataFrame:
    thresholds = np.linspace(0.05, 0.6, 56) if thresholds is None else thresholds
    y = np.asarray(y)
    prev = y.mean()
    return pd.DataFrame({
        "threshold": thresholds,
        "model": net_benefit(y, p, thresholds),
        "treat_all": prev - (1 - prev) * thresholds / (1 - thresholds),
        "treat_none": 0.0,
    })


def dca_summary(curve: pd.DataFrame, lo: float = 0.15, hi: float = 0.45) -> dict:
    band = curve[(curve.threshold >= lo) & (curve.threshold <= hi)]
    best_default = np.maximum(band.treat_all, 0)
    return {
        "threshold_range": [lo, hi],
        "mean_net_benefit_model": float(band.model.mean()),
        "mean_net_benefit_best_default": float(best_default.mean()),
        "share_of_range_model_beats_defaults": float(np.mean(band.model > best_default)),
    }


# ----------------------------------------------------------------- conformal
@dataclass
class ConformalClassifier:
    """Split-conformal classifier (LAC score) with optional class-conditional quantiles."""

    alpha: float = 0.1
    mondrian: bool = True

    def fit(self, p_cal: np.ndarray, y_cal: np.ndarray) -> ConformalClassifier:
        p_cal, y_cal = np.asarray(p_cal), np.asarray(y_cal).astype(int)
        probs = np.column_stack([1 - p_cal, p_cal])
        scores = 1 - probs[np.arange(len(y_cal)), y_cal]
        if self.mondrian:
            self.qhat_ = {c: self._quantile(scores[y_cal == c]) for c in (0, 1)}
        else:
            q = self._quantile(scores)
            self.qhat_ = {0: q, 1: q}
        return self

    def _quantile(self, s: np.ndarray) -> float:
        n = len(s)
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        return float(np.quantile(s, level, method="higher"))

    def predict_sets(self, p: np.ndarray) -> np.ndarray:
        """Boolean (n, 2) matrix: column c is True if class c is in the prediction set."""
        p = np.asarray(p)
        probs = np.column_stack([1 - p, p])
        return np.column_stack([1 - probs[:, c] <= self.qhat_[c] for c in (0, 1)])

    @staticmethod
    def label(sets: np.ndarray) -> np.ndarray:
        return np.where(sets[:, 0] & sets[:, 1], "uncertain",
                        np.where(sets[:, 1], "likely responder",
                                 np.where(sets[:, 0], "likely non-responder", "empty")))


def conformal_report(p_cal, y_cal, p_test, y_test, alpha: float = 0.1) -> dict:
    out = {}
    y_test = np.asarray(y_test).astype(int)
    for name, mondrian in [("marginal", False), ("class_conditional", True)]:
        cp = ConformalClassifier(alpha=alpha, mondrian=mondrian).fit(p_cal, y_cal)
        sets = cp.predict_sets(p_test)
        covered = sets[np.arange(len(y_test)), y_test]
        labels = ConformalClassifier.label(sets)
        out[name] = {
            "target_coverage": 1 - alpha,
            "coverage": float(covered.mean()),
            "coverage_responders": float(covered[y_test == 1].mean()),
            "coverage_non_responders": float(covered[y_test == 0].mean()),
            "mean_set_size": float(sets.sum(axis=1).mean()),
            "share_singleton": float((sets.sum(axis=1) == 1).mean()),
            "label_counts": {k: int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
        }
    return out


# ----------------------------------------------------------------- subgroup audit
SUBGROUPS = {
    "Sex": ("female", {0: "Male", 1: "Female"}),
    "TBI history": ("tbi_history", {0: "No TBI", 1: "TBI"}),
    "Housing": ("housing_unstable", {0: "Stable", 1: "Unstable"}),
    "Primary substance": ("substance", None),
}


def subgroup_audit(df_test: pd.DataFrame, y: np.ndarray, p: np.ndarray,
                   min_n: int = 25) -> pd.DataFrame:
    y, p = np.asarray(y), np.asarray(p)
    rows = []
    thr = float(np.mean(y))  # flag the top-risk share equal to prevalence
    for family, (col, names) in SUBGROUPS.items():
        values = df_test[col].astype(str) if names is None else df_test[col].map(names)
        for level in sorted(values.unique()):
            m = (values == level).to_numpy()
            if m.sum() < min_n or len(np.unique(y[m])) < 2:
                continue
            flag = p[m] >= np.quantile(p, 1 - thr)
            rows.append({
                "family": family,
                "group": level,
                "n": int(m.sum()),
                "prevalence": float(y[m].mean()),
                "auroc": float(roc_auc_score(y[m], p[m])),
                "calibration_in_the_large": float(p[m].mean() - y[m].mean()),
                "tpr": float(flag[y[m] == 1].mean()),
                "fpr": float(flag[y[m] == 0].mean()),
            })
    return pd.DataFrame(rows)
