"""SHAP explanations for the response and relapse models.

Two ideas beyond a standard summary plot:

* **Grouped SHAP** - one-hot columns (substance, protocol) are summed back into their
  parent feature so categorical predictors are ranked fairly against continuous ones.
* **Oracle check** - because the cohort is synthetic, we can compute SHAP values of
  the *true* data-generating response function and test whether the fitted XGBoost
  model recovers the planted clinical predictors (rank correlation of importances).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

from neuromod.data.synthetic import NOISE_FEATURES, SUBSTANCES, TARGETS, _response_logit
from neuromod.models.features import group_columns


@dataclass
class ShapResult:
    values: np.ndarray  # (n, n_encoded_columns)
    grouped: pd.DataFrame  # (n, n_parent_features)
    data: pd.DataFrame  # encoded feature values for the explained rows
    importance: pd.Series  # mean |SHAP| per parent feature, sorted desc


def grouped_shap(values: np.ndarray, columns: list[str]) -> pd.DataFrame:
    groups = group_columns(columns)
    return pd.DataFrame({g: values[:, ix].sum(axis=1) for g, ix in groups.items()})


def explain_tree_model(model, X: pd.DataFrame) -> ShapResult:
    """TreeExplainer SHAP on the model margin (log-odds or log-hazard)."""
    booster = model.model if hasattr(model, "model") else model
    explainer = shap.TreeExplainer(booster)
    values = explainer.shap_values(X)
    if isinstance(values, list):  # older shap versions, binary classifier
        values = values[1]
    grouped = grouped_shap(values, list(X.columns))
    importance = grouped.abs().mean().sort_values(ascending=False)
    return ShapResult(values=values, grouped=grouped, data=X, importance=importance)


def _decode(X: pd.DataFrame) -> pd.DataFrame:
    """Invert ``encode`` so the true generator can be evaluated on (masked) rows."""
    df = X.copy()
    for col, cats in {"substance": SUBSTANCES, "target": TARGETS}.items():
        onehot = np.column_stack([X[f"{col}={c}"].to_numpy() for c in cats[1:]])
        idx = np.where(onehot.max(axis=1) > 0.5, onehot.argmax(axis=1) + 1, 0)
        df[col] = np.array(cats, dtype=object)[idx]
    return df


def oracle_shap(X: pd.DataFrame, background: pd.DataFrame, max_rows: int = 300,
                seed: int = 0) -> ShapResult:
    """Interventional SHAP of the true response log-odds (permutation explainer)."""

    def true_logit(arr: np.ndarray) -> np.ndarray:
        frame = _decode(pd.DataFrame(arr, columns=X.columns))
        return _response_logit(frame, frame["active"].to_numpy(dtype=float))

    rows = X.sample(min(max_rows, len(X)), random_state=seed)
    bg = shap.maskers.Independent(background.sample(min(100, len(background)),
                                                    random_state=seed).to_numpy())
    explainer = shap.PermutationExplainer(true_logit, bg, seed=seed)
    values = explainer(rows.to_numpy(), max_evals=10 * X.shape[1], silent=True).values
    grouped = grouped_shap(values, list(X.columns))
    importance = grouped.abs().mean().sort_values(ascending=False)
    return ShapResult(values=values, grouped=grouped, data=rows, importance=importance)


def recovery_metrics(model_imp: pd.Series, oracle_imp: pd.Series, top_k: int = 8) -> dict:
    feats = oracle_imp.index
    rho = spearmanr(model_imp.reindex(feats).fillna(0), oracle_imp)[0]
    top_model = set(model_imp.index[:top_k])
    top_true = set(oracle_imp.index[:top_k])
    noise_ranks = [int(list(model_imp.index).index(f)) + 1 for f in NOISE_FEATURES
                   if f in model_imp.index]
    return {
        "spearman_rho_importance": float(rho),
        f"top{top_k}_overlap": len(top_model & top_true) / top_k,
        "noise_feature_ranks": dict(zip(
            [f for f in NOISE_FEATURES if f in model_imp.index], noise_ranks)),
        "n_features": int(len(model_imp)),
    }
