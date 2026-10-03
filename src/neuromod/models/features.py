"""Feature preparation shared by the response, survival and HTE models."""

from __future__ import annotations

import pandas as pd

from neuromod.data.synthetic import BASELINE_FEATURES, SUBSTANCES, TARGETS

CATEGORICAL = {"substance": SUBSTANCES, "target": TARGETS}

PRETTY_NAMES = {
    "age": "Age",
    "female": "Female sex",
    "combat_deployments": "Combat deployments",
    "pcl5": "PTSD severity (PCL-5)",
    "phq9": "Depression (PHQ-9)",
    "baseline_craving": "Baseline craving (VAS)",
    "days_used_past30": "Days used, past 30",
    "years_use": "Years of use",
    "prior_tx_attempts": "Prior treatment attempts",
    "tbi_history": "TBI history",
    "psqi": "Sleep quality (PSQI)",
    "on_ssri": "On SSRI",
    "on_mat": "On SUD medication (MAT)",
    "social_support": "Social support (MSPSS)",
    "housing_unstable": "Housing instability",
    "rmt": "Resting motor threshold",
    "bmi": "BMI",
    "education_years": "Education (years)",
    "substance": "Primary substance",
    "active": "Active rTMS (vs sham)",
    "target": "Stimulation target/protocol",
    "sessions_completed": "Sessions completed",
}


def encode(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """One-hot encode categoricals with a fixed, data-independent column order.

    The reference level (first category) is dropped, so the encoding is identical
    for any subset of rows (train/test splits, single patients).
    """
    features = features or BASELINE_FEATURES
    out = df[[f for f in features if f not in CATEGORICAL]].astype(float).copy()
    for col, cats in CATEGORICAL.items():
        if col not in features:
            continue
        values = df[col].astype(str)
        for cat in cats[1:]:
            out[f"{col}={cat}"] = (values == cat).astype(float)
    return out


def parent_feature(column: str) -> str:
    """Map an encoded column back to its source feature (``substance=opioid`` -> ``substance``)."""
    return column.split("=", 1)[0]


def group_columns(columns: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for i, c in enumerate(columns):
        groups.setdefault(parent_feature(c), []).append(i)
    return groups
