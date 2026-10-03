"""Synthetic veteran cohort for rTMS-for-addiction modeling.

!!! SYNTHETIC DATA — NOT REAL PATIENTS !!!

No real patient data is used anywhere in this repository. This module simulates a
sham-controlled rTMS trial in veterans with a substance use disorder (SUD) and
comorbid PTSD. Distributions and effect *directions* are informed by the published
literature in ``data/corpus`` (see ``docs/synthetic_cohort.md`` for the rationale
behind each parameter), but every number produced here is simulated. Results from
models trained on this data demonstrate the *methodology*; they are not clinical
findings.

Because the data-generating process is known, the cohort ships with its own ground
truth (true per-patient response probability, true individual treatment effect and
true log-hazard), which lets us test whether XGBoost + SHAP actually recover the
predictors that were planted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SUBSTANCES = ["alcohol", "nicotine", "cocaine", "opioid", "methamphetamine"]
SUBSTANCE_PROBS = [0.42, 0.22, 0.12, 0.14, 0.10]
TARGETS = ["sham", "L-DLPFC 10Hz", "L-DLPFC iTBS", "mPFC/ACC deep TMS"]

# Columns that would be known before treatment starts (model inputs).
BASELINE_FEATURES = [
    "age",
    "female",
    "combat_deployments",
    "pcl5",
    "phq9",
    "baseline_craving",
    "days_used_past30",
    "years_use",
    "prior_tx_attempts",
    "tbi_history",
    "psqi",
    "on_ssri",
    "on_mat",
    "social_support",
    "housing_unstable",
    "rmt",
    "bmi",
    "education_years",
    "substance",
    "active",
    "target",
    "sessions_completed",
]

# Ground-truth coefficients on the response log-odds scale. Centering constants keep
# the intercept interpretable (a "typical" sham patient).
RESPONSE_COEFS: dict[str, float] = {
    "intercept": -1.05,
    "active": 0.85,  # main active-vs-sham effect (moderate, per NIBS craving meta-analyses)
    "active_x_craving": 0.018,  # larger effect with higher baseline craving
    "active_x_sessions": 2.0,  # saturating dose-response (sigmoid in sessions completed)
    "active_x_tbi": -0.70,  # TBI blunts the *active* effect (altered cortical excitability)
    "pcl5": -0.012,  # PTSD severity attenuates response ...
    "pcl5_hinge60": -0.09,  # ... much more steeply above PCL-5 = 60
    "phq9": -0.035,
    "tbi_history": -0.20,
    "psqi": -0.04,  # poor sleep
    "sleep_x_ptsd": -0.70,  # severe insomnia (PSQI > 14) with severe PTSD (PCL-5 > 55)
    "on_mat": 0.30,  # concurrent medication for SUD
    "social_support": 0.22,
    "housing_unstable": -0.55,
    "prior_tx_attempts": -0.06,
}
SUBSTANCE_RESPONSE = {
    "alcohol": 0.0,
    "nicotine": 0.20,
    "cocaine": -0.10,
    "opioid": -0.15,
    "methamphetamine": -0.25,
}
TARGET_RESPONSE = {  # additive, only for active arms
    "sham": 0.0,
    "L-DLPFC 10Hz": 0.0,
    "L-DLPFC iTBS": 0.05,
    "mPFC/ACC deep TMS": 0.15,
}

# Ground-truth coefficients on the log-hazard scale for time-to-relapse.
HAZARD_COEFS: dict[str, float] = {
    "responder": -0.95,  # acute responders relapse later
    "pcl5": 0.022,
    "days_used_past30": 0.025,
    "tbi_history": 0.30,
    "housing_unstable": 0.45,
    "prior_tx_attempts": 0.10,
    "social_support": -0.20,
    "on_mat": -0.30,
    "psqi": 0.04,
    "pcl5_hinge60": 0.05,  # relapse risk accelerates with very severe PTSD
    "housing_x_low_support": 0.60,  # unstable housing without a support network
}
CENTER = {
    "baseline_craving": 60.0,
    "sessions_completed": 15.0,
    "pcl5": 52.0,
    "phq9": 13.0,
    "psqi": 12.0,
    "social_support": 4.5,
    "prior_tx_attempts": 2.0,
    "days_used_past30": 15.0,
}

FOLLOW_UP_WEEKS = 52.0


@dataclass
class SyntheticCohort:
    """A simulated cohort plus the ground truth that generated it."""

    data: pd.DataFrame
    truth: pd.DataFrame
    seed: int
    meta: dict = field(default_factory=dict)

    @property
    def X(self) -> pd.DataFrame:  # noqa: N802 - conventional ML name
        return self.data[BASELINE_FEATURES]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _response_logit(df: pd.DataFrame, active: np.ndarray) -> np.ndarray:
    c = RESPONSE_COEFS

    def col(name: str) -> np.ndarray:
        return df[name].to_numpy(dtype=float)

    z = np.full(len(df), c["intercept"])
    def dose(sessions: np.ndarray) -> np.ndarray:
        # little benefit below ~10 sessions, plateau after ~14 (centered at 15 sessions)
        return _sigmoid((sessions - 10.0) / 1.5) - _sigmoid(5.0 / 1.5)

    z = z + active * (
        c["active"]
        + c["active_x_craving"] * (col("baseline_craving") - CENTER["baseline_craving"])
        + c["active_x_sessions"] * dose(col("sessions_completed"))
        + c["active_x_tbi"] * col("tbi_history")
        + df["target"].astype(str).map(TARGET_RESPONSE).to_numpy(dtype=float)
    )
    z = z + c["pcl5_hinge60"] * np.maximum(col("pcl5") - 60.0, 0.0)
    z = z + c["sleep_x_ptsd"] * ((col("psqi") > 14) & (col("pcl5") > 55))
    for k in ["pcl5", "phq9", "psqi", "social_support", "prior_tx_attempts"]:
        z = z + c[k] * (col(k) - CENTER[k])
    for k in ["tbi_history", "on_mat", "housing_unstable"]:
        z = z + c[k] * col(k)
    z = z + df["substance"].astype(str).map(SUBSTANCE_RESPONSE).to_numpy(dtype=float)
    return z


def simulate_cohort(n: int = 2000, seed: int = 7) -> SyntheticCohort:
    """Simulate a 1:1 randomized active-vs-sham rTMS trial with 52-week follow-up."""
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(42, 11, n), 22, 75).round()
    female = rng.binomial(1, 0.12, n)
    combat_deployments = np.minimum(rng.poisson(1.6, n), 8)

    # PTSD (PCL-5, all >= 33 provisional-diagnosis cut-off) and correlated depression
    pcl5 = np.clip(rng.normal(52, 11, n), 33, 80).round()
    phq9 = np.clip(4 + 0.18 * pcl5 + rng.normal(0, 3.5, n), 0, 27).round()
    substance = rng.choice(SUBSTANCES, size=n, p=SUBSTANCE_PROBS)

    baseline_craving = np.clip(rng.normal(58, 16, n) + 0.15 * (pcl5 - 52), 5, 100).round()
    days_used_past30 = np.clip(rng.normal(15, 8, n) + 0.2 * (baseline_craving - 58), 0, 30).round()
    years_use = np.clip((age - 18) * rng.uniform(0.25, 0.9, n), 1, None).round()
    prior_tx_attempts = np.minimum(rng.poisson(2.0, n), 10)
    tbi_history = rng.binomial(1, 0.33, n)
    psqi = np.clip(rng.normal(11.5, 3.5, n) + 0.06 * (pcl5 - 52), 0, 21).round()
    on_ssri = rng.binomial(1, 0.42, n)
    mat_p = np.where(np.isin(substance, ["alcohol", "opioid", "nicotine"]), 0.45, 0.12)
    on_mat = rng.binomial(1, mat_p)
    social_support = np.clip(rng.normal(4.5, 1.2, n) - 0.01 * (pcl5 - 52), 1, 7).round(1)
    housing_unstable = rng.binomial(1, 0.13, n)
    rmt = np.clip(rng.normal(48, 8, n), 30, 80).round()  # resting motor threshold, %MSO (noise)
    bmi = np.clip(rng.normal(29, 5, n), 17, 50).round(1)  # noise feature
    education_years = np.clip(rng.normal(13.5, 2, n), 8, 22).round()  # noise feature

    active = rng.binomial(1, 0.5, n)
    active_targets = rng.choice(TARGETS[1:], size=n, p=[0.45, 0.30, 0.25])
    target = np.where(active == 1, active_targets, "sham")

    # Adherence: 20 planned sessions; housing instability, depression lower adherence.
    adh_logit = (1.6 - 1.0 * housing_unstable - 0.06 * (phq9 - 13) + 0.25 * (social_support - 4.5)
                 + rng.normal(0, 1.3, n))  # patient-level heterogeneity in attendance
    sessions_completed = rng.binomial(20, _sigmoid(adh_logit))

    df = pd.DataFrame(
        {
            "patient_id": [f"SYN-{i:05d}" for i in range(n)],
            "age": age,
            "female": female,
            "combat_deployments": combat_deployments,
            "pcl5": pcl5,
            "phq9": phq9,
            "baseline_craving": baseline_craving,
            "days_used_past30": days_used_past30,
            "years_use": years_use,
            "prior_tx_attempts": prior_tx_attempts,
            "tbi_history": tbi_history,
            "psqi": psqi,
            "on_ssri": on_ssri,
            "on_mat": on_mat,
            "social_support": social_support,
            "housing_unstable": housing_unstable,
            "rmt": rmt,
            "bmi": bmi,
            "education_years": education_years,
            "substance": pd.Categorical(substance, categories=SUBSTANCES),
            "active": active,
            "target": pd.Categorical(target, categories=TARGETS),
            "sessions_completed": sessions_completed,
        }
    )

    # --- Acute response (>= 50% craving reduction at end of treatment) ---
    z1 = _response_logit(df, np.ones(n))
    z0 = _response_logit(df.assign(target="sham"), np.zeros(n))
    z = np.where(active == 1, z1, z0)
    p_response = _sigmoid(z)
    responder = rng.binomial(1, p_response)

    # --- Time to relapse: Weibull proportional hazards, 52-week follow-up ---
    h = HAZARD_COEFS
    log_hazard = h["responder"] * responder
    for k in ["pcl5", "days_used_past30", "prior_tx_attempts", "social_support", "psqi"]:
        log_hazard = log_hazard + h[k] * (df[k].to_numpy() - CENTER[k])
    for k in ["tbi_history", "housing_unstable", "on_mat"]:
        log_hazard = log_hazard + h[k] * df[k].to_numpy()
    log_hazard = log_hazard + h["pcl5_hinge60"] * np.maximum(df["pcl5"].to_numpy() - 60, 0)
    log_hazard = log_hazard + h["housing_x_low_support"] * (
        (df["housing_unstable"].to_numpy() == 1) & (df["social_support"].to_numpy() < 4.0))
    shape, scale = 0.85, 26.0  # early-weighted hazard; median ~17 wk for a typical patient
    u = rng.uniform(size=n)
    t_relapse = scale * (-np.log(u) / np.exp(log_hazard)) ** (1 / shape)
    t_dropout = rng.exponential(140.0, n)  # loss to follow-up
    time = np.minimum.reduce([t_relapse, t_dropout, np.full(n, FOLLOW_UP_WEEKS)])
    relapse = (t_relapse <= np.minimum(t_dropout, FOLLOW_UP_WEEKS)).astype(int)

    df["responder"] = responder
    df["relapse"] = relapse
    df["weeks_to_relapse_or_censor"] = np.round(np.maximum(time, 0.1), 2)
    # Sustained cessation = still abstinent and observed to the end of follow-up.
    df["sustained_cessation"] = ((relapse == 0) & (time >= FOLLOW_UP_WEEKS - 1e-9)).astype(int)

    truth = pd.DataFrame(
        {
            "patient_id": df["patient_id"],
            "p_response": p_response,
            "p_response_if_active": _sigmoid(z1),
            "p_response_if_sham": _sigmoid(z0),
            "true_cate": _sigmoid(z1) - _sigmoid(z0),
            "true_log_hazard": log_hazard,
        }
    )
    return SyntheticCohort(
        data=df,
        truth=truth,
        seed=seed,
        meta={"n": n, "synthetic": True, "follow_up_weeks": FOLLOW_UP_WEEKS},
    )


def true_response_drivers() -> list[str]:
    """Features that truly drive response in the generator (for SHAP recovery tests)."""
    return [
        "active",
        "baseline_craving",
        "sessions_completed",
        "pcl5",
        "phq9",
        "tbi_history",
        "psqi",
        "on_mat",
        "social_support",
        "housing_unstable",
        "prior_tx_attempts",
        "substance",
        "target",
    ]


NOISE_FEATURES = ["rmt", "bmi", "education_years", "on_ssri", "combat_deployments", "female"]
