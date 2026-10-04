# The synthetic veteran cohort

> **Every patient in this repository is simulated.** No real patient records were
> used. The simulator exists so the modeling pipeline can be built, tested and shared
> openly, and so we can check whether the models recover effects we planted on
> purpose. Nothing below is a clinical finding.

Source: [`src/neuromod/data/synthetic.py`](../src/neuromod/data/synthetic.py)

## Design

A simulated 1:1 randomized, sham-controlled rTMS trial in veterans with a substance
use disorder (SUD) and comorbid PTSD:

* **Arms.** Sham, or active rTMS split across three protocols: left DLPFC 10 Hz,
  left DLPFC iTBS, and deep TMS of mPFC/ACC. These are the targets most often used
  in the trials in [`data/corpus`](../data/corpus).
* **Acute course.** 20 planned sessions. Attendance varies by patient and drops with
  housing instability and depression.
* **Outcome 1: acute response.** At least a 50% reduction in craving at the end of
  treatment (binary).
* **Outcome 2: time to relapse.** Weibull proportional hazards over a 52-week
  follow-up, with administrative censoring and random loss to follow-up. *Sustained
  cessation* means no relapse through week 52.

## Parameters and rationale

Each effect's direction and rough size were chosen to be plausible against the
literature. They are modeling assumptions, not estimates taken from any one study.

| Parameter | Assumption in the simulator | Rationale |
|---|---|---|
| Active vs sham | Calibrated so the marginal active-vs-sham log odds ratio equals the pooled SUD meta-analysis (g = 0.69, log OR 1.24; about +32 percentage points) | Set by `neuromod meta` from 8 real sham-controlled trials; see "Effect calibration" below |
| Baseline craving × active | Larger benefit with higher baseline craving | Craving is the main target of DLPFC/mPFC protocols, so there is a ceiling/floor logic |
| Sessions × active | Saturating dose-response: little benefit below about 10 sessions | Trials use multi-session courses (about 10–20+) and assume cumulative effects |
| PTSD severity (PCL-5) | Attenuates response, steeply above 60 | Severe PTSD is a well-known complicating factor in SUD treatment |
| TBI history | Blunts the active effect | Mild TBI is common in post-9/11 veterans and may change cortical excitability |
| Sleep (PSQI) × PTSD | Severe insomnia with severe PTSD lowers response | Sleep disturbance is a core feature of PTSD and is linked to relapse |
| Depression, prior treatment attempts | Small negative effects | Common prognostic factors in addiction treatment |
| On SUD medication (MAT), social support | Positive effects on response and on time to relapse | Standard protective factors |
| Housing instability | Lower adherence, lower response, higher relapse hazard (worst without social support) | Social determinants of relapse |
| Primary substance | Small offsets (nicotine slightly higher, methamphetamine lower) | Response differs by substance across the corpus |
| Resting motor threshold, BMI, education, SSRI use, deployments, sex | **No effect (noise features)** | Planted as negatives, to test whether SHAP ranks them low |

PCL-5 is drawn so that every patient meets the provisional PTSD cut-off (≥ 33).
Depression and craving are correlated with PTSD severity. The default cohort has
n = 2,000 and seed = 7, and is fully deterministic.

## Effect calibration

When `reports/meta/meta_results.json` exists, `neuromod ml` reads the pooled Hedges' g
from the SUD meta-analysis, converts it to a log odds ratio (g × π/√3, Chinn 2000), and
finds the active-arm coefficient whose *marginal* log odds ratio in the simulated cohort
matches it (bisection in `calibrate_active_coef`). The marginal effect is the target
because that is what a trial measures: the conditional coefficient is larger, since the
logistic model is non-collapsible. `neuromod ml --no-calibration` uses the fixed default
coefficient instead. The calibration is recorded under `cohort.effect_calibration` in
`reports/metrics.json`.

## Ground truth shipped with the data

`simulate_cohort()` also returns, for every patient:

* `p_response`: the true response probability
* `p_response_if_active` and `p_response_if_sham`: both potential outcomes
* `true_cate`: the true individual treatment effect
* `true_log_hazard`

These are what make the **oracle SHAP** comparison (`shap_vs_ground_truth.png`) and
the **T-learner CATE recovery** (`hte.cate_pearson_r` in `reports/metrics.json`)
possible. Real trial data would never provide them.

## Using real data instead

The modeling code reads a plain DataFrame. To run it on a real (IRB-approved,
de-identified) dataset, map its columns to `BASELINE_FEATURES` plus `responder`,
`relapse` and `weeks_to_relapse_or_censor`. Then call `train_response_model`,
`train_survival_models` and `explain_tree_model` directly. The oracle and CATE-truth
checks do not apply to real data.
