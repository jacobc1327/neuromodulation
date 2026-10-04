import numpy as np

from neuromod.data.synthetic import BASELINE_FEATURES, simulate_cohort


def test_deterministic_and_labeled(small_cohort):
    again = simulate_cohort(n=600, seed=11)
    assert small_cohort.data.equals(again.data)
    assert small_cohort.meta["synthetic"] is True
    assert small_cohort.data["patient_id"].str.startswith("SYN-").all()


def test_schema_and_ranges(small_cohort):
    df = small_cohort.data
    assert set(BASELINE_FEATURES) <= set(df.columns)
    assert df["pcl5"].between(33, 80).all()  # everyone meets the PTSD screening cut-off
    assert df["sessions_completed"].between(0, 20).all()
    assert df["weeks_to_relapse_or_censor"].between(0, 52).all()
    assert (df.loc[df.active == 0, "target"] == "sham").all()
    assert (df.loc[df.active == 1, "target"] != "sham").all()


def test_active_arm_outperforms_sham():
    df = simulate_cohort(n=4000, seed=3).data
    assert df.loc[df.active == 1, "responder"].mean() > df.loc[df.active == 0, "responder"].mean()


def test_truth_is_consistent(small_cohort):
    t = small_cohort.truth
    assert np.allclose(t["true_cate"], t["p_response_if_active"] - t["p_response_if_sham"])
    assert t["p_response"].between(0, 1).all()


def test_effect_calibration_hits_target_log_or():
    c = simulate_cohort(n=1500, seed=5, active_log_or=0.8)
    assert abs(c.meta["marginal_log_or"] - 0.8) < 1e-3
    assert c.meta["active_coef"] != simulate_cohort(n=1500, seed=5).meta["active_coef"]
