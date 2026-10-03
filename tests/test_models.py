import numpy as np

from neuromod.explain.shap_analysis import explain_tree_model, oracle_shap, recovery_metrics
from neuromod.models.features import encode, group_columns
from neuromod.models.hte import n_per_arm_two_proportions, t_learner
from neuromod.models.response import train_response_model
from neuromod.models.survival import train_survival_models


def test_encoding_is_stable_across_subsets(small_cohort):
    df = small_cohort.data
    assert list(encode(df).columns) == list(encode(df.iloc[:3]).columns)
    groups = group_columns(list(encode(df).columns))
    assert len(groups["substance"]) == 4 and len(groups["target"]) == 3


def test_response_model_beats_chance(small_cohort):
    r = train_response_model(small_cohort.data, n_iter=3)
    assert r.metrics["xgboost"]["auroc"] > 0.55
    assert 0 <= r.metrics["xgboost"]["brier"] < 0.25


def test_survival_models(small_cohort):
    s = train_survival_models(small_cohort.data)
    for name, m in s.metrics.items():
        if name.startswith("_"):
            continue
        assert m["harrell_c"] > 0.55, name
        assert 0 < m["integrated_brier_score"] < 0.3, name


def test_shap_additivity_and_oracle(small_cohort):
    r = train_response_model(small_cohort.data, n_iter=2)
    res = explain_tree_model(r.model, r.X_test)
    margin = r.model.predict(r.X_test, output_margin=True)
    base = margin - res.values.sum(axis=1)
    assert np.allclose(base, base[0], atol=1e-3)  # SHAP values sum to the margin
    oracle = oracle_shap(r.X_test, r.X_train, max_rows=60)
    for noise in ["bmi", "rmt", "education_years"]:
        assert oracle.importance[noise] < 1e-9  # generator ignores noise features
    rec = recovery_metrics(res.importance, oracle.importance)
    assert -1 <= rec["spearman_rho_importance"] <= 1


def test_sample_size_formula():
    # classic textbook value: p1=0.5 vs p0=0.3 -> ~93 per arm (two-sided 0.05, 80% power)
    assert 90 <= n_per_arm_two_proportions(0.5, 0.3) <= 95
    assert n_per_arm_two_proportions(0.3, 0.3) == float("inf")


def test_t_learner(small_cohort):
    df, truth = small_cohort.data, small_cohort.truth
    idx = np.arange(len(df))
    h = t_learner(df, truth, idx[:450], idx[450:])
    assert len(h.cate_pred) == 150
    assert h.enrichment["enrolled_fraction"].iloc[0] == 1.0
