import numpy as np
import pandas as pd

from neuromod.models.clinical_eval import (
    ConformalClassifier,
    conformal_report,
    decision_curve,
    net_benefit,
    subgroup_audit,
)


def test_net_benefit_of_perfect_model_equals_prevalence():
    y = np.array([1, 0, 1, 0, 0, 1, 0, 0])
    nb = net_benefit(y, y.astype(float), np.array([0.2, 0.5]))
    assert np.allclose(nb, y.mean())
    curve = decision_curve(y, y.astype(float))
    assert (curve.model >= curve.treat_all - 1e-12).all()


def test_conformal_coverage_guarantee():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 6000)
    y = rng.binomial(1, p)
    rep = conformal_report(p[:3000], y[:3000], p[3000:], y[3000:], alpha=0.1)
    for kind in ("marginal", "class_conditional"):
        assert rep[kind]["coverage"] >= 0.88
    cc = rep["class_conditional"]
    assert cc["coverage_responders"] >= 0.87 and cc["coverage_non_responders"] >= 0.87
    sets = ConformalClassifier(alpha=0.1).fit(p[:3000], y[:3000]).predict_sets(np.array([0.01, 0.99]))
    assert sets.any(axis=1).all()


def test_subgroup_audit_shapes():
    rng = np.random.default_rng(1)
    n = 400
    df = pd.DataFrame({
        "female": rng.integers(0, 2, n), "tbi_history": rng.integers(0, 2, n),
        "housing_unstable": rng.integers(0, 2, n),
        "substance": rng.choice(["alcohol", "nicotine"], n),
    })
    p = rng.uniform(0, 1, n)
    y = rng.binomial(1, p)
    audit = subgroup_audit(df, y, p)
    assert set(audit.family) == {"Sex", "TBI history", "Housing", "Primary substance"}
    assert audit.auroc.between(0, 1).all()
