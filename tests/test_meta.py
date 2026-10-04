import math

import numpy as np
import pytest

from neuromod.meta.effects import from_events, from_means, from_t, hedges_j, smd_to_log_odds
from neuromod.meta.pooling import egger_test, leave_one_out, meta_regression, random_effects


def test_hedges_g_from_means_matches_textbook():
    # Borenstein et al. (2009) worked example: d = 0.5970, var(d) = 0.0418 -> var(g) ~ 0.0410
    e = from_means(103, 5.5, 50, 100, 4.5, 50, lower_is_better=False)
    assert e.g == pytest.approx(0.5970 * hedges_j(98), abs=1e-3)
    assert e.var == pytest.approx(0.0410, abs=5e-4)


def test_direction_conventions():
    assert from_means(10, 5, 20, 15, 5, 20, lower_is_better=True).g > 0  # active lower = better
    assert from_t(2.0, 20, 20).g > 0
    assert from_events(10, 20, 4, 20).g > 0
    assert from_events(10, 20, 4, 20, higher_is_better=False).g < 0


def test_t_and_means_agree():
    m = from_means(10, 4, 30, 13, 4, 30)
    t = (13 - 10) / (4 * math.sqrt(2 / 30))
    assert from_t(t, 30, 30).g == pytest.approx(m.g, rel=1e-6)


def test_events_conversion_roundtrip():
    e = from_events(15, 50, 8, 50)
    lor = math.log((15 * 42) / (35 * 8))
    assert smd_to_log_odds(e.g) == pytest.approx(lor)


def test_random_effects_homogeneous_and_heterogeneous():
    y = np.array([0.3, 0.3, 0.3, 0.3])
    v = np.array([0.04, 0.05, 0.06, 0.03])
    r = random_effects(y, v)
    assert r.estimate == pytest.approx(0.3)
    assert r.tau2 == pytest.approx(0.0, abs=1e-9) and r.i2 == 0
    het = random_effects([-0.5, 0.2, 0.9, 1.6], [0.02, 0.02, 0.02, 0.02], method="DL")
    assert het.tau2 > 0.4 and het.i2 > 0.9
    assert het.pi[0] < het.ci[0] and het.pi[1] > het.ci[1]


def test_reml_close_to_dl_and_tools_run():
    rng = np.random.default_rng(0)
    v = rng.uniform(0.02, 0.2, 15)
    y = rng.normal(0.4, np.sqrt(v + 0.05))
    reml, dl = random_effects(y, v), random_effects(y, v, method="DL")
    assert abs(reml.estimate - dl.estimate) < 0.1
    assert 0 <= egger_test(y, v)["p"] <= 1
    assert len(leave_one_out(y, v, [str(i) for i in range(15)])) == 15
    x = rng.integers(0, 2, (15, 1))
    mr = meta_regression(y, v, x, ["deep_tms"])
    assert set(mr["coefficients"]) == {"intercept", "deep_tms"}
