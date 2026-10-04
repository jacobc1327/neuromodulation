from pathlib import Path

import pytest

from neuromod.demo import DemoModels

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.fixture(scope="module")
def demo():
    return DemoModels(n=600, seed=3)


def test_demo_predictions_are_coherent(demo):
    base = demo.predict({})
    assert 0 < base.p_active < 1 and 0 < base.p_sham < 1
    assert base.conformal_label in {"likely responder", "likely non-responder", "uncertain"}
    assert base.survival["This patient"].is_monotonic_decreasing
    assert len(base.shap) > 10
    worse = demo.predict({"pcl5": 78, "tbi_history": 1, "housing_unstable": 1,
                          "sessions_completed": 3})
    assert worse.p_active < base.p_active


def test_streamlit_app_renders():
    st_testing = pytest.importorskip("streamlit.testing.v1")
    at = st_testing.AppTest.from_file(str(APP), default_timeout=240).run()
    assert not at.exception
    assert any("P(response)" in m.label for m in at.metric)
