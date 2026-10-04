"""Interactive demo: rTMS response/relapse calculator, literature assistant, meta-analysis.

Run locally:   streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")
os.environ.setdefault("NEUROMOD_OFFLINE", "" if os.getenv("NEUROMOD_LLM_MODEL") else "1")

from neuromod.data.synthetic import SUBSTANCES, TARGETS  # noqa: E402
from neuromod.demo import DEFAULT_PATIENT, DemoModels  # noqa: E402
from neuromod.models.features import PRETTY_NAMES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#8a8984"

st.set_page_config(page_title="rTMS for Addiction", page_icon="🧠", layout="wide")


@st.cache_resource(show_spinner="Training models on the synthetic cohort ...")
def load_models() -> DemoModels:
    return DemoModels()


@st.cache_resource(show_spinner="Building the literature index ...")
def load_assistant():
    from neuromod.rag.pipeline import RTMSResearchAssistant

    return RTMSResearchAssistant(metrics_path=REPORTS / "metrics.json")


def _json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


st.title("Noninvasive Neuromodulation for Addiction")
st.caption("rTMS response and relapse modeling for veterans with a substance use disorder and "
           "comorbid PTSD, plus a citation-grounded assistant over the rTMS literature.")
st.warning("**Research demo on synthetic data.** The patient models are trained on a simulated "
           "cohort and are not for clinical use. The literature corpus and meta-analysis use "
           "real published studies.", icon="⚠️")

tab_calc, tab_lit, tab_meta, tab_report = st.tabs(
    ["Patient calculator", "Literature assistant", "Meta-analysis", "Model report"])

# --------------------------------------------------------------------- calculator
with tab_calc:
    models = load_models()
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.subheader("Patient profile")
        p = dict(DEFAULT_PATIENT)
        p["substance"] = st.selectbox("Primary substance", SUBSTANCES,
                                      index=SUBSTANCES.index(p["substance"]))
        p["pcl5"] = st.slider(PRETTY_NAMES["pcl5"], 33, 80, p["pcl5"])
        p["phq9"] = st.slider(PRETTY_NAMES["phq9"], 0, 27, p["phq9"])
        p["baseline_craving"] = st.slider(PRETTY_NAMES["baseline_craving"], 0, 100,
                                          p["baseline_craving"])
        p["psqi"] = st.slider(PRETTY_NAMES["psqi"], 0, 21, p["psqi"])
        p["social_support"] = st.slider(PRETTY_NAMES["social_support"], 1.0, 7.0,
                                        p["social_support"], 0.1)
        p["days_used_past30"] = st.slider(PRETTY_NAMES["days_used_past30"], 0, 30,
                                          p["days_used_past30"])
        p["prior_tx_attempts"] = st.slider(PRETTY_NAMES["prior_tx_attempts"], 0, 10,
                                           p["prior_tx_attempts"])
        c1, c2 = st.columns(2)
        p["tbi_history"] = int(c1.toggle("TBI history", bool(p["tbi_history"])))
        p["housing_unstable"] = int(c2.toggle("Unstable housing", bool(p["housing_unstable"])))
        p["on_mat"] = int(c1.toggle("On SUD medication", bool(p["on_mat"])))
        p["female"] = int(c2.toggle("Female", bool(p["female"])))
        st.subheader("Treatment plan")
        p["target"] = st.selectbox("Protocol", TARGETS[1:], index=0)
        p["sessions_completed"] = st.slider("Sessions expected to complete", 0, 20,
                                            p["sessions_completed"])

    rep = models.predict(p)
    with right:
        m1, m2, m3 = st.columns(3)
        m1.metric("P(response) with active rTMS", f"{rep.p_active:.0%}")
        m2.metric("P(response) with sham", f"{rep.p_sham:.0%}")
        m3.metric("Predicted benefit", f"{(rep.p_active - rep.p_sham) * 100:+.0f} pp")
        st.markdown(f"**Conformal call (90% coverage):** {rep.conformal_label}")

        st.markdown("##### Why this prediction? (SHAP, log-odds of response)")
        top = rep.shap.head(10).copy()
        top["direction"] = top.contribution.map(lambda v: "raises" if v > 0 else "lowers")
        top["label"] = top.apply(lambda r: f"{r.feature} = {r.value}", axis=1)
        bars = alt.Chart(top).mark_bar(cornerRadius=3).encode(
            x=alt.X("contribution:Q", title="SHAP contribution"),
            y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=280)),
            color=alt.Color("direction:N", scale=alt.Scale(domain=["raises", "lowers"],
                                                           range=[BLUE, ORANGE]),
                            legend=alt.Legend(title="Effect on response")),
            tooltip=["feature", "value", alt.Tooltip("contribution:Q", format=".3f")],
        ).properties(height=320)
        st.altair_chart(bars, use_container_width=True)

        st.markdown("##### Relapse-free survival (XGBoost-Cox)")
        long = rep.survival.melt("week", var_name="curve", value_name="relapse_free")
        line = alt.Chart(long).mark_line(strokeWidth=2.5).encode(
            x=alt.X("week:Q", title="Weeks after treatment start"),
            y=alt.Y("relapse_free:Q", title="Probability relapse-free",
                    scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("curve:N", scale=alt.Scale(domain=["This patient", "Cohort median"],
                                                           range=[BLUE, MUTED]),
                            legend=alt.Legend(title=None, orient="top-right")),
            tooltip=["curve", "week", alt.Tooltip("relapse_free:Q", format=".2f")],
        ).properties(height=280)
        st.altair_chart(line, use_container_width=True)

# --------------------------------------------------------------------- literature
with tab_lit:
    st.subheader("Ask the rTMS literature")
    st.caption("Hybrid FAISS + BM25 retrieval over 42 studies. Every sentence must cite a "
               "retrieved study that supports it.")
    mode = st.radio("Task", ["ask", "hypotheses", "design"], horizontal=True,
                    format_func={"ask": "Evidence synthesis", "hypotheses": "Hypotheses",
                                 "design": "Study design"}.get)
    examples = {
        "ask": "Does rTMS reduce craving in alcohol use disorder?",
        "hypotheses": "rTMS for veterans with PTSD and substance use disorder",
        "design": "rTMS trial for veterans with alcohol use disorder and comorbid PTSD",
    }
    q = st.text_input("Question", examples[mode])
    k = st.slider("Studies to retrieve", 3, 10, 6)
    if st.button("Run", type="primary") and q.strip():
        ans = load_assistant().run(q, mode=mode, k=k)
        g = ans.grounding
        c1, c2, c3 = st.columns(3)
        c1.metric("Claims checked", g["n_claims"])
        c2.metric("Grounded", f"{g['grounding_score']:.0%}")
        c3.metric("Citation coverage", f"{g['citation_coverage']:.0%}")
        st.markdown(ans.answer)
        with st.expander("References", expanded=True):
            for i, s in enumerate(ans.sources, 1):
                link = f" [link]({s['link']})" if s.get("link") else ""
                st.markdown(f"**[S{i}]** {s['reference']}{link}")

# --------------------------------------------------------------------- meta-analysis
with tab_meta:
    meta = _json(REPORTS / "meta" / "meta_results.json")
    if not meta:
        st.info("Run `neuromod meta` to generate the meta-analysis.")
    else:
        st.subheader("Random-effects meta-analysis of sham-controlled trials")
        for key, res in meta["analyses"].items():
            pooled = res["pooled"]
            st.markdown(f"**{res['title']}**: g = {pooled['g']:.2f} "
                        f"(95% CI {pooled['ci95'][0]:.2f} to {pooled['ci95'][1]:.2f}), "
                        f"k = {pooled['k']}, I² = {pooled['I2']:.0%}")
            fig = REPORTS / "figures" / f"forest_{key}.png"
            if fig.exists():
                st.image(str(fig))
        st.caption(meta.get("note", ""))
        table = REPORTS / "meta" / "study_effects.csv"
        if table.exists():
            st.dataframe(pd.read_csv(table), use_container_width=True, hide_index=True)

# --------------------------------------------------------------------- report
with tab_report:
    metrics = _json(REPORTS / "metrics.json")
    if metrics:
        r = metrics["response_model"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Response AUROC (XGBoost)", f"{r['xgboost']['auroc']:.3f}")
        c2.metric("Oracle ceiling", f"{r['oracle_auroc']:.3f}")
        c3.metric("SHAP vs truth (rho)",
                  f"{metrics['shap']['ground_truth_recovery']['spearman_rho_importance']:.2f}")
        c4.metric("Trial size cut by enrichment",
                  f"{metrics['hte']['randomized_sample_reduction_top50pct']:.0%}")
    for name in ["response_roc_calibration", "decision_curve", "subgroup_audit",
                 "shap_response_beeswarm", "km_risk_groups", "enrichment_design"]:
        fig = REPORTS / "figures" / f"{name}.png"
        if fig.exists():
            st.image(str(fig))
