"""End-to-end ML pipeline: simulate -> response model -> relapse models -> SHAP -> HTE."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from neuromod.data.synthetic import BASELINE_FEATURES, simulate_cohort
from neuromod.explain.shap_analysis import explain_tree_model, oracle_shap, recovery_metrics
from neuromod.models.clinical_eval import (
    conformal_report,
    dca_summary,
    decision_curve,
    subgroup_audit,
)
from neuromod.models.hte import t_learner
from neuromod.models.response import train_response_model
from neuromod.models.survival import train_survival_models
from neuromod.viz import plots


def run_ml_pipeline(out_dir: str | Path = "reports", n: int = 2000, seed: int = 7,
                    n_iter: int = 25, figures: bool = True, verbose: bool = True) -> dict:
    out = Path(out_dir)
    fig_dir = out / "figures"
    log = print if verbose else (lambda *a, **k: None)

    log(f"[1/6] Simulating synthetic veteran cohort (n={n}, seed={seed}) ...")
    cohort = simulate_cohort(n=n, seed=seed)
    df, truth = cohort.data, cohort.truth
    (out / "data").mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "data" / "synthetic_cohort.csv", index=False)

    log("[2/6] Training XGBoost response model (randomized search, 5-fold CV) ...")
    resp = train_response_model(df, n_iter=n_iter, seed=seed)
    p_oracle = truth["p_response"].to_numpy()[resp.idx_test]
    resp.metrics["oracle_auroc"] = float(roc_auc_score(resp.y_test, p_oracle))
    resp.metrics["xgboost_pct_of_oracle_auroc"] = float(
        (resp.metrics["xgboost"]["auroc"] - 0.5) / (resp.metrics["oracle_auroc"] - 0.5))
    resp.metrics["best_params"] = resp.best_params

    log("[3/6] Training relapse models (CoxPH, RSF, GB-Cox, XGBoost-Cox) ...")
    surv = train_survival_models(df, seed=seed)

    log("[4/6] SHAP: model explanations + ground-truth recovery check ...")
    shap_resp = explain_tree_model(resp.model, resp.X_test)
    shap_oracle = oracle_shap(resp.X_test, resp.X_train, seed=seed)
    recovery = recovery_metrics(shap_resp.importance, shap_oracle.importance)
    xgb_cox = surv.models["XGBoost Cox"]
    shap_relapse = explain_tree_model(xgb_cox, surv.X_test)

    log("[5/6] Clinical evaluation: decision curves, conformal sets, subgroup audit ...")
    df_test = df.iloc[resp.idx_test]
    active = df_test["active"].to_numpy() == 1
    dca = decision_curve(resp.y_test[active], resp.p_test[active])
    rng = np.random.default_rng(seed)
    cal = rng.random(len(resp.y_test)) < 0.5  # half of the held-out set calibrates
    conformal = conformal_report(resp.p_test[cal], resp.y_test[cal], resp.p_test[~cal],
                                 resp.y_test[~cal], alpha=0.1)
    audit = subgroup_audit(df_test[BASELINE_FEATURES], resp.y_test, resp.p_test)

    log("[6/6] Heterogeneous treatment effects + enrichment design ...")
    train_idx = np.setdiff1d(np.arange(len(df)), resp.idx_test)
    hte = t_learner(df, truth, train_idx, resp.idx_test, seed=seed)

    results = {
        "synthetic_data": True,
        "cohort": {
            "n": n,
            "seed": seed,
            "response_rate_active": float(df.loc[df.active == 1, "responder"].mean()),
            "response_rate_sham": float(df.loc[df.active == 0, "responder"].mean()),
            "relapse_rate_52wk": float(df["relapse"].mean()),
            "sustained_cessation_rate": float(df["sustained_cessation"].mean()),
        },
        "response_model": resp.metrics,
        "relapse_models": surv.metrics,
        "shap": {
            "response_top_predictors": shap_resp.importance.round(4).head(10).to_dict(),
            "relapse_top_predictors": shap_relapse.importance.round(4).head(10).to_dict(),
            "ground_truth_recovery": recovery,
        },
        "clinical_eval": {
            "decision_curve_active_arm": dca_summary(dca),
            "conformal_alpha_0.1": conformal,
            "subgroup_audit": audit.round(4).to_dict(orient="records"),
        },
        "hte": hte.metrics,
        "enrichment_table": hte.enrichment.round(4).to_dict(orient="records"),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=2, default=float))

    if figures:
        plots.roc_and_calibration(resp.y_test, resp.p_test, resp.p_test_baseline, p_oracle,
                                  fig_dir / "response_roc_calibration.png", resp.metrics)
        plots.shap_beeswarm(shap_resp.grouped, shap_resp.data,
                            fig_dir / "shap_response_beeswarm.png",
                            "What drives rTMS response? (XGBoost, TreeSHAP)",
                            "SHAP value (impact on log-odds of response)")
        plots.shap_dependence(shap_resp.grouped, shap_resp.data, "baseline_craving", "active",
                              fig_dir / "shap_dependence_craving.png",
                              "Craving x treatment interaction learned by the model")
        plots.shap_dependence(shap_resp.grouped, shap_resp.data, "sessions_completed",
                              "active", fig_dir / "shap_dependence_sessions.png",
                              "Dose-response: sessions completed")
        plots.shap_vs_oracle(shap_resp.importance, shap_oracle.importance,
                             fig_dir / "shap_vs_ground_truth.png",
                             recovery["spearman_rho_importance"])
        plots.shap_beeswarm(shap_relapse.grouped, shap_relapse.data,
                            fig_dir / "shap_relapse_beeswarm.png",
                            "Predictors of relapse vs sustained cessation (XGBoost-Cox)",
                            "SHAP value (impact on log-hazard of relapse)")
        risk = xgb_cox.predict(surv.X_test)
        plots.km_by_risk_group(surv.y_test["time"], surv.y_test["event"], risk,
                               fig_dir / "km_risk_groups.png",
                               "Held-out relapse-free survival by predicted risk tertile")
        plots.survival_model_comparison(surv.metrics, fig_dir / "survival_model_comparison.png")
        plots.enrichment_curve(hte.enrichment, fig_dir / "enrichment_design.png")
        plots.decision_curve_plot(dca, fig_dir / "decision_curve.png",
                                  "Decision curve: who should start a 20-session course?")
        plots.subgroup_auroc_plot(audit, resp.metrics["xgboost"]["auroc"],
                                  fig_dir / "subgroup_audit.png")
    log(f"Done. Metrics -> {out / 'metrics.json'}")
    return results
