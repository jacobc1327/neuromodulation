"""Figures for the README / report. Static PNGs rendered with matplotlib.

Palette: a validated, colorblind-safe categorical order (blue, orange, aqua) and a
blue<->red diverging map with a neutral gray midpoint for SHAP feature values.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402
from sksurv.nonparametric import kaplan_meier_estimator  # noqa: E402

from neuromod.models.features import PRETTY_NAMES  # noqa: E402

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list(
    "bluered", ["#256abf", "#86b6ef", "#d9d8d4", "#f0a3a2", "#c63a39"]
)


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "font.size": 10,
        "legend.frameon": False,
        "legend.labelcolor": INK2,
        "lines.linewidth": 2,
    })


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _pretty(name: str) -> str:
    return PRETTY_NAMES.get(name, name)


def roc_and_calibration(y, p_xgb, p_lr, p_oracle, path: Path, metrics: dict):
    _style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    for p, label, color, ls in [
        (p_xgb, f"XGBoost  AUROC {metrics['xgboost']['auroc']:.3f}", BLUE, "-"),
        (p_lr, f"Logistic regression  {metrics['logistic_regression']['auroc']:.3f}", ORANGE, "-"),
        (p_oracle, f"Oracle (true risk)  {metrics['oracle_auroc']:.3f}", MUTED, "--"),
    ]:
        fpr, tpr, _ = roc_curve(y, p)
        a1.plot(fpr, tpr, color=color, ls=ls, label=label)
    a1.plot([0, 1], [0, 1], color=GRID, lw=1)
    a1.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1))
    a1.set_title("Acute response: discrimination")
    a1.legend(loc="lower right", fontsize=9)

    for p, label, color in [(p_xgb, "XGBoost", BLUE), (p_lr, "Logistic regression", ORANGE)]:
        fp, mp = calibration_curve(y, p, n_bins=8, strategy="quantile")
        a2.plot(mp, fp, marker="o", ms=6, color=color, label=label,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
    a2.plot([0, 1], [0, 1], color=GRID, lw=1)
    a2.set(xlabel="Predicted probability", ylabel="Observed response rate",
           xlim=(0, 0.9), ylim=(0, 0.9))
    a2.set_title("Calibration (held-out)")
    a2.legend(loc="upper left", fontsize=9)
    fig.text(0.0, -0.04, "Synthetic cohort - methodology demonstration, not clinical results.",
             color=MUTED, fontsize=8)
    _save(fig, path)


def shap_beeswarm(grouped: pd.DataFrame, data: pd.DataFrame, path: Path, title: str,
                  xlabel: str, max_display: int = 14, seed: int = 0):
    """Beeswarm on grouped SHAP values; color = feature value (categoricals in gray)."""
    _style()
    rng = np.random.default_rng(seed)
    order = grouped.abs().mean().sort_values(ascending=False).index[:max_display][::-1]
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(order) + 1.2))
    for i, feat in enumerate(order):
        vals = grouped[feat].to_numpy()
        if feat in data.columns:
            raw = data[feat].to_numpy(dtype=float)
            lo, hi = np.nanpercentile(raw, [5, 95])
            colors = DIVERGING(np.clip((raw - lo) / (hi - lo + 1e-9), 0, 1))
        else:
            colors = MUTED
        # simple density-aware jitter
        hist, edges = np.histogram(vals, bins=40)
        bin_idx = np.clip(np.digitize(vals, edges) - 1, 0, len(hist) - 1)
        width = hist[bin_idx] / hist.max()
        y = i + rng.uniform(-0.38, 0.38, len(vals)) * width
        ax.scatter(vals, y, c=colors, s=9, linewidths=0, alpha=0.85, rasterized=True)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(len(order)), [_pretty(f) for f in order])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    sm = plt.cm.ScalarMappable(cmap=DIVERGING)
    cb = fig.colorbar(sm, ax=ax, pad=0.01, aspect=40, ticks=[0, 1])
    cb.ax.set_yticklabels(["low", "high"])
    cb.set_label("Feature value (gray = categorical)", color=INK2)
    cb.outline.set_visible(False)
    _save(fig, path)


def shap_dependence(grouped: pd.DataFrame, data: pd.DataFrame, feature: str, by: str,
                    path: Path, title: str, labels=("Sham", "Active rTMS")):
    _style()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for val, color, label in [(0, ORANGE, labels[0]), (1, BLUE, labels[1])]:
        m = data[by].to_numpy() == val
        ax.scatter(data[feature].to_numpy()[m], grouped[feature].to_numpy()[m], s=12,
                   color=color, alpha=0.7, linewidths=0, label=label)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel(_pretty(feature))
    ax.set_ylabel(f"SHAP value for {_pretty(feature).lower()}\n(log-odds of response)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    _save(fig, path)


def shap_vs_oracle(model_imp: pd.Series, oracle_imp: pd.Series, path: Path, rho: float):
    _style()
    feats = list(oracle_imp.sort_values().index)
    m = model_imp.reindex(feats).fillna(0).to_numpy()
    o = oracle_imp.reindex(feats).to_numpy()
    m, o = m / m.max(), o / o.max()
    y = np.arange(len(feats))
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(feats) + 1.2))
    ax.barh(y + 0.2, o, height=0.38, color=MUTED, label="Ground truth (oracle SHAP)")
    ax.barh(y - 0.2, m, height=0.38, color=BLUE, label="XGBoost (TreeSHAP)")
    ax.set_yticks(y, [_pretty(f) for f in feats])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean |SHAP|, scaled to max = 1")
    ax.set_title(f"Does SHAP recover the planted predictors?  Spearman rho = {rho:.2f}")
    ax.legend(loc="lower right")
    _save(fig, path)


def km_by_risk_group(time, event, risk, path: Path, title: str):
    _style()
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    q = np.quantile(risk, [1 / 3, 2 / 3])
    groups = np.digitize(risk, q)
    for g, color, label in [(0, AQUA, "Low predicted risk"), (1, BLUE, "Medium"),
                            (2, ORANGE, "High predicted risk")]:
        m = groups == g
        t, s = kaplan_meier_estimator(event[m].astype(bool), time[m])
        ax.step(t, s, where="post", color=color, label=f"{label} (n={m.sum()})")
    ax.set(xlabel="Weeks after treatment start", ylabel="Relapse-free (sustained cessation)",
           ylim=(0, 1.02), xlim=(0, 52))
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    _save(fig, path)


def survival_model_comparison(metrics: dict, path: Path):
    _style()
    names = [k for k in metrics if not k.startswith("_")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    y = np.arange(len(names))
    a1.barh(y, [metrics[n]["uno_c"] for n in names], color=BLUE, height=0.5)
    a1.set_xlim(0.5, max(metrics[n]["uno_c"] for n in names) + 0.05)
    a1.set_title("Uno's C-index (higher is better)")
    a2.barh(y, [metrics[n]["integrated_brier_score"] for n in names], color=ORANGE, height=0.5)
    a2.set_title("Integrated Brier score (lower is better)")
    for ax, key, fmt in [(a1, "uno_c", "{:.3f}"), (a2, "integrated_brier_score", "{:.3f}")]:
        for i, n in enumerate(names):
            v = metrics[n][key]
            ax.text(v, i, " " + fmt.format(v), va="center", color=INK2, fontsize=9)
        ax.grid(axis="y", visible=False)
    a1.set_yticks(y, names)
    _save(fig, path)


def enrichment_curve(enrichment: pd.DataFrame, path: Path):
    _style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    f = enrichment["enrolled_fraction"] * 100
    a1.plot(f, enrichment["true_effect"] * 100, marker="o", color=BLUE, ms=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    a1.set(xlabel="% of screened veterans enrolled (top predicted benefit)",
           ylabel="True active - sham response (pp)")
    a1.invert_xaxis()
    a1.set_title("Model-guided enrichment raises the effect")
    a2.plot(f, enrichment["n_randomized"], marker="o", color=BLUE, ms=6, label="Randomized",
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    a2.plot(f, enrichment["n_screened"], marker="o", color=ORANGE, ms=6, label="Screened",
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    a2.set(xlabel="% of screened veterans enrolled", ylabel="Participants for 80% power")
    a2.invert_xaxis()
    a2.set_title("Sample size trade-off")
    a2.legend(loc="upper left")
    _save(fig, path)


def corpus_overview(studies: pd.DataFrame, path: Path | str):
    path = Path(path)
    _style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    by_cond = studies["condition_group"].value_counts().sort_values()
    a1.barh(by_cond.index, by_cond.values, color=BLUE, height=0.55)
    n_peer = int(studies.get("peer_reviewed", pd.Series([True] * len(studies))).fillna(True).sum())
    a1.set_title(f"Corpus: {len(studies)} studies ({n_peer} peer-reviewed)")
    a1.grid(axis="y", visible=False)
    for i, v in enumerate(by_cond.values):
        a1.text(v, i, f" {v}", va="center", color=INK2, fontsize=9)
    by_type = studies["study_type"].value_counts().sort_values()
    a2.barh(by_type.index, by_type.values, color=AQUA, height=0.55)
    a2.set_title("By design")
    a2.grid(axis="y", visible=False)
    for i, v in enumerate(by_type.values):
        a2.text(v, i, f" {v}", va="center", color=INK2, fontsize=9)
    _save(fig, path)
