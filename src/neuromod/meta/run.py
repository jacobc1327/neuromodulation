"""Run the meta-analyses on the curated effect-size extraction in ``data/meta``.

Two analyses, each restricted to sham-controlled trials from the corpus:

* ``sud``  - rTMS/deep TMS/TBS vs sham on craving or use in substance use disorders
* ``ptsd`` - rTMS vs sham on PTSD symptom severity

Every extracted number in ``data/meta/*.json`` carries a verbatim quote and the URL it
came from, so each row can be audited.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from neuromod.meta.effects import compute_effect
from neuromod.meta.pooling import egger_test, leave_one_out, meta_regression, random_effects

DATA = Path(__file__).resolve().parents[3] / "data" / "meta"
CORPUS = Path(__file__).resolve().parents[3] / "data" / "corpus" / "studies.jsonl"

ANALYSES = {
    "sud": ("sud_effects.json", "Substance use disorders: craving / use (active vs sham)"),
    "ptsd": ("ptsd_effects.json", "PTSD symptom severity (active vs sham)"),
}


def _corpus() -> dict[str, dict]:
    if not CORPUS.exists():
        return {}
    return {r["id"]: r for r in map(json.loads, CORPUS.read_text().splitlines()) if r}


def _label(rec: dict, corpus: dict) -> str:
    c = corpus.get(rec["id"])
    if c:
        first = c["authors"].split(",")[0].split()[0]
        return f"{first} {c['year']}"
    return rec.get("label") or rec["id"]


def _moderators(rec: dict, corpus: dict) -> dict:
    c = corpus.get(rec["id"], {})
    text = " ".join(str(c.get(k) or "") for k in ("target", "protocol")).lower()
    text += " " + str(rec.get("target") or "").lower()
    return {
        "deep_tms": int("deep" in text or "h-coil" in text or "h7" in text or "h4" in text
                        or "h1" in text),
        "medial_target": int(any(t in text for t in ("medial", "mpfc", "cingul", "insula",
                                                     "frontal pole"))),
        "theta_burst": int("theta" in text or "tbs" in text),
    }


def effects_table(records: list[dict], corpus: dict) -> pd.DataFrame:
    rows = []
    for rec in records:
        if not rec.get("extractable", True):
            continue
        eff = compute_effect(rec)
        if eff is None:
            continue
        lo, hi = eff.ci()
        rows.append({
            "id": rec["id"], "label": _label(rec, corpus), "g": eff.g, "var": eff.var,
            "se": eff.se, "lo": lo, "hi": hi, "method": eff.method,
            "n_active": rec.get("n_active"), "n_sham": rec.get("n_sham"),
            "outcome": rec.get("outcome"), "design": rec.get("design", "parallel"),
            **_moderators(rec, corpus), "source_url": rec.get("source_url"),
        })
    return pd.DataFrame(rows).sort_values("label").reset_index(drop=True)


def run_meta(out_dir: str | Path = "reports", figures: bool = True,
             data_dir: Path = DATA) -> dict:
    out = Path(out_dir)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    corpus = _corpus()
    results, tables = {}, []
    for key, (fname, title) in ANALYSES.items():
        path = data_dir / fname
        if not path.exists():
            continue
        records = json.loads(path.read_text())
        tab = effects_table(records, corpus)
        excluded = [{"id": r["id"], "reason": r.get("notes") or r.get("reason", "")}
                    for r in records if not r.get("extractable", True)]
        if len(tab) < 2:
            continue
        y, v = tab.g.to_numpy(), tab["var"].to_numpy()
        pooled = random_effects(y, v)
        tab["weight"] = pooled.weights
        res = {
            "title": title,
            "pooled": pooled.as_dict(),
            "pooled_dl_no_hk": random_effects(y, v, method="DL", hartung_knapp=False).as_dict(),
            "egger": egger_test(y, v) if len(tab) >= 3 else None,
            "leave_one_out": [{**d, "ci95": list(d["ci95"])} for d in
                              leave_one_out(y, v, list(tab.label))],
            "n_participants": int(tab.n_active.fillna(0).sum() + tab.n_sham.fillna(0).sum()),
            "excluded": excluded,
        }
        # Keep moderators with at least two studies at each level, then drop any
        # column that duplicates one already kept.
        mods, seen = [], set()
        for m in ("deep_tms", "medial_target", "theta_burst"):
            col = tuple(tab[m])
            if 2 <= tab[m].sum() <= len(tab) - 2 and col not in seen:
                mods.append(m)
                seen.add(col)
        if mods and len(tab) >= len(mods) + 4:
            reg = meta_regression(y, v, tab[mods].to_numpy(), mods)
            if reg is not None:
                res["meta_regression"] = reg
        results[key] = res
        tables.append(tab.assign(analysis=key))
        if figures:
            from neuromod.viz import plots

            plots.forest_plot(tab, res["pooled"], out / "figures" / f"forest_{key}.png", title)
            if len(tab) >= 3:
                plots.funnel_plot(y, np.sqrt(v), pooled.estimate,
                                  out / "figures" / f"funnel_{key}.png", res["egger"]["p"])
    summary = {
        "analyses": results,
        "note": ("Effects are Hedges' g (positive favors active rTMS). Random effects, REML "
                 "tau^2, Hartung-Knapp 95% CIs. Binary outcomes converted from log odds "
                 "ratios (Chinn 2000). Crossover trials are analysed as parallel groups "
                 "(conservative). Every extracted number is quoted with its source in "
                 "data/meta."),
    }
    (out / "meta" / "meta_results.json").write_text(json.dumps(summary, indent=2,
                                                                default=float))
    if tables:
        cols = ["analysis", "label", "g", "lo", "hi", "weight", "method", "n_active", "n_sham",
                "outcome", "design", "deep_tms", "medial_target", "theta_burst", "source_url"]
        pd.concat(tables)[cols].round(3).to_csv(out / "meta" / "study_effects.csv", index=False)
    return summary
