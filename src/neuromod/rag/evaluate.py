"""Retrieval and grounding evaluation for the RAG pipeline."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from neuromod.rag.retriever import RTMSIndex

DEFAULT_EVAL = Path(__file__).resolve().parents[3] / "data" / "corpus" / "eval_questions.jsonl"


def load_eval(path: str | Path = DEFAULT_EVAL) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _metrics_for(ranked: list[str], relevant: set[str], k: int) -> dict:
    top = ranked[:k]
    hits = [1 if sid in relevant else 0 for sid in top]
    recall = sum(hits) / len(relevant)
    rr = next((1 / (i + 1) for i, h in enumerate(hits) if h), 0.0)
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return {"recall": recall, "mrr": rr, "ndcg": dcg / idcg, "hit": float(any(hits))}


def evaluate_retrieval(index: RTMSIndex, questions: list[dict], k: int = 5,
                       modes=("bm25", "dense", "hybrid")) -> dict:
    out = {}
    for mode in modes:
        retriever = index.retriever(k=k, mode=mode)
        rows = []
        for q in questions:
            ranked = [d.metadata["study_id"] for d in retriever.invoke(q["question"])]
            rows.append(_metrics_for(ranked, set(q["relevant"]), k))
        out[mode] = {f"{m}@{k}": float(np.mean([r[m] for r in rows]))
                     for m in ["recall", "mrr", "ndcg", "hit"]}
    out["_meta"] = {"n_questions": len(questions), "n_studies": len(index.studies), "k": k}
    return out


def evaluate_grounding(assistant, questions: list[dict], mode: str = "ask") -> dict:
    scores, coverage, valid = [], [], []
    for q in questions:
        ans = assistant.run(q["question"], mode=mode)
        scores.append(ans.grounding["grounding_score"])
        coverage.append(ans.grounding["citation_coverage"])
        ids = {s["id"] for s in ans.sources}
        valid.append(len(ids & set(q["relevant"])) > 0)
    return {"mean_grounding_score": float(np.mean(scores)),
            "mean_citation_coverage": float(np.mean(coverage)),
            "answers_citing_a_relevant_study": float(np.mean(valid)),
            "n_questions": len(questions)}
