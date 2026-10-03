"""Citation-grounded RAG over the rTMS literature (LangChain + FAISS).

Three tasks share one retrieval step:

* ``ask``        - evidence synthesis for a clinical question
* ``hypotheses`` - testable hypotheses motivated by the retrieved studies
* ``design``     - a study-design brief (PICO, protocol, sample size, stratification)

Generation is pluggable. With ``ANTHROPIC_API_KEY`` set (and ``neuromod[llm]``
installed) a Claude model writes the answer through a LangChain chain. Without a key
the pipeline falls back to a deterministic extractive/templated generator, so every
command works offline. Either way the answer passes through the citation checker in
``grounding.py`` and only cites studies that were actually retrieved.
"""

from __future__ import annotations

import json
import os
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from neuromod.rag.corpus import DEFAULT_CORPUS, Study, split_sentences
from neuromod.rag.grounding import check_grounding
from neuromod.rag.retriever import RTMSIndex, tokenize

DEFAULT_LLM = "claude-sonnet-5-5"
MODES = ("ask", "hypotheses", "design")

SYSTEM_PROMPT = """You are a research assistant for a Duke Bass Connections team studying \
noninvasive neuromodulation (rTMS) for addiction in veterans with comorbid PTSD.
Use ONLY the numbered sources provided. Rules:
1. End every sentence that states a finding with one or more citations like [S1] or [S2][S4].
2. Never cite a source number that is not listed. Never invent studies, numbers or authors.
3. If the sources do not answer the question, say so explicitly.
4. Be concise and clinically precise; note sample sizes and designs when relevant."""

TASK_PROMPTS = {
    "ask": "Question: {question}\n\nWrite a short evidence synthesis (4-8 sentences) answering "
           "the question, noting agreement, disagreement and evidence quality.",
    "hypotheses": "Research focus: {question}\n\nFirst summarise the relevant evidence in 3-5 "
                  "cited sentences under '## Evidence'. Then under '## Hypotheses' propose 3 "
                  "specific, testable hypotheses (population, intervention, comparator, outcome), "
                  "each citing the studies that motivate it.",
    "design": "Study to design: {question}\n\nUnder '## Evidence' summarise the most relevant "
              "protocols in 3-5 cited sentences. Then under '## Design brief' propose: population, "
              "intervention (target, frequency, sessions), comparator, primary/secondary outcomes, "
              "sample-size reasoning, stratification variables and follow-up, citing sources.",
}


@dataclass
class RAGAnswer:
    question: str
    mode: str
    answer: str
    sources: list[dict]
    grounding: dict
    backend: str
    extras: dict = field(default_factory=dict)

    def to_markdown(self) -> str:
        refs = "\n".join(f"[S{i}] {s['reference']}" + (f" {s['link']}" if s.get("link") else "")
                         for i, s in enumerate(self.sources, 1))
        g = self.grounding
        return (f"{self.answer}\n\nReferences\n{refs}\n\n"
                f"_Grounding: {g['grounding_score']:.0%} of {g['n_claims']} claims supported by "
                f"their cited source; citation coverage {g['citation_coverage']:.0%}. "
                f"Backend: {self.backend}._")


class RTMSResearchAssistant:
    def __init__(self, index: RTMSIndex | None = None, k: int = 6,
                 llm=None, metrics_path: str | Path | None = "reports/metrics.json"):
        self.index = index or RTMSIndex.from_corpus(DEFAULT_CORPUS)
        self.k = k
        self.llm = llm if llm is not None else _default_llm()
        self.metrics_path = Path(metrics_path) if metrics_path else None

    # ------------------------------------------------------------------ public API
    def run(self, question: str, mode: str = "ask", k: int | None = None,
            strict: bool = False) -> RAGAnswer:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        docs = self.index.retriever(k=k or self.k).invoke(question)
        studies = [self.index.by_id[d.metadata["study_id"]] for d in docs]
        source_texts = [d.page_content for d in docs]
        if self.llm is not None:
            answer = self._generate_llm(question, mode, docs)
            backend = f"llm:{getattr(self.llm, 'model', getattr(self.llm, 'model_name', 'chat'))}"
        else:
            answer = {"ask": self._extractive_answer,
                      "hypotheses": self._extractive_hypotheses,
                      "design": self._extractive_design}[mode](question, studies)
            backend = "extractive (offline)"
        evidence = _evidence_section(answer, mode)
        report = check_grounding(evidence, source_texts)
        if strict and report.unsupported:
            for s in report.unsupported:
                answer = answer.replace(s.sentence, "").replace("\n\n\n", "\n\n")
            report = check_grounding(_evidence_section(answer, mode), source_texts)
        sources = [{"id": s.id, "short_cite": s.short_cite, "title": s.title,
                    "reference": s.reference(), "link": s.link,
                    "score": d.metadata.get("score")} for s, d in zip(studies, docs)]
        return RAGAnswer(question, mode, answer.strip(), sources, report.as_dict(), backend)

    def ask(self, question: str, **kw) -> RAGAnswer:
        return self.run(question, "ask", **kw)

    def hypotheses(self, question: str, **kw) -> RAGAnswer:
        return self.run(question, "hypotheses", **kw)

    def design(self, question: str, **kw) -> RAGAnswer:
        return self.run(question, "design", **kw)

    # ------------------------------------------------------------------ LLM path
    def _generate_llm(self, question: str, mode: str, docs: list[Document]) -> str:
        context = "\n\n".join(f"[S{i}] {d.page_content}" for i, d in enumerate(docs, 1))
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "Sources:\n{context}\n\n" + TASK_PROMPTS[mode]),
        ])
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": question})

    # ------------------------------------------------------------------ offline path
    def _rank_sentences(self, question: str, studies: list[Study], max_total: int = 6,
                        max_per_study: int = 2) -> list[tuple[int, str]]:
        emb = self.index.embeddings
        q_vec = np.asarray(emb.embed_query(question))
        q_tok = set(tokenize(question))
        cands = []
        for i, s in enumerate(studies):
            for sent in split_sentences(s.abstract or s.summary):
                vec = np.asarray(emb.embed_query(sent))
                overlap = len(q_tok & set(tokenize(sent))) / max(1, len(q_tok))
                rank_prior = 1.0 / (1 + i)  # retrieval order
                cands.append((float(q_vec @ vec) + overlap + 0.3 * rank_prior, i, sent))
        cands.sort(key=lambda c: -c[0])
        picked, per = [], Counter()
        for _, i, sent in cands:
            if per[i] >= max_per_study:
                continue
            picked.append((i, sent))
            per[i] += 1
            if len(picked) >= max_total:
                break
        return picked

    @staticmethod
    def _label(s: Study) -> str:
        bits = [s.study_type] if s.study_type else []
        if s.n:
            bits.append(f"n={s.n}")
        return f"{s.short_cite}" + (f" ({', '.join(bits)})" if bits else "")

    def _evidence_lines(self, question: str, studies: list[Study], max_total: int = 6) -> list[str]:
        lines = []
        for i, sent in self._rank_sentences(question, studies, max_total=max_total):
            lines.append(f"- {self._label(studies[i])}: {sent.rstrip('.')} [S{i + 1}].")
        return lines

    def _extractive_answer(self, question: str, studies: list[Study]) -> str:
        lines = self._evidence_lines(question, studies)
        types = Counter(s.study_type for s in studies if s.study_type)
        quality = ", ".join(f"{v} {k}" for k, v in types.most_common())
        return (f"## Evidence ({len(studies)} studies retrieved: {quality})\n" + "\n".join(lines)
                + "\n\n_Offline extractive mode: sentences are quoted from study summaries "
                  "verbatim; set ANTHROPIC_API_KEY for an abstractive synthesis._")

    def _extractive_hypotheses(self, question: str, studies: list[Study]) -> str:
        ev = self._evidence_lines(question, studies, max_total=5)
        idx = {s.id: i + 1 for i, s in enumerate(studies)}
        rcts = [s for s in studies if s.study_type and "RCT" in s.study_type]
        ptsd = [s for s in studies if "ptsd" in (s.condition or "").lower()]
        sud = [s for s in studies if s not in ptsd]
        targets = Counter(_norm_target(s.target) for s in rcts if s.target)
        top_target = targets.most_common(1)[0][0] if targets else "left DLPFC"

        def cites(group):
            return "".join(f"[S{idx[s.id]}]" for s in group[:3]) or ""

        hyps = [
            f"H1 (efficacy): In veterans with a substance use disorder and comorbid PTSD, active "
            f"rTMS to the {top_target} will reduce cue-induced craving more than sham over a "
            f"standard acute course {cites([s for s in rcts if _norm_target(s.target) == top_target])}.",
            f"H2 (moderation): Baseline PTSD severity will moderate the craving response, since "
            f"PTSD and addiction trials target overlapping prefrontal circuits "
            f"{cites(ptsd[:2] + sud[:1])}.",
            "H3 (durability): Acute craving responders will show longer time-to-relapse over "
            f"52 weeks than non-responders, a gap given the short follow-up in most trials "
            f"{cites(rcts[:2])}.",
        ]
        shap_note = self._shap_note()
        if shap_note:
            hyps.append(shap_note)
        return "## Evidence\n" + "\n".join(ev) + "\n\n## Hypotheses\n" + "\n".join(
            f"- {h}" for h in hyps)

    def _extractive_design(self, question: str, studies: list[Study]) -> str:
        ev = self._evidence_lines(question, studies, max_total=5)
        idx = {s.id: i + 1 for i, s in enumerate(studies)}
        rcts = [s for s in studies if s.study_type and "RCT" in s.study_type] or studies
        targets = Counter(_norm_target(s.target) for s in rcts if s.target)
        top_target, n_t = targets.most_common(1)[0] if targets else ("left DLPFC", 0)
        t_cites = "".join(f"[S{idx[s.id]}]" for s in rcts
                          if _norm_target(s.target) == top_target)
        ns = [s.n for s in rcts if s.n]
        sessions = [int(m) for s in rcts if s.protocol
                    for m in re.findall(r"(\d+)\s*(?:daily\s+)?sessions", s.protocol)]
        brief = [
            "Population: veterans with a DSM-5 substance use disorder and comorbid PTSD "
            "(PCL-5 >= 33), stratified by primary substance.",
            f"Intervention: active rTMS to the {top_target}, the most common target among the "
            f"retrieved RCTs ({n_t} of {len(rcts)}) {t_cites}.",
            "Comparator: sham coil matched for sound and scalp sensation, double-blind.",
        ]
        if sessions:
            brief.append(f"Dose: retrieved protocols used a median of "
                         f"{statistics.median(sessions):g} sessions (range {min(sessions)}-"
                         f"{max(sessions)}).")
        if ns:
            brief.append(f"Sample size context: retrieved RCTs enrolled a median of "
                         f"{statistics.median(ns):g} participants (range {min(ns)}-{max(ns)}), "
                         "so a multi-site design is likely needed for relapse endpoints.")
        brief += [
            "Outcomes: primary = craving (VAS) at end of treatment; secondary = biochemically "
            "verified abstinence, time-to-relapse over 52 weeks, PCL-5 change.",
        ]
        enrich = self._enrichment_note()
        if enrich:
            brief.append(enrich)
        shap = self._shap_note(prefix="Stratification")
        if shap:
            brief.append(shap)
        return "## Evidence\n" + "\n".join(ev) + "\n\n## Design brief\n" + "\n".join(
            f"- {b}" for b in brief)

    # --------------------------------------------------- link to the ML pipeline
    def _metrics(self) -> dict | None:
        if self.metrics_path and self.metrics_path.exists():
            return json.loads(self.metrics_path.read_text())
        return None

    def _shap_note(self, prefix: str = "H4 (simulation-derived)") -> str | None:
        m = self._metrics()
        if not m:
            return None
        top = [f for f in m["shap"]["response_top_predictors"] if f not in ("active", "target")][:3]
        return (f"{prefix}: the synthetic-cohort XGBoost model ranks {', '.join(top)} as the "
                "strongest SHAP predictors of response; treat these as candidate stratification "
                "variables to test, not as evidence (synthetic data).")

    def _enrichment_note(self) -> str | None:
        m = self._metrics()
        if not m:
            return None
        h = m["hte"]
        return ("Enrichment (synthetic simulation): enrolling the top 50% of predicted "
                f"responders cut the randomized sample for 80% power from "
                f"{h['n_randomized_all_comers']:.0f} to {h['n_randomized_top50pct']:.0f} "
                f"({h['randomized_sample_reduction_top50pct']:.0%} fewer).")


def _norm_target(t: str | None) -> str:
    if not t:
        return "unspecified"
    t = t.lower()
    if "dlpfc" in t or "dorsolateral" in t:
        side = "left " if "left" in t else "right " if "right" in t else ""
        if "bilateral" in t:
            side = "bilateral "
        return f"{side}DLPFC".strip()
    if "mpfc" in t or "medial prefrontal" in t or "acc" in t or "cingulate" in t:
        return "mPFC/ACC"
    if "insula" in t:
        return "insula"
    return t


def _evidence_section(answer: str, mode: str) -> str:
    if mode == "ask":
        return answer
    m = re.search(r"##\s*Evidence(.*?)(?=\n##\s|\Z)", answer, flags=re.S)
    return m.group(1) if m else answer


def _default_llm():
    if not os.getenv("ANTHROPIC_API_KEY") or os.getenv("NEUROMOD_OFFLINE"):
        return None
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        return None
    return ChatAnthropic(model=os.getenv("NEUROMOD_LLM_MODEL", DEFAULT_LLM), temperature=0,
                         max_tokens=1500)
