"""Citation grounding: every claim must cite a retrieved source that supports it."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from neuromod.rag.retriever import tokenize

CITE = re.compile(r"\[S(\d+)\]")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(*-])|\n+")


@dataclass
class SentenceCheck:
    sentence: str
    citations: list[int]
    supported: bool
    support: float
    reason: str = ""


@dataclass
class GroundingReport:
    sentences: list[SentenceCheck] = field(default_factory=list)

    @property
    def n_claims(self) -> int:
        return len(self.sentences)

    @property
    def citation_coverage(self) -> float:
        """Share of claim sentences carrying at least one valid citation."""
        if not self.sentences:
            return 0.0
        return sum(bool(s.citations) for s in self.sentences) / len(self.sentences)

    @property
    def grounding_score(self) -> float:
        """Share of claim sentences whose citations lexically support them."""
        if not self.sentences:
            return 0.0
        return sum(s.supported for s in self.sentences) / len(self.sentences)

    @property
    def unsupported(self) -> list[SentenceCheck]:
        return [s for s in self.sentences if not s.supported]

    def as_dict(self) -> dict:
        return {
            "n_claims": self.n_claims,
            "citation_coverage": round(self.citation_coverage, 3),
            "grounding_score": round(self.grounding_score, 3),
            "unsupported": [s.sentence for s in self.unsupported],
        }


def _support(claim: str, source: str) -> float:
    """Fraction of the claim's content words that appear in the cited source."""
    c = set(tokenize(claim))
    if not c:
        return 1.0
    s = set(tokenize(source))
    return len(c & s) / len(c)


def check_grounding(answer: str, sources: list[str], threshold: float = 0.5) -> GroundingReport:
    """Verify each sentence of ``answer`` against the numbered ``sources`` (1-indexed).

    Headings, bullets without content and the reference list are ignored; every other
    sentence is a claim that must cite a valid source with enough lexical support.
    """
    report = GroundingReport()
    body = answer.split("\nReferences")[0]
    for raw in _SENT_SPLIT.split(body):
        sent = raw.strip().lstrip("-*• ").strip()
        if len(sent) < 25 or sent.endswith(":") or sent.startswith(("#", "_")):
            continue
        cites = [int(n) for n in CITE.findall(sent)]
        valid = [n for n in cites if 1 <= n <= len(sources)]
        claim = CITE.sub("", sent)
        if not cites:
            report.sentences.append(SentenceCheck(sent, [], False, 0.0, "no citation"))
            continue
        if len(valid) < len(cites):
            report.sentences.append(SentenceCheck(sent, valid, False, 0.0,
                                                  "cites a source that was not retrieved"))
            continue
        support = max(_support(claim, sources[n - 1]) for n in valid)
        ok = support >= threshold
        report.sentences.append(SentenceCheck(sent, valid, ok, round(support, 3),
                                              "" if ok else "weak support in cited source"))
    return report
