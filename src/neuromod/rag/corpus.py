"""Load the rTMS literature corpus and turn it into LangChain documents."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CORPUS = Path(__file__).resolve().parents[3] / "data" / "corpus" / "studies.jsonl"

REQUIRED = ["id", "title", "authors", "year", "journal", "summary"]


@dataclass(frozen=True)
class Study:
    id: str
    title: str
    authors: str
    year: int
    journal: str
    summary: str
    doi: str | None = None
    pmid: str | None = None
    url: str | None = None
    study_type: str | None = None
    condition: str | None = None
    condition_group: str | None = None
    population: str | None = None
    n: int | None = None
    target: str | None = None
    protocol: str | None = None
    limitations: str | None = None
    peer_reviewed: bool = True
    abstract: str | None = None  # filled in by scripts/fetch_pubmed.py when available

    @property
    def first_author(self) -> str:
        return self.authors.split(",")[0].split()[0]

    @property
    def short_cite(self) -> str:
        return f"{self.first_author} et al., {self.year}"

    @property
    def link(self) -> str | None:
        if self.doi:
            return f"https://doi.org/{self.doi}"
        if self.pmid:
            return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"
        return self.url

    def reference(self) -> str:
        parts = [f"{self.authors} ({self.year}). {self.title}. {self.journal}."]
        if self.doi:
            parts.append(f"doi:{self.doi}")
        if self.pmid:
            parts.append(f"PMID:{self.pmid}")
        return " ".join(parts)

    def card(self) -> str:
        """Structured text used for retrieval (one per study)."""
        lines = [f"Title: {self.title}", f"Citation: {self.short_cite}, {self.journal}"]
        for label, val in [("Design", self.study_type), ("Condition", self.condition),
                           ("Population", self.population),
                           ("Sample size", self.n), ("Stimulation target", self.target),
                           ("Protocol", self.protocol)]:
            if val not in (None, ""):
                lines.append(f"{label}: {val}")
        lines.append(f"Summary: {self.abstract or self.summary}")
        if self.limitations:
            lines.append(f"Limitations: {self.limitations}")
        return "\n".join(lines)


def load_studies(path: str | Path = DEFAULT_CORPUS) -> list[Study]:
    studies, seen = [], set()
    fields = set(Study.__dataclass_fields__)
    for line_no, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        rec = json.loads(line)
        missing = [k for k in REQUIRED if not rec.get(k)]
        if missing:
            raise ValueError(f"{path}:{line_no} missing fields {missing}")
        if rec["id"] in seen:
            raise ValueError(f"duplicate study id {rec['id']}")
        seen.add(rec["id"])
        studies.append(Study(**{k: v for k, v in rec.items() if k in fields}))
    return studies


def to_documents(studies: list[Study], chunk_size: int = 900,
                 chunk_overlap: int = 120) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                              separators=["\n", ". ", " "])
    docs = []
    for s in studies:
        for i, chunk in enumerate(splitter.split_text(s.card())):
            # keep the citation header on every chunk so each is self-describing
            if i > 0:
                chunk = f"Title: {s.title}\nCitation: {s.short_cite}\n{chunk}"
            docs.append(Document(page_content=chunk, metadata={
                "study_id": s.id, "chunk": i, "title": s.title, "year": s.year,
                "short_cite": s.short_cite, "doi": s.doi, "pmid": s.pmid,
                "study_type": s.study_type, "condition": s.condition,
                "condition_group": s.condition_group, "n": s.n, "target": s.target,
                "protocol": s.protocol, "link": s.link,
            }))
    return docs


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if len(s.strip()) > 20]
