"""Hybrid retrieval: FAISS dense search + BM25, fused with reciprocal rank fusion."""

from __future__ import annotations

import math
import pickle
import re
import warnings
from collections import Counter
from pathlib import Path

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from langchain_community.vectorstores import FAISS

from neuromod.rag.corpus import DEFAULT_CORPUS, Study, load_studies, to_documents
from neuromod.rag.embeddings import LSAEmbeddings, get_embeddings, normalize

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "in", "and", "or", "for", "to", "with", "on", "is", "are",
         "was", "were", "by", "as", "at", "be", "does", "do", "what", "which", "how", "that",
         "this", "from", "vs", "versus", "than", "it", "its", "can", "who"}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(normalize(text)) if t not in _STOP]


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(t) for t in texts]
        self.lens = [len(d) for d in self.docs]
        self.avgdl = sum(self.lens) / max(1, len(self.lens))
        self.tf = [Counter(d) for d in self.docs]
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for tf, dl in zip(self.tf, self.lens):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out


class HybridRetriever(BaseRetriever):
    """LangChain retriever over the rTMS corpus.

    ``mode``: ``"hybrid"`` (default), ``"dense"`` (FAISS only) or ``"bm25"``.
    Results are de-duplicated to one chunk per study.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vectorstore: FAISS
    bm25: BM25
    docs: list[Document]
    k: int = 6
    fetch_k: int = 20
    mode: str = "hybrid"
    rrf_k: int = 60

    def _dense_ranking(self, query: str) -> list[int]:
        hits = self.vectorstore.similarity_search_with_score(query, k=min(self.fetch_k,
                                                                         len(self.docs)))
        return [h[0].metadata["_idx"] for h in hits]

    def _bm25_ranking(self, query: str) -> list[int]:
        scores = self.bm25.scores(query)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [i for i in order[: self.fetch_k] if scores[i] > 0]

    def rank(self, query: str) -> list[tuple[int, float]]:
        rankings = []
        if self.mode in ("hybrid", "dense"):
            rankings.append(self._dense_ranking(query))
        lexical: list[int] = []
        if self.mode in ("hybrid", "bm25"):
            lexical = self._bm25_ranking(query)
            rankings.append(lexical)
        fused: dict[int, float] = {}
        for ranking in rankings:
            for r, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.rrf_k + r + 1)
        # ties (common with two rankers) are broken by lexical rank: exact term matches
        # such as a substance or target name are the more precise signal
        lex_rank = {idx: r for r, idx in enumerate(lexical)}
        return sorted(fused.items(), key=lambda kv: (-round(kv[1], 9),
                                                     lex_rank.get(kv[0], len(self.docs))))

    def _get_relevant_documents(self, query: str, *,
                                run_manager: CallbackManagerForRetrieverRun | None = None
                                ) -> list[Document]:
        out, seen = [], set()
        for idx, score in self.rank(query):
            doc = self.docs[idx]
            sid = doc.metadata["study_id"]
            if sid in seen:
                continue
            seen.add(sid)
            out.append(Document(page_content=doc.page_content,
                                metadata={**doc.metadata, "score": round(score, 5)}))
            if len(out) >= self.k:
                break
        return out


class RTMSIndex:
    """Builds and persists the FAISS index + BM25 for the corpus."""

    def __init__(self, studies: list[Study], embeddings: Embeddings | None = None):
        self.studies = studies
        self.by_id = {s.id: s for s in studies}
        self.docs = to_documents(studies)
        for i, d in enumerate(self.docs):
            d.metadata["_idx"] = i
        self.embeddings = embeddings or get_embeddings()
        if isinstance(self.embeddings, LSAEmbeddings) and self.embeddings.svd is None:
            self.embeddings.fit([d.page_content for d in self.docs])
        self.vectorstore = FAISS.from_documents(self.docs, self.embeddings)
        self.bm25 = BM25([d.page_content for d in self.docs])

    @classmethod
    def from_corpus(cls, path: str | Path = DEFAULT_CORPUS, **kw) -> RTMSIndex:
        return cls(load_studies(path), **kw)

    def retriever(self, k: int = 6, mode: str = "hybrid") -> HybridRetriever:
        return HybridRetriever(vectorstore=self.vectorstore, bm25=self.bm25, docs=self.docs,
                               k=k, mode=mode)

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(d / "faiss"))
        with open(d / "index.pkl", "wb") as f:
            pickle.dump({"studies": self.studies, "embeddings": self.embeddings}, f)

    @classmethod
    def load(cls, directory: str | Path) -> RTMSIndex:
        with open(Path(directory) / "index.pkl", "rb") as f:
            state = pickle.load(f)  # noqa: S301 - our own locally built artifact
        return cls(state["studies"], embeddings=state["embeddings"])
