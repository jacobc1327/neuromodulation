"""Embedding backends implementing LangChain's ``Embeddings`` interface.

The default, ``LSAEmbeddings`` (TF-IDF + truncated SVD), is fully offline and
deterministic, so the whole RAG pipeline runs in CI and on a laptop with no
network or API key. A sentence-transformers model can be swapped in with
``NEUROMOD_EMBEDDINGS=hf`` (``pip install neuromod[embeddings]``).
"""

from __future__ import annotations

import os

import numpy as np
from langchain_core.embeddings import Embeddings
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# Domain synonyms folded together before vectorizing so lexical variants match.
SYNONYMS = {
    "transcranial magnetic stimulation": "tms",
    "repetitive tms": "rtms",
    "dorsolateral prefrontal cortex": "dlpfc",
    "posttraumatic stress disorder": "ptsd",
    "post-traumatic stress disorder": "ptsd",
    "intermittent theta burst": "itbs",
    "intermittent theta-burst": "itbs",
    "theta-burst": "theta burst",
    "alcohol use disorder": "aud alcohol",
    "cigarette": "smoking nicotine",
    "tobacco": "smoking nicotine",
    "veteran": "veterans military",
}


def normalize(text: str) -> str:
    t = text.lower()
    for k, v in SYNONYMS.items():
        t = t.replace(k, f"{k} {v}")
    return t


class LSAEmbeddings(Embeddings):
    """Latent semantic analysis embeddings fitted on the corpus itself."""

    def __init__(self, n_components: int = 96, random_state: int = 0):
        self.n_components = n_components
        self.random_state = random_state
        self.vectorizer: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None

    def fit(self, texts: list[str]) -> LSAEmbeddings:
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1,
                                          stop_words="english", preprocessor=normalize)
        tfidf = self.vectorizer.fit_transform(texts)
        k = max(2, min(self.n_components, tfidf.shape[0] - 1, tfidf.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=self.random_state).fit(tfidf)
        return self

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self.vectorizer is None or self.svd is None:
            raise RuntimeError("LSAEmbeddings must be fit() on the corpus first")
        z = self.svd.transform(self.vectorizer.transform(texts))
        z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
        return z.astype("float32").tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


def get_embeddings(kind: str | None = None) -> Embeddings:
    kind = (kind or os.getenv("NEUROMOD_EMBEDDINGS", "lsa")).lower()
    if kind == "hf":
        from langchain_huggingface import HuggingFaceEmbeddings  # optional dependency

        model = os.getenv("NEUROMOD_HF_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(model_name=model,
                                     encode_kwargs={"normalize_embeddings": True})
    return LSAEmbeddings()
