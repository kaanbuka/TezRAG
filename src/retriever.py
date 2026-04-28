from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.config import Config

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, config: Config):
        self.config = config
        self._client = chromadb.PersistentClient(
            path=str(config.chroma_path_resolved),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = SentenceTransformer(config.embedding_model)
        self._reranker: CrossEncoder | None = None

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            logger.info("Reranker yükleniyor: %s", self.config.reranker_model)
            self._reranker = CrossEncoder(self.config.reranker_model)
        return self._reranker

    def count(self) -> int:
        try:
            return self._collection.count()
        except Exception:
            return 0

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        k = top_k or self.config.top_k_retrieval
        if self.count() == 0:
            return []

        prefixed = f"query: {query}"
        q_emb = self._embedder.encode(
            [prefixed],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

        res = self._collection.query(
            query_embeddings=q_emb,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        out: list[dict[str, Any]] = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            out.append({"text": doc, "metadata": meta or {}, "distance": dist})
        return out

    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        if not results:
            return []
        n = top_n or self.config.top_k_rerank
        pairs = [(query, r["text"]) for r in results]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        for r, s in zip(results, scores):
            r["rerank_score"] = float(s)
        results.sort(key=lambda r: r["rerank_score"], reverse=True)
        return results[:n]

    def search(self, query: str) -> list[dict[str, Any]]:
        retrieved = self.retrieve(query)
        if not retrieved:
            return []
        return self.rerank(query, retrieved)
