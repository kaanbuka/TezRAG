from __future__ import annotations

from typing import Any

from src.config import Config
from src.generator import generate_answer
from src.retriever import Retriever


class RAGPipeline:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.retriever = Retriever(self.config)

    def query(self, question: str) -> dict[str, Any]:
        question = (question or "").strip()
        if not question:
            return {"answer": "Lütfen bir soru giriniz.", "sources": []}

        chunks = self.retriever.search(question)
        if not chunks:
            return {
                "answer": (
                    "İndekslenmiş belge bulunamadı ya da ilgili parça yok. "
                    "Lütfen önce PDF belgelerinizi indeksleyin."
                ),
                "sources": [],
            }

        return generate_answer(question, chunks, self.config)

    def indexed_count(self) -> int:
        return self.retriever.count()
