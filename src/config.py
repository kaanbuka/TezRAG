from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "intfloat/multilingual-e5-base"
    )
    reranker_model: str = os.getenv(
        "RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
    )
    llm_model: str = os.getenv("LLM_MODEL", "qwen2.5:7b")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k_retrieval: int = 10
    top_k_rerank: int = 4

    chroma_path: str = os.getenv("CHROMA_PATH", "./chroma_db")
    collection_name: str = os.getenv("COLLECTION_NAME", "tezrag")

    temperature: float = 0.2
    max_tokens: int = 1024
    request_timeout: int = 180

    @property
    def chroma_path_resolved(self) -> Path:
        return Path(self.chroma_path).expanduser().resolve()
