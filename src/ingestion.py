from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import Config

logger = logging.getLogger(__name__)

TURKISH_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]


def extract_text_from_pdf(path: Path) -> list[dict]:
    pages: list[dict] = []
    try:
        import pdfplumber

        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append({"page": i, "text": text})
        if pages:
            return pages
    except Exception as exc:
        logger.warning("pdfplumber failed for %s: %s — falling back to pypdf", path.name, exc)

    from pypdf import PdfReader

    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            pages.append({"page": i, "text": text})
    return pages


def chunk_document(
    pages: list[dict],
    source_name: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=TURKISH_SEPARATORS,
        length_function=len,
    )

    chunks: list[dict] = []
    chunk_idx = 0
    for page_data in pages:
        page_num = page_data["page"]
        for piece in splitter.split_text(page_data["text"]):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                {
                    "text": piece,
                    "metadata": {
                        "source": source_name,
                        "page": page_num,
                        "chunk_index": chunk_idx,
                    },
                }
            )
            chunk_idx += 1
    return chunks


def _file_signature(path: Path) -> str:
    stat = path.stat()
    raw = f"{path.name}:{int(stat.st_mtime)}".encode()
    return hashlib.sha1(raw).hexdigest()[:16]


def _chunk_id(signature: str, chunk_idx: int) -> str:
    return f"{signature}-{chunk_idx}"


def _iter_pdfs(pdf_dir: Path) -> Iterable[Path]:
    for ext in ("*.pdf", "*.PDF"):
        yield from pdf_dir.glob(ext)


def _already_indexed(collection, signature: str) -> bool:
    try:
        existing = collection.get(where={"signature": signature}, limit=1)
        return bool(existing.get("ids"))
    except Exception:
        return False


def ingest_pdfs(pdf_dir: Path | str, config: Config) -> dict:
    pdf_dir = Path(pdf_dir).expanduser().resolve()
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF dizini bulunamadı: {pdf_dir}")

    pdfs = sorted(_iter_pdfs(pdf_dir))
    if not pdfs:
        return {"files_processed": 0, "chunks_added": 0, "skipped": 0}

    client = chromadb.PersistentClient(
        path=str(config.chroma_path_resolved),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=config.collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    embedder = SentenceTransformer(config.embedding_model)

    files_processed = 0
    chunks_added = 0
    skipped = 0

    for pdf_path in tqdm(pdfs, desc="PDF indexing", unit="file"):
        signature = _file_signature(pdf_path)
        if _already_indexed(collection, signature):
            skipped += 1
            continue

        pages = extract_text_from_pdf(pdf_path)
        if not pages:
            logger.warning("Boş PDF, atlanıyor: %s", pdf_path.name)
            continue

        chunks = chunk_document(
            pages,
            source_name=pdf_path.name,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        if not chunks:
            continue

        texts = [f"passage: {c['text']}" for c in chunks]
        embeddings = embedder.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

        ids = [_chunk_id(signature, c["metadata"]["chunk_index"]) for c in chunks]
        metadatas = []
        for c in chunks:
            md = dict(c["metadata"])
            md["signature"] = signature
            metadatas.append(md)
        documents = [c["text"] for c in chunks]

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        files_processed += 1
        chunks_added += len(chunks)

    return {
        "files_processed": files_processed,
        "chunks_added": chunks_added,
        "skipped": skipped,
        "total_files": len(pdfs),
    }


def list_indexed_sources(config: Config) -> list[str]:
    try:
        client = chromadb.PersistentClient(
            path=str(config.chroma_path_resolved),
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_or_create_collection(name=config.collection_name)
        data = collection.get(include=["metadatas"])
        sources = {m.get("source") for m in data.get("metadatas", []) if m}
        return sorted(s for s in sources if s)
    except Exception as exc:
        logger.warning("Kaynak listesi alınamadı: %s", exc)
        return []
