from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import Config
from src.ingestion import ingest_pdfs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PDF belgelerini ChromaDB'ye indeksle."
    )
    parser.add_argument(
        "pdf_dir",
        type=str,
        nargs="?",
        default="data/pdfs",
        help="PDF dizini (varsayılan: data/pdfs)",
    )
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        print(f"[HATA] Dizin yok: {pdf_dir}", file=sys.stderr)
        return 1

    config = Config()
    print(f"İndeksleme başlıyor: {pdf_dir}")
    print(f"Embedding modeli : {config.embedding_model}")
    print(f"ChromaDB yolu    : {config.chroma_path_resolved}")
    print(f"Koleksiyon       : {config.collection_name}")
    print("-" * 60)

    stats = ingest_pdfs(pdf_dir, config)

    print("-" * 60)
    print(f"Toplam bulunan PDF     : {stats.get('total_files', 0)}")
    print(f"İşlenen dosya sayısı   : {stats['files_processed']}")
    print(f"Atlanan (zaten indeks) : {stats['skipped']}")
    print(f"Eklenen chunk sayısı   : {stats['chunks_added']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
