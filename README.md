# TezRAG - Turkish Academic PDF Q&A (RAG)

**English** · [Türkçe](README.tr.md)

[![Python](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**TezRAG** is a fully **local** **Retrieval-Augmented Generation (RAG)** application for **Turkish academic PDFs**. Documents are stored in a vector database; each question triggers semantic retrieval of relevant passages, then a **local language model** generates **Turkish, source-grounded** answers. No third-party LLM API key is required (inference via **Ollama**).

---

## Screenshot

![TezRAG Streamlit UI: PDF indexing, Turkish Q&A with citations](docs/tezrag-ui.png)

---

## Skills & competencies demonstrated

| Area | What this project implements |
|------|------------------------------|
| **RAG pipeline** | Question → dense retrieval → cross-encoder reranking → grounded generation |
| **Vector storage** | ChromaDB (persistent), cosine similarity, HNSW |
| **Embeddings** | `multilingual-e5-base` ; `query:` / `passage:` prefixes per model requirements |
| **Reranking** | Re-scoring candidates with `bge-reranker-v2-m3` (two-stage retrieve + rerank) |
| **Data preparation** | PDF text extraction (pdfplumber, pypdf fallback), Turkish-aware chunking, page & source metadata |
| **Idempotent indexing** | File signatures to skip unchanged PDFs on re-runs |
| **LLM integration** | Ollama HTTP API (`/api/generate`), configurable model and options |
| **Prompt design** | Numbered citations, system rules to reduce off-context hallucinations (Turkish academic tone) |
| **UI** | Streamlit: PDF upload, indexing, Q&A, source display |

> **One-liner:** End-to-end RAG for Turkish documents; indexing, semantic retrieval, reranking, and local LLM generation in one stack.

---

## Architecture (overview)

```text
PDF → text extraction → chunks (+ metadata)
                    ↓
            Embedding (E5, passage:)
                    ↓
            ChromaDB (vector search, top-k)
                    ↓
            Cross-encoder rerank (top-k → top-n)
                    ↓
            Prompt + context → Ollama (local LLM) → Turkish answer + source list
```

---

## Tech stack

| Component | Choice |
|-----------|--------|
| Orchestration / UI | Streamlit |
| Vector DB | ChromaDB |
| Embeddings | Sentence Transformers - `intfloat/multilingual-e5-base` |
| Reranker | CrossEncoder - `BAAI/bge-reranker-v2-m3` |
| PDF | pdfplumber, pypdf (fallback) |
| Chunking | LangChain `RecursiveCharacterTextSplitter` (Turkish-oriented separators) |
| LLM | Ollama (default: `qwen2.5:7b`) |
| Configuration | `python-dotenv`, `src/config.py` |

---

## Highlights

- **Privacy & cost:** Documents and embedding/reranker models stay local; no mandatory cloud LLM API.
- **Two-stage retrieval:** Broad candidate set (e.g. 10) → rerank to a smaller set (e.g. 4) for sharper context.
- **Turkish-oriented chunking:** Splits respect sentence/paragraph boundaries; tunable size and overlap for academic Q&A.
- **Source transparency:** UI can show file, page, and rerank scores.
- **CLI indexing:** Bulk ingest via `scripts/index_pdfs.py`; interactive upload via Streamlit.

---

## Requirements

- **Python 3.11+**
- **[Ollama](https://ollama.com)** installed and running (`ollama serve`)
- On first run, embedding and reranker models download from Hugging Face (no API key for open weights)

---

## Setup

```bash
git clone <TezRAG>
cd <cloned-repo-folder>

python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env          # Optional; defaults are often enough
```

Example Ollama model:

```bash
ollama pull qwen2.5:7b
```

---

## Usage

**1) Add PDFs**

Copy PDFs into `data/pdfs/` or upload them in the app sidebar.

**2) Index (CLI)**

```bash
python -m scripts.index_pdfs data/pdfs/
```

**3) Launch the UI**

```bash
streamlit run app.py
```

Ask questions in the browser; at least one PDF must be indexed first. If Ollama is unreachable, the sidebar shows a warning; ensure the server is running (`ollama serve`).

---

## Configuration

| Setting | Default | Notes |
|---------|---------|--------|
| Chunk size / overlap | 800 / 150 | Editable in `src/config.py` |
| Retrieval → rerank | 10 → 4 | `top_k_retrieval`, `top_k_rerank` |
| LLM temperature / tokens | 0.2 / 1024 | Generation behavior |

Override via `.env`; see `.env.example` for template variables.

---

## Project layout

```text
├── app.py                 # Streamlit app
├── scripts/
│   └── index_pdfs.py      # Batch PDF indexing
├── src/
│   ├── config.py          # Environment & hyperparameters
│   ├── ingestion.py       # PDF → chunks → ChromaDB
│   ├── retriever.py       # Embeddings, query, rerank
│   ├── generator.py       # Answer generation via Ollama
│   └── rag_pipeline.py  # Pipeline wiring
├── data/pdfs/             # Place PDFs here (*.pdf gitignored)
├── chroma_db/             # Local vector store (gitignored)
├── docs/
│   └── tezrag-ui.png      # README screenshot
├── requirements.txt
├── .env.example
└── LICENSE
```

---

## License

This project is released under the [MIT License](LICENSE).
