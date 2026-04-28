from __future__ import annotations

from pathlib import Path

import requests
import streamlit as st

from src.config import Config
from src.ingestion import ingest_pdfs, list_indexed_sources
from src.rag_pipeline import RAGPipeline

PDF_DIR = Path("data/pdfs")
PDF_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="TezRAG — Türkçe Akademik RAG",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource(show_spinner="Modeller yükleniyor…")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline(Config())


def check_ollama(config: Config) -> tuple[bool, str]:
    try:
        r = requests.get(f"{config.ollama_host.rstrip('/')}/api/tags", timeout=3)
        r.raise_for_status()
        tags = r.json().get("models", [])
        names = [t.get("name", "") for t in tags]
        if not any(config.llm_model in n for n in names):
            return (
                False,
                f"Ollama çalışıyor ama `{config.llm_model}` yüklü değil. "
                f"Çalıştır: `ollama pull {config.llm_model}`",
            )
        return True, "Ollama hazır."
    except Exception as exc:
        return (
            False,
            f"Ollama'ya erişilemedi ({config.ollama_host}). "
            f"`ollama serve` çalışıyor mu? Detay: {exc}",
        )


with st.sidebar:
    st.title("📚 TezRAG")
    st.caption("Türkçe akademik PDF sorgulama sistemi")

    pipeline = get_pipeline()
    config = pipeline.config

    ok, msg = check_ollama(config)
    if ok:
        st.success(msg)
    else:
        st.error(msg)

    st.divider()
    st.subheader("PDF Yükle ve İndeksle")

    uploads = st.file_uploader(
        "PDF dosyalarını seçin",
        type=["pdf"],
        accept_multiple_files=True,
        help="Birden fazla dosya seçebilirsiniz.",
    )

    if st.button("İndeksle", type="primary", use_container_width=True):
        if not uploads:
            st.warning("Önce en az bir PDF seçin.")
        else:
            saved = 0
            for uf in uploads:
                target = PDF_DIR / uf.name
                target.write_bytes(uf.getbuffer())
                saved += 1
            with st.spinner(f"{saved} PDF indeksleniyor…"):
                stats = ingest_pdfs(PDF_DIR, config)
            st.success(
                f"Tamam. İşlenen: {stats['files_processed']}  •  "
                f"Atlanan: {stats['skipped']}  •  "
                f"Chunk: {stats['chunks_added']}"
            )
            st.cache_resource.clear()

    st.divider()
    st.subheader("İndekslenmiş Belgeler")
    sources = list_indexed_sources(config)
    if sources:
        st.write(f"Toplam **{len(sources)}** belge, "
                 f"**{pipeline.indexed_count()}** chunk.")
        for s in sources:
            st.markdown(f"- `{s}`")
    else:
        st.info("Henüz indekslenmiş belge yok.")


st.title("Soru-Cevap")

if not sources:
    st.info(
        "👋 Henüz indekslenmiş PDF yok. "
        "Sol panelden dosya yükleyip **İndeksle** düğmesine basın."
    )
else:
    st.caption(f"Model: `{config.llm_model}`  •  Top-K: "
               f"{config.top_k_retrieval}→{config.top_k_rerank}")

question = st.text_area(
    "Sorunuz",
    placeholder="Örn: Makine öğrenmesinde aşırı öğrenme (overfitting) nedir?",
    height=100,
)

ask = st.button("Sor", type="primary", disabled=not sources)

if ask:
    q = (question or "").strip()
    if not q:
        st.warning("Lütfen bir soru girin.")
    else:
        try:
            with st.spinner("Cevap üretiliyor…"):
                result = pipeline.query(q)
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()

        st.markdown("### Cevap")
        st.markdown(result["answer"])

        if result["sources"]:
            with st.expander(f"📎 Kaynaklar ({len(result['sources'])})", expanded=False):
                for s in result["sources"]:
                    score = s.get("rerank_score")
                    score_str = f"  •  skor: {score:.3f}" if score is not None else ""
                    st.markdown(
                        f"**[{s['id']}] `{s['source']}` — Sayfa {s['page']}**"
                        f"{score_str}"
                    )
                    st.caption(s["text_preview"])
                    st.divider()
