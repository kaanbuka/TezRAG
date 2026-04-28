from __future__ import annotations

from typing import Any

import requests

from src.config import Config

SYSTEM_PROMPT = """Sen Türkçe akademik kaynaklardan bilgi sunan bir asistansın.

KURALLAR:
1. Cevabı SADECE aşağıda verilen bağlam (context) parçalarına dayanarak Türkçe ver.
2. Kendi genel bilgini kullanma, tahmin yürütme.
3. Her iddianı hemen yanına köşeli parantez içinde kaynak numarasıyla belirt: [1], [2], [3] gibi.
4. Bir iddia birden fazla parçadan destek alıyorsa: [1][3] şeklinde yaz.
5. Cevap bağlamda yoksa yalnızca şunu yaz: "Verilen kaynaklarda bu bilgi bulunmamaktadır."
6. Teknik terimlerin özgün biçimini koru.
7. Cevap net, bilimsel ve öz olsun; uydurma yapma.
"""


def _format_context(chunks: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for i, c in enumerate(chunks, start=1):
        meta = c.get("metadata", {})
        source = meta.get("source", "bilinmiyor")
        page = meta.get("page", "?")
        blocks.append(
            f"[{i}] Kaynak: {source} — Sayfa: {page}\n{c['text'].strip()}"
        )
    return "\n\n---\n\n".join(blocks)


def _build_prompt(query: str, chunks: list[dict[str, Any]]) -> str:
    context = _format_context(chunks)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"BAĞLAM:\n{context}\n\n"
        f"SORU: {query}\n\n"
        f"CEVAP (Türkçe, kaynak numaralarıyla):"
    )


def _preview(text: str, n: int = 240) -> str:
    t = " ".join(text.split())
    return t if len(t) <= n else t[:n].rstrip() + "…"


def generate_answer(
    query: str,
    chunks: list[dict[str, Any]],
    config: Config,
) -> dict[str, Any]:
    if not chunks:
        return {
            "answer": "Verilen kaynaklarda bu bilgi bulunmamaktadır.",
            "sources": [],
        }

    prompt = _build_prompt(query, chunks)
    payload = {
        "model": config.llm_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": config.temperature,
            "num_predict": config.max_tokens,
        },
    }
    url = f"{config.ollama_host.rstrip('/')}/api/generate"

    try:
        resp = requests.post(url, json=payload, timeout=config.request_timeout)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Ollama sunucusuna bağlanılamadı ({config.ollama_host}). "
            f"`ollama serve` çalışıyor mu? Detay: {exc}"
        ) from exc
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(
            f"Ollama HTTP hatası: {exc}. Model yüklü mü? "
            f"(`ollama pull {config.llm_model}`)"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(f"Ollama zaman aşımı: {exc}") from exc

    data = resp.json()
    answer = (data.get("response") or "").strip()

    sources = []
    for i, c in enumerate(chunks, start=1):
        meta = c.get("metadata", {})
        sources.append(
            {
                "id": i,
                "source": meta.get("source", "bilinmiyor"),
                "page": meta.get("page", "?"),
                "text_preview": _preview(c["text"]),
                "rerank_score": c.get("rerank_score"),
            }
        )

    return {"answer": answer, "sources": sources}
