from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]

CHUNKS_FILE = ROOT / "knowledge_base" / "chunks.jsonl"
INDEX_FILE = ROOT / "knowledge_base" / "faiss.index"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Yandex OpenAI-compatible API settings
YANDEX_BASE_URL = "https://llm.api.cloud.yandex.net/v1"
YANDEX_FOLDER_ID = os.getenv("YANDEX_CLOUD_FOLDER_ID", "").strip()
YANDEX_API_KEY = os.getenv("YANDEX_CLOUD_API_KEY", "").strip()
YANDEX_MODEL_URI = None  # constructed dynamically

TOP_K = 5
MAX_CONTEXT_CHARS = 12000  # cap prompt size
MAX_TOKENS = 256
TEMPERATURE = 0.2


SYSTEM_PROMPT = """You are a RAG assistant for a private knowledge base.
Follow these rules strictly:
1) Use ONLY the information in [CONTEXT]. Do not use outside knowledge.
2) If the answer is not explicitly supported by [CONTEXT], say: "I don't know based on the provided documents."
3) Be concise and factual. No speculation.
4) Always include a short reasoning trace as numbered steps that reference evidence (chunk ids).
   The steps must be based on the retrieved text, not hidden reasoning.
5) Always end with a "Sources" section listing the chunk ids you used.
""".strip()


@dataclass
class Chunk:
    id: int
    text: str
    source: str | None = None
    start_char: int | None = None
    end_char: int | None = None
    score: float | None = None


def load_chunks(path: Path) -> dict[int, dict]:
    by_id: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            by_id[int(rec["id"])] = rec
    return by_id


def retrieve(
    question: str,
    embedder: SentenceTransformer,
    index: faiss.Index,
    chunks_by_id: dict[int, dict],
    top_k: int = TOP_K,
) -> list[Chunk]:
    q = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    scores, ids = index.search(q, top_k)

    out: list[Chunk] = []
    for cid, score in zip(ids[0], scores[0]):
        cid = int(cid)
        rec = chunks_by_id.get(cid)
        if not rec:
            continue
        out.append(
            Chunk(
                id=cid,
                text=(rec.get("text", "") or "").strip(),
                source=rec.get("source") or rec.get("source_file"),
                start_char=rec.get("start_char"),
                end_char=rec.get("end_char"),
                score=float(score),
            )
        )
    return out


def format_context(chunks: list[Chunk]) -> str:
    parts: list[str] = []
    total = 0

    for ch in chunks:
        header = f"[id={ch.id}]"
        meta = []
        if ch.source:
            meta.append(f"source={ch.source}")
        if ch.start_char is not None and ch.end_char is not None:
            meta.append(f"pos={ch.start_char}:{ch.end_char}")
        if meta:
            header += " " + " ".join(meta)

        block = header + "\n" + ch.text
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        total += len(block)

    return "\n\n".join(parts)


def build_prompt(question: str, docs: str) -> str:
    # строго по требуемому формату
    return (
        f"[System]\n{SYSTEM_PROMPT}\n\n"
        f"[CONTEXT]\n<<<\n{docs}\n>>>\n\n"
        f"Question: {question}\n"
    )


def yandex_generate(prompt: str) -> str:
    if not YANDEX_FOLDER_ID or not YANDEX_API_KEY:
        raise RuntimeError(
            "Missing env vars. Set:\n"
            "  YANDEX_CLOUD_FOLDER_ID=<your folder id>\n"
            "  YANDEX_CLOUD_API_KEY=<your service account api key>"
        )

    client = OpenAI(
        api_key=YANDEX_API_KEY,
        base_url=YANDEX_BASE_URL,
        project=YANDEX_FOLDER_ID,
    )

    model = f"gpt://{YANDEX_FOLDER_ID}/yandexgpt/latest"

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    return (resp.choices[0].message.content or "").strip()


def repl() -> None:
    if not CHUNKS_FILE.exists():
        raise SystemExit(f"Missing: {CHUNKS_FILE}")
    if not INDEX_FILE.exists():
        raise SystemExit(f"Missing: {INDEX_FILE}")

    chunks_by_id = load_chunks(CHUNKS_FILE)
    index = faiss.read_index(str(INDEX_FILE))
    embedder = SentenceTransformer(EMBED_MODEL, device="cpu")

    print("RAG REPL (type 'exit' to quit)")
    print(f"Embedder: {EMBED_MODEL}")
    print(f"LLM: YandexGPT via {YANDEX_BASE_URL}")
    print(f"TopK: {TOP_K} | max_tokens: {MAX_TOKENS}\n")

    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        retrieved = retrieve(q, embedder, index, chunks_by_id, top_k=TOP_K)
        docs = format_context(retrieved)

        if not docs.strip():
            print("I don't know based on the provided documents.\n")
            continue

        prompt = build_prompt(q, docs)
        try:
            answer = yandex_generate(prompt)
        except Exception as e:
            print(f"[ERROR] YandexGPT request failed: {type(e).__name__}: {e}\n")
            continue

        print(answer)
        print()


def main() -> None:
    # One-shot mode:
    #   python scripts/rag_chat.py --once "your question"
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        question = " ".join(sys.argv[2:]).strip()
        if not question:
            raise SystemExit('Usage: python scripts/rag_chat.py --once "your question"')

        if not CHUNKS_FILE.exists():
            raise SystemExit(f"Missing: {CHUNKS_FILE}")
        if not INDEX_FILE.exists():
            raise SystemExit(f"Missing: {INDEX_FILE}")

        chunks_by_id = load_chunks(CHUNKS_FILE)
        index = faiss.read_index(str(INDEX_FILE))
        embedder = SentenceTransformer(EMBED_MODEL, device="cpu")

        retrieved = retrieve(question, embedder, index, chunks_by_id, top_k=TOP_K)
        docs = format_context(retrieved)

        if not docs.strip():
            print("I don't know based on the provided documents.")
            return

        prompt = build_prompt(question, docs)
        print(yandex_generate(prompt))
        return

    repl()


if __name__ == "__main__":
    main()
