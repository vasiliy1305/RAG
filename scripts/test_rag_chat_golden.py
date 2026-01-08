# scripts/test_rag_chat_golden.py
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer


# --- project paths ---
ROOT = Path(__file__).resolve().parents[1]
CHUNKS_FILE = ROOT / "knowledge_base" / "chunks.json"
INDEX_FILE = ROOT / "knowledge_base" / "faiss.index"
LOG_FILE = ROOT / "logs" / "rag_golden_run.jsonl"
GOLDEN_FILE_DEFAULT = ROOT / "golden_questions.txt"

# --- models/settings (match your rag_chat.py) ---
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
YANDEX_BASE_URL = "https://llm.api.cloud.yandex.net/v1"
YANDEX_FOLDER_ID = os.getenv("YANDEX_CLOUD_FOLDER_ID", "").strip()
YANDEX_API_KEY = os.getenv("YANDEX_CLOUD_API_KEY", "").strip()

TOP_K = int(os.getenv("TOP_K", "5"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
MAX_TOKENS = int(os.getenv("YANDEX_MAX_TOKENS", "256"))
TEMPERATURE = float(os.getenv("YANDEX_TEMPERATURE", "0.2"))

IDK_PHRASE = "I don't know based on the provided documents."


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


def parse_golden(path: Path) -> List[Tuple[str, str]]:
    """
    Each line: [K] question...  or [M] question...
    K = should answer (not IDK)
    M = should NOT answer (IDK)
    """
    items: List[Tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if len(s) >= 4 and s[0] == "[" and s[2] == "]" and s[1].upper() in ("K", "M"):
            items.append((s[1].upper(), s[3:].strip()))
        else:
            raise ValueError(f"Bad golden line: {raw!r} (expected '[K] ...' or '[M] ...')")
    return items


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
    top_k: int,
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


def is_idk(ans: str) -> bool:
    return IDK_PHRASE.lower() in (ans or "").lower()


def main() -> None:
    golden_path = Path(os.getenv("GOLDEN_FILE", str(GOLDEN_FILE_DEFAULT)))
    if not golden_path.exists():
        raise SystemExit(f"Golden file not found: {golden_path}")

    if not CHUNKS_FILE.exists():
        raise SystemExit(f"Missing: {CHUNKS_FILE}")
    if not INDEX_FILE.exists():
        raise SystemExit(f"Missing: {INDEX_FILE}")

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    if os.getenv("TRUNCATE_LOG", "1") == "1":
        LOG_FILE.write_text("", encoding="utf-8")

    items = parse_golden(golden_path)

    chunks_by_id = load_chunks(CHUNKS_FILE)
    index = faiss.read_index(str(INDEX_FILE))
    embedder = SentenceTransformer(EMBED_MODEL, device="cpu")

    passed = failed = 0
    k_total = m_total = 0
    k_pass = m_pass = 0

    print(f"Golden: {golden_path} ({len(items)} questions)")
    print(f"Log:    {LOG_FILE}")
    print(f"TopK={TOP_K} max_tokens={MAX_TOKENS}\n")

    for i, (exp, q) in enumerate(items, 1):
        t0 = time.time()
        err = None

        retrieved = retrieve(q, embedder, index, chunks_by_id, top_k=TOP_K)
        docs = format_context(retrieved)
        retrieved_ids = [c.id for c in retrieved]

        if not docs.strip():
            ans = IDK_PHRASE
        else:
            prompt = build_prompt(q, docs)
            try:
                ans = yandex_generate(prompt)
            except Exception as e:
                ans = ""
                err = f"{type(e).__name__}: {e}"

        dt_ms = int((time.time() - t0) * 1000)

        if exp == "K":
            k_total += 1
            ok = (not is_idk(ans)) and not err
            k_pass += int(ok)
        else:
            m_total += 1
            ok = is_idk(ans) and not err
            m_pass += int(ok)

        verdict = "pass" if ok else "fail"
        passed += int(ok)
        failed += int(not ok)

        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "i": i,
            "expected": exp,
            "question": q,
            "chunks_found": bool(docs.strip()),
            "retrieved_ids": retrieved_ids,
            "answer": ans,
            "answer_len": len(ans),
            "latency_ms": dt_ms,
            "error": err,
            "verdict": verdict,
        }
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"[{i}/{len(items)}] {verdict} ({dt_ms}ms): {q}")

    print("\n=== SUMMARY ===")
    print(f"passed={passed} failed={failed} total={len(items)}")
    if k_total:
        print(f"K (should answer): {k_pass}/{k_total}")
    if m_total:
        print(f"M (should NOT):    {m_pass}/{m_total}")


if __name__ == "__main__":
    main()
