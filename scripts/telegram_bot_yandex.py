from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters


# -------------------------
# Paths / Models
# -------------------------
ROOT = Path(__file__).resolve().parents[1]
CHUNKS_FILE = ROOT / "knowledge_base" / "chunks.jsonl"
INDEX_FILE = ROOT / "knowledge_base" / "faiss.index"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# -------------------------
# Telegram
# -------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

# -------------------------
# YandexGPT (OpenAI-compatible)
# -------------------------
YANDEX_BASE_URL = "https://llm.api.cloud.yandex.net/v1"
YANDEX_FOLDER_ID = os.getenv("YANDEX_CLOUD_FOLDER_ID", "").strip()
YANDEX_API_KEY = os.getenv("YANDEX_CLOUD_API_KEY", "").strip()
YANDEX_MAX_TOKENS = int(os.getenv("YANDEX_MAX_TOKENS", "256"))
YANDEX_TEMPERATURE = float(os.getenv("YANDEX_TEMPERATURE", "0.2"))

# -------------------------
# Retrieval / prompt settings
# -------------------------
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "9000"))

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
    # строго по твоему формату
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
        temperature=YANDEX_TEMPERATURE,
        max_tokens=YANDEX_MAX_TOKENS,
    )

    return (resp.choices[0].message.content or "").strip()


class RagBot:
    def __init__(self) -> None:
        if not CHUNKS_FILE.exists():
            raise SystemExit(f"Missing: {CHUNKS_FILE}")
        if not INDEX_FILE.exists():
            raise SystemExit(f"Missing: {INDEX_FILE}")

        self.chunks_by_id = load_chunks(CHUNKS_FILE)
        self.index = faiss.read_index(str(INDEX_FILE))
        self.embedder = SentenceTransformer(EMBED_MODEL, device="cpu")

    def answer(self, question: str) -> str:
        retrieved = retrieve(question, self.embedder, self.index, self.chunks_by_id, top_k=TOP_K)
        docs = format_context(retrieved)

        if not docs.strip():
            return "I don't know based on the provided documents."

        prompt = build_prompt(question, docs)
        return yandex_generate(prompt)


ragbot: RagBot | None = None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = (
        "Hi! Send me a question and I will answer using the private knowledge base (RAG).\n\n"
        "Provider: YandexGPT\n"
        f"TopK: {TOP_K}\n"
        "Commands: /start /health"
    )
    await update.message.reply_text(msg)


async def health(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ok = True
    lines = []

    # KB
    try:
        lines.append(f"chunks.jsonl: {'OK' if CHUNKS_FILE.exists() else 'MISSING'}")
        lines.append(f"faiss.index: {'OK' if INDEX_FILE.exists() else 'MISSING'}")
    except Exception as e:
        ok = False
        lines.append(f"KB error: {e}")

    # Yandex env
    lines.append(f"YANDEX_CLOUD_FOLDER_ID set: {bool(YANDEX_FOLDER_ID)}")
    lines.append(f"YANDEX_CLOUD_API_KEY set: {bool(YANDEX_API_KEY)}")
    lines.append(f"MAX_TOKENS: {YANDEX_MAX_TOKENS}")

    await update.message.reply_text(("OK\n" if ok else "NOT OK\n") + "\n".join(lines))


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global ragbot
    q = (update.message.text or "").strip()
    if not q:
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    loop = asyncio.get_running_loop()
    try:
        ans = await loop.run_in_executor(None, ragbot.answer, q)
    except Exception as e:
        ans = f"Error: {type(e).__name__}: {e}"

    # Telegram message length safety
    if len(ans) > 3500:
        ans = ans[:3500] + "\n...\n(truncated)"

    await update.message.reply_text(ans, parse_mode=ParseMode.HTML, disable_web_page_preview=True)


def main() -> None:
    global ragbot

    if not TELEGRAM_BOT_TOKEN:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN env var")

    ragbot = RagBot()

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("health", health))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    print("Telegram bot started (YandexGPT).")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
