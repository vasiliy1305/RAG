from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
CHUNKS_FILE = ROOT / "knowledge_base" / "chunks.jsonl"
INDEX_FILE = ROOT / "knowledge_base" / "faiss.index"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks_by_id(path: Path) -> dict[int, dict]:
    by_id: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            cid = int(rec["id"])
            by_id[cid] = rec
    return by_id


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python scripts/rag_retrieve.py "query" [k]')

    query = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) >= 3 else 5

    if not CHUNKS_FILE.exists():
        raise SystemExit(f"Missing chunks: {CHUNKS_FILE}")
    if not INDEX_FILE.exists():
        raise SystemExit(f"Missing index: {INDEX_FILE}")

    chunks = load_chunks_by_id(CHUNKS_FILE)
    index = faiss.read_index(str(INDEX_FILE))

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    scores, ids = index.search(q_emb, k)

    print(f"Query: {query}")
    print(f"Top-{k} chunks:\n")

    for rank, (cid, score) in enumerate(zip(ids[0], scores[0]), 1):
        cid = int(cid)
        rec = chunks.get(cid, {})
        text = rec.get("text", "")
        preview = text[:350].replace("\n", " ") + ("..." if len(text) > 350 else "")

        source = rec.get("source") or rec.get("source_file") or ""
        start = rec.get("start_char")
        end = rec.get("end_char")

        pos = ""
        if start is not None and end is not None:
            pos = f" pos={start}:{end}"

        src = f" source={source}" if source else ""
        print(f"{rank}. id={cid} score={float(score):.4f}{src}{pos}")
        print(f"   {preview}\n")


if __name__ == "__main__":
    main()
