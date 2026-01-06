from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
CHUNKS_FILE = ROOT / "knowledge_base" / "chunks.json"
INDEX_FILE = ROOT / "knowledge_base" / "faiss.index"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks(path: Path) -> dict[int, dict]:
    by_id: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            by_id[int(rec["id"])] = rec
    return by_id


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python scripts/search_faiss.py "your query" [k]')

    query = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) >= 3 else 5

    if not INDEX_FILE.exists():
        raise SystemExit(f"Missing index: {INDEX_FILE}")
    if not CHUNKS_FILE.exists():
        raise SystemExit(f"Missing chunks: {CHUNKS_FILE}")

    chunks = load_chunks(CHUNKS_FILE)

    index = faiss.read_index(str(INDEX_FILE))
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    scores, ids = index.search(q, k)

    print(f"Query: {query}")
    print(f"Top-{k}:")
    for rank, (cid, score) in enumerate(zip(ids[0], scores[0]), 1):
        rec = chunks.get(int(cid))
        preview = rec["text"][:320].replace("\n", " ") + "..." if rec else "<missing>"
        print(f"{rank}. id={int(cid)}  score={float(score):.4f}")
        print(f"   {preview}")


if __name__ == "__main__":
    main()
