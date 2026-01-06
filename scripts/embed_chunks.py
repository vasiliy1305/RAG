from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
CHUNKS_FILE = ROOT / "knowledge_base" / "chunks.jsonl"
OUT_EMB = ROOT / "knowledge_base" / "embeddings.npy"
OUT_META = ROOT / "knowledge_base" / "embeddings_meta.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64  # if RAM is tight, use 16 or 32
DEVICE = "cpu"  # force CPU to avoid CUDA warnings


def load_chunks(path: Path) -> list[dict]:
    """Load chunks.jsonl and return records sorted by integer id."""
    chunks: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "id" not in rec or "text" not in rec:
                continue
            rec["id"] = int(rec["id"])
            chunks.append(rec)
    chunks.sort(key=lambda x: x["id"])
    return chunks


def main() -> None:
    if not CHUNKS_FILE.exists():
        raise SystemExit(f"File not found: {CHUNKS_FILE}")

    chunks = load_chunks(CHUNKS_FILE)
    if not chunks:
        raise SystemExit("No chunks loaded from chunks.jsonl")

    texts = [c["text"] for c in chunks]

    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    np.save(OUT_EMB, embeddings)

    meta = {
        "model": MODEL_NAME,
        "device": DEVICE,
        "num_chunks": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "normalized": True,
        "batch_size": BATCH_SIZE,
        "chunks_file": str(CHUNKS_FILE.name),
        "embeddings_file": str(OUT_EMB.name),
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done.")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Saved: {OUT_EMB.resolve()}")
    print(f"Saved: {OUT_META.resolve()}")


if __name__ == "__main__":
    main()
