from __future__ import annotations

from pathlib import Path
import numpy as np
import faiss


ROOT = Path(__file__).resolve().parents[1]
EMB_FILE = ROOT / "knowledge_base" / "embeddings.npy"
INDEX_FILE = ROOT / "knowledge_base" / "faiss.index"


def main() -> None:
    if not EMB_FILE.exists():
        raise SystemExit(f"Missing embeddings: {EMB_FILE}")

    emb = np.load(EMB_FILE).astype("float32")
    n, d = emb.shape

    # embeddings normalized -> inner product == cosine similarity
    index = faiss.IndexFlatIP(d)
    index.add(emb)

    faiss.write_index(index, str(INDEX_FILE))

    print("Done.")
    print(f"Embeddings: {n} x {d}")
    print(f"Index size: {index.ntotal}")
    print(f"Saved: {INDEX_FILE.resolve()}")


if __name__ == "__main__":
    main()
