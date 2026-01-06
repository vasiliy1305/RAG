from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
CHUNKS_FILE = ROOT / "knowledge_base" / "chunks.json"


def main() -> None:
    if not CHUNKS_FILE.exists():
        raise SystemExit(f"File not found: {CHUNKS_FILE}")

    word_counts = []

    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            wc = record.get("word_count")
            if isinstance(wc, int):
                word_counts.append(wc)

    if not word_counts:
        raise SystemExit("No chunks found in chunks.jsonl")

    print("Chunk statistics (words):")
    print(f"Total chunks: {len(word_counts)}")
    print(f"Min length : {min(word_counts)}")
    print(f"Max length : {max(word_counts)}")
    print(f"Avg length : {mean(word_counts):.2f}")


if __name__ == "__main__":
    main()
