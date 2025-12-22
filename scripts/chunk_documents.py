from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "knowledge_base" / "final"
OUTPUT_FILE = ROOT / "knowledge_base" / "chunks.jsonl"

MIN_WORDS = 150
MAX_WORDS = 250


def count_words(text: str) -> int:
    return len(text.split())


def split_into_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def chunk_paragraphs(paragraphs: list[str]) -> list[str]:
    chunks = []
    current = []
    current_words = 0

    for p in paragraphs:
        w = count_words(p)

        if current_words + w > MAX_WORDS and current_words >= MIN_WORDS:
            chunks.append("\n\n".join(current))
            current = [p]
            current_words = w
        else:
            current.append(p)
            current_words += w

    if current_words >= MIN_WORDS:
        chunks.append("\n\n".join(current))

    return chunks


def main() -> None:
    files = list(INPUT_DIR.glob("*.md"))
    if not files:
        raise SystemExit("No files found in knowledge_base/final")

    total_chunks = 0

    with OUTPUT_FILE.open("w", encoding="utf-8") as out:
        for file_path in files:
            text = file_path.read_text(encoding="utf-8", errors="ignore")

            paragraphs = split_into_paragraphs(text)
            chunks = chunk_paragraphs(paragraphs)

            for i, chunk in enumerate(chunks):
                record = {
                    "chunk_id": f"{file_path.stem}_{i:04d}",
                    "source_file": file_path.name,
                    "chunk_index": i,
                    "word_count": count_words(chunk),
                    "text": chunk,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

    print("Done.")
    print(f"Documents processed: {len(files)}")
    print(f"Total chunks: {total_chunks}")
    print(f"Saved to: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
