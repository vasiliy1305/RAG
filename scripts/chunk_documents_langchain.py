from __future__ import annotations

import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "knowledge_base" / "final"
OUTPUT_FILE = ROOT / "knowledge_base" / "chunks.json"

# Target: ~100–300 words ≈ 600–1800 chars (very rough).
# We'll chunk by characters (common approach), then you can report avg word count.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150


def main() -> None:
    files = sorted(INPUT_DIR.glob("*.md"))
    if not files:
        raise SystemExit(f"No files found in: {INPUT_DIR}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunk_id = 0
    total = 0

    with OUTPUT_FILE.open("w", encoding="utf-8") as out:
        for fp in files:
            text = fp.read_text(encoding="utf-8", errors="ignore")

            # LangChain splitter returns list[str]; we also need positions.
            # We'll compute positions by scanning the original text in order.
            chunks = splitter.split_text(text)

            cursor = 0
            for i, ch in enumerate(chunks):
                # Find chunk position starting from current cursor
                start = text.find(ch, cursor)
                if start == -1:
                    # Fallback: if overlap/separators change, try from beginning (rare)
                    start = text.find(ch)
                if start == -1:
                    # If still not found, skip but keep pipeline running
                    continue
                end = start + len(ch)
                cursor = max(cursor, start)

                record = {
                    "id": chunk_id,
                    "source": fp.name,
                    "chunk_index": i,
                    "start_char": start,
                    "end_char": end,
                    "text": ch,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

                chunk_id += 1
                total += 1

    print("Done.")
    print(f"Documents processed: {len(files)}")
    print(f"Total chunks: {total}")
    print(f"Saved to: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
