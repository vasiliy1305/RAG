from __future__ import annotations

import re
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Change to clean_text if needed
TEXT_DIR = ROOT / "knowledge_base" / "clean_text"
OUT_DIR = ROOT / "knowledge_base" 


WORD_RE = re.compile(r"[a-zA-Z]+")  # only words, no digits


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def main() -> None:
    if not TEXT_DIR.exists():
        raise SystemExit(f"Text directory not found: {TEXT_DIR}")

    files = list(TEXT_DIR.glob("*.md"))
    if not files:
        raise SystemExit(f"No text files found in: {TEXT_DIR}")

    counter = Counter()

    for p in files:
        text = p.read_text(encoding="utf-8", errors="ignore")
        words = tokenize(text)
        counter.update(words)

    vocab = dict(counter.most_common())

    # Save JSON with frequencies
    json_path = OUT_DIR / "vocabulary.json"
    json_path.write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Save plain word list (frequency-sorted)
    txt_path = OUT_DIR / "vocabulary.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        for word, count in counter.most_common():
            f.write(f"{word}\t{count}\n")

    print(f"Done.")
    print(f"Documents processed: {len(files)}")
    print(f"Unique words: {len(counter)}")
    print(f"Saved:")
    print(f" - {json_path}")
    print(f" - {txt_path}")


if __name__ == "__main__":
    main()
