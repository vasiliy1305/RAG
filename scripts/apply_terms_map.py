from __future__ import annotations

import json
import re
from pathlib import Path


# === Project paths ===
ROOT = Path(__file__).resolve().parents[1]

CLEAN_DIR = ROOT / "knowledge_base" / "clean_text"
FINAL_DIR = ROOT / "knowledge_base" / "final"
TERMS_FILE = ROOT / "knowledge_base" / "terms_map.json"


def load_terms_map(path: Path) -> list[tuple[str, str]]:
    """
    Load terms_map.json (keys are expected to be lower-case).
    Return list of (source, target) sorted by source length DESC
    to avoid partial collisions.
    """
    data = json.loads(path.read_text(encoding="utf-8"))

    pairs = sorted(
        ((k.lower(), v) for k, v in data.items()),
        key=lambda x: len(x[0]),
        reverse=True,
    )
    return pairs


def replace_terms_case_insensitive(text: str, terms: list[tuple[str, str]]) -> str:
    """
    Replace terms in text in a case-insensitive and word-boundary-safe way.
    The replacement value is taken exactly as in terms_map.json.
    """
    for src, dst in terms:
        pattern = re.compile(rf"\b{re.escape(src)}\b", flags=re.IGNORECASE)
        text = pattern.sub(dst, text)
    return text


def main() -> None:
    if not TERMS_FILE.exists():
        raise SystemExit(f"terms_map.json not found: {TERMS_FILE}")

    if not CLEAN_DIR.exists():
        raise SystemExit(f"clean_text directory not found: {CLEAN_DIR}")

    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    terms = load_terms_map(TERMS_FILE)

    files = list(CLEAN_DIR.glob("*.md"))
    if not files:
        raise SystemExit("No .md files found in clean_text/")

    processed = 0

    for src_path in files:
        text = src_path.read_text(encoding="utf-8", errors="ignore")
        new_text = replace_terms_case_insensitive(text, terms)

        out_path = FINAL_DIR / src_path.name
        out_path.write_text(new_text, encoding="utf-8")

        processed += 1

    print("Done.")
    print(f"Processed files: {processed}")
    print(f"Output directory: {FINAL_DIR.resolve()}")


if __name__ == "__main__":
    main()
