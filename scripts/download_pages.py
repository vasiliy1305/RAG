from __future__ import annotations

import time
import hashlib
from pathlib import Path

import requests


# === Project paths (independent from current working directory) ===
ROOT = Path(__file__).resolve().parents[1]     # .../RAG
URLS_FILE = ROOT / "sources" / "pages.txt"
RAW_DIR = ROOT / "knowledge_base" / "raw_html"

# === Download settings ===
SLEEP_SEC = 1.0
TIMEOUT_SEC = 30
RETRIES = 3


def stable_id(url: str) -> str:
    """
    Generate a stable short identifier for a URL.
    Used as a filename to ensure reproducibility.
    """
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]


def fetch(session: requests.Session, url: str) -> str:
    """
    Download HTML content with retries.
    """
    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            response = session.get(url, timeout=TIMEOUT_SEC)
            response.raise_for_status()
            response.encoding = response.apparent_encoding or "utf-8"
            return response.text
        except Exception as e:
            last_err = e
            time.sleep(0.7 * attempt)
    raise RuntimeError(f"Failed to download {url}: {last_err}") from last_err


def main() -> None:
    """
    Read URLs from sources/pages.txt and download raw HTML pages
    into knowledge_base/raw_html.
    """
    if not URLS_FILE.exists():
        raise SystemExit(f"File not found: {URLS_FILE}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    urls = [
        line.strip()
        for line in URLS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    if not urls:
        raise SystemExit("No URLs found in pages.txt")

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "RAG-KB-Downloader/1.0",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )

    ok, failed = 0, 0

    for i, url in enumerate(urls, 1):
        try:
            html = fetch(session, url)
            doc_id = stable_id(url)

            out_path = RAW_DIR / f"{doc_id}.html"
            out_path.write_text(html, encoding="utf-8")

            print(f"[{i}/{len(urls)}] OK   -> {out_path}")
            ok += 1
        except Exception as e:
            print(f"[{i}/{len(urls)}] FAIL {url}\n    {e}")
            failed += 1
        finally:
            time.sleep(SLEEP_SEC)

    print(f"\nDone. Success: {ok}, Failed: {failed}")
    print(f"HTML files saved to: {RAW_DIR.resolve()}")


if __name__ == "__main__":
    main()
