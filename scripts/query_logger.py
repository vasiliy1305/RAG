# scripts/query_logger.py
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

_lock = threading.Lock()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_query_jsonl(
    log_path: str | Path,
    *,
    question: str,
    chunks_found: bool,
    answer: str,
    success: bool,
    sources_used: Iterable[Any] = (),
    sources_retrieved: Iterable[Any] = (),
    filtered: Iterable[Any] = (),
    extra: dict[str, Any] | None = None,
    timestamp: str | None = None,
) -> None:
    """
    Append one JSONL record for a single user query.

    Required fields (per assignment):
      - question
      - timestamp
      - chunks_found
      - answer_length
      - success
      - sources (here: sources_used + sources_retrieved)

    Notes:
      - JSONL: one JSON object per line.
      - Uses a process-local lock to avoid interleaving writes from threads.
      - Creates parent directories automatically.
    """
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    ts = timestamp or _utc_now_iso()
    ans = answer or ""

    rec: dict[str, Any] = {
        "timestamp": ts,
        "question": question,
        "chunks_found": bool(chunks_found),
        "answer_length": len(ans),
        "success": bool(success),
        "sources_used": list(sources_used),
        "sources_retrieved": list(sources_retrieved),
        "filtered": list(filtered),
    }

    if extra:
        rec["extra"] = extra

    line = json.dumps(rec, ensure_ascii=False)

    # Thread-safe append (for multi-threaded bot). For multi-process safety on Linux,
    # appending a single line is usually atomic, but we still keep it simple here.
    with _lock:
        with p.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                # Some filesystems don't support fsync well; safe to ignore.
                pass
