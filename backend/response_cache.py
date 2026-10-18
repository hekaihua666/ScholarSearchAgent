"""Disk-persistent full-response cache for /api/search.

Caches the complete structured response keyed by the normalized query, so that
repeated queries — during live demos, video recording, or re-judging — return
instantly with **zero API and zero LLM cost**.

Only real computed results are stored; this never fabricates data. It simply
remembers a genuine prior result so the same question does not have to hit the
external APIs again.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
_CACHE_DIR.mkdir(exist_ok=True)

# 7 days — papers don't need to be fresh within a competition demo window.
_TTL = 7 * 24 * 3600

# Bump when the response schema changes so old cache files are ignored automatically.
_SCHEMA_VERSION = 16


def _normalize(query: str) -> str:
    return " ".join(query.strip().lower().split())


def _path(query: str) -> Path:
    key = hashlib.sha256(_normalize(query).encode("utf-8")).hexdigest()[:16]
    return _CACHE_DIR / f"{key}.json"


def load(query: str) -> dict[str, Any] | None:
    """Return the cached payload for query, or None if missing/expired/corrupt."""
    path = _path(query)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if time.time() - raw.get("_cached_at", 0) > _TTL:
        return None
    if raw.get("_schema") != _SCHEMA_VERSION:
        return None
    return raw.get("payload")


def save(query: str, payload: dict[str, Any]) -> None:
    """Persist a real computed response. Failures are swallowed (cache is best-effort)."""
    try:
        _path(query).write_text(
            json.dumps(
                {"_cached_at": time.time(), "_schema": _SCHEMA_VERSION,
                 "query": query, "payload": payload},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass


def invalidate(query: str) -> None:
    """Remove a single cached entry (e.g. when it fails to deserialize)."""
    try:
        _path(query).unlink(missing_ok=True)
    except Exception:
        pass


def clear() -> int:
    """Delete all cached responses. Returns the number of files removed."""
    removed = 0
    for f in _CACHE_DIR.glob("*.json"):
        try:
            f.unlink()
            removed += 1
        except Exception:
            pass
    return removed
