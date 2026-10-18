"""Cached cross-encoder selector with a transparent CPU fallback."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
CPU_FALLBACK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _lexical_fallback(query: str, document: str) -> float:
    pattern = re.compile(r"[a-z][a-z0-9+_.-]*", re.I)
    q = {x.lower() for x in pattern.findall(query)}
    d = {x.lower() for x in pattern.findall(document)}
    return len(q & d) / math.sqrt(len(q) * len(d)) if q and d else 0.0


class CrossEncoderSelector:
    def __init__(self, model_id: str = DEFAULT_MODEL, cache_path: Path | None = None, max_length: int = 512,
                 local_files_only: bool = False):
        self.model_id = model_id
        self.max_length = max_length
        self.local_files_only = local_files_only
        self.cache_path = cache_path
        self.cache: dict[str, float] = {}
        self.model: Any = None
        self.runtime = "uninitialized"
        if cache_path and cache_path.exists():
            self.cache = json.loads(cache_path.read_text(encoding="utf-8"))

    def _key(self, query: str, document: str) -> str:
        raw = json.dumps([self.model_id, self.max_length, query, document], ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _load(self) -> None:
        if self.runtime != "uninitialized":
            return
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            self.model = CrossEncoder(
                self.model_id, max_length=self.max_length, device="cpu",
                local_files_only=self.local_files_only,
            )
            self.runtime = "cross_encoder_cpu"
        except Exception as exc:
            self.runtime = f"lexical_fallback:{type(exc).__name__}"

    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 16) -> tuple[list[float], dict[str, Any]]:
        started = time.perf_counter()
        self._load()
        scores: list[float | None] = []
        misses: list[tuple[int, tuple[str, str]]] = []
        for index, pair in enumerate(pairs):
            key = self._key(*pair)
            if key in self.cache:
                scores.append(self.cache[key])
            else:
                scores.append(None)
                misses.append((index, pair))
        if misses and self.model is not None:
            raw = self.model.predict([pair for _, pair in misses], batch_size=batch_size, show_progress_bar=False)
            values = [float(value) for value in raw]
        else:
            values = [_lexical_fallback(*pair) for _, pair in misses]
        for (index, pair), value in zip(misses, values):
            scores[index] = value
            self.cache[self._key(*pair)] = value
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, indent=2), encoding="utf-8")
        return [float(value or 0) for value in scores], {
            "model": self.model_id,
            "runtime": self.runtime,
            "pairs": len(pairs),
            "cache_hits": len(pairs) - len(misses),
            "wall_seconds": round(time.perf_counter() - started, 4),
        }
