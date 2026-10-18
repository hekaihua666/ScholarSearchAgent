"""Official-protocol evaluation context and deterministic external-call replay.

Normal API requests are unaffected. Benchmark runners opt in with
``evaluation_session`` and receive a per-query cutoff plus LIVE/FROZEN call
recording. Snapshots contain real API/LLM outputs and can be replayed without
network access.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterator, Literal, TypeVar


T = TypeVar("T")
Mode = Literal["live", "replay"]


def _json_copy(value: T) -> T:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _key(label: str, arguments: dict[str, Any]) -> str:
    raw = json.dumps({"label": label, "arguments": arguments}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


@dataclass
class EvaluationSession:
    benchmark: str
    query_id: str
    cutoff: datetime
    mode: Mode
    snapshot_path: Path
    calls: dict[str, dict[str, Any]] = field(default_factory=dict)
    date_filter_stats: dict[str, dict[str, int]] = field(default_factory=dict)
    approximate_date_examples: list[dict[str, Any]] = field(default_factory=list)
    missing_date_examples: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(
        cls, benchmark: str, query_id: str, cutoff: datetime, mode: Mode, snapshot_path: Path
    ) -> "EvaluationSession":
        session = cls(benchmark, query_id, cutoff, mode, snapshot_path)
        if mode == "replay":
            raw = json.loads(snapshot_path.read_text(encoding="utf-8"))
            expected = raw.get("cutoff")
            if expected != cutoff.isoformat():
                raise ValueError(f"snapshot cutoff mismatch: {expected} != {cutoff.isoformat()}")
            session.calls = raw.get("calls") or {}
        return session

    async def call(
        self, label: str, arguments: dict[str, Any], operation: Callable[[], Awaitable[T]]
    ) -> T:
        key = _key(label, arguments)
        if self.mode == "replay":
            entry = self.calls.get(key)
            if not entry:
                raise KeyError(f"frozen call missing: {label} {arguments}")
            if entry.get("error"):
                raise RuntimeError(f"replayed external error: {entry['error']}")
            return _json_copy(entry.get("output"))
        started = time.perf_counter()
        try:
            output = await operation()
        except Exception as exc:
            self.calls[key] = {
                "label": label, "arguments": arguments,
                "elapsed_seconds": round(time.perf_counter() - started, 6),
                "error": f"{type(exc).__name__}: {exc}",
            }
            raise
        self.calls[key] = {
            "label": label, "arguments": arguments,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "output": _json_copy(output), "error": None,
        }
        return _json_copy(output)

    def filter_papers(self, papers: list[dict[str, Any]], channel: str) -> list[dict[str, Any]]:
        stats = self.date_filter_stats.setdefault(
            channel, {"input": 0, "kept_exact": 0, "kept_year_approx": 0,
                      "filtered_future": 0, "filtered_missing_date": 0}
        )
        kept: list[dict[str, Any]] = []
        for paper in papers:
            stats["input"] += 1
            publication_date = parse_publication_date(paper.get("publication_date"))
            if publication_date is not None:
                if publication_date <= self.cutoff.date():
                    stats["kept_exact"] += 1
                    kept.append(paper)
                else:
                    stats["filtered_future"] += 1
                continue
            year = paper.get("year")
            try:
                approximate = date(int(year), 1, 1)
            except (TypeError, ValueError):
                stats["filtered_missing_date"] += 1
                if len(self.missing_date_examples) < 30:
                    self.missing_date_examples.append({"channel": channel, "title": paper.get("title")})
                continue
            # Explicit policy matching Semantic Scholar's documented treatment:
            # unknown day/month is interpreted as January 1 of its known year.
            if approximate <= self.cutoff.date():
                stats["kept_year_approx"] += 1
                paper = dict(paper)
                paper["evaluation_date_approximation"] = f"{int(year):04d}-01-01"
                kept.append(paper)
                if len(self.approximate_date_examples) < 30:
                    self.approximate_date_examples.append({
                        "channel": channel, "title": paper.get("title"), "year": year,
                    })
            else:
                stats["filtered_future"] += 1
        return kept

    def save(self, metadata: dict[str, Any] | None = None) -> None:
        if self.mode != "live":
            return
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "benchmark": self.benchmark,
            "query_id": self.query_id,
            "cutoff": self.cutoff.isoformat(),
            "calls": self.calls,
            "date_filter_stats": self.date_filter_stats,
            "approximate_date_examples": self.approximate_date_examples,
            "missing_date_examples": self.missing_date_examples,
            "metadata": metadata or {},
        }
        self.snapshot_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


_SESSION: ContextVar[EvaluationSession | None] = ContextVar("evaluation_session", default=None)


@contextmanager
def evaluation_session(session: EvaluationSession) -> Iterator[EvaluationSession]:
    token = _SESSION.set(session)
    try:
        yield session
    finally:
        _SESSION.reset(token)


def current_session() -> EvaluationSession | None:
    return _SESSION.get()


def current_cutoff() -> datetime | None:
    session = current_session()
    return session.cutoff if session else None


def current_cutoff_iso() -> str | None:
    cutoff = current_cutoff()
    return cutoff.date().isoformat() if cutoff else None


async def frozen_call(
    label: str, arguments: dict[str, Any], operation: Callable[[], Awaitable[T]]
) -> T:
    session = current_session()
    if session is None:
        return await operation()
    return await session.call(label, arguments, operation)


def filter_for_cutoff(papers: list[dict[str, Any]], channel: str) -> list[dict[str, Any]]:
    session = current_session()
    return session.filter_papers(papers, channel) if session else papers


def parse_publication_date(value: Any) -> date | None:
    text = str(value or "").strip()[:10]
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None

