"""Minimal SQLite FTS5 local-paper channel, disabled until a corpus is configured."""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from backend.paper_identity import paper_identity


def stable_key(paper: dict[str, Any]) -> str:
    identity = paper_identity(paper)
    if identity.arxiv_ids:
        return "arxiv:" + sorted(identity.arxiv_ids)[0]
    if identity.dois:
        return "doi:" + sorted(identity.dois)[0]
    if paper.get("openalex_id"):
        return "openalex:" + str(paper["openalex_id"]).lower()
    return "title:" + identity.title


class LocalHybridIndex:
    def __init__(self, path: Path):
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS papers (paper_key TEXT PRIMARY KEY, title TEXT, abstract TEXT, "
            "year INTEGER, publication_date TEXT, metadata TEXT)"
        )
        self.connection.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(paper_key UNINDEXED, title, abstract)"
        )

    def add(self, papers: Iterable[dict[str, Any]]) -> int:
        inserted = 0
        for paper in papers:
            key = stable_key(paper)
            if not key or key == "title:":
                continue
            cursor = self.connection.execute(
                "INSERT OR IGNORE INTO papers(paper_key,title,abstract,year,publication_date,metadata) VALUES(?,?,?,?,?,?)",
                (key, paper.get("title") or "", paper.get("abstract") or "", paper.get("year"),
                 paper.get("publication_date") or "", json.dumps(paper, ensure_ascii=False)),
            )
            if cursor.rowcount:
                self.connection.execute(
                    "INSERT INTO papers_fts(paper_key,title,abstract) VALUES(?,?,?)",
                    (key, paper.get("title") or "", paper.get("abstract") or ""),
                )
                inserted += 1
        self.connection.commit()
        return inserted

    def search(self, query: str, cutoff: date, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT p.metadata, bm25(papers_fts, 0.0, 2.0, 1.0) AS rank_score "
            "FROM papers_fts JOIN papers p USING(paper_key) "
            "WHERE papers_fts MATCH ? AND "
            "((p.publication_date != '' AND p.publication_date <= ?) OR "
            "(p.publication_date = '' AND p.year IS NOT NULL AND p.year <= ?)) "
            "ORDER BY rank_score LIMIT ?",
            (query, cutoff.isoformat(), cutoff.year, limit),
        ).fetchall()
        output = []
        for rank, row in enumerate(rows, 1):
            paper = json.loads(row["metadata"])
            paper.update({"source_api": "Local Hybrid", "source_query": query, "_source_rank": rank,
                          "local_bm25": -float(row["rank_score"])})
            output.append(paper)
        return output

    def close(self) -> None:
        self.connection.close()
