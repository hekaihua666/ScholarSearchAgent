"""Oracle-only audit of dev20 gold papers missing from the frozen candidate pool.

This module is deliberately isolated under ``scripts``.  It reads gold labels and
performs exact identity probes only to diagnose source coverage; it never imports
or mutates production query planning, caches, ranking, or selection configuration.
"""

from __future__ import annotations

import asyncio
import csv
import html
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "reports/joint_optimization_v4"
DIAGNOSTICS = ROOT / "reports/overnight_iteration/per_gold_diagnostics.csv"
DATASET = Path(r"D:\AICompetition\references\github\SPAR\benchmark\AutoScholarQuery_test.jsonl")
FROZEN = ROOT / "reports/frozen_eval/autoscholar20_trace_v4_official_20260824/queries"
LOCAL_ROOTS = [
    Path(r"D:\AICompetition\references\github\pasa"),
    Path(r"D:\AICompetition\references\github\SPAR"),
]
ARXIV_NS = {"a": "http://www.w3.org/2005/Atom"}
TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)


def norm_title(value: str) -> str:
    return " ".join(TOKEN_RE.findall((value or "").lower()))


def tokens(value: str) -> set[str]:
    return set(TOKEN_RE.findall((value or "").lower()))


def lexical_similarity(a: str, b: str) -> float:
    left, right = tokens(a), tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / math.sqrt(len(left) * len(right))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_missing() -> list[dict[str, str]]:
    with DIAGNOSTICS.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row.get("loss_class") == "retrieval_miss_all_channels"]


def local_inventory_and_hits(missing: list[dict[str, str]]) -> tuple[dict[str, Any], dict[str, list[str]]]:
    wanted = {norm_title(row["gold_title"]): row["gold_title"] for row in missing}
    hits: dict[str, list[str]] = {title: [] for title in wanted.values()}
    inventory: list[dict[str, Any]] = []
    name_patterns = ("id2paper", "arxiv_data", "paper_database", "papers.json", "paper.db")
    for root in LOCAL_ROOTS:
        if not root.exists():
            inventory.append({"root": str(root), "exists": False, "candidate_stores": []})
            continue
        stores = [
            p for p in root.rglob("*")
            if p.is_file() and any(pattern in p.name.lower() for pattern in name_patterns)
        ]
        inventory.append(
            {
                "root": str(root),
                "exists": True,
                "candidate_stores": [{"path": str(p), "bytes": p.stat().st_size} for p in stores],
            }
        )
        # Only inspect plausible metadata stores.  Repository source/benchmark labels
        # are intentionally not counted as a retrieval corpus.
        for store in stores:
            if store.stat().st_size > 1_500_000_000:
                continue
            try:
                text = store.read_text(encoding="utf-8", errors="ignore").lower()
            except OSError:
                continue
            for normalized, original in wanted.items():
                if normalized and normalized in norm_title(text):
                    hits[original].append(str(store))
    return {"roots": inventory}, hits


def frozen_context(qid: str) -> dict[str, Any]:
    audit_path = FROZEN / qid / "live_audit.json"
    if not audit_path.exists():
        return {"variants": [], "papers": []}
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    variants: list[str] = []
    papers: list[dict[str, Any]] = []
    for event in audit.get("events") or []:
        if event.get("name") == "query_planned":
            variants.extend(event.get("sub_queries") or [])
            variants.extend(event.get("title_hypotheses") or [])
    for stage in audit.get("stages") or []:
        papers.extend(stage.get("papers") or [])
    # Deduplicate identities while preserving every citation-neighborhood field.
    unique: dict[str, dict[str, Any]] = {}
    for paper in papers:
        key = str(paper.get("openalex_id") or paper.get("doi") or norm_title(str(paper.get("title") or "")))
        if key:
            current = unique.setdefault(key, {})
            for field, value in paper.items():
                if value not in (None, "", [], {}):
                    current[field] = value
    return {"variants": list(dict.fromkeys(variants)), "papers": list(unique.values())}


def openalex_key() -> str:
    load_dotenv(ROOT / ".env")
    single = os.getenv("OPENALEX_API_KEY", "").strip()
    pool = [x.strip() for x in os.getenv("OPENALEX_API_KEYS", "").split(",") if x.strip()]
    return single or (pool[0] if pool else "")


async def probe_openalex(client: httpx.AsyncClient, arxiv_id: str, title: str, sem: asyncio.Semaphore) -> dict[str, Any]:
    params: dict[str, Any] = {
        "select": "id,title,display_name,publication_year,publication_date,doi,ids,referenced_works,related_works",
    }
    key = openalex_key()
    if key:
        params["api_key"] = key
    async with sem:
        for identifier in (f"arxiv:{arxiv_id}",):
            try:
                response = await client.get(f"https://api.openalex.org/works/{identifier}", params=params)
                if response.status_code == 200:
                    data = response.json()
                    return {"found": True, "method": "arxiv_id", "status": 200, "work": data}
                if response.status_code not in (404, 400):
                    return {"found": False, "method": "arxiv_id", "status": response.status_code, "error": response.text[:160]}
            except Exception as exc:  # network diagnostic, not production behavior
                return {"found": False, "method": "arxiv_id", "status": 0, "error": type(exc).__name__}
        try:
            response = await client.get(
                "https://api.openalex.org/works",
                params={**params, "search": title, "per-page": 5},
            )
            works = response.json().get("results", []) if response.status_code == 200 else []
            target = norm_title(title)
            match = next((work for work in works if norm_title(work.get("title") or work.get("display_name") or "") == target), None)
            return {"found": bool(match), "method": "exact_title", "status": response.status_code, "work": match or {}}
        except Exception as exc:
            return {"found": False, "method": "exact_title", "status": 0, "error": type(exc).__name__}


async def probe_crossref(client: httpx.AsyncClient, title: str, sem: asyncio.Semaphore) -> dict[str, Any]:
    async with sem:
        try:
            response = await client.get(
                "https://api.crossref.org/works",
                params={"query.title": title, "rows": 5, "select": "DOI,title,published,created"},
            )
            items = response.json().get("message", {}).get("items", []) if response.status_code == 200 else []
            target = norm_title(title)
            match = next(
                (item for item in items if norm_title(" ".join(item.get("title") or [])) == target),
                None,
            )
            return {"found": bool(match), "status": response.status_code, "item": match or {}}
        except Exception as exc:
            return {"found": False, "status": 0, "error": type(exc).__name__}


async def probe_semantic_scholar(client: httpx.AsyncClient, missing: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    payload = {"ids": [f"ARXIV:{row['gold_arxiv_id']}" for row in missing]}
    try:
        response = await client.post(
            "https://api.semanticscholar.org/graph/v1/paper/batch",
            params={"fields": "paperId,title,year,externalIds,citationCount,referenceCount"},
            json=payload,
        )
        if response.status_code != 200:
            return {row["gold_arxiv_id"]: {"found": False, "status": response.status_code} for row in missing}
        values = response.json()
        return {
            row["gold_arxiv_id"]: {"found": bool(item), "status": 200, "paper": item or {}}
            for row, item in zip(missing, values)
        }
    except Exception as exc:
        return {row["gold_arxiv_id"]: {"found": False, "status": 0, "error": type(exc).__name__} for row in missing}


async def probe_arxiv(client: httpx.AsyncClient, missing: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for start in range(0, len(missing), 15):
        batch = missing[start : start + 15]
        ids = [row["gold_arxiv_id"] for row in batch]
        try:
            response = await client.get(
                "https://export.arxiv.org/api/query",
                params={"id_list": ",".join(ids), "max_results": len(ids)},
            )
            if response.status_code == 200:
                root = ET.fromstring(response.text)
                for entry in root.findall("a:entry", ARXIV_NS):
                    value = (entry.findtext("a:id", default="", namespaces=ARXIV_NS).rsplit("/", 1)[-1]).split("v", 1)[0]
                    found[value] = {
                        "found": True,
                        "status": 200,
                        "title": " ".join((entry.findtext("a:title", default="", namespaces=ARXIV_NS)).split()),
                        "published": entry.findtext("a:published", default="", namespaces=ARXIV_NS),
                    }
            for identifier in ids:
                found.setdefault(identifier, {"found": False, "status": response.status_code})
        except Exception as exc:
            for identifier in ids:
                found.setdefault(identifier, {"found": False, "status": 0, "error": type(exc).__name__})
        await asyncio.sleep(1.1)
    # The Atom endpoint can return a transient 429 even when the public abstract
    # endpoint is healthy.  Probe the stable identity page as an independent
    # fallback; this still tests source coverage, not query retrieval quality.
    sem = asyncio.Semaphore(4)

    async def probe_abs(row: dict[str, str]) -> tuple[str, dict[str, Any]]:
        identifier = row["gold_arxiv_id"]
        if found.get(identifier, {}).get("found"):
            return identifier, found[identifier]
        async with sem:
            try:
                response = await client.get(f"https://arxiv.org/abs/{identifier}")
                title_match = re.search(
                    r'<meta\s+name="citation_title"\s+content="([^"]+)"',
                    response.text,
                    re.I,
                )
                date_match = re.search(
                    r'<meta\s+name="citation_date"\s+content="([^"]+)"',
                    response.text,
                    re.I,
                )
                title = html.unescape(title_match.group(1)).strip() if title_match else ""
                return identifier, {
                    "found": response.status_code == 200 and bool(title),
                    "status": response.status_code,
                    "title": title,
                    "published": date_match.group(1) if date_match else "",
                    "method": "abs_identity_page",
                }
            except Exception as exc:
                return identifier, {"found": False, "status": 0, "error": type(exc).__name__}

    for identifier, value in await asyncio.gather(*(probe_abs(row) for row in missing)):
        if value.get("found") or not found.get(identifier, {}).get("found"):
            found[identifier] = value
    return found


def publication_date(openalex: dict[str, Any], arxiv: dict[str, Any]) -> str:
    work = openalex.get("work") or {}
    if work.get("publication_date"):
        return str(work["publication_date"])
    if arxiv.get("published"):
        return str(arxiv["published"])[:10]
    return ""


def cutoff_for(row: dict[str, Any]) -> str:
    raw = str((row.get("source_meta") or {}).get("published_time") or "")
    return (datetime.strptime(raw[:8], "%Y%m%d") - timedelta(days=7)).date().isoformat()


def citation_distance(work: dict[str, Any], papers: list[dict[str, Any]]) -> tuple[str, str]:
    gold_id = str(work.get("id") or "")
    gold_refs = set(work.get("referenced_works") or [])
    gold_related = set(work.get("related_works") or [])
    candidate_ids = {str(p.get("openalex_id") or "") for p in papers if p.get("openalex_id")}
    if gold_id:
        for paper in papers:
            if gold_id in set(paper.get("referenced_works") or []) | set(paper.get("related_works") or []):
                return "1", "candidate_to_gold"
    if candidate_ids & (gold_refs | gold_related):
        return "1", "gold_to_candidate"
    for paper in papers:
        neighborhood = set(paper.get("referenced_works") or []) | set(paper.get("related_works") or [])
        bridge = neighborhood & (gold_refs | gold_related)
        if bridge:
            return "2", "shared_bridge"
    return "unreached", ""


async def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    missing = load_missing()
    dataset = {row["qid"]: row for row in read_jsonl(DATASET)}
    local_inventory, local_hits = local_inventory_and_hits(missing)
    timeout = httpx.Timeout(35.0, connect=12.0)
    headers = {"User-Agent": "YanwenZhiso-v4-oracle-audit/1.0 (mailto:jiazixu123@gmail.com)"}
    oa_sem, cr_sem = asyncio.Semaphore(4), asyncio.Semaphore(3)
    async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
        arxiv_task = asyncio.create_task(probe_arxiv(client, missing))
        s2_task = asyncio.create_task(probe_semantic_scholar(client, missing))
        oa_tasks = [probe_openalex(client, row["gold_arxiv_id"], row["gold_title"], oa_sem) for row in missing]
        cr_tasks = [probe_crossref(client, row["gold_title"], cr_sem) for row in missing]
        openalex_results = await asyncio.gather(*oa_tasks)
        crossref_results = await asyncio.gather(*cr_tasks)
        arxiv_results, s2_results = await arxiv_task, await s2_task

    output: list[dict[str, Any]] = []
    for source, oa, cr in zip(missing, openalex_results, crossref_results):
        qid, title, arxiv_id = source["query_id"], source["gold_title"], source["gold_arxiv_id"]
        query_row = dataset[qid]
        context = frozen_context(qid)
        ax = arxiv_results.get(arxiv_id, {"found": False, "status": 0})
        s2 = s2_results.get(arxiv_id, {"found": False, "status": 0})
        date = publication_date(oa, ax)
        cutoff = cutoff_for(query_row)
        legal = bool(date) and date <= cutoff
        distance, citation_direction = citation_distance(oa.get("work") or {}, context["papers"])
        variant_scores = [(variant, lexical_similarity(title, variant)) for variant in context["variants"]]
        best_variant, best_variant_score = max(variant_scores, key=lambda item: item[1], default=("", 0.0))
        query_score = lexical_similarity(title, query_row["question"])
        title_already_in_frozen = any(norm_title(p.get("title") or "") == norm_title(title) for p in context["papers"])
        any_online = bool(oa.get("found") or cr.get("found") or ax.get("found") or s2.get("found"))
        local = local_hits.get(title) or []
        if title_already_in_frozen:
            category = "D"
        elif date and not legal:
            category = "E"
        elif distance == "1":
            category = "C"
        elif local and not any_online:
            category = "B"
        elif any_online:
            category = "A"
        elif not title or not arxiv_id:
            category = "G"
        else:
            category = "F"
        output.append(
            {
                "query_id": qid,
                "gold_title": title,
                "gold_arxiv_id": arxiv_id,
                "category": category,
                "openalex_found": int(bool(oa.get("found"))),
                "openalex_status": oa.get("status", 0),
                "openalex_id": (oa.get("work") or {}).get("id", ""),
                "crossref_found": int(bool(cr.get("found"))),
                "crossref_status": cr.get("status", 0),
                "doi": ((oa.get("work") or {}).get("doi") or (cr.get("item") or {}).get("DOI") or ""),
                "semantic_scholar_found": int(bool(s2.get("found"))),
                "semantic_scholar_status": s2.get("status", 0),
                "arxiv_found": int(bool(ax.get("found"))),
                "arxiv_status": ax.get("status", 0),
                "local_pasa_or_spar_store_found": int(bool(local)),
                "local_store_paths": ";".join(local),
                "google_serper_available": 0,
                "publication_date": date,
                "cutoff": cutoff,
                "date_legal": int(legal),
                "identity_failure": int(title_already_in_frozen),
                "citation_distance": distance,
                "citation_direction": citation_direction,
                "query_title_lexical_cosine": round(query_score, 4),
                "best_variant_lexical_cosine": round(best_variant_score, 4),
                "best_existing_variant": best_variant,
                "existing_variant_plausible": int(best_variant_score >= 0.45),
                "dense_similarity": "not_available_in_audit_runtime",
                "reranker_similarity": "not_available_in_audit_runtime",
            }
        )

    csv_path = OUT / "03_retrievability_audit.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)

    categories = Counter(row["category"] for row in output)
    source_counts = {
        "OpenAlex": sum(row["openalex_found"] for row in output),
        "Semantic Scholar": sum(row["semantic_scholar_found"] for row in output),
        "Crossref": sum(row["crossref_found"] for row in output),
        "arXiv": sum(row["arxiv_found"] for row in output),
        "local PaSa/SPAR store": sum(row["local_pasa_or_spar_store_found"] for row in output),
    }
    covered = sum(any(row[field] for field in ("openalex_found", "semantic_scholar_found", "crossref_found", "arxiv_found", "local_pasa_or_spar_store_found")) for row in output)
    legal_covered = sum(
        row["date_legal"] and any(row[field] for field in ("openalex_found", "semantic_scholar_found", "crossref_found", "arxiv_found", "local_pasa_or_spar_store_found"))
        for row in output
    )
    plausible = sum(row["existing_variant_plausible"] for row in output)
    citation = sum(row["citation_distance"] == "1" for row in output)
    summary = [
        "# Candidate-pool 外 gold 可检索性审计",
        "",
        f"审计对象：{len(output)} 篇。工具仅用于离线 oracle 诊断，未写入生产缓存、查询计划或排序配置。",
        "",
        "## 来源覆盖",
        "",
        "| 来源 | 可精确取回 |",
        "|---|---:|",
        *[f"| {name} | {count}/{len(output)} |" for name, count in source_counts.items()],
        "",
        f"至少一个当前可用来源可取回：**{covered}/{len(output)}**；同时满足日期 cutoff：**{legal_covered}/{len(output)}**。",
        "",
        "## 分类",
        "",
        "| 类别 | 数量 | 含义 |",
        "|---|---:|---|",
        f"| A | {categories['A']} | source-covered-query-missed |",
        f"| B | {categories['B']} | local-corpus-only |",
        f"| C | {categories['C']} | citation-reachable |",
        f"| D | {categories['D']} | metadata-or-identity-failure |",
        f"| E | {categories['E']} | date-filtered-legitimately |",
        f"| F | {categories['F']} | source-not-covered |",
        f"| G | {categories['G']} | ambiguous-or-corrupt-record |",
        "",
        "## 决策结论",
        "",
        f"- 现有 query variants 与 missing title 有较强词项重合（cosine≥0.45）：{plausible}/{len(output)}。",
        f"- 从冻结候选的一跳引用/相关邻域可达：{citation}/{len(output)}。",
        "- 本机未发现可直接使用的 PaSa `id2paper.json` 或 SPAR `arxiv_data.db`。" if source_counts['local PaSa/SPAR store'] == 0 else "- 本机发现了可直接使用的本地论文语料。",
        f"- 精确来源覆盖给出的诊断上限为 {legal_covered}/{len(output)}；这不是生产召回率，因为精确 ID/标题探针不会进入线上系统。",
        "- 优先动作按证据排序：约束驱动 query portfolio；arXiv 专用 lexical lane；可达样本的一跳 citation expansion；最后才考虑下载并建设本地语料索引。",
        "- dense/reranker 相似度本轮审计运行时没有已安装模型，因此保持缺失值，不用词项分数冒充语义模型分数。",
        "- Semantic Scholar batch endpoint 本轮返回 429，表中的 0 表示探针不可用，不表示论文一定不在该来源；Google/Serper 未配置。",
        "",
        "## 本地语料盘点",
        "",
        "```json",
        json.dumps(local_inventory, ensure_ascii=False, indent=2),
        "```",
    ]
    (OUT / "03_retrievability_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps({"missing": len(output), "covered": covered, "legal_covered": legal_covered, "sources": source_counts, "categories": categories}, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    asyncio.run(run())
