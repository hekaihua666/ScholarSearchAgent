"""Build query-grouped selector train/calibration data without validation leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.selection.relevance_selector import tokenize
from scripts.audit_evaluation import load_jsonl, source_cutoff


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def intent(query: str) -> str:
    q = query.lower()
    if any(x in q for x in ("survey", "review", "overview")): return "survey"
    if any(x in q for x in ("latest", "recent", "newest", "state of the art")): return "frontier_latest"
    if any(x in q for x in ("compare", "comparison", "versus", " vs ")): return "method_comparison"
    if any(x in q for x in ("foundation", "seminal", "first paper", "introduced")): return "foundation"
    if len(tokenize(query)) >= 12: return "multi_constraint"
    return "broad"


def stratified_take(rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        n = len(row.get("answer") or [])
        g = "g1" if n <= 1 else "g2" if n == 2 else "g3plus"
        length = "short" if len(tokenize(row.get("question"))) < 12 else "long"
        buckets[(g, length)].append(row)
    for values in buckets.values(): rng.shuffle(values)
    selected: list[dict[str, Any]] = []
    keys = sorted(buckets)
    while len(selected) < count:
        progressed = False
        for key in keys:
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].pop()); progressed = True
        if not progressed: raise RuntimeError("not enough eligible rows")
    rng.shuffle(selected)
    return selected


def load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def fetch_arxiv(ids: list[str], cache_path: Path) -> dict[str, dict[str, Any]]:
    cache = load_metadata(cache_path)
    missing = sorted({x.split("v")[0] for x in ids if x and x.split("v")[0] not in cache})
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for start in range(0, len(missing), 100):
        batch = missing[start:start + 100]
        for attempt in range(4):
            try:
                response = httpx.get(
                    "https://export.arxiv.org/api/query",
                    params={"id_list": ",".join(batch), "max_results": len(batch)},
                    headers={"User-Agent": "ScholarSearchAgent/2.0 (competition evaluation)"}, timeout=60,
                )
                response.raise_for_status(); root = ET.fromstring(response.text)
                for entry in root.findall("a:entry", ns):
                    raw_id = (entry.findtext("a:id", default="", namespaces=ns).rsplit("/", 1)[-1]).split("v")[0]
                    cache[raw_id] = {
                        "title": " ".join(entry.findtext("a:title", default="", namespaces=ns).split()),
                        "abstract": " ".join(entry.findtext("a:summary", default="", namespaces=ns).split()),
                        "published": entry.findtext("a:published", default="", namespaces=ns),
                    }
                break
            except Exception:
                if attempt == 3: break
                time.sleep(4 * (attempt + 1))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        if start + 100 < len(missing): time.sleep(3.2)
    return cache


def documents(rows: list[dict[str, Any]], metadata: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        for i, title in enumerate(row.get("answer") or []):
            arxiv = ((row.get("answer_arxiv_id") or []) + [""] * (i + 1))[i].split("v")[0]
            key = arxiv or re.sub(r"\W+", "", title.lower())
            meta = metadata.get(arxiv) or {}
            result[key] = {"paper_key": key, "arxiv_id": arxiv, "title": title,
                           "abstract": meta.get("abstract") or "", "year": (meta.get("published") or "")[:4]}
    return list(result.values())


def similarity(query_tokens: set[str], paper: dict[str, Any]) -> float:
    title = tokenize(paper["title"]); body = tokenize(paper.get("abstract"))
    title_cov = len(query_tokens & title) / len(query_tokens) if query_tokens else 0
    body_cov = len(query_tokens & body) / len(query_tokens) if query_tokens else 0
    return 0.65 * title_cov + 0.35 * body_cov


def build_pairs(rows: list[dict[str, Any]], pool: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    by_key = {p["paper_key"]: p for p in pool}
    for row in rows:
        q = row["question"]; qtokens = tokenize(q)
        positive_keys = set()
        for i, title in enumerate(row.get("answer") or []):
            ids = row.get("answer_arxiv_id") or []
            arxiv = (ids[i] if i < len(ids) else "").split("v")[0]
            key = arxiv or re.sub(r"\W+", "", title.lower()); positive_keys.add(key)
            paper = by_key.get(key, {"paper_key": key, "arxiv_id": arxiv, "title": title, "abstract": "", "year": ""})
            pairs.append({"query_id": row["qid"], "query": q, "intent": intent(q), **paper,
                          "label": 1, "negative_type": "positive", "sample_weight": 1.0})
        candidates = [p for p in pool if p["paper_key"] not in positive_keys]
        hard = sorted(candidates, key=lambda p: (similarity(qtokens, p), p["paper_key"]), reverse=True)[:4]
        remaining = [p for p in candidates if p not in hard]
        random_neg = rng.sample(remaining, min(3, len(remaining)))
        for paper in hard:
            pairs.append({"query_id": row["qid"], "query": q, "intent": intent(q), **paper,
                          "label": 0, "negative_type": "lexical_semantic_hard_weak", "sample_weight": 0.45})
        for paper in random_neg:
            pairs.append({"query_id": row["qid"], "query": q, "intent": intent(q), **paper,
                          "label": 0, "negative_type": "random_weak", "sample_weight": 0.2})
    return pairs


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows: stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path(r"D:\AICompetition\references\github\SPAR\benchmark\AutoScholarQuery_test.jsonl"))
    ap.add_argument("--train-count", type=int, default=500)
    ap.add_argument("--calibration-count", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--fetch-abstracts", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "reports/f1_selector_v3")
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.dataset); by_id = {row["qid"]: row for row in rows}
    query_sets = json.loads((ROOT / "configs/query_sets.json").read_text(encoding="utf-8"))
    dev_ids = set(query_sets["dev20"]); validation_ids = set(query_sets["validation30"])
    eligible = [row for row in rows if row["qid"] not in dev_ids | validation_ids and source_cutoff(row) is not None]
    rng = random.Random(args.seed)
    selected = stratified_take(eligible, args.train_count + args.calibration_count, rng)
    train, calibration = selected[:args.train_count], selected[args.train_count:]
    ids = [x for row in selected for x in (row.get("answer_arxiv_id") or [])]
    metadata_path = args.output_dir / "arxiv_metadata.json"
    metadata = fetch_arxiv(ids, metadata_path) if args.fetch_abstracts else load_metadata(metadata_path)
    pool = documents(selected, metadata)
    train_pairs = build_pairs(train, pool, random.Random(args.seed + 1))
    calibration_pairs = build_pairs(calibration, pool, random.Random(args.seed + 2))
    write_jsonl(args.output_dir / "selector_train_pairs.jsonl", train_pairs)
    write_jsonl(args.output_dir / "selector_calibration_pairs.jsonl", calibration_pairs)

    def qmanifest(subset: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"query_id": row["qid"], "gold_count": len(row.get("answer") or []),
                 "gold_bin": "g1" if len(row.get("answer") or []) <= 1 else "g2" if len(row.get("answer") or []) == 2 else "g3plus",
                 "intent": intent(row["question"]), "domain": "AI/machine-learning", "cutoff": source_cutoff(row).isoformat()}
                for row in subset]
    manifest = {
        "schema_version": 1, "dataset": str(args.dataset), "dataset_sha256": sha256(args.dataset),
        "sampling_seed": args.seed, "split_source": "local AutoScholarQuery_test development re-split",
        "split_caveat": "Used because official train/dev files are not present locally; dev20 and validation30 are strictly excluded.",
        "dev20_ids_sha256": hashlib.sha256("\n".join(sorted(dev_ids)).encode()).hexdigest(),
        "validation30_ids_sha256": hashlib.sha256("\n".join(sorted(validation_ids)).encode()).hexdigest(),
        "selector_train": qmanifest(train), "selector_calibration": qmanifest(calibration),
        "overlap_checks": {"train_calibration": len({r['qid'] for r in train} & {r['qid'] for r in calibration}),
                           "train_dev20": len({r['qid'] for r in train} & dev_ids),
                           "train_validation30": len({r['qid'] for r in train} & validation_ids),
                           "calibration_validation30": len({r['qid'] for r in calibration} & validation_ids)},
        "metadata_cache": str(metadata_path), "metadata_records": len(metadata),
    }
    (args.output_dir / "02_selector_split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    neg = Counter(x["negative_type"] for x in train_pairs + calibration_pairs)
    missing = sum(not x.get("abstract") for x in train_pairs + calibration_pairs)
    total = len(train_pairs) + len(calibration_pairs)
    report = f"""# Selector 数据集报告

## 防泄漏划分

- selector_train：{len(train)} queries，{len(train_pairs)} query-paper pairs。
- selector_calibration：{len(calibration)} queries，{len(calibration_pairs)} pairs。
- dev20 与 validation30 均按 query ID 完全排除；重叠检查：`{manifest['overlap_checks']}`。
- 数据源：本地 AutoScholarQuery_test 的开发重划分。由于本机没有官方 train/dev 文件，不能把该划分冒充官方训练协议。

## 样本构造

- 正样本：官方 gold paper，权重 1.0。
- lexical/semantic hard weak negatives：其他 query 的高相似 gold，权重 0.45。
- random weak negatives：其他 query 的随机 gold，权重 0.2。
- 类型计数：`{dict(neg)}`。
- 摘要缺失：{missing}/{total} ({missing/total:.1%})；已缓存 arXiv 元数据 {len(metadata)} 条。

## 标注风险

AutoScholarQuery gold 不是现实世界全部相关论文。其他 query 的 gold 对当前 query 可能仍相关，因此所有跨 query 负样本均按 weak negative 降权，不作为强负样本。validation30 的逐查询 gold 没有输出、没有进入样本、没有用于阈值选择。
"""
    (args.output_dir / "03_selector_dataset_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"train_queries": len(train), "calibration_queries": len(calibration),
                      "train_pairs": len(train_pairs), "calibration_pairs": len(calibration_pairs),
                      "metadata": len(metadata), "overlaps": manifest["overlap_checks"]}, indent=2))


if __name__ == "__main__":
    main()
