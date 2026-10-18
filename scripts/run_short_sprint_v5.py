"""Execute the bounded short-sprint-v5 selector experiment.

The script deliberately reuses frozen v4 production captures.  For train500
and calibration150 queries without a production capture it builds one fixed
local hard-negative proxy from the already cached 1,404-paper metadata corpus;
it does not call an API or an LLM.  dev20 is evaluated only after every model,
fusion weight and set-decision parameter has been frozen on train/calibration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.selection.pairwise_selector_v5 import (
    FEATURE_NAMES,
    PairwiseSelectorV5,
    assert_safe_feature_schema,
    build_pairwise_examples,
    mine_hard_negative_indices,
)
from backend.selection.set_decision import select_final_set
from backend.selection.relevance_selector import tokenize


DATASET = Path(r"D:\AICompetition\references\github\SPAR\benchmark\AutoScholarQuery_test.jsonl")
V3 = ROOT / "reports/f1_selector_v3"
V4 = ROOT / "reports/joint_optimization_v4"
REPORT = ROOT / "reports/short_sprint_v5"
ARTIFACT = ROOT / "artifacts/short_sprint_v5"
CONFIG = ROOT / "configs/short_sprint_v5"
SEED = 20260830


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fields or sorted({key for row in rows for key in row}))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def parquet_safe(row: dict[str, Any]) -> dict[str, Any]:
    integer_fields = {"label", "candidate_rank", "candidate_pool_size", "live_rank", "year"}
    float_fields = {"trace_score", "lexical_score", "llm_relevance_score", "constraint_coverage_score", "sample_weight"}
    boolean_fields = {"gold_materialized", "mined_hard_negative"}
    safe: dict[str, Any] = {}
    for key, value in row.items():
        if value is None:
            safe[key] = None
        elif key in integer_fields:
            try:
                safe[key] = int(value)
            except (TypeError, ValueError):
                safe[key] = None
        elif key in float_fields:
            try:
                safe[key] = float(value)
            except (TypeError, ValueError):
                safe[key] = None
        elif key in boolean_fields:
            safe[key] = bool(value)
        elif isinstance(value, (list, tuple, dict)):
            safe[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            safe[key] = str(value)
    return safe


def write_parquet(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_rows = [parquet_safe(row) for row in rows]
    fields = sorted({key for row in safe_rows for key in row})
    rectangular = [{key: row.get(key) for key in fields} for row in safe_rows]
    pq.write_table(pa.Table.from_pylist(rectangular), path, compression="zstd")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normal_title(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def arxiv_id(value: Any) -> str:
    text = str(value or "").lower().strip()
    match = re.search(r"(?:arxiv[:./ ]*)?((?:\d{4}\.\d{4,5}|[a-z-]+/\d{7}))(?:v\d+)?", text)
    return match.group(1) if match else ""


def paper_key(row: dict[str, Any]) -> str:
    aid = arxiv_id(row.get("arxiv_id") or row.get("doi") or row.get("paper_id"))
    return f"arxiv:{aid}" if aid else f"title:{normal_title(row.get('title'))}"


def parse_date(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    result = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= low) & (probabilities < high if high < 1 else probabilities <= high)
        if mask.any():
            result += float(mask.mean()) * abs(float(probabilities[mask].mean()) - float(labels[mask].mean()))
    return result


def load_split_context() -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    manifest = json.loads((V3 / "02_selector_split_manifest.json").read_text(encoding="utf-8"))
    dataset = {row["qid"]: row for row in read_jsonl(DATASET)}
    metadata = json.loads((V3 / "arxiv_metadata.json").read_text(encoding="utf-8"))
    return manifest, dataset, metadata


def split_ids(manifest: dict[str, Any], name: str) -> list[str]:
    return [item["query_id"] for item in manifest[name]]


def audit_split_isolation(manifest: dict[str, Any]) -> dict[str, int]:
    query_sets = json.loads((ROOT / "configs/query_sets.json").read_text(encoding="utf-8"))
    train = set(split_ids(manifest, "selector_train"))
    calibration = set(split_ids(manifest, "selector_calibration"))
    dev = set(query_sets["dev20"])
    validation = set(query_sets["validation30"])
    blind_dir = ROOT / "reports/frozen_eval/blind_holdout50_v4_v3/queries"
    blind = {path.name for path in blind_dir.iterdir() if path.is_dir()} if blind_dir.exists() else set()
    return {
        "train_calibration": len(train & calibration),
        "train_dev": len(train & dev),
        "train_validation": len(train & validation),
        "train_blind": len(train & blind),
        "calibration_dev": len(calibration & dev),
        "calibration_validation": len(calibration & validation),
        "calibration_blind": len(calibration & blind),
        "dev_blind": len(dev & blind),
    }


def legal_gold(
    qid: str, item: dict[str, Any], dataset: dict[str, dict[str, Any]], metadata: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    source = dataset[qid]
    cutoff = parse_date(item["cutoff"])
    rows: list[dict[str, Any]] = []
    stats = {"total": 0, "materialized": 0, "missing_metadata": 0, "date_filtered": 0, "identity_failure": 0}
    ids = source.get("answer_arxiv_id") or []
    for index, title in enumerate(source.get("answer") or []):
        stats["total"] += 1
        aid = arxiv_id(ids[index] if index < len(ids) else "")
        if not aid:
            stats["identity_failure"] += 1; continue
        meta = metadata.get(aid)
        if not meta:
            stats["missing_metadata"] += 1; continue
        published = parse_date(meta.get("published"))
        if cutoff is not None and published is not None and published > cutoff:
            stats["date_filtered"] += 1; continue
        stats["materialized"] += 1
        rows.append({
            "query_id": qid, "query": source["question"], "cutoff": item["cutoff"],
            "paper_id": f"arxiv:{aid}", "arxiv_id": aid, "doi": None,
            "title": meta.get("title") or title, "abstract": meta.get("abstract") or "",
            "year": str(meta.get("published") or "")[:4], "published": meta.get("published"),
            "label": 1, "negative_type": "positive", "candidate_origin": "gold_materialized",
            "gold_materialized": True, "source_api": "cached_arxiv_metadata",
        })
    return rows, stats


def build_corpus(manifest: dict[str, Any], dataset: dict[str, dict[str, Any]], metadata: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for split in ("selector_train", "selector_calibration"):
        for item in manifest[split]:
            for row in legal_gold(item["query_id"], item, dataset, metadata)[0]:
                by_key[paper_key(row)] = {key: row.get(key) for key in (
                    "paper_id", "arxiv_id", "doi", "title", "abstract", "year", "published", "source_api"
                )}
    return sorted(by_key.values(), key=paper_key)


def load_real_candidates() -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in (V4 / "prod_train30_scored100.jsonl", V4 / "prod_cal20_scored100.jsonl"):
        for row in read_jsonl(path):
            result[str(row["query_id"])].append(row)
    for rows in result.values():
        rows.sort(key=lambda row: int(row.get("candidate_rank") or 10_000))
    return result


def classify_negative(query: str, row: dict[str, Any], rank: int) -> str:
    title = str(row.get("title") or "").lower()
    q = query.lower()
    if not row.get("abstract"):
        return "missing_abstract"
    if any(word in q for word in ("survey", "review", "overview")) != any(word in title for word in ("survey", "review", "overview")):
        return "paper_type_mismatch"
    qtokens, ptokens = tokenize(query), tokenize(f"{row.get('title') or ''} {row.get('abstract') or ''}")
    overlap = len(qtokens & ptokens) / max(len(qtokens), 1)
    if overlap >= 0.45:
        return "high_title_or_abstract_similarity"
    if rank <= 5:
        return "top1_5_benchmark_negative"
    if rank <= 20:
        return "top6_20_benchmark_negative"
    return "top21_60_benchmark_negative"


@dataclass
class PreparedData:
    train_rows: list[dict[str, Any]]
    train_all_candidates: list[dict[str, Any]]
    calibration_rows: list[dict[str, Any]]
    coverage: list[dict[str, Any]]
    negative_stats: list[dict[str, Any]]
    isolation: dict[str, int]


def prepare_data() -> PreparedData:
    manifest, dataset, metadata = load_split_context()
    isolation = audit_split_isolation(manifest)
    if any(isolation.values()):
        raise RuntimeError(f"split leakage detected: {isolation}")
    corpus = build_corpus(manifest, dataset, metadata)
    documents = [f"{row.get('title') or ''} {row.get('abstract') or ''}" for row in corpus]
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True, max_features=60000)
    document_matrix = vectorizer.fit_transform(documents)
    real = load_real_candidates()
    coverage_rows: list[dict[str, Any]] = []
    all_negative_types: Counter[str] = Counter()

    def build_split(name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        selected_rows: list[dict[str, Any]] = []
        every_candidate: list[dict[str, Any]] = []
        items = manifest[name]
        questions = [dataset[item["query_id"]]["question"] for item in items]
        similarities = (vectorizer.transform(questions) @ document_matrix.T).toarray()
        for query_index, item in enumerate(items):
            qid = item["query_id"]; query = dataset[qid]["question"]
            positives, stats = legal_gold(qid, item, dataset, metadata)
            positive_keys = {paper_key(row) for row in positives}
            candidates: list[dict[str, Any]] = []
            seen: set[str] = set()
            if qid in real:
                source = real[qid][:60]; origin = "frozen_production_v4"
                for rank, raw in enumerate(source, 1):
                    row = {key: raw.get(key) for key in (
                        "paper_id", "arxiv_id", "doi", "openalex_id", "title", "abstract", "year",
                        "source_api", "source_round", "trace_score", "lexical_score", "llm_relevance_score",
                        "constraint_coverage_score", "candidate_rank", "candidate_pool_size", "live_rank",
                    )}
                    row.update({"query_id": qid, "query": query, "cutoff": item["cutoff"], "candidate_origin": origin})
                    key = paper_key(row)
                    if not key.endswith("title:") and key not in seen:
                        seen.add(key); candidates.append(row)
            else:
                origin = "fixed_local_metadata_proxy"
                order = np.argsort(-similarities[query_index], kind="stable")[:60]
                for rank, corpus_index in enumerate(order, 1):
                    base = dict(corpus[int(corpus_index)])
                    lexical = float(similarities[query_index, int(corpus_index)])
                    base.update({
                        "query_id": qid, "query": query, "cutoff": item["cutoff"], "candidate_origin": origin,
                        "candidate_rank": rank, "candidate_pool_size": 60, "lexical_score": lexical,
                        "trace_score": lexical, "llm_relevance_score": None, "source_round": "offline_proxy",
                    })
                    key = paper_key(base)
                    if key not in seen:
                        seen.add(key); candidates.append(base)

            retrieved_gold = 0
            for rank, row in enumerate(candidates, 1):
                row["candidate_rank"] = rank
                row["candidate_pool_size"] = len(candidates)
                row["label"] = int(paper_key(row) in positive_keys)
                if row["label"]:
                    retrieved_gold += 1; row["negative_type"] = "positive"; row["gold_materialized"] = False
                else:
                    row["negative_type"] = classify_negative(query, row, rank); row["gold_materialized"] = False
                    all_negative_types[row["negative_type"]] += 1

            for positive in positives:
                key = paper_key(positive)
                if key not in seen:
                    injected = dict(positive)
                    injected.update({
                        "candidate_rank": len(candidates) + 1, "candidate_pool_size": len(candidates) + 1,
                        "trace_score": None, "lexical_score": None, "llm_relevance_score": None,
                    })
                    candidates.append(injected); seen.add(key)

            # Keep original retrieval order.  Reordering positives before
            # benchmark-negatives would let stable tie-breaking leak labels when
            # a channel score is missing for every candidate in a query.
            # Pair construction later caps each positive at ten negatives.
            selected_rows.extend(candidates)
            every_candidate.extend(candidates)
            coverage_rows.append({
                "split": "train" if name == "selector_train" else "calibration", "query_id": qid,
                "total_legal_gold": stats["total"] - stats["date_filtered"] - stats["identity_failure"] - stats["missing_metadata"],
                "materialized_positives": stats["materialized"], "retrieved_positive_count": retrieved_gold,
                "missing_metadata": stats["missing_metadata"], "date_filtered": stats["date_filtered"],
                "identity_failures": stats["identity_failure"], "candidate_origin": origin,
            })
        return selected_rows, every_candidate

    train_rows, train_all = build_split("selector_train")
    calibration_rows, _ = build_split("selector_calibration")
    negative_stats = [{"negative_type": name, "count": count} for name, count in sorted(all_negative_types.items())]
    write_parquet(ARTIFACT / "selector_train_pairs.parquet", train_rows)
    write_parquet(ARTIFACT / "selector_calibration_candidates.parquet", calibration_rows)
    write_csv(REPORT / "01_positive_coverage.csv", coverage_rows)
    write_csv(REPORT / "01_hard_negative_stats.csv", negative_stats)
    totals = Counter()
    for row in coverage_rows:
        for key in ("total_legal_gold", "materialized_positives", "retrieved_positive_count", "missing_metadata", "date_filtered", "identity_failures"):
            totals[f"{row['split']}_{key}"] += int(row[key])
    (REPORT / "01_selector_data_audit.md").write_text(f"""# Selector 数据审计

| 项目 | train500 | calibration150 |
|---|---:|---:|
| 合法 gold | {totals['train_total_legal_gold']} | {totals['calibration_total_legal_gold']} |
| materialized positives | {totals['train_materialized_positives']} | {totals['calibration_materialized_positives']} |
| 原生产/代理候选已召回正例 | {totals['train_retrieved_positive_count']} | {totals['calibration_retrieved_positive_count']} |
| metadata 缺失 | {totals['train_missing_metadata']} | {totals['calibration_missing_metadata']} |
| 日期排除 | {totals['train_date_filtered']} | {totals['calibration_date_filtered']} |
| identity 失败 | {totals['train_identity_failures']} | {totals['calibration_identity_failures']} |

- 旧 v4 冻结 production50 只有 35 个已召回正例；本轮 train500 合法正例全部物化。
- 30 个训练查询、20 个校准查询直接使用冻结生产 Top60；其余查询使用同一份缓存元数据语料构建固定 TF-IDF Top60 hard-negative proxy，没有新增 API/LLM 调用。
- 非 gold 候选统一解释为 `benchmark-negative`，不解释为现实世界确定不相关。
- split overlap：`{isolation}`；dev20、validation30、blind50 的 gold 均未读取或进入训练。
- `gold_materialized` 仅用于数据审计，`FEATURE_NAMES={list(FEATURE_NAMES)}` 中不含该字段或任何标签派生字段。
""", encoding="utf-8")
    return PreparedData(train_rows, train_all, calibration_rows, coverage_rows, negative_stats, isolation)


def grouped(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[str(row["query_id"])].append(row)
    return result


def metric_summary(
    rows: Sequence[dict[str, Any]], selector: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    gold_counts: dict[str, int] | None = None,
) -> dict[str, float | int]:
    values = []
    tp_total = fp_total = fn_total = 0
    for qid, candidates in grouped(rows).items():
        selected = selector(candidates)
        tp = sum(int(row.get("label") or 0) for row in selected)
        predicted = len(selected)
        gold = gold_counts[qid] if gold_counts else sum(int(row.get("label") or 0) for row in candidates)
        fp, fn = predicted - tp, max(0, gold - tp)
        precision = tp / predicted if predicted else 0.0
        recall = tp / gold if gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        values.append((precision, recall, f1, predicted, tp))
        tp_total += tp; fp_total += fp; fn_total += fn
    n = max(len(values), 1)
    return {
        "precision": sum(value[0] for value in values) / n,
        "recall": sum(value[1] for value in values) / n,
        "f1": sum(value[2] for value in values) / n,
        "TP": tp_total, "FP": fp_total, "FN": fn_total,
        "mean_output_count": sum(value[3] for value in values) / n,
        "mean_tp": sum(value[4] for value in values) / n,
        "query_count": len(values),
    }


def old_s1_probabilities(rows: list[dict[str, Any]]) -> None:
    with (V4 / "selector_model.pkl").open("rb") as handle:
        bundle = pickle.load(handle)
    model = bundle["models"]["S1_llm_calibrated"]
    x = np.asarray([[float(row.get("llm_relevance_score") or 0.0)] for row in rows], dtype=float)
    probabilities = model.predict_proba(x)[:, 1]
    for row, probability in zip(rows, probabilities):
        row["old_s1_probability"] = float(probability)


def gold_count_map() -> dict[str, int]:
    return {row["qid"]: len(row.get("answer") or []) for row in read_jsonl(DATASET)}


def same_snapshot_baselines(dev_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    old_s1_probabilities(dev_rows)
    counts = gold_count_map()

    def top_live(k: int):
        return lambda rows: sorted(
            [row for row in rows if row.get("live_rank") and int(row["live_rank"]) <= 20],
            key=lambda row: int(row["live_rank"]),
        )[:k]

    def old_s1(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ordered = sorted(rows, key=lambda row: float(row["old_s1_probability"]), reverse=True)
        selected = [row for row in ordered if float(row["old_s1_probability"]) >= 0.99][:8]
        return selected if len(selected) >= 2 else ordered[:2]

    def llm090(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        eligible = [row for row in rows if row.get("live_rank") and int(row["live_rank"]) <= 20]
        return [row for row in sorted(eligible, key=lambda row: float(row.get("llm_relevance_score") or 0), reverse=True)
                if float(row.get("llm_relevance_score") or 0) >= 0.9]

    selectors: list[tuple[str, Callable[[list[dict[str, Any]]], list[dict[str, Any]]]]] = [
        ("Top20", top_live(20)), *[(f"K={k}", top_live(k)) for k in range(1, 6)],
        ("current_S1", old_s1), ("current_LLM_0.90", llm090),
    ]
    output = []
    for name, selector in selectors:
        row = {"configuration": name, **metric_summary(dev_rows, selector, counts)}
        row["per_query_output_count"] = json.dumps({qid: len(selector(items)) for qid, items in grouped(dev_rows).items()}, sort_keys=True)
        output.append(row)
    write_csv(REPORT / "00_same_snapshot_baselines.csv", output)
    by_name = {row["configuration"]: row for row in output}
    k2, s1 = by_name["K=2"], by_name["current_S1"]
    (REPORT / "00_baseline_note.md").write_text(f"""# dev20 同快照基线

- 全部配置使用 `dev20_current_scored.jsonl`、同一 identity/evaluator 和同一 gold count。
- K=2：P/R/F1={k2['precision']:.4f}/{k2['recall']:.4f}/{k2['f1']:.4f}，TP/FP={k2['TP']}/{k2['FP']}。
- S1：P/R/F1={s1['precision']:.4f}/{s1['recall']:.4f}/{s1['f1']:.4f}，TP/FP={s1['TP']}/{s1['FP']}。
- S1 相对 K=2：TP {int(s1['TP'])-int(k2['TP']):+d}，FP {int(s1['FP'])-int(k2['FP']):+d}，平均输出 {float(s1['mean_output_count'])-float(k2['mean_output_count']):+.2f}。
- 该结果只用于固定参考线，未用于训练、融合权重或阈值选择。
""", encoding="utf-8")
    return output


def rank_metrics(rows: Sequence[dict[str, Any]], scores: np.ndarray) -> dict[str, float]:
    score_by_id = {id(row): float(score) for row, score in zip(rows, scores)}
    reciprocal, r1, r3, r5, r10 = [], [], [], [], []
    for candidates in grouped(rows).values():
        ordered = sorted(candidates, key=lambda row: score_by_id[id(row)], reverse=True)
        gold = sum(int(row["label"]) for row in ordered)
        ranks = [index + 1 for index, row in enumerate(ordered) if int(row["label"]) == 1]
        reciprocal.append(1 / min(ranks) if ranks else 0.0)
        for bucket, k in ((r1, 1), (r3, 3), (r5, 5), (r10, 10)):
            bucket.append(sum(int(row["label"]) for row in ordered[:k]) / gold if gold else 0.0)
    return {"mrr": float(np.mean(reciprocal)), "recall@1": float(np.mean(r1)), "recall@3": float(np.mean(r3)),
            "recall@5": float(np.mean(r5)), "recall@10": float(np.mean(r10))}


def pairwise_accuracy(model: PairwiseSelectorV5, rows: Sequence[dict[str, Any]]) -> float:
    scores = model.decision_function(rows)
    score_by_id = {id(row): float(score) for row, score in zip(rows, scores)}
    correct = total = 0
    for candidates in grouped(rows).values():
        positives = [row for row in candidates if int(row["label"]) == 1]
        negatives = [row for row in candidates if int(row["label"]) == 0][:10]
        for positive in positives:
            for negative in negatives:
                correct += int(score_by_id[id(positive)] > score_by_id[id(negative)]); total += 1
    return correct / total if total else 0.0


def cv_predictions(rows: list[dict[str, Any]], mode: str, *, mined: bool = False) -> tuple[np.ndarray, float]:
    groups_array = np.asarray([row["query_id"] for row in rows])
    predictions = np.zeros(len(rows), dtype=float)
    splitter = GroupKFold(n_splits=3)
    dummy = np.zeros(len(rows))
    for train_idx, test_idx in splitter.split(dummy, dummy, groups_array):
        training = [dict(rows[int(index)]) for index in train_idx]
        if mode == "pointwise":
            model = PairwiseSelectorV5.fit_pointwise(training)
        else:
            model = PairwiseSelectorV5.fit_pairwise(training)
            if mined:
                for index in mine_hard_negative_indices(model, training, per_query=3):
                    training[index]["mined_hard_negative"] = True
                model = PairwiseSelectorV5.fit_pairwise(training, seed=SEED + 1)
        testing = [rows[int(index)] for index in test_idx]
        predictions[test_idx] = model.decision_function(testing)
    labels = np.asarray([int(row["label"]) for row in rows])
    return predictions, float(average_precision_score(labels, predictions))


@dataclass
class ModelBundle:
    models: dict[str, PairwiseSelectorV5]
    chosen_name: str
    comparison: list[dict[str, Any]]


def train_models(data: PreparedData) -> ModelBundle:
    train = [dict(row) for row in data.train_rows]
    calibration = [dict(row) for row in data.calibration_rows]
    models: dict[str, PairwiseSelectorV5] = {}
    comparison: list[dict[str, Any]] = []
    definitions = (("pointwise", "pointwise", False), ("pairwise", "pairwise", False), ("pairwise_mined", "pairwise", True))
    for name, mode, mined in definitions:
        oof, cv_ap = cv_predictions(train, mode, mined=mined)
        if mode == "pointwise":
            model = PairwiseSelectorV5.fit_pointwise(train)
        else:
            model = PairwiseSelectorV5.fit_pairwise(train)
            if mined:
                indices = mine_hard_negative_indices(model, train, per_query=3)
                for index in indices:
                    train[index]["mined_hard_negative"] = True
                model = PairwiseSelectorV5.fit_pairwise(train, seed=SEED + 1)
        raw_cal = model.decision_function(calibration)
        labels_cal = np.asarray([int(row["label"]) for row in calibration])
        cal_ap = float(average_precision_score(labels_cal, raw_cal))
        model.fit_calibrator(calibration)
        probabilities = model.predict_proba(calibration)
        metrics = rank_metrics(calibration, raw_cal)
        comparison.append({
            "model": name, "query_grouped_cv_pr_auc": cv_ap,
            "calibration_pr_auc": cal_ap, "pairwise_accuracy": pairwise_accuracy(model, calibration),
            "brier": float(brier_score_loss(labels_cal, probabilities)), "ece": ece(labels_cal, probabilities),
            "train_positive_count": sum(int(row["label"]) for row in train),
            "train_benchmark_negative_count": sum(1-int(row["label"]) for row in train),
            **metrics,
        })
        models[name] = model
    comparison.append({"model": "xgboost_rank_pairwise", "status": "skipped_dependency_not_installed"})
    write_csv(REPORT / "02_model_comparison.csv", comparison)
    eligible = [row for row in comparison if row.get("calibration_pr_auc") is not None]
    chosen = max(eligible, key=lambda row: (float(row["calibration_pr_auc"]), float(row["query_grouped_cv_pr_auc"])))
    chosen_name = str(chosen["model"])
    importance = [{"feature": name, "weight": weight, "abs_weight": abs(weight)}
                  for name, weight in models[chosen_name].feature_weights().items()]
    importance.sort(key=lambda row: float(row["abs_weight"]), reverse=True)
    write_csv(REPORT / "02_feature_importance.csv", importance)
    return ModelBundle(models, chosen_name, comparison)


def query_normalised(rows: Sequence[dict[str, Any]], field: str, *, missing: float = 0.0) -> np.ndarray:
    values = np.zeros(len(rows), dtype=float)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows): groups[str(row["query_id"])].append(index)
    for indices in groups.values():
        raw = [float(rows[index].get(field)) if rows[index].get(field) is not None else missing for index in indices]
        low, high = min(raw), max(raw); span = max(high - low, 1e-9)
        for index, value in zip(indices, raw): values[index] = (value - low) / span if high > low else value
    return values


def selected_by_scores(
    candidates: list[dict[str, Any]], score_map: dict[int, float], strategy: str, params: dict[str, Any],
) -> list[dict[str, Any]]:
    ordered = sorted(candidates, key=lambda row: (
        -score_map[id(row)], int(row.get("candidate_rank") or 10_000),
        str(row.get("paper_id") or row.get("title") or ""),
    ))
    probabilities = [score_map[id(row)] for row in ordered]
    if strategy == "fixed_k2":
        return ordered[:2]
    if strategy == "global_threshold":
        decision = select_final_set(probabilities, strategy="global_threshold", threshold=params["threshold"], max_results=6)
    elif strategy == "expected_f1":
        decision = select_final_set(probabilities, strategy="expected_f1", max_results=6)
    else:
        decision = select_final_set(probabilities, strategy="score_gap", threshold=params["threshold"],
                                    low_threshold=params["low_threshold"], score_gap=params["score_gap"], max_results=6)
    return [ordered[index] for index in decision.selected_indices]


def evaluate_score_decisions(rows: list[dict[str, Any]], scores: np.ndarray, name: str) -> list[dict[str, Any]]:
    score_map = {id(row): float(score) for row, score in zip(rows, scores)}
    trials: list[tuple[str, dict[str, Any]]] = [("fixed_k2", {}), ("expected_f1", {})]
    trials += [("global_threshold", {"threshold": threshold}) for threshold in np.arange(0.2, 0.91, 0.1)]
    trials += [("threshold_score_gap", {"threshold": threshold, "low_threshold": max(0.05, threshold - 0.25), "score_gap": gap})
               for threshold in (0.4, 0.6, 0.8) for gap in (0.10, 0.20)]
    output = []
    for strategy, params in trials:
        selector = lambda candidates, s=strategy, p=params: selected_by_scores(candidates, score_map, s, p)
        output.append({"score_configuration": name, "decision": strategy, "params": json.dumps(params, sort_keys=True),
                       **metric_summary(rows, selector)})
    return output


@dataclass
class FrozenChoice:
    model_name: str
    weights: tuple[float, float, float]
    score_configuration: str
    decision: str
    params: dict[str, Any]
    calibration_metrics: dict[str, Any]
    standalone_choices: dict[str, dict[str, Any]]


def best_decision(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda row: (
        round(float(row["f1"]), 4), float(row["recall"]), int(row["TP"]), -float(row["mean_output_count"])
    ))


def calibrate_fusion_and_decision(data: PreparedData, bundle: ModelBundle) -> tuple[FrozenChoice, list[dict[str, Any]]]:
    rows = data.calibration_rows
    model = bundle.models[bundle.chosen_name]
    pair = model.predict_proba(rows)
    llm = query_normalised(rows, "llm_relevance_score", missing=0.0)
    trace = query_normalised(rows, "trace_score", missing=0.0)
    configs: list[tuple[str, tuple[float, float, float]]] = [
        ("F0_pairwise_only", (1.0, 0.0, 0.0)), ("F1_current_llm_only", (0.0, 1.0, 0.0)),
    ]
    configs += [("F2_pairwise_llm", (a, 1-a, 0.0)) for a in np.arange(0.2, 1.0, 0.2)]
    for a in np.arange(0.2, 1.0, 0.2):
        for b in np.arange(0.0, 1.01-a, 0.2):
            c = 1.0-a-b
            if c >= -1e-9:
                configs.append(("F3_pairwise_llm_trace", (float(a), float(b), float(max(0, c)))))
    registry: list[dict[str, Any]] = []
    standalone: dict[str, dict[str, Any]] = {}
    for model_name, candidate_model in bundle.models.items():
        candidate_scores = candidate_model.predict_proba(rows)
        trials = evaluate_score_decisions(rows, candidate_scores, f"standalone_{model_name}")
        for row in trials:
            row.update({"model": model_name, "weight_pairwise": 1.0, "weight_llm": 0.0, "weight_trace": 0.0})
            registry.append(row)
        model_best = best_decision(trials)
        standalone[model_name] = {
            "decision": model_best["decision"], "params": json.loads(str(model_best["params"])),
            "calibration_metrics": {key: model_best[key] for key in (
                "precision", "recall", "f1", "TP", "FP", "FN", "mean_output_count"
            )},
        }
    for name, weights in configs:
        scores = weights[0]*pair + weights[1]*llm + weights[2]*trace
        for row in evaluate_score_decisions(rows, scores, name):
            row.update({"weight_pairwise": weights[0], "weight_llm": weights[1], "weight_trace": weights[2]})
            registry.append(row)
    # Prefer F1; for practically tied results retain more TP/recall and then fewer outputs.
    fusion_rows = [row for row in registry if not str(row["score_configuration"]).startswith("standalone_")]
    best = best_decision(fusion_rows)
    choice = FrozenChoice(
        model_name=bundle.chosen_name,
        weights=(float(best["weight_pairwise"]), float(best["weight_llm"]), float(best["weight_trace"])),
        score_configuration=str(best["score_configuration"]), decision=str(best["decision"]),
        params=json.loads(str(best["params"])), calibration_metrics={key: best[key] for key in (
            "precision", "recall", "f1", "TP", "FP", "FN", "mean_output_count"
        )}, standalone_choices=standalone,
    )
    write_csv(REPORT / "03_calibration_and_set_decision.csv", registry)
    return choice, registry


def score_components(rows: list[dict[str, Any]], model: PairwiseSelectorV5, weights: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    pair = model.predict_proba(rows)
    llm = query_normalised(rows, "llm_relevance_score", missing=0.0)
    trace = query_normalised(rows, "trace_score", missing=0.0)
    return pair, weights[0]*pair + weights[1]*llm + weights[2]*trace


def final_dev_check(
    dev_rows: list[dict[str, Any]], baselines: list[dict[str, Any]], bundle: ModelBundle, choice: FrozenChoice,
) -> tuple[list[dict[str, Any]], bool, dict[str, Any]]:
    counts = gold_count_map(); model = bundle.models[choice.model_name]
    _, fused = score_components(dev_rows, model, choice.weights)
    fused_map = {id(row): float(score) for row, score in zip(dev_rows, fused)}

    def model_selector(score_map: dict[int, float]):
        return lambda candidates: selected_by_scores(candidates, score_map, choice.decision, choice.params)

    baseline_map = {row["configuration"]: row for row in baselines}
    output = []
    for label in ("Top20", "K=2", "current_S1", "current_LLM_0.90"):
        output.append({"configuration": label, **{k: v for k, v in baseline_map[label].items() if k != "configuration"}})
    for model_name, label in (("pointwise", "pointwise"), ("pairwise", "pairwise"), ("pairwise_mined", "pairwise+mining")):
        standalone_choice = choice.standalone_choices[model_name]
        scores = bundle.models[model_name].predict_proba(dev_rows)
        score_map = {id(row): float(score) for row, score in zip(dev_rows, scores)}
        selector = lambda candidates, sm=score_map, sc=standalone_choice: selected_by_scores(
            candidates, sm, str(sc["decision"]), dict(sc["params"])
        )
        output.append({"configuration": label, **metric_summary(dev_rows, selector, counts)})
    output.append({"configuration": "best_fused_selector", **metric_summary(dev_rows, model_selector(fused_map), counts)})
    write_csv(REPORT / "04_dev20_final_check.csv", output)
    rows_by_name = {row["configuration"]: row for row in output}
    candidate = rows_by_name["best_fused_selector"]; k2 = rows_by_name["K=2"]; s1 = rows_by_name["current_S1"]
    rule_a = candidate["f1"] >= k2["f1"] + 0.015 and candidate["recall"] >= k2["recall"] - 0.01
    rule_b = candidate["f1"] >= k2["f1"] - 0.005 and candidate["recall"] >= k2["recall"] + 0.05
    rule_c = abs(candidate["mean_output_count"] - k2["mean_output_count"]) <= 0.5 and candidate["TP"] >= k2["TP"] + 3 and candidate["FP"] <= k2["FP"] + 5
    changes = [candidate[key] - s1[key] for key in ("precision", "recall", "f1")]
    rule_d = sum(change > 0 for change in changes) >= 2 and min(changes) >= -0.01
    accepted = bool(rule_a or rule_b or rule_c or rule_d)
    acceptance = {"A": bool(rule_a), "B": bool(rule_b), "C": bool(rule_c), "D": bool(rule_d), "accepted": accepted}
    return output, accepted, acceptance


def markdown_table(rows: Sequence[dict[str, Any]]) -> str:
    lines = ["| 配置 | P | R | F1 | TP | FP | FN | 平均输出 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['configuration']} | {float(row['precision']):.4f} | {float(row['recall']):.4f} | {float(row['f1']):.4f} | {int(row['TP'])} | {int(row['FP'])} | {int(row['FN'])} | {float(row['mean_output_count']):.2f} |")
    return "\n".join(lines)


def write_outputs(
    data: PreparedData, baselines: list[dict[str, Any]], bundle: ModelBundle, choice: FrozenChoice,
    dev_results: list[dict[str, Any]], accepted: bool, acceptance: dict[str, Any],
) -> None:
    model = bundle.models[choice.model_name]
    model_path = ARTIFACT / "pairwise_selector.pkl"; model.save(model_path)
    CONFIG.mkdir(parents=True, exist_ok=True)
    (CONFIG / "pairwise_selector.yaml").write_text(
        "schema_version: 1\nprofile: short_sprint_v5\n"
        f"model: {choice.model_name}\nmodel_path: {model_path.as_posix()}\n"
        f"fusion_weights:\n  pairwise: {choice.weights[0]:.3f}\n  llm: {choice.weights[1]:.3f}\n  trace: {choice.weights[2]:.3f}\n"
        f"set_decision: {choice.decision}\nset_params: {json.dumps(choice.params, sort_keys=True)}\nmax_results: 6\nproduct_min_k: 1\n",
        encoding="utf-8",
    )
    optional = [{"status": "skipped", "reason": "selector_not_accepted" if not accepted else "no_fixed_local_full_corpus_beyond_existing_1404_metadata_proxy",
                 "candidate_gold_delta": 0, "candidate_recall_delta": 0.0, "final_f1_delta": 0.0}]
    write_csv(REPORT / "05_optional_local_fts_result.csv", optional)
    train_positive = sum(int(row["label"]) for row in data.train_rows)
    train_negative = len(data.train_rows) - train_positive
    mined_count = sum(1 for row in data.train_rows if row.get("mined_hard_negative"))
    materialized_train = sum(int(row["materialized_positives"]) for row in data.coverage if row["split"] == "train")
    deduplicated_aliases = materialized_train - train_positive
    materialized_calibration = sum(int(row["materialized_positives"]) for row in data.coverage if row["split"] == "calibration")
    unique_calibration_positive = sum(int(row["label"]) for row in data.calibration_rows)
    calibration_aliases = materialized_calibration - unique_calibration_positive
    current = {row["configuration"]: row for row in dev_results}
    final = current["best_fused_selector"]
    report = f"""# short-sprint-v5 最终报告

## 数据改造

- 旧 v4 production50 已召回正例：35。
- train500 合法 materialized gold 记录：{materialized_train}；按 query-paper identity 去重后正例：{train_positive}（合并重复别名 {deduplicated_aliases} 条）；benchmark-negative candidates：{train_negative}。
- calibration150 合法 materialized gold 记录：{materialized_calibration}；identity 去重后正例：{unique_calibration_positive}（合并重复别名 {calibration_aliases} 条）。
- 一轮 mined hard negatives：{mined_count}；split isolation：`{data.isolation}`。
- 日期、metadata 和 identity 明细见 `01_positive_coverage.csv`。

## 模型

- pointwise、pairwise、pairwise+mining 的 CV/校准结果见 `02_model_comparison.csv`。
- 校准选择：`{choice.model_name}`；融合 `{choice.score_configuration}`，权重 `{choice.weights}`。
- 集合决策：`{choice.decision}`，参数 `{choice.params}`；全部由 train500+calibration150 定稿。

## dev20 一次性检查

{markdown_table(dev_results)}

- 相对 K=2：TP {int(final['TP'])-int(current['K=2']['TP']):+d}，FP {int(final['FP'])-int(current['K=2']['FP']):+d}。
- 相对 S1：TP {int(final['TP'])-int(current['current_S1']['TP']):+d}，FP {int(final['FP'])-int(current['current_S1']['FP']):+d}。
- 接受规则：`{acceptance}`；development_candidate_v5：**{'形成' if accepted else '未形成'}**。
- 没有达到接受规则时，v4 默认配置保持不变；训练数据管线和模型作为开发资产保留。

## optional local FTS5

- 本轮不建设新大索引。现有 1,404 篇 metadata proxy 已用于 hard-negative 训练；没有更完整的固定本地语料可做有意义的 retrieval challenger。

## 推迟到最终论文阶段

- 新盲测、bootstrap CI、完整通道消融、外部同协议复现、显著性检验和成本曲线。
"""
    (REPORT / "final_report.md").write_text(report, encoding="utf-8")
    state = {
        "schema_version": 1, "branch": "short-sprint-v5", "baseline_commit": "f5bcca3",
        "dataset_sha256": sha256(DATASET), "chosen_model": choice.model_name,
        "fusion_weights": choice.weights, "decision": choice.decision, "decision_params": choice.params,
        "standalone_choices": choice.standalone_choices,
        "calibration_metrics": choice.calibration_metrics, "dev_metrics": final,
        "acceptance": acceptance, "development_candidate_v5": accepted,
        "blind50_used": False, "validation30_gold_used": False,
        "artifacts": {"model": str(model_path), "config": str(CONFIG / "pairwise_selector.yaml")},
    }
    (REPORT / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True); ARTIFACT.mkdir(parents=True, exist_ok=True); CONFIG.mkdir(parents=True, exist_ok=True)
    assert_safe_feature_schema()
    dev_rows = read_jsonl(V4 / "dev20_current_scored.jsonl")
    baselines = same_snapshot_baselines(dev_rows) if not args.skip_baseline else list(csv.DictReader((REPORT / "00_same_snapshot_baselines.csv").open(encoding="utf-8-sig")))
    data = prepare_data()
    bundle = train_models(data)
    # Persist which existing negatives were selected by the single mining pass.
    base = PairwiseSelectorV5.fit_pairwise(data.train_rows)
    for index in mine_hard_negative_indices(base, data.train_rows, per_query=3):
        data.train_rows[index]["mined_hard_negative"] = True
    write_parquet(ARTIFACT / "selector_train_pairs.parquet", data.train_rows)
    choice, _ = calibrate_fusion_and_decision(data, bundle)
    dev_results, accepted, acceptance = final_dev_check(dev_rows, baselines, bundle, choice)
    write_outputs(data, baselines, bundle, choice, dev_results, accepted, acceptance)
    print(json.dumps({
        "train_positive": sum(int(row["label"]) for row in data.train_rows),
        "train_negative": sum(1-int(row["label"]) for row in data.train_rows),
        "chosen_model": choice.model_name, "choice": asdict(choice),
        "dev": dev_results, "accepted": accepted, "acceptance": acceptance,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
