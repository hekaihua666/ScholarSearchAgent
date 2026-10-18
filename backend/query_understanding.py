"""Domain-agnostic academic query understanding for the MVP.

This module is deliberately rule-based. It should be easy to replace with a
real LLM later, but the current design avoids hard-coding one research area.
"""

from __future__ import annotations

import re
from typing import Any


DEFAULT_TIME_RANGE = "not specified"


ACADEMIC_ERROR_MESSAGE = (
    "当前系统面向科研场景，请输入与论文、算法、模型、数据集、实验或研究问题相关的学术查询。"
)


def _contains_any(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = re.sub(r"\s+", " ", item.strip(" ,;，；。？?"))
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


def is_academic_query(query: str) -> bool:
    """Block only clearly non-research queries; allow uncertain queries."""
    normalized = query.strip()
    if not normalized:
        return False

    academic_intent_keywords = [
        "论文",
        "文献",
        "学术",
        "科研",
        "研究",
        "综述",
        "算法",
        "模型",
        "数据集",
        "实验",
        "基准",
        "评测",
        "检索",
        "推荐",
        "问答",
        "工程",
        "技术",
        "问题",
        "是什么",
        "如何",
        "方法",
        "应用",
        "理论",
        "证明",
        "优化",
        "最优化",
        "凸优化",
        "图",
        "矩阵",
        "函数",
        "定理",
        "收敛",
        "复杂度",
        "分类",
        "预测",
        "检测",
        "识别",
        "paper",
        "literature",
        "academic",
        "scholarly",
        "scientific",
        "research",
        "survey",
        "algorithm",
        "model",
        "dataset",
        "experiment",
        "benchmark",
        "retrieval",
        "recommendation",
        "question answering",
        "qa",
        "optimization",
        "theorem",
        "convergence",
        "complexity",
        "classification",
        "prediction",
        "detection",
        "recognition",
    ]
    if _contains_any(normalized, academic_intent_keywords):
        return True

    # Unknown uppercase abbreviations are often methods, datasets, or problems.
    if re.search(r"[A-Z]{2,}", normalized):
        return True

    non_academic_keywords = [
        "原神",
        "王者荣耀",
        "和平精英",
        "游戏攻略",
        "火神攻略",
        "抽卡",
        "角色强度",
        "娱乐八卦",
        "电影",
        "电视剧",
        "明星",
        "今天吃什么",
        "美食",
        "旅游",
        "天气",
        "genshin",
        "gacha",
    ]
    if _contains_any(normalized, non_academic_keywords):
        return False

    return True


TERM_MAPPINGS: list[tuple[str, str, str]] = [
    # Methods and models
    ("检索增强生成", "retrieval augmented generation", "method"),
    ("大语言模型", "large language model", "method"),
    ("智能体", "agent", "method"),
    ("查询分解", "query decomposition", "method"),
    ("多轮检索", "iterative retrieval", "method"),
    ("迭代式", "iterative retrieval", "method"),
    ("引文追踪", "citation tracing", "method"),
    ("引文链", "citation chain expansion", "method"),
    ("图神经网络", "graph neural network", "method"),
    ("支持向量机", "support vector machine", "method"),
    ("深度学习", "deep learning", "method"),
    ("机器学习", "machine learning", "method"),
    ("Frank-Wolfe", "Frank-Wolfe", "method"),
    ("Frank Wolfe", "Frank-Wolfe", "method"),
    ("SVM", "SVM", "method"),
    ("GNN", "GNN", "method"),
    ("LLM", "LLM", "method"),
    ("RAG", "RAG", "method"),
    # Tasks
    ("学术论文检索", "academic paper search", "task"),
    ("论文检索", "academic paper retrieval", "task"),
    ("文献检索", "literature retrieval", "task"),
    ("学术文献检索", "literature retrieval", "task"),
    ("科研问答", "scientific question answering", "task"),
    ("科学问答", "scientific question answering", "task"),
    ("问答", "question answering", "task"),
    ("论文推荐", "paper recommendation", "task"),
    ("推荐", "recommendation", "task"),
    ("检索", "retrieval", "task"),
    ("分类", "classification", "task"),
    ("预测", "prediction", "task"),
    ("交通预测", "traffic prediction", "task"),
    ("检测", "detection", "task"),
    ("缺陷检测", "defect detection", "task"),
    ("欺诈检测", "fraud detection", "task"),
    ("分割", "segmentation", "task"),
    ("优化问题", "optimization problem", "task"),
    ("优化", "optimization", "task"),
    # Domains
    ("学术搜索", "academic search", "domain"),
    ("复杂学术查询", "complex scholarly query", "domain"),
    ("学术检索", "academic search", "domain"),
    ("科研", "scientific research", "domain"),
    ("学术", "scholarly search", "domain"),
    ("交通", "transportation", "domain"),
    ("信用卡", "credit card", "domain"),
    ("医学影像", "medical imaging", "domain"),
    ("医疗影像", "medical imaging", "domain"),
    ("图像识别", "image recognition", "domain"),
    ("计算机视觉", "computer vision", "domain"),
    ("自然语言处理", "natural language processing", "domain"),
    ("金融", "finance", "domain"),
    ("制造", "manufacturing", "domain"),
    ("图论", "graph theory", "domain"),
    ("最大团", "maximum clique", "domain"),
    ("最小包围椭球", "minimum volume enclosing ellipsoid", "domain"),
    ("最小体积包围椭球", "minimum volume enclosing ellipsoid", "domain"),
    ("最小包围球", "minimum enclosing ball", "domain"),
    # English phrases
    ("retrieval augmented generation", "retrieval augmented generation", "method"),
    ("retrieval-augmented generation", "retrieval augmented generation", "method"),
    ("graph neural network", "graph neural network", "method"),
    ("support vector machine", "support vector machine", "method"),
    ("question answering", "question answering", "task"),
    ("literature retrieval", "literature retrieval", "task"),
    ("academic paper search", "academic paper search", "task"),
    ("medical imaging", "medical imaging", "domain"),
    ("computer vision", "computer vision", "domain"),
    ("graph theory", "graph theory", "domain"),
    ("optimization", "optimization", "task"),
]


ENGLISH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
    "what",
    "which",
    "who",
    "whose",
    "how",
    "is",
}


def _extract_time_range(query: str) -> str:
    if _contains_any(query, ["近五年", "最近五年", "过去五年", "last five years", "recent five years"]):
        return "last 5 years"
    if _contains_any(query, ["近三年", "最近三年", "过去三年", "last three years", "recent three years"]):
        return "last 3 years"
    if _contains_any(query, ["近十年", "最近十年", "过去十年", "last decade", "recent decade"]):
        return "last 10 years"

    year_range = re.search(r"(20\d{2}|19\d{2})\s*[-到至]\s*(20\d{2}|19\d{2})", query)
    if year_range:
        return f"{year_range.group(1)}-{year_range.group(2)}"

    since_year = re.search(r"(?:since|自|从)\s*(20\d{2}|19\d{2})", query, re.IGNORECASE)
    if since_year:
        return f"since {since_year.group(1)}"

    return DEFAULT_TIME_RANGE


def _extract_mapped_terms(query: str) -> dict[str, list[str]]:
    terms = {"method": [], "task": [], "domain": []}
    for source, target, category in TERM_MAPPINGS:
        if source.lower() in query.lower():
            terms[category].append(target)
    return {key: _dedupe(value) for key, value in terms.items()}


def _extract_english_terms(query: str) -> list[str]:
    terms: list[str] = []
    for match in re.finditer(r"[A-Za-z][A-Za-z0-9]*(?:[- ][A-Za-z0-9]+)*", query):
        term = match.group(0).strip()
        if term.lower() not in ENGLISH_STOPWORDS and len(term) > 1:
            terms.append(term)
    return _dedupe(terms)


def _fallback_keywords(query: str) -> list[str]:
    cleaned = re.sub(
        r"(有哪些|是什么|什么是|代表性论文|研究|要求包含|中的|在|和|或|的|问题|方法)",
        " ",
        query,
    )
    chinese_chunks = re.findall(r"[\u4e00-\u9fff]{2,}", cleaned)
    english_terms = _extract_english_terms(query)
    return _dedupe([*english_terms, *chinese_chunks])[:4]


def _build_research_topic(query: str, terms: dict[str, list[str]]) -> str:
    pieces = _dedupe([*terms["method"], *terms["domain"], *terms["task"]])
    if pieces:
        return " ".join(pieces[:5])

    english_terms = _extract_english_terms(query)
    if english_terms:
        return " ".join(english_terms[:5])

    fallback = _fallback_keywords(query)
    if fallback:
        return " ".join(fallback)

    return query[:80]


def _join_or_default(items: list[str]) -> str:
    return ", ".join(_dedupe(items)) if items else "not specified"


def _make_sub_queries(
    research_topic: str,
    methods: list[str],
    tasks: list[str],
    domains: list[str],
    query: str,
) -> list[str]:
    method_text = " ".join(methods[:2])
    task_text = " ".join(tasks[:2])
    domain_text = " ".join(domains[:2])
    fallback_terms = " ".join(_fallback_keywords(query))
    if not fallback_terms:
        fallback_terms = research_topic

    candidates = [
        research_topic,
        f"{research_topic} {method_text}" if method_text else f"{research_topic} algorithm",
        f"{research_topic} {task_text}" if task_text else f"{research_topic} application",
        f"{research_topic} {domain_text}" if domain_text else f"{fallback_terms} scholarly search",
        f"{research_topic} benchmark survey literature review",
        f"{fallback_terms} scientific literature retrieval",
        f"{fallback_terms} paper survey algorithm application",
    ]
    return _dedupe(candidates)[:5]


def understand_query(query: str) -> dict[str, Any]:
    """Extract a domain-agnostic academic query understanding object."""
    normalized = query.strip()
    if not normalized:
        return {
            "research_topic": "",
            "method_constraints": "",
            "task_constraints": "",
            "time_range": DEFAULT_TIME_RANGE,
            "domain_constraints": "",
            "sub_queries": [],
        }

    terms = _extract_mapped_terms(normalized)
    methods = terms["method"]
    tasks = terms["task"]
    domains = terms["domain"]

    # Preserve unknown uppercase technical abbreviations as method candidates.
    for abbreviation in re.findall(r"[A-Z]{2,}", normalized):
        if abbreviation not in methods:
            methods.append(abbreviation)

    research_topic = _build_research_topic(normalized, terms)
    sub_queries = _make_sub_queries(research_topic, methods, tasks, domains, normalized)

    return {
        "research_topic": research_topic,
        "method_constraints": _join_or_default(methods),
        "task_constraints": _join_or_default(tasks),
        "time_range": _extract_time_range(normalized),
        "domain_constraints": _join_or_default(domains),
        "sub_queries": sub_queries,
    }
