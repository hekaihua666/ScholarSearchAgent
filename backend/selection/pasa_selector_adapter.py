"""Parsing adapter for PaSa Selector-style Yes/No generations."""

from __future__ import annotations

import re
from typing import Any


MODEL_ID = "bytedance-research/pasa-7b-selector"


def parse_selector_output(text: str) -> dict[str, Any]:
    normalized = re.sub(r"\s+", " ", (text or "").strip()).lower()
    if re.search(r"\b(yes|relevant|include)\b", normalized) and not re.search(r"\b(no|not relevant|exclude)\b", normalized):
        return {"label": 1, "score": 1.0, "parsed": True}
    if re.search(r"\b(no|not relevant|exclude)\b", normalized):
        return {"label": 0, "score": 0.0, "parsed": True}
    match = re.search(r"(?:score|probability)\s*[:=]\s*(0(?:\.\d+)?|1(?:\.0+)?)", normalized)
    if match:
        score = float(match.group(1))
        return {"label": int(score >= 0.5), "score": score, "parsed": True}
    return {"label": None, "score": None, "parsed": False}


def runtime_status() -> dict[str, Any]:
    try:
        import torch  # type: ignore
        import transformers  # type: ignore
    except ImportError as exc:
        return {"available": False, "model": MODEL_ID, "reason": f"missing_dependency:{exc.name}"}
    if not torch.cuda.is_available():
        return {"available": False, "model": MODEL_ID, "reason": "7B_model_not_feasible_on_cpu_only_runtime"}
    return {"available": True, "model": MODEL_ID, "transformers": transformers.__version__}
