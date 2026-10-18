"""Small dependency-light probability calibrators and calibration metrics."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def _clip(p: float) -> float:
    return min(1 - 1e-9, max(1e-9, float(p)))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x); return 1 / (1 + z)
    z = math.exp(x); return z / (1 + z)


@dataclass
class PlattCalibrator:
    slope: float = 1.0
    intercept: float = 0.0
    method: str = "platt"

    def fit(self, scores: Iterable[float], labels: Iterable[int], *, steps: int = 3000, lr: float = 0.05) -> "PlattCalibrator":
        xs = [float(x) for x in scores]; ys = [int(y) for y in labels]
        if not xs or len(xs) != len(ys): raise ValueError("scores/labels must be non-empty and aligned")
        pos = max(1, sum(ys)); neg = max(1, len(ys) - sum(ys))
        self.intercept = math.log(pos / neg)
        for step in range(steps):
            ga = gb = 0.0
            for x, y in zip(xs, ys):
                p = _sigmoid(self.slope * x + self.intercept)
                ga += (p - y) * x; gb += p - y
            rate = lr / math.sqrt(1 + step / 100)
            self.slope -= rate * ga / len(xs); self.intercept -= rate * gb / len(xs)
        return self

    def predict_one(self, score: float) -> float:
        return _sigmoid(self.slope * float(score) + self.intercept)

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "slope": self.slope, "intercept": self.intercept}


@dataclass
class TemperatureCalibrator:
    temperature: float = 1.0
    method: str = "temperature"

    def fit(self, scores: Iterable[float], labels: Iterable[int]) -> "TemperatureCalibrator":
        xs = [_clip(x) for x in scores]; ys = [int(y) for y in labels]
        best = (float("inf"), 1.0)
        for t100 in range(20, 501, 5):
            t = t100 / 100
            loss = 0.0
            for x, y in zip(xs, ys):
                logit = math.log(x / (1 - x)) / t; p = _clip(_sigmoid(logit))
                loss -= y * math.log(p) + (1 - y) * math.log(1 - p)
            if loss < best[0]: best = (loss, t)
        self.temperature = best[1]; return self

    def predict_one(self, score: float) -> float:
        p = _clip(score); return _sigmoid(math.log(p / (1 - p)) / self.temperature)

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "temperature": self.temperature}


@dataclass
class IsotonicCalibrator:
    thresholds: list[float] | None = None
    values: list[float] | None = None
    method: str = "isotonic"

    def fit(self, scores: Iterable[float], labels: Iterable[int]) -> "IsotonicCalibrator":
        pairs = sorted((float(x), int(y)) for x, y in zip(scores, labels))
        if not pairs: raise ValueError("empty calibration data")
        blocks: list[list[float]] = []
        for x, y in pairs:
            blocks.append([x, x, float(y), 1.0])
            while len(blocks) >= 2 and blocks[-2][2] / blocks[-2][3] > blocks[-1][2] / blocks[-1][3]:
                b = blocks.pop(); a = blocks.pop()
                blocks.append([a[0], b[1], a[2] + b[2], a[3] + b[3]])
        self.thresholds = [b[1] for b in blocks]
        self.values = [b[2] / b[3] for b in blocks]
        return self

    def predict_one(self, score: float) -> float:
        assert self.thresholds and self.values
        for threshold, value in zip(self.thresholds, self.values):
            if score <= threshold: return value
        return self.values[-1]

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "thresholds": self.thresholds, "values": self.values}


def calibrator_from_dict(raw: dict[str, Any]) -> Any:
    method = raw.get("method")
    if method == "platt": return PlattCalibrator(float(raw["slope"]), float(raw["intercept"]))
    if method == "temperature": return TemperatureCalibrator(float(raw["temperature"]))
    if method == "isotonic": return IsotonicCalibrator(list(raw["thresholds"]), list(raw["values"]))
    raise ValueError(f"unknown calibrator: {method}")


def save_calibrator(calibrator: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibrator.to_dict(), indent=2), encoding="utf-8")


def calibration_metrics(probabilities: Iterable[float], labels: Iterable[int], bins: int = 10) -> dict[str, float]:
    ps = [_clip(p) for p in probabilities]; ys = [int(y) for y in labels]
    if not ps or len(ps) != len(ys): raise ValueError("probabilities/labels must align")
    brier = sum((p - y) ** 2 for p, y in zip(ps, ys)) / len(ps)
    ece = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        idx = [j for j, p in enumerate(ps) if lo <= p < hi or (i == bins - 1 and p == 1)]
        if idx:
            conf = sum(ps[j] for j in idx) / len(idx); acc = sum(ys[j] for j in idx) / len(idx)
            ece += len(idx) / len(ps) * abs(conf - acc)
    return {"brier": round(brier, 6), "ece": round(ece, 6)}
