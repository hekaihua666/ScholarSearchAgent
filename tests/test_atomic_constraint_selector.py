from __future__ import annotations

from pathlib import Path

from backend.selection.atomic_constraint_selector import ACEConfig, judge_paper
from backend.selection.cross_encoder_selector import CrossEncoderSelector
from backend.selection.evidence_verifier import needs_verification, validate_evidence_span
from backend.selection.pasa_selector_adapter import parse_selector_output


def graph(*criteria):
    return {"criteria": list(criteria)}


def criterion(cid: str, description: str, kind: str, importance: str = "required"):
    return {
        "criterion_id": cid,
        "description": description,
        "type": kind,
        "importance": importance,
        "verification_target": "title|abstract",
        "unknown_allowed": True,
    }


def test_ace_hard_gate_cannot_be_rescued_by_trace_score() -> None:
    paper = {"title": "A 2024 method", "abstract": "A highly cited method.", "year": 2024, "final_score": 1.0}
    result = judge_paper(paper, graph(criterion("c1", "2018-2020", "time")))
    assert result["hard_gate"] == "fail"
    assert result["raw_probability"] == 0.0
    assert result["decision"] == "reject"


def test_unknown_is_not_contradiction_and_geometric_is_supported() -> None:
    paper = {"title": "Graph models", "abstract": "We study representation learning.", "year": 2020}
    result = judge_paper(
        paper,
        graph(criterion("c1", "target network", "method")),
        ACEConfig(unknown_score=0.4, aggregate="geometric"),
    )
    assert result["hard_gate"] == "unknown"
    assert 0 < result["criterion_score"] < 1
    assert needs_verification(result)


def test_preferred_contradiction_does_not_fail_required_gate() -> None:
    paper = {"title": "Target Networks", "abstract": "We use target networks.", "year": 2024}
    result = judge_paper(
        paper,
        graph(
            criterion("c1", "target networks", "method", "required"),
            criterion("c2", "2018-2020", "time", "preferred"),
        ),
    )
    assert result["hard_gate"] != "fail"
    assert result["criteria"][1]["state"] == "CONTRADICTED"


def test_evidence_span_must_be_verbatim() -> None:
    paper = {"title": "Target Networks", "abstract": "We evaluate target networks for deep reinforcement learning."}
    assert validate_evidence_span("evaluate target networks", paper)
    assert not validate_evidence_span("invented evidence", paper)


def test_relation_entailment_has_evidence() -> None:
    paper = {"title": "A New Agent", "abstract": "We propose a novel agent for literature retrieval."}
    result = judge_paper(paper, graph(criterion("c1", "introduces", "relation")))
    assert result["criteria"][0]["state"] == "ENTAILED"
    assert result["criteria"][0]["evidence_span"]


def test_pasa_adapter_parses_and_rejects_ambiguous() -> None:
    assert parse_selector_output("Yes, relevant")["label"] == 1
    assert parse_selector_output("No, not relevant")["label"] == 0
    assert not parse_selector_output("uncertain")["parsed"]


def test_cross_encoder_missing_model_falls_back(tmp_path: Path) -> None:
    selector = CrossEncoderSelector(
        "definitely/not-a-local-model", tmp_path / "cache.json", local_files_only=True
    )
    scores, metrics = selector.predict([("graph anomaly", "graph anomaly detection")])
    assert 0 <= scores[0] <= 1
    assert metrics["runtime"].startswith("lexical_fallback")
    scores2, metrics2 = selector.predict([("graph anomaly", "graph anomaly detection")])
    assert scores2 == scores
    assert metrics2["cache_hits"] == 1
