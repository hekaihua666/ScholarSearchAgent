from backend.helios_lite import HELIOSLiteController, HELIOSState, admit_candidates


def state(**updates):
    base = dict(query_intent="broad", hard_constraints=["task"], soft_preferences=[],
                coverage_vector={"task": 0.2}, high_confidence_papers=["p1"],
                unexpanded_papers=["p1"], api_budget_remaining=8)
    base.update(updates)
    return HELIOSState(**base)


def test_budget_forces_stop():
    action, _ = HELIOSLiteController().choose(state(api_budget_remaining=0))
    assert action.action == "stop"


def test_cost_changes_action_utility():
    s = state()
    costly = HELIOSLiteController(use_cost=True).estimate("second_round_search", s)
    free = HELIOSLiteController(use_cost=False).estimate("second_round_search", s)
    assert free.predicted_utility > costly.predicted_utility


def test_admission_rejects_unsupported_candidate():
    accepted, rejected = admit_candidates([{"title": "noise"}], enabled=True)
    assert not accepted and rejected[0]["reason"] == "insufficient_query_evidence"
