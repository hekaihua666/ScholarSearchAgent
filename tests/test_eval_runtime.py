import asyncio
from datetime import datetime

from backend.eval_runtime import EvaluationSession, evaluation_session, frozen_call


def test_cutoff_filters_exact_future_and_uses_jan1_for_year_only(tmp_path):
    session = EvaluationSession.create(
        "bench", "q1", datetime(2023, 6, 15), "live", tmp_path / "snapshot.json"
    )
    papers = [
        {"title": "old", "publication_date": "2023-06-01", "year": 2023},
        {"title": "future", "publication_date": "2023-07-01", "year": 2023},
        {"title": "year-only", "year": 2023},
        {"title": "missing"},
    ]
    kept = session.filter_papers(papers, "unit")
    assert [paper["title"] for paper in kept] == ["old", "year-only"]
    assert session.date_filter_stats["unit"]["filtered_future"] == 1
    assert session.date_filter_stats["unit"]["filtered_missing_date"] == 1


def test_frozen_call_replays_without_executing_operation(tmp_path):
    path = tmp_path / "snapshot.json"
    live = EvaluationSession.create("bench", "q1", datetime(2023, 1, 1), "live", path)

    async def record():
        with evaluation_session(live):
            value = await frozen_call("source", {"q": "x"}, lambda: asyncio.sleep(0, result=[1, 2]))
        live.save()
        return value

    assert asyncio.run(record()) == [1, 2]

    replay = EvaluationSession.create("bench", "q1", datetime(2023, 1, 1), "replay", path)

    async def fail_if_called():
        raise AssertionError("network operation should not run during replay")

    async def replay_value():
        with evaluation_session(replay):
            return await frozen_call("source", {"q": "x"}, fail_if_called)

    assert asyncio.run(replay_value()) == [1, 2]

