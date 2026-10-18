import pytest
from pydantic import ValidationError

from backend.main import SearchRequest


def test_efficient_mode_is_supported():
    assert SearchRequest(query="graph neural networks", mode="efficient").mode == "efficient"
    assert SearchRequest(query="graph neural networks", mode="helios").mode == "helios"


def test_relational_retrieval_is_default_and_lean_remains_available():
    assert SearchRequest(query="graph neural networks").retrieval_profile == "relational"
    assert SearchRequest(
        query="graph neural networks", retrieval_profile="lean"
    ).retrieval_profile == "lean"


def test_unknown_mode_is_rejected():
    with pytest.raises(ValidationError):
        SearchRequest(query="graph neural networks", mode="unbounded")

    with pytest.raises(ValidationError):
        SearchRequest(query="graph neural networks", retrieval_profile="unbounded")
