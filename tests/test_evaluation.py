from backend.evaluation import macro_average, retrieval_metrics


def test_retrieval_metrics():
    row = retrieval_metrics(["a", "b", "c"], ["a", "c", "d"], k=3)
    assert row["true_positive"] == 2
    assert row["precision@3"] == 0.666667
    assert row["recall@3"] == 0.666667


def test_macro_average():
    a = retrieval_metrics(["a"], ["a"], 1)
    b = retrieval_metrics(["x"], ["a"], 1)
    assert macro_average([a, b])["f1@1"] == 0.5
