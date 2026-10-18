from backend.selection.linear_model import LinearSelectorModel, text_features


def test_linear_model_learns_simple_text_signal(tmp_path):
    rows = []
    for i in range(20):
        rows.append({"query": "graph anomaly detection", "title": "graph anomaly detection", "abstract": "detect graph anomalies", "label": 1, "sample_weight": 1})
        rows.append({"query": "graph anomaly detection", "title": "language translation", "abstract": "machine translation", "label": 0, "sample_weight": 1})
    model = LinearSelectorModel.fit(rows, steps=500)
    assert model.predict_one(rows[0]["query"], rows[0]) > model.predict_one(rows[1]["query"], rows[1])
    assert len(text_features("a b", {"title": "a", "abstract": "b"})) == len(model.feature_names)
