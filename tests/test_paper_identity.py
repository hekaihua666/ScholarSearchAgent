from backend.paper_identity import (
    match_identity, normalize_arxiv_id, normalize_doi, official_pasa_title,
)


def test_arxiv_versions_and_arxiv_doi_normalize_to_same_id():
    assert normalize_arxiv_id("https://arxiv.org/pdf/2302.09419v2.pdf") == "2302.09419"
    assert normalize_arxiv_id("https://doi.org/10.48550/arXiv.2302.09419") == "2302.09419"


def test_doi_normalization():
    assert normalize_doi("https://doi.org/10.1145/3605943") == "10.1145/3605943"


def test_identity_checks_all_fields_not_only_openalex_id():
    predicted = {
        "openalex_id": "https://openalex.org/W123",
        "doi": "https://doi.org/10.48550/arXiv.2302.09419",
        "title": "Different display title",
    }
    gold = {"arxiv_id": "2302.09419v1", "title": "Gold title"}
    assert match_identity(predicted, gold) == "arxiv_id"


def test_official_pasa_title_ignores_digits_and_punctuation():
    assert official_pasa_title("GPT-4: A Test") == official_pasa_title("GPT A Test")

