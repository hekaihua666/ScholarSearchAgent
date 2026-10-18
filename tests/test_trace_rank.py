from backend.trace_rank import build_relation_graph, diversify, fuse_candidates, score_candidates


def _paper(title, rank, query, doi="", citations=0, year=2024):
    return {"title": title, "abstract": title + " retrieval benchmark method", "doi": doi,
            "source_query": query, "source_api": "OpenAlex", "_openalex_rank": rank,
            "cited_by_count": citations, "year": year, "authors": ["A"]}


def test_fusion_retains_multiple_hits():
    papers = [_paper("Agentic paper retrieval", 1, "agent retrieval", "10/x"),
              _paper("Agentic paper retrieval", 3, "iterative search", "10/x")]
    fused = fuse_candidates(papers)
    assert len(fused) == 1
    assert len(fused[0]["retrieval_hits"]) == 2


def test_fusion_merges_doi_and_title_only_cross_source_records():
    with_doi = _paper("Same Research Paper", 1, "q1", "10/example")
    title_only = _paper("Same Research Paper", 2, "q2")
    title_only["source_api"] = "arXiv"
    fused = fuse_candidates([with_doi, title_only])
    assert len(fused) == 1
    assert len(fused[0]["retrieval_hits"]) == 2


def test_fusion_merges_literal_backslash_n_title_variant():
    first = _paper("A Unified Model for\\n Scientific Search", 1, "q1")
    second = _paper("A Unified Model for Scientific Search", 2, "q2")
    second["source_api"] = "arXiv"
    fused = fuse_candidates([first, second])
    assert len(fused) == 1
    assert len(fused[0]["retrieval_hits"]) == 2


def test_constraint_match_outranks_unrelated_high_citation():
    understanding = {"research_topic": "academic paper retrieval", "method_constraints": "iterative agent",
                     "task_constraints": "complex query search", "domain_constraints": "scientific literature"}
    relevant = _paper("Iterative agent for academic paper retrieval", 2, "academic retrieval", citations=10)
    unrelated = _paper("Protein structure prediction", 1, "academic retrieval", citations=10000)
    ranked = score_candidates(fuse_candidates([relevant, unrelated]), understanding)
    assert ranked[0]["title"].startswith("Iterative agent")


def test_diversity_and_graph_are_bounded():
    u = {"research_topic": "paper retrieval", "method_constraints": "agent", "task_constraints": "", "domain_constraints": ""}
    ranked = score_candidates(fuse_candidates([_paper(f"Agent paper retrieval {i}", i + 1, "q") for i in range(6)]), u)
    selected = diversify(ranked, limit=3)
    graph = build_relation_graph(selected)
    assert len(selected) == 3
    assert len(graph["nodes"]) == 3
