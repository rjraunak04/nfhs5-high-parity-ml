from fertility_risk.knowledge import load_knowledge, search_knowledge


def test_knowledge_base_contains_only_approved_sources():
    chunks = load_knowledge()
    sources = {chunk["source"] for chunk in chunks}

    assert chunks
    assert sources == {
        "docs/ARCHITECTURE.md",
        "docs/DATA_CARD.md",
        "docs/MODEL_CARD.md",
        "research/README.md",
    }


def test_threshold_question_retrieves_threshold_evidence():
    evidence = search_knowledge("0.4890 aur 0.4694 threshold ka difference kya hai?")

    assert evidence
    assert evidence[0]["source"] == "docs/MODEL_CARD.md"
    assert evidence[0]["section"] == "Threshold clarification"
    assert "0.4890" in evidence[0]["excerpt"]


def test_data_policy_question_retrieves_data_card():
    evidence = search_knowledge("Can raw DHS microdata be committed to GitHub?")

    assert evidence
    assert any(item["section"] == "Repository data policy" for item in evidence)
