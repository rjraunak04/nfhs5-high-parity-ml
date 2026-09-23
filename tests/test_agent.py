import pytest
from pydantic import ValidationError

from fertility_risk.agent import ResearchAgent
from fertility_risk.schemas import AgentRequest


def valid_record():
    return {
        "current_age": 31,
        "residence": 2,
        "education": 2,
        "wealth": 3,
        "in_union": 1,
    }


def test_agent_assessment_is_auditable(demo_bundle):
    response = ResearchAgent(demo_bundle).run(
        AgentRequest(intent="assess_risk", record=valid_record())
    )

    assert 0 <= response.result["prediction"]["probability"] <= 1
    assert [call.tool for call in response.tool_calls] == [
        "predict_risk",
        "explain_prediction",
        "generate_report",
    ]
    assert response.demo_only is True
    assert "not medical advice" in response.disclaimer
    assert response.report["model_version"] == response.model_version
    assert response.report["tools_executed"] == ["predict_risk", "explain_prediction"]


def test_agent_compares_two_scenarios(demo_bundle):
    comparison = {**valid_record(), "education": 3, "wealth": 5}
    response = ResearchAgent(demo_bundle).run(
        AgentRequest(
            intent="compare_scenarios",
            record=valid_record(),
            comparison_record=comparison,
        )
    )

    assert "probability_difference" in response.result
    assert response.tool_calls[0].tool == "compare_scenarios"


def test_agent_methodology_uses_curated_content(demo_bundle):
    response = ResearchAgent(demo_bundle).run(
        AgentRequest(intent="methodology", question="How was the model validated?")
    )
    assert "Hyperparameter selection inside inner folds" in response.answer
    assert response.result["evidence"][0]["source"] == "docs/MODEL_CARD.md"
    assert response.tool_calls[0].tool == "get_methodology"


def test_agent_rejects_unknown_intent():
    with pytest.raises(ValidationError):
        AgentRequest(intent="diagnose", record=valid_record())


def test_agent_requires_record_for_assessment():
    with pytest.raises(ValidationError, match="record is required"):
        AgentRequest(intent="assess_risk")
