from types import SimpleNamespace

import pytest

from fertility_risk.planner import LLMIntent, plan_message, plan_message_with_fallback


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Is profile ka risk score explain karo", "assess_risk"),
        ("Compare both scenarios aur farak batao", "compare_scenarios"),
        ("Model validation and limitations kya hain?", "methodology"),
        ("Tell me about this project", "methodology"),
    ],
)
def test_natural_language_planner(message, expected):
    plan = plan_message(message)
    assert plan.intent == expected
    assert plan.required_tools
    assert 0 <= plan.confidence <= 1


class StubResponses:
    def __init__(self, intent="methodology", raises=False):
        self.intent = intent
        self.raises = raises
        self.kwargs = None

    def parse(self, **kwargs):
        self.kwargs = kwargs
        if self.raises:
            raise RuntimeError("provider unavailable")
        parsed = LLMIntent(intent=self.intent, reason="Structured test decision")
        return SimpleNamespace(output_parsed=parsed)


def test_llm_planner_can_only_select_allow_listed_intent():
    responses = StubResponses(intent="assess_risk")
    client = SimpleNamespace(responses=responses)

    plan = plan_message_with_fallback("Please understand this profile", client=client)

    assert plan.intent == "assess_risk"
    assert plan.provider == "openai"
    assert plan.required_tools == ["predict_risk", "explain_prediction"]
    assert responses.kwargs["text_format"] is LLMIntent
    assert responses.kwargs["store"] is False


def test_llm_failure_falls_back_without_breaking_agent():
    client = SimpleNamespace(responses=StubResponses(raises=True))
    plan = plan_message_with_fallback("Compare both scenarios", client=client)

    assert plan.intent == "compare_scenarios"
    assert plan.provider == "deterministic"
