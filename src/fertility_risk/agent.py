"""Deterministic single-agent orchestrator for the research demonstration."""

from __future__ import annotations

from typing import Any

from .agent_tools import (
    compare_scenarios,
    explain_prediction,
    generate_report,
    get_methodology,
    predict_risk,
)
from .schemas import AgentRequest, AgentResponse, AgentToolCall

DISCLAIMER = (
    "Research demonstration only. This output is not medical advice, a fertility forecast, "
    "or evidence of causation."
)


class ResearchAgent:
    """Route a validated request to a small allow-list of auditable tools."""

    def __init__(self, bundle: dict[str, Any]):
        self.bundle = bundle
        self.metadata = bundle["metadata"]

    def run(self, request: AgentRequest) -> AgentResponse:
        if request.intent == "assess_risk":
            if request.record is None:
                raise ValueError("record is required for assess_risk")
            prediction = predict_risk(self.bundle, request.record)
            explanation = explain_prediction(self.bundle, request.record, prediction)
            result = {"prediction": prediction, "explanation": explanation}
            calls = [AgentToolCall(tool="predict_risk"), AgentToolCall(tool="explain_prediction")]
            answer = (
                f"The synthetic demo score is {prediction['probability']:.1%} and is classified "
                f"as {prediction['classification']} at the locked demo threshold."
            )
        elif request.intent == "compare_scenarios":
            if request.record is None or request.comparison_record is None:
                raise ValueError(
                    "record and comparison_record are required for compare_scenarios"
                )
            result = compare_scenarios(self.bundle, request.record, request.comparison_record)
            calls = [AgentToolCall(tool="compare_scenarios")]
            difference = result["probability_difference"]
            answer = (
                "The comparison scenario's synthetic model score is "
                f"{difference:+.1%} relative to the baseline. This is not a causal effect."
            )
        else:
            result = get_methodology(request.question)
            calls = [AgentToolCall(tool="get_methodology")]
            answer = result["answer"]

        tools = [call.tool for call in calls]
        report = generate_report(
            intent=request.intent,
            result=result,
            model_version=str(self.metadata["model_version"]),
            demo_only=bool(self.metadata["demo_only"]),
            tools=tools,
            disclaimer=DISCLAIMER,
        )
        calls.append(AgentToolCall(tool="generate_report"))
        return AgentResponse(
            intent=request.intent,
            answer=answer,
            result=result,
            tool_calls=calls,
            model_version=str(self.metadata["model_version"]),
            demo_only=bool(self.metadata["demo_only"]),
            disclaimer=DISCLAIMER,
            report=report,
        )
