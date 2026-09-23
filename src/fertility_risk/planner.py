"""Transparent natural-language planning for the single research agent."""

from __future__ import annotations

import os
import re
from typing import Literal, Protocol

from pydantic import BaseModel

from .schemas import AgentPlan

INTENT_KEYWORDS = {
    "compare_scenarios": {
        "compare",
        "comparison",
        "difference",
        "versus",
        "vs",
        "scenario",
        "farak",
        "antar",
        "muqabla",
    },
    "assess_risk": {
        "assess",
        "assessment",
        "classify",
        "predict",
        "prediction",
        "probability",
        "risk",
        "score",
        "profile",
        "explain",
        "batao",
    },
    "methodology": {
        "method",
        "methodology",
        "validation",
        "validated",
        "limitation",
        "limitations",
        "data",
        "dataset",
        "feature",
        "threshold",
        "model",
        "research",
        "kaise",
    },
}

TOOLS_BY_INTENT = {
    "assess_risk": ["predict_risk", "explain_prediction"],
    "compare_scenarios": ["compare_scenarios"],
    "methodology": ["get_methodology"],
}


class LLMIntent(BaseModel):
    """The only decision an optional LLM is allowed to make."""

    intent: Literal["assess_risk", "compare_scenarios", "methodology"]
    reason: str


class ResponsesParser(Protocol):
    def parse(self, **kwargs): ...


class OpenAIClient(Protocol):
    responses: ResponsesParser


def plan_message(message: str) -> AgentPlan:
    """Classify a short English/Hinglish request using an auditable keyword score."""

    tokens = set(re.findall(r"[a-z0-9]+", message.lower()))
    scores = {
        intent: len(tokens.intersection(keywords))
        for intent, keywords in INTENT_KEYWORDS.items()
    }

    # Comparison is the most specific intent and wins when explicitly requested.
    if scores["compare_scenarios"]:
        intent = "compare_scenarios"
    elif scores["assess_risk"] > scores["methodology"]:
        intent = "assess_risk"
    else:
        intent = "methodology"

    matched = sorted(tokens.intersection(INTENT_KEYWORDS[intent]))
    confidence = 0.95 if len(matched) >= 2 else 0.8 if matched else 0.6
    reason = (
        f"Matched request terms: {', '.join(matched)}."
        if matched
        else "No action keyword matched; routed to the safe methodology workflow."
    )
    return AgentPlan(
        intent=intent,
        confidence=confidence,
        reason=reason,
        required_tools=TOOLS_BY_INTENT[intent],
    )


def llm_planning_enabled() -> bool:
    """Require an explicit feature flag as well as a key to avoid surprise API usage."""

    enabled = os.getenv("FERTILITY_ENABLE_LLM", "false").strip().lower()
    return enabled in {"1", "true", "yes"} and bool(os.getenv("OPENAI_API_KEY"))


def plan_message_with_fallback(
    message: str,
    *,
    client: OpenAIClient | None = None,
    model: str | None = None,
) -> AgentPlan:
    """Use structured LLM routing when enabled; otherwise keep deterministic behavior."""

    if client is None and not llm_planning_enabled():
        return plan_message(message)

    try:
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        response = client.responses.parse(
            model=model or os.getenv("FERTILITY_LLM_MODEL", "gpt-4o-mini"),
            input=[
                {
                    "role": "system",
                    "content": (
                        "Route a research-demo request to exactly one allowed intent. "
                        "Use assess_risk for one-profile scores or explanations, "
                        "compare_scenarios for comparisons, and methodology for project, "
                        "data, validation, limitation, or unsupported questions. "
                        "Never calculate a probability or propose a different tool."
                    ),
                },
                {"role": "user", "content": message},
            ],
            text_format=LLMIntent,
            store=False,
        )
        parsed = response.output_parsed
        if parsed is None:
            return plan_message(message)
        return AgentPlan(
            intent=parsed.intent,
            confidence=0.9,
            reason=parsed.reason,
            required_tools=TOOLS_BY_INTENT[parsed.intent],
            provider="openai",
        )
    except Exception:
        # Public deployments must remain available when the provider is unavailable.
        return plan_message(message)
