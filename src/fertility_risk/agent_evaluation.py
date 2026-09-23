"""Repeatable offline evaluations for the deterministic agent planner."""

from __future__ import annotations

from typing import Any

from .planner import TOOLS_BY_INTENT, plan_message

EVALUATION_CASES = [
    ("Is profile ka risk score explain karo", "assess_risk"),
    ("Predict this synthetic record", "assess_risk"),
    ("What is the probability for this profile?", "assess_risk"),
    ("Compare both scenarios aur farak batao", "compare_scenarios"),
    ("Baseline versus comparison", "compare_scenarios"),
    ("What is the difference between these records?", "compare_scenarios"),
    ("Model validation and limitations kya hain?", "methodology"),
    ("Explain the threshold", "methodology"),
    ("Can DHS data be committed?", "methodology"),
    ("Tell me about this project", "methodology"),
]


def evaluate_planner() -> dict[str, Any]:
    """Measure intent accuracy and verify that tool mappings cannot drift."""

    rows = []
    correct = 0
    tool_alignment = True
    for message, expected in EVALUATION_CASES:
        plan = plan_message(message)
        passed = plan.intent == expected
        correct += int(passed)
        tool_alignment &= plan.required_tools == TOOLS_BY_INTENT[plan.intent]
        rows.append(
            {
                "message": message,
                "expected": expected,
                "predicted": plan.intent,
                "passed": passed,
                "provider": plan.provider,
            }
        )

    accuracy = correct / len(EVALUATION_CASES)
    return {
        "cases": len(EVALUATION_CASES),
        "correct": correct,
        "intent_accuracy": accuracy,
        "tool_alignment": tool_alignment,
        "safe_unknown_fallback": plan_message("hello there").intent == "methodology",
        "passed": accuracy == 1.0 and tool_alignment,
        "details": rows,
    }
