"""Allow-listed tools available to the single-agent research assistant."""

from __future__ import annotations

from typing import Any

from .inference import predict_records
from .knowledge import search_knowledge
from .schemas import PredictionRequest

METHODOLOGY_SUMMARY = {
    "outcome": "Observed high parity at interview, defined as children ever born >= 3.",
    "public_features": [
        "current age",
        "residence",
        "education",
        "wealth quintile",
        "current union status",
    ],
    "validation": (
        "The research workflow uses leakage-safe preprocessing, nested validation, a protected "
        "holdout, and a separately locked geographic validation on Nepal DHS data."
    ),
    "boundary": (
        "The public artifact is trained on synthetic data. It is a software demonstration, not "
        "a clinical tool, causal model, or forecast of future fertility."
    ),
}


def predict_risk(bundle: dict[str, Any], record: PredictionRequest) -> dict[str, Any]:
    """Score one already-validated record with the versioned ML pipeline."""

    return predict_records(bundle, [record.model_dump()])[0]


def explain_prediction(
    bundle: dict[str, Any], record: PredictionRequest, prediction: dict[str, Any]
) -> dict[str, Any]:
    """Return a conservative, non-causal explanation of a demo prediction.

    The demo bundle exposes global feature importance, not local SHAP values. The tool therefore
    reports that distinction explicitly instead of presenting global importance as person-specific.
    """

    importance = bundle["metadata"].get("global_feature_importance", {})
    ranked = sorted(importance.items(), key=lambda item: item[1], reverse=True)
    return {
        "classification": prediction["classification"],
        "probability": prediction["probability"],
        "provided_inputs": record.model_dump(),
        "globally_important_features": [name for name, _ in ranked[:3]],
        "explanation_scope": (
            "Global model importance only; these features are not local contributions or causes."
        ),
    }


def compare_scenarios(
    bundle: dict[str, Any], baseline: PredictionRequest, comparison: PredictionRequest
) -> dict[str, Any]:
    """Score two synthetic scenarios and report their model-score difference."""

    baseline_result, comparison_result = predict_records(
        bundle, [baseline.model_dump(), comparison.model_dump()]
    )
    return {
        "baseline": baseline_result,
        "comparison": comparison_result,
        "probability_difference": round(
            comparison_result["probability"] - baseline_result["probability"], 6
        ),
        "interpretation": (
            "A model-score comparison between synthetic records; it is not a causal effect."
        ),
    }


def get_methodology(question: str | None = None) -> dict[str, Any]:
    """Retrieve cited project evidence rather than generating unsupported claims."""

    evidence = search_knowledge(question)
    if evidence:
        answer = " ".join(
            f"[{item['source']} — {item['section']}] {item['excerpt']}" for item in evidence[:2]
        )
    else:
        answer = f"{METHODOLOGY_SUMMARY['validation']} {METHODOLOGY_SUMMARY['boundary']}"
    return {
        "question": question,
        "answer": answer,
        "evidence": evidence,
        **METHODOLOGY_SUMMARY,
    }


def generate_report(
    *,
    intent: str,
    result: dict[str, Any],
    model_version: str,
    demo_only: bool,
    tools: list[str],
    disclaimer: str,
) -> dict[str, Any]:
    """Build a portable audit report without storing personal or server-side history."""

    return {
        "report_type": "NFHS high-parity research-demo agent report",
        "intent": intent,
        "model_version": model_version,
        "demo_only": demo_only,
        "tools_executed": tools,
        "result": result,
        "disclaimer": disclaimer,
    }
