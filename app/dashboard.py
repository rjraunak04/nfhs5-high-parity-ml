"""Streamlit interface for the synthetic research demonstration model."""

from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from fertility_risk.agent import ResearchAgent
from fertility_risk.constants import (
    EDUCATION_LABELS,
    MODEL_FEATURES,
    RESIDENCE_LABELS,
    UNION_LABELS,
    WEALTH_LABELS,
)
from fertility_risk.inference import load_or_create_demo_bundle, predict_records
from fertility_risk.planner import plan_message_with_fallback
from fertility_risk.schemas import AgentRequest, AgentResponse, PredictionRequest

MODEL_PATH = os.getenv("FERTILITY_MODEL_PATH", "models/demo_transport_model.joblib")

st.set_page_config(page_title="High-Parity ML Demo", page_icon="📊", layout="wide")


@st.cache_resource
def get_bundle():
    return load_or_create_demo_bundle(MODEL_PATH, expected_features=MODEL_FEATURES)


def reverse_lookup(mapping, label):
    """Map a human-readable dashboard label back to its model code."""
    return next(key for key, value in mapping.items() if value == label)


def build_record(prefix: str, *, defaults: dict | None = None) -> PredictionRequest:
    """Render one reusable synthetic-profile form and return validated inputs."""
    values = defaults or {}
    current_age = st.slider(
        "Current age",
        min_value=15,
        max_value=49,
        value=values.get("current_age", 30),
        key=f"{prefix}_age",
    )
    residence_label = st.selectbox(
        "Residence",
        list(RESIDENCE_LABELS.values()),
        index=values.get("residence_index", 0),
        key=f"{prefix}_residence",
    )
    education_label = st.selectbox(
        "Education",
        list(EDUCATION_LABELS.values()),
        index=values.get("education_index", 2),
        key=f"{prefix}_education",
    )
    wealth_label = st.selectbox(
        "Wealth quintile",
        list(WEALTH_LABELS.values()),
        index=values.get("wealth_index", 2),
        key=f"{prefix}_wealth",
    )
    union_label = st.selectbox(
        "Current union status",
        list(UNION_LABELS.values()),
        index=values.get("union_index", 1),
        key=f"{prefix}_union",
    )
    return PredictionRequest(
        current_age=current_age,
        residence=reverse_lookup(RESIDENCE_LABELS, residence_label),
        education=reverse_lookup(EDUCATION_LABELS, education_label),
        wealth=reverse_lookup(WEALTH_LABELS, wealth_label),
        in_union=reverse_lookup(UNION_LABELS, union_label),
    )


def render_agent_response(response: AgentResponse) -> None:
    """Display the answer, structured evidence, and auditable tool trace."""
    st.success(response.answer)
    if response.intent == "assess_risk":
        prediction = response.result["prediction"]
        left, right = st.columns(2)
        left.metric("Synthetic demo probability", f"{prediction['probability']:.1%}")
        right.metric("Classification", prediction["classification"])
    elif response.intent == "compare_scenarios":
        result = response.result
        first, second, difference = st.columns(3)
        first.metric("Baseline", f"{result['baseline']['probability']:.1%}")
        second.metric("Comparison", f"{result['comparison']['probability']:.1%}")
        difference.metric("Score difference", f"{result['probability_difference']:+.1%}")

    with st.expander("Agent evidence and tool trace"):
        st.write("Tools called:", " → ".join(call.tool for call in response.tool_calls))
        st.json(response.result)
        st.caption(
            f"Model: {response.model_version} | Synthetic demo: {response.demo_only}"
        )
    st.download_button(
        "Download agent report (JSON)",
        data=json.dumps(response.report, indent=2),
        file_name="nfhs_agent_report.json",
        mime="application/json",
        key=f"download_{response.intent}",
    )
    st.info(response.disclaimer)


bundle = get_bundle()
metadata = bundle["metadata"]
agent = ResearchAgent(bundle)

st.title("Explainable High-Parity Classification")
st.caption("Production-style ML interface built from NFHS-5/DHS research")

st.warning(
    "Research demonstration only. This score classifies a synthetic representation of observed "
    "high parity at interview; it is not a forecast, diagnosis, or recommendation."
)

single_tab, batch_tab, agent_tab = st.tabs(
    ["Single record", "CSV batch", "Agent assistant"]
)

with single_tab:
    with st.form("prediction_form"):
        current_age = st.slider("Current age", min_value=15, max_value=49, value=30)
        residence_label = st.selectbox("Residence", list(RESIDENCE_LABELS.values()))
        education_label = st.selectbox("Education", list(EDUCATION_LABELS.values()), index=2)
        wealth_label = st.selectbox("Wealth quintile", list(WEALTH_LABELS.values()), index=2)
        union_label = st.selectbox("Current union status", list(UNION_LABELS.values()), index=1)
        submitted = st.form_submit_button("Generate research-demo score", type="primary")

    if submitted:
        request = PredictionRequest(
            current_age=current_age,
            residence=reverse_lookup(RESIDENCE_LABELS, residence_label),
            education=reverse_lookup(EDUCATION_LABELS, education_label),
            wealth=reverse_lookup(WEALTH_LABELS, wealth_label),
            in_union=reverse_lookup(UNION_LABELS, union_label),
        )
        prediction = predict_records(bundle, [request.model_dump()])[0]
        st.metric("Synthetic demo probability", f"{prediction['probability']:.1%}")
        st.progress(prediction["probability"])
        st.write(f"Classification at demo threshold: **{prediction['classification']}**")
        st.caption(
            f"Demo threshold: {prediction['threshold']:.3f} | "
            f"Version: {prediction['model_version']}"
        )

with batch_tab:
    st.subheader("Score a validated CSV batch")
    st.write("Upload 1–500 rows containing exactly these columns:", MODEL_FEATURES)
    st.download_button(
        "Download input template",
        data=pd.DataFrame(columns=MODEL_FEATURES).to_csv(index=False),
        file_name="high_parity_batch_template.csv",
        mime="text/csv",
    )
    upload = st.file_uploader("Choose CSV", type="csv")

    if upload is not None:
        try:
            batch = pd.read_csv(upload)
            if not 1 <= len(batch) <= 500:
                raise ValueError("CSV must contain between 1 and 500 rows.")
            if set(batch.columns) != set(MODEL_FEATURES):
                raise ValueError(f"CSV columns must be exactly: {MODEL_FEATURES}")

            records = [
                PredictionRequest(**row).model_dump()
                for row in batch[MODEL_FEATURES].to_dict(orient="records")
            ]
            result = batch.copy()
            scored = predict_records(bundle, records)
            result["probability"] = [row["probability"] for row in scored]
            result["classification"] = [row["classification"] for row in scored]
            result["model_version"] = [row["model_version"] for row in scored]

            st.success(f"Validated and scored {len(result)} rows.")
            st.dataframe(result, use_container_width=True)
            st.download_button(
                "Download predictions",
                data=result.to_csv(index=False),
                file_name="high_parity_demo_predictions.csv",
                mime="text/csv",
                type="primary",
            )
        except (ValueError, ValidationError, pd.errors.ParserError) as exc:
            st.error(f"The CSV could not be scored: {exc}")

with agent_tab:
    st.subheader("Auditable research agent")
    st.write(
        "Choose a workflow. The agent can use only approved tools and always shows its tool trace."
    )
    workflow = st.radio(
        "Agent workflow",
        [
            "Ask in English/Hinglish",
            "Assess one scenario",
            "Compare two scenarios",
            "Ask about methodology",
        ],
        horizontal=True,
    )

    if workflow == "Ask in English/Hinglish":
        with st.form("agent_natural_language_form"):
            message = st.text_area(
                "Message",
                value="Is profile ka risk score explain karo",
                max_chars=500,
            )
            st.markdown("#### Scenario context")
            record = build_record("agent_chat_record")
            include_comparison = st.checkbox(
                "Include a second scenario for comparison",
                key="agent_chat_include_comparison",
            )
            comparison = None
            if include_comparison:
                st.markdown("#### Comparison scenario")
                comparison = build_record(
                    "agent_chat_comparison",
                    defaults={"current_age": 35, "education_index": 3, "wealth_index": 4},
                )
            run_message = st.form_submit_button("Plan and run", type="primary")

        if run_message:
            plan = plan_message_with_fallback(message)
            with st.expander("Planner decision", expanded=True):
                st.write(f"Selected intent: **{plan.intent}**")
                st.write(f"Planning confidence: **{plan.confidence:.0%}**")
                st.write(f"Planner provider: **{plan.provider}**")
                st.write(plan.reason)
                st.write("Required tools:", " → ".join(plan.required_tools))

            if plan.intent == "compare_scenarios" and comparison is None:
                st.error(
                    "The planner detected a comparison request. Select “Include a second "
                    "scenario for comparison” and submit again."
                )
            else:
                response = agent.run(
                    AgentRequest(
                        intent=plan.intent,
                        record=record if plan.intent != "methodology" else None,
                        comparison_record=comparison,
                        question=message,
                    )
                )
                render_agent_response(response)

    elif workflow == "Assess one scenario":
        with st.form("agent_assessment_form"):
            record = build_record("agent_assess")
            run_assessment = st.form_submit_button("Run agent assessment", type="primary")
        if run_assessment:
            response = agent.run(AgentRequest(intent="assess_risk", record=record))
            render_agent_response(response)

    elif workflow == "Compare two scenarios":
        with st.form("agent_comparison_form"):
            baseline_column, comparison_column = st.columns(2)
            with baseline_column:
                st.markdown("#### Baseline scenario")
                baseline = build_record("agent_baseline")
            with comparison_column:
                st.markdown("#### Comparison scenario")
                comparison = build_record(
                    "agent_comparison",
                    defaults={"current_age": 35, "education_index": 3, "wealth_index": 4},
                )
            run_comparison = st.form_submit_button("Compare with agent", type="primary")
        if run_comparison:
            response = agent.run(
                AgentRequest(
                    intent="compare_scenarios",
                    record=baseline,
                    comparison_record=comparison,
                )
            )
            render_agent_response(response)

    else:
        with st.form("agent_methodology_form"):
            question = st.text_area(
                "Methodology question",
                value="How was the model validated and what are its limitations?",
                max_chars=500,
            )
            ask_methodology = st.form_submit_button("Ask research agent", type="primary")
        if ask_methodology:
            response = agent.run(AgentRequest(intent="methodology", question=question))
            render_agent_response(response)

with st.expander("What the model uses"):
    st.write(
        "The deployed demo uses only five harmonised variables: current age, residence, education, "
        "wealth quintile, and union status. Caste, religion, and state are intentionally excluded "
        "from the public interface."
    )
    st.json(metadata["global_feature_importance"])
    st.caption("These are global model importances, not causal or person-specific effects.")

with st.expander("Manuscript-reported evidence"):
    st.markdown(
        "- India primary holdout AUC: **0.8839**\n"
        "- Nepal geographic-validation AUC: **0.8488**\n"
        "- Nepal calibration slope: **0.7272** (recalibration required)"
    )
    st.caption("These values come from approved DHS data and are not metrics of the synthetic demo artifact.")
