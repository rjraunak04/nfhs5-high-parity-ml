"""Streamlit interface for the synthetic research demonstration model."""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from fertility_risk.constants import (
    EDUCATION_LABELS,
    MODEL_FEATURES,
    RESIDENCE_LABELS,
    UNION_LABELS,
    WEALTH_LABELS,
)
from fertility_risk.inference import load_model_bundle, predict_records
from fertility_risk.schemas import PredictionRequest

MODEL_PATH = os.getenv("FERTILITY_MODEL_PATH", "models/demo_transport_model.joblib")

st.set_page_config(page_title="High-Parity ML Demo", page_icon="📊", layout="wide")


@st.cache_resource
def get_bundle():
    return load_model_bundle(MODEL_PATH, expected_features=MODEL_FEATURES)


bundle = get_bundle()
metadata = bundle["metadata"]

st.title("Explainable High-Parity Classification")
st.caption("Production-style ML interface built from NFHS-5/DHS research")

st.warning(
    "Research demonstration only. This score classifies a synthetic representation of observed "
    "high parity at interview; it is not a forecast, diagnosis, or recommendation."
)

single_tab, batch_tab = st.tabs(["Single record", "CSV batch"])

with single_tab:
    with st.form("prediction_form"):
        current_age = st.slider("Current age", min_value=15, max_value=49, value=30)
        residence_label = st.selectbox("Residence", list(RESIDENCE_LABELS.values()))
        education_label = st.selectbox("Education", list(EDUCATION_LABELS.values()), index=2)
        wealth_label = st.selectbox("Wealth quintile", list(WEALTH_LABELS.values()), index=2)
        union_label = st.selectbox("Current union status", list(UNION_LABELS.values()), index=1)
        submitted = st.form_submit_button("Generate research-demo score", type="primary")

    if submitted:
        def reverse_lookup(mapping, label):
            return next(key for key, value in mapping.items() if value == label)

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
