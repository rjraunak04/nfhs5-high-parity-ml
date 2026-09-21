"""FastAPI service for the synthetic five-feature research demonstration."""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI

from fertility_risk.constants import MODEL_FEATURES
from fertility_risk.inference import load_or_create_demo_bundle, predict_records
from fertility_risk.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)

DEFAULT_MODEL_PATH = "models/demo_transport_model.joblib"

app = FastAPI(
    title="NFHS-5 High-Parity Research Demo API",
    version="0.1.0",
    description=(
        "Typed inference API for a synthetic demonstration model. "
        "It is not a clinical tool and does not predict future fertility."
    ),
)


@app.get("/", tags=["operations"])
def root() -> dict:
    """Small discoverability endpoint for humans and deployment probes."""
    return {
        "service": app.title,
        "version": app.version,
        "documentation": "/docs",
        "health": "/health",
        "demo_only": True,
    }


@lru_cache(maxsize=1)
def get_bundle():
    return load_or_create_demo_bundle(
        os.getenv("FERTILITY_MODEL_PATH", DEFAULT_MODEL_PATH),
        expected_features=MODEL_FEATURES,
    )


@app.get("/health", response_model=HealthResponse, tags=["operations"])
def health() -> HealthResponse:
    metadata = get_bundle()["metadata"]
    return HealthResponse(
        status="ok",
        model_version=metadata["model_version"],
        demo_only=metadata["demo_only"],
    )


@app.get("/v1/model", tags=["model"])
def model_information() -> dict:
    return get_bundle()["metadata"]


@app.get("/v1/features", tags=["model"])
def feature_contract() -> dict:
    """Return the public input contract without exposing training data."""
    schema = PredictionRequest.model_json_schema()
    return {
        "feature_order": MODEL_FEATURES,
        "properties": schema["properties"],
        "additional_properties_allowed": False,
        "max_batch_size": 500,
    }


@app.post("/v1/predict", response_model=PredictionResponse, tags=["inference"])
def predict(request: PredictionRequest) -> PredictionResponse:
    result = predict_records(get_bundle(), [request.model_dump()])[0]
    return PredictionResponse(**result)


@app.post("/v1/predict/batch", response_model=BatchPredictionResponse, tags=["inference"])
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    records = [record.model_dump() for record in request.records]
    predictions = [PredictionResponse(**row) for row in predict_records(get_bundle(), records)]
    return BatchPredictionResponse(predictions=predictions, count=len(predictions))
