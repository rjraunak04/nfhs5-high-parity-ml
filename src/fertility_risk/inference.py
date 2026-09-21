"""Validated loading and prediction for versioned model bundles."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn

from .constants import MODEL_FEATURES


def load_model_bundle(
    path: str | Path,
    *,
    expected_features: list[str] | None = None,
) -> dict[str, Any]:
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model bundle not found at {model_path}. Run: python scripts/train_demo_model.py"
        )
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle or "metadata" not in bundle:
        raise ValueError("Invalid model bundle: pipeline and metadata are required")
    metadata = bundle["metadata"]
    if not isinstance(metadata.get("features"), list) or not metadata["features"]:
        raise ValueError("Invalid model bundle: a non-empty feature contract is required")
    if expected_features is not None and metadata["features"] != expected_features:
        raise ValueError("Model feature contract does not match this service version")
    trained_version = metadata.get("scikit_learn_version")
    if trained_version and trained_version != sklearn.__version__:
        warnings.warn(
            f"Model was trained with scikit-learn {trained_version}; running {sklearn.__version__}",
            RuntimeWarning,
            stacklevel=2,
        )
    return bundle


def predict_records(bundle: dict[str, Any], records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(records)
    if not rows:
        raise ValueError("At least one record is required")
    frame = pd.DataFrame(rows)
    missing = [feature for feature in MODEL_FEATURES if feature not in frame]
    extra = [column for column in frame if column not in MODEL_FEATURES]
    if missing or extra:
        raise ValueError(f"Feature contract mismatch; missing={missing}, extra={extra}")
    frame = frame[MODEL_FEATURES]
    probabilities = bundle["pipeline"].predict_proba(frame)[:, 1]
    metadata = bundle["metadata"]
    threshold = float(metadata["threshold"])
    return [
        {
            "probability": round(float(probability), 6),
            "classification": "higher_score" if probability >= threshold else "lower_score",
            "threshold": round(threshold, 6),
            "model_version": str(metadata["model_version"]),
            "demo_only": bool(metadata["demo_only"]),
        }
        for probability in probabilities
    ]
