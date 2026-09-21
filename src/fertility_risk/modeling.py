"""Preprocessing, model construction, and deterministic synthetic demo training."""

from __future__ import annotations

import platform
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .constants import MODEL_FEATURES
from .evaluation import binary_metrics, select_youden_threshold

CATEGORICAL_FEATURES = ["residence", "education", "wealth"]
NUMERIC_FEATURES = ["current_age", "in_union"]


def make_preprocessor(
    categorical_features: list[str] | None = None,
    numeric_features: list[str] | None = None,
) -> ColumnTransformer:
    categorical = categorical_features or CATEGORICAL_FEATURES
    numeric = numeric_features or NUMERIC_FEATURES
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True))]),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def make_model(
    params: dict[str, Any] | None = None,
    *,
    categorical_features: list[str] | None = None,
    numeric_features: list[str] | None = None,
    random_state: int = 42,
) -> Pipeline:
    defaults: dict[str, Any] = {
        "n_estimators": 180,
        "max_depth": 12,
        "min_samples_leaf": 8,
        "max_features": "sqrt",
    }
    defaults.update(params or {})
    classifier = RandomForestClassifier(
        **defaults,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(
        [
            ("prep", make_preprocessor(categorical_features, numeric_features)),
            ("clf", classifier),
        ]
    )


def generate_synthetic_demo_data(
    rows: int = 6000,
    *,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate non-personal DHS-shaped data for software demonstrations only."""

    rng = np.random.default_rng(random_state)
    frame = pd.DataFrame(
        {
            "current_age": rng.integers(15, 50, rows),
            "residence": rng.choice([1, 2], rows, p=[0.36, 0.64]),
            "education": rng.choice([0, 1, 2, 3], rows, p=[0.16, 0.14, 0.53, 0.17]),
            "wealth": rng.choice([1, 2, 3, 4, 5], rows, p=[0.2, 0.21, 0.21, 0.2, 0.18]),
            "in_union": rng.binomial(1, 0.68, rows),
        }
    )
    logit = (
        -5.4
        + 0.13 * (frame["current_age"] - 20)
        + 1.05 * frame["in_union"]
        + 0.28 * (frame["residence"] == 2)
        + 0.25 * (3 - frame["education"])
        + 0.16 * (3 - frame["wealth"])
    )
    probability = 1 / (1 + np.exp(-logit))
    outcome = pd.Series(rng.binomial(1, probability), name="fertility_class", dtype="int8")
    return frame[MODEL_FEATURES], outcome


def grouped_feature_importance(model: Pipeline) -> dict[str, float]:
    """Aggregate encoded random-forest importances back to raw input features."""

    names = model.named_steps["prep"].get_feature_names_out()
    values = model.named_steps["clf"].feature_importances_
    totals = {feature: 0.0 for feature in MODEL_FEATURES}
    for encoded_name, value in zip(names, values, strict=True):
        clean = encoded_name.split("__", maxsplit=1)[-1]
        matched = next((feature for feature in MODEL_FEATURES if clean == feature or clean.startswith(f"{feature}_")), None)
        if matched is not None:
            totals[matched] += float(value)
    return dict(sorted(totals.items(), key=lambda item: item[1], reverse=True))


def train_demo_bundle(rows: int = 6000, *, random_state: int = 42) -> dict[str, Any]:
    """Train a deterministic, self-contained model bundle on synthetic data."""

    features, outcome = generate_synthetic_demo_data(rows, random_state=random_state)
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        outcome,
        test_size=0.2,
        stratify=outcome,
        random_state=random_state,
    )
    model = make_model(random_state=random_state)
    cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    oof_probability = cross_val_predict(
        model,
        train_x,
        train_y,
        cv=cross_validation,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    threshold = select_youden_threshold(train_y, oof_probability)
    model.fit(train_x, train_y)
    test_probability = model.predict_proba(test_x)[:, 1]
    metrics = binary_metrics(test_y, test_probability, threshold)
    return {
        "pipeline": model,
        "metadata": {
            "model_version": "demo-synthetic-v1",
            "training_source": "deterministic synthetic DHS-shaped records",
            "demo_only": True,
            "features": MODEL_FEATURES,
            "threshold": threshold,
            "random_state": random_state,
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "training_rows": int(len(train_x)),
            "test_rows": int(len(test_x)),
            "metrics": metrics,
            "global_feature_importance": grouped_feature_importance(model),
            "intended_use": "software demonstration and portfolio review only",
        },
    }
