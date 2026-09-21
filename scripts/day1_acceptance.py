"""Dependency-light acceptance checks for the Day-1 ML foundation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from fertility_risk.data import build_transport_dataset, validate_no_leakage
from fertility_risk.inference import load_model_bundle, predict_records
from fertility_risk.schemas import PredictionRequest


def main() -> None:
    bundle = load_model_bundle("models/demo_transport_model.joblib")
    valid = PredictionRequest(
        current_age=31,
        residence=2,
        education=2,
        wealth=3,
        in_union=1,
    )
    prediction = predict_records(bundle, [valid.model_dump()])[0]
    assert 0 <= prediction["probability"] <= 1
    assert prediction["demo_only"] is True

    try:
        PredictionRequest(
            current_age=52,
            residence=2,
            education=2,
            wealth=3,
            in_union=1,
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("Age outside 15-49 was accepted")

    try:
        validate_no_leakage(["current_age", "v201"])
    except ValueError:
        pass
    else:
        raise AssertionError("Outcome-proximal variable v201 was accepted")

    raw = pd.DataFrame(
        {
            "v012": [25, 31, 44],
            "v025": [1, 2, 2],
            "v106": [3, 2, 0],
            "v190": [5, 3, 1],
            "v501": [0, 1, 2],
            "v201": [None, 2, 4],
        }
    )
    features, outcome = build_transport_dataset(raw)
    assert len(features) == 2
    assert features["in_union"].tolist() == [1, 1]
    assert outcome.tolist() == [0, 1]

    research = json.loads(Path("research/manuscript_metrics.json").read_text(encoding="utf-8"))
    assert research["india_primary"]["holdout_auc"] == 0.8839
    assert research["transport"]["nepal_auc"] == 0.8488

    print("DAY 1 ACCEPTANCE: PASS")
    print(json.dumps(prediction, indent=2))


if __name__ == "__main__":
    main()
