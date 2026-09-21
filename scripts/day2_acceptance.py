"""Offline Day 2 acceptance checks that require no web server dependencies."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import yaml

from fertility_risk.constants import MODEL_FEATURES
from fertility_risk.inference import load_model_bundle, predict_records
from fertility_risk.schemas import BatchPredictionRequest, PredictionRequest

ROOT = Path(__file__).resolve().parents[1]


def parsed_routes(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    routes: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.args
                and isinstance(decorator.args[0], ast.Constant)
            ):
                routes.add(str(decorator.args[0].value))
    return routes


def main() -> None:
    required_routes = {
        "/",
        "/health",
        "/v1/model",
        "/v1/features",
        "/v1/predict",
        "/v1/predict/batch",
    }
    assert required_routes <= parsed_routes(ROOT / "app/api.py")

    valid = {
        "current_age": 31,
        "residence": 2,
        "education": 2,
        "wealth": 3,
        "in_union": 1,
    }
    request = PredictionRequest(**valid)
    assert list(request.model_dump()) == MODEL_FEATURES
    assert BatchPredictionRequest(records=[request]).records

    bundle = load_model_bundle(
        ROOT / "models/demo_transport_model.joblib",
        expected_features=MODEL_FEATURES,
    )
    prediction = predict_records(bundle, [valid])[0]
    assert 0 <= prediction["probability"] <= 1
    assert prediction["demo_only"] is True

    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    assert set(compose["services"]) == {"api", "dashboard"}
    assert compose["services"]["api"]["ports"] == ["8000:8000"]
    assert compose["services"]["dashboard"]["ports"] == ["8501:8501"]

    dashboard = (ROOT / "app/dashboard.py").read_text(encoding="utf-8")
    for phrase in ("Research demonstration only", "CSV batch", "Download predictions"):
        assert phrase in dashboard

    print("DAY 2 ACCEPTANCE: PASS")
    print(json.dumps(prediction, indent=2))


if __name__ == "__main__":
    main()
