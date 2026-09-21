import pytest

from fertility_risk.inference import load_model_bundle, predict_records


def test_model_bundle_roundtrip(demo_model_path):
    bundle = load_model_bundle(demo_model_path)
    predictions = predict_records(
        bundle,
        [
            {
                "current_age": 31,
                "residence": 2,
                "education": 2,
                "wealth": 3,
                "in_union": 1,
            }
        ],
    )
    assert 0 <= predictions[0]["probability"] <= 1
    assert predictions[0]["demo_only"] is True


def test_inference_requires_exact_feature_contract(demo_bundle):
    with pytest.raises(ValueError, match="Feature contract mismatch"):
        predict_records(demo_bundle, [{"current_age": 31}])
