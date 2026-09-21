import pandas as pd
import pytest

from fertility_risk.data import build_transport_dataset, validate_no_leakage


def test_leakage_guard_blocks_outcome_proximal_features():
    with pytest.raises(ValueError, match="Outcome-proximal"):
        validate_no_leakage(["current_age", "v201"])


def test_transport_builder_drops_missing_outcome_and_derives_union():
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
