import pytest
from pydantic import ValidationError

from fertility_risk.schemas import PredictionRequest


def test_valid_prediction_request():
    request = PredictionRequest(
        current_age=31,
        residence=2,
        education=2,
        wealth=3,
        in_union=1,
    )
    assert request.current_age == 31


@pytest.mark.parametrize("age", [14, 50])
def test_age_outside_study_population_is_rejected(age):
    with pytest.raises(ValidationError):
        PredictionRequest(
            current_age=age,
            residence=2,
            education=2,
            wealth=3,
            in_union=1,
        )


def test_extra_fields_are_rejected():
    with pytest.raises(ValidationError):
        PredictionRequest(
            current_age=31,
            residence=2,
            education=2,
            wealth=3,
            in_union=1,
            children_ever_born=4,
        )
