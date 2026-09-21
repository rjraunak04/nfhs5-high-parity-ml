from pathlib import Path

import joblib
import pytest

from fertility_risk.modeling import train_demo_bundle


@pytest.fixture(scope="session")
def demo_bundle():
    return train_demo_bundle(rows=1600)


@pytest.fixture(scope="session")
def demo_model_path(tmp_path_factory, demo_bundle) -> Path:
    path = tmp_path_factory.mktemp("models") / "demo.joblib"
    joblib.dump(demo_bundle, path)
    return path
