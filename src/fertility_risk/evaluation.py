"""Model evaluation helpers that avoid test-set threshold tuning."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def select_youden_threshold(y_true, probabilities) -> float:
    """Select a decision threshold from training/OOF predictions only."""

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, probabilities)
    finite = np.isfinite(thresholds)
    scores = true_positive_rate[finite] - false_positive_rate[finite]
    return float(thresholds[finite][int(np.argmax(scores))])


def binary_metrics(y_true, probabilities, threshold: float) -> dict[str, float]:
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        "auc_roc": float(roc_auc_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "threshold": float(threshold),
    }


def bootstrap_auc_interval(
    y_true,
    probabilities,
    *,
    repetitions: int = 500,
    random_state: int = 42,
) -> tuple[float, float]:
    """Return a non-parametric percentile interval for AUC."""

    y_array = np.asarray(y_true)
    p_array = np.asarray(probabilities)
    rng = np.random.default_rng(random_state)
    values: list[float] = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(y_array), len(y_array))
        if np.unique(y_array[indices]).size < 2:
            continue
        values.append(float(roc_auc_score(y_array[indices], p_array[indices])))
    if not values:
        raise ValueError("Bootstrap AUC interval requires both outcome classes")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))
