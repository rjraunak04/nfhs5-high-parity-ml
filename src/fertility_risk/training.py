"""Leakage-aware nested validation and final model training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from .evaluation import binary_metrics, bootstrap_auc_interval, select_youden_threshold
from .modeling import make_model


@dataclass(frozen=True)
class TrainingConfig:
    random_state: int = 42
    holdout_fraction: float = 0.20
    outer_folds: int = 5
    inner_folds: int = 3
    tuning_trials: int = 10
    max_nested_rows: int = 100_000
    bootstrap_repetitions: int = 500

    @classmethod
    def for_mode(cls, mode: str) -> TrainingConfig:
        if mode == "quick":
            return cls(
                outer_folds=3,
                inner_folds=2,
                tuning_trials=2,
                max_nested_rows=10_000,
                bootstrap_repetitions=100,
            )
        if mode != "full":
            raise ValueError("mode must be 'quick' or 'full'")
        return cls()


def stratified_sample(
    features: pd.DataFrame,
    outcome: pd.Series,
    max_rows: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if len(features) <= max_rows:
        return features.reset_index(drop=True), outcome.reset_index(drop=True)
    sampled_x, _, sampled_y, _ = train_test_split(
        features,
        outcome,
        train_size=max_rows,
        stratify=outcome,
        random_state=random_state,
    )
    return sampled_x.reset_index(drop=True), sampled_y.reset_index(drop=True)


def _feature_types(features: list[str]) -> tuple[list[str], list[str]]:
    numeric = [feature for feature in ["current_age", "in_union"] if feature in features]
    categorical = [feature for feature in features if feature not in numeric]
    return categorical, numeric


def tune_random_forest(
    features: pd.DataFrame,
    outcome: pd.Series,
    *,
    inner_folds: int,
    tuning_trials: int,
    random_state: int,
) -> tuple[dict[str, Any], float]:
    """Tune only inside the supplied training partition."""

    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError('Install training dependencies with: pip install -e ".[train]"') from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    feature_names = list(features.columns)
    categorical, numeric = _feature_types(feature_names)
    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=random_state)

    def objective(trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 4, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        }
        fold_scores: list[float] = []
        for train_index, valid_index in inner_cv.split(features, outcome):
            model = make_model(
                params,
                categorical_features=categorical,
                numeric_features=numeric,
                random_state=random_state,
            )
            model.fit(features.iloc[train_index], outcome.iloc[train_index])
            probability = model.predict_proba(features.iloc[valid_index])[:, 1]
            fold_scores.append(float(roc_auc_score(outcome.iloc[valid_index], probability)))
        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=tuning_trials, show_progress_bar=False)
    return dict(study.best_params), float(study.best_value)


def nested_oof_validation(
    features: pd.DataFrame,
    outcome: pd.Series,
    config: TrainingConfig,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return unbiased outer-fold metrics and out-of-fold probabilities."""

    outer_cv = StratifiedKFold(
        n_splits=config.outer_folds,
        shuffle=True,
        random_state=config.random_state,
    )
    oof_probability = np.full(len(outcome), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []
    categorical, numeric = _feature_types(list(features.columns))
    for fold, (train_index, valid_index) in enumerate(outer_cv.split(features, outcome), start=1):
        params, inner_auc = tune_random_forest(
            features.iloc[train_index],
            outcome.iloc[train_index],
            inner_folds=config.inner_folds,
            tuning_trials=config.tuning_trials,
            random_state=config.random_state + fold,
        )
        model = make_model(
            params,
            categorical_features=categorical,
            numeric_features=numeric,
            random_state=config.random_state,
        )
        model.fit(features.iloc[train_index], outcome.iloc[train_index])
        probability = model.predict_proba(features.iloc[valid_index])[:, 1]
        oof_probability[valid_index] = probability
        fold_rows.append(
            {
                "fold": fold,
                "outer_auc": float(roc_auc_score(outcome.iloc[valid_index], probability)),
                "best_inner_auc": inner_auc,
                "best_params": params,
            }
        )
    if np.isnan(oof_probability).any():
        raise RuntimeError("Nested validation failed to create every out-of-fold prediction")
    return pd.DataFrame(fold_rows), oof_probability


def train_research_bundle(
    features: pd.DataFrame,
    outcome: pd.Series,
    *,
    config: TrainingConfig,
    feature_set: str,
    data_fingerprint: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Train, evaluate, and package a research model without touching the holdout early."""

    train_x, test_x, train_y, test_y = train_test_split(
        features,
        outcome,
        test_size=config.holdout_fraction,
        stratify=outcome,
        random_state=config.random_state,
    )
    nested_x, nested_y = stratified_sample(
        train_x,
        train_y,
        config.max_nested_rows,
        config.random_state,
    )
    fold_metrics, oof_probability = nested_oof_validation(nested_x, nested_y, config)
    threshold = select_youden_threshold(nested_y, oof_probability)
    final_params, final_inner_auc = tune_random_forest(
        nested_x,
        nested_y,
        inner_folds=config.inner_folds,
        tuning_trials=config.tuning_trials,
        random_state=config.random_state + 100,
    )
    categorical, numeric = _feature_types(list(features.columns))
    final_model = make_model(
        final_params,
        categorical_features=categorical,
        numeric_features=numeric,
        random_state=config.random_state,
    )
    final_model.fit(train_x, train_y)
    holdout_probability = final_model.predict_proba(test_x)[:, 1]
    holdout_metrics = binary_metrics(test_y, holdout_probability, threshold)
    auc_low, auc_high = bootstrap_auc_interval(
        test_y,
        holdout_probability,
        repetitions=config.bootstrap_repetitions,
        random_state=config.random_state,
    )
    holdout_metrics["auc_95_ci_low"] = auc_low
    holdout_metrics["auc_95_ci_high"] = auc_high
    metadata = {
        "model_version": f"research-{feature_set}-v1",
        "training_source": "approved DHS microdata; data are not included",
        "demo_only": False,
        "feature_set": feature_set,
        "features": list(features.columns),
        "threshold": threshold,
        "random_state": config.random_state,
        "data_fingerprint": data_fingerprint,
        "rows": int(len(features)),
        "positive_prevalence": float(outcome.mean()),
        "nested_auc_mean": float(fold_metrics["outer_auc"].mean()),
        "nested_auc_sd": float(fold_metrics["outer_auc"].std(ddof=1)),
        "final_inner_auc": final_inner_auc,
        "holdout_metrics": holdout_metrics,
        "final_params": final_params,
        "training_config": asdict(config),
        "interpretation": "cross-sectional classification of observed high parity at interview",
    }
    return {"pipeline": final_model, "metadata": metadata}, fold_metrics
