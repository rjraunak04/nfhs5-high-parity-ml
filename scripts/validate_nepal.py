"""Evaluate a locked India transport model once on approved Nepal DHS data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from fertility_risk.data import (
    build_transport_dataset,
    fast_file_fingerprint,
    read_dta_columns,
)
from fertility_risk.evaluation import binary_metrics, bootstrap_auc_interval
from fertility_risk.inference import load_model_bundle


def calibration_intercept_slope(outcome, probabilities) -> tuple[float, float]:
    try:
        import statsmodels.api as sm
    except ImportError as exc:
        raise RuntimeError('Install training dependencies with: pip install -e ".[train]"') from exc

    clipped = np.clip(np.asarray(probabilities), 1e-6, 1 - 1e-6)
    logit_probability = np.log(clipped / (1 - clipped))
    fitted = sm.Logit(np.asarray(outcome), sm.add_constant(logit_probability)).fit(disp=0)
    return float(fitted.params[0]), float(fitted.params[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--nepal-dta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/nepal"))
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()

    bundle = load_model_bundle(args.model)
    if bundle["metadata"].get("feature_set") != "transport":
        raise ValueError("Nepal validation requires a locked India transport model bundle")

    raw = read_dta_columns(
        args.nepal_dta,
        ["v012", "v025", "v106", "v190", "v501", "v201", "v005"],
    )
    features, outcome = build_transport_dataset(raw)
    probability = bundle["pipeline"].predict_proba(features)[:, 1]
    threshold = float(bundle["metadata"]["threshold"])
    results = binary_metrics(outcome, probability, threshold)
    auc_low, auc_high = bootstrap_auc_interval(
        outcome,
        probability,
        repetitions=args.bootstrap,
        random_state=42,
    )
    intercept, slope = calibration_intercept_slope(outcome, probability)
    results.update(
        {
            "auc_95_ci_low": auc_low,
            "auc_95_ci_high": auc_high,
            "calibration_intercept": intercept,
            "calibration_slope": slope,
            "rows": int(len(features)),
            "positive_prevalence": float(outcome.mean()),
            "data_fingerprint": fast_file_fingerprint(args.nepal_dta),
            "model_version": bundle["metadata"]["model_version"],
            "validation_design": "locked India model applied without Nepal retraining",
        }
    )

    weights = pd.to_numeric(raw.loc[features.index, "v005"], errors="coerce") / 1_000_000
    valid_weights = weights.notna() & (weights > 0)
    if valid_weights.sum() > 100:
        results["weighted_auc"] = float(
            roc_auc_score(
                outcome.loc[valid_weights],
                probability[valid_weights.to_numpy()],
                sample_weight=weights.loc[valid_weights],
            )
        )
        results["weighted_brier"] = float(
            brier_score_loss(
                outcome.loc[valid_weights],
                probability[valid_weights.to_numpy()],
                sample_weight=weights.loc[valid_weights],
            )
        )

    subgroup_rows = []
    subgroups = {
        "Residence urban": features["residence"] == 1,
        "Residence rural": features["residence"] == 2,
        "Age 15 to 24": features["current_age"].between(15, 24),
        "Age 25 to 34": features["current_age"].between(25, 34),
        "Age 35 to 49": features["current_age"].between(35, 49),
    }
    for name, mask in subgroups.items():
        subgroup_y = outcome.loc[mask]
        if mask.sum() < 100 or subgroup_y.nunique() < 2:
            continue
        subgroup_probability = probability[mask.to_numpy()]
        subgroup_rows.append(
            {
                "subgroup": name,
                "n": int(mask.sum()),
                "positive_prevalence": float(subgroup_y.mean()),
                "auc_roc": float(roc_auc_score(subgroup_y, subgroup_probability)),
                "brier_score": float(brier_score_loss(subgroup_y, subgroup_probability)),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "nepal_validation.json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(subgroup_rows).to_csv(args.output_dir / "nepal_subgroups.csv", index=False)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
