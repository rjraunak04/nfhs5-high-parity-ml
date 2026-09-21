"""Train a reproducible research model from approved India NFHS-5 microdata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from fertility_risk.data import (
    build_primary_india_dataset,
    build_transport_dataset,
    fast_file_fingerprint,
    read_dta_columns,
)
from fertility_risk.training import TrainingConfig, train_research_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--india-dta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/research"))
    parser.add_argument("--feature-set", choices=["primary", "transport"], default="primary")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    args = parser.parse_args()

    if args.feature_set == "primary":
        requested = ["v012", "v025", "v106", "v130", "v190", "v024", "s116", "v201"]
        builder = build_primary_india_dataset
    else:
        requested = ["v012", "v025", "v106", "v190", "v501", "v201"]
        builder = build_transport_dataset

    frame = read_dta_columns(args.india_dta, requested)
    features, outcome = builder(frame)
    config = TrainingConfig.for_mode(args.mode)
    fingerprint = fast_file_fingerprint(args.india_dta)
    bundle, fold_metrics = train_research_bundle(
        features,
        outcome,
        config=config,
        feature_set=args.feature_set,
        data_fingerprint=fingerprint,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output_dir / f"{args.feature_set}_model_bundle.joblib", compress=3)
    (args.output_dir / f"{args.feature_set}_metadata.json").write_text(
        json.dumps(bundle["metadata"], indent=2) + "\n",
        encoding="utf-8",
    )
    fold_metrics.assign(best_params=fold_metrics["best_params"].map(json.dumps)).to_csv(
        args.output_dir / f"{args.feature_set}_nested_folds.csv",
        index=False,
    )
    print(json.dumps(bundle["metadata"], indent=2))


if __name__ == "__main__":
    main()
