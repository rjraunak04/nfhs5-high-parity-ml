"""Build the safe synthetic artifact and sample batch used by the public demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from fertility_risk.modeling import generate_synthetic_demo_data, train_demo_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--rows", type=int, default=6000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    bundle = train_demo_bundle(rows=args.rows)
    artifact_path = args.output_dir / "demo_transport_model.joblib"
    metadata_path = args.output_dir / "demo_model_metadata.json"
    joblib.dump(bundle, artifact_path, compress=3)
    metadata_path.write_text(json.dumps(bundle["metadata"], indent=2) + "\n", encoding="utf-8")

    sample, _ = generate_synthetic_demo_data(rows=12, random_state=2026)
    sample.to_csv(args.data_dir / "demo_batch.csv", index=False)
    print(f"Saved demo artifact: {artifact_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
