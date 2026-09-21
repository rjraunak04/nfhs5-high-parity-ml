"""Generate governed global SHAP evidence for a local model and CSV batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fertility_risk.inference import load_model_bundle


def main() -> None:
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError('Install explanation dependencies with: pip install -e ".[explain]"') from exc

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/explanations"))
    parser.add_argument("--max-rows", type=int, default=1000)
    args = parser.parse_args()

    bundle = load_model_bundle(args.model)
    feature_names = bundle["metadata"]["features"]
    frame = pd.read_csv(args.input_csv)
    missing = [feature for feature in feature_names if feature not in frame]
    if missing:
        raise ValueError(f"Explanation batch is missing features: {missing}")
    frame = frame[feature_names].head(args.max_rows)

    preprocessor = bundle["pipeline"].named_steps["prep"]
    classifier = bundle["pipeline"].named_steps["clf"]
    transformed = preprocessor.transform(frame)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    encoded_names = list(preprocessor.get_feature_names_out())
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(transformed)
    if isinstance(shap_values, list):
        values = np.asarray(shap_values[1])
    else:
        values = np.asarray(shap_values)
        if values.ndim == 3:
            values = values[:, :, 1]
    mean_absolute = np.abs(values).mean(axis=0)
    ranking = dict(
        sorted(
            zip(encoded_names, mean_absolute.astype(float), strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "mean_absolute_shap.json").write_text(
        json.dumps(ranking, indent=2) + "\n",
        encoding="utf-8",
    )
    top = list(ranking.items())[:15][::-1]
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.barh([name for name, _ in top], [value for _, value in top], color="#2E75B6")
    axis.set_xlabel("Mean absolute SHAP value")
    axis.set_title("Global model associations (not causal effects)")
    figure.tight_layout()
    figure.savefig(args.output_dir / "global_shap.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved governed explanation outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
