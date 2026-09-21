"""Data loading, cleaning, and leakage-protection utilities."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .constants import (
    BLOCKED_OUTCOME_PROXIMAL,
    MODEL_FEATURES,
    PRIMARY_INDIA_FEATURES,
    RAW_TO_CLEAN,
)


def validate_no_leakage(features: Iterable[str]) -> None:
    """Reject outcome and outcome-proximal variables from a predictor set."""

    normalized = {str(feature).strip().lower() for feature in features}
    blocked = normalized.intersection({item.lower() for item in BLOCKED_OUTCOME_PROXIMAL})
    if blocked:
        raise ValueError(f"Outcome-proximal predictors are not allowed: {sorted(blocked)}")


def fast_file_fingerprint(path: str | Path, block_size: int = 1024 * 1024) -> str:
    """Create a reproducibility fingerprint without hashing an entire multi-GB file."""

    file_path = Path(path)
    stat = file_path.stat()
    digest = hashlib.sha256()
    digest.update(f"{file_path.name}|{stat.st_size}".encode())
    with file_path.open("rb") as handle:
        digest.update(handle.read(block_size))
        if stat.st_size > block_size:
            handle.seek(max(0, stat.st_size - block_size))
            digest.update(handle.read(block_size))
    return digest.hexdigest()


def read_dta_columns(path: str | Path, requested: Iterable[str]) -> pd.DataFrame:
    """Read only requested Stata columns with pyreadstat to control memory use."""

    try:
        import pyreadstat
    except ImportError as exc:
        raise RuntimeError('Install training dependencies with: pip install -e ".[train]"') from exc

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"DHS Stata file not found: {file_path}")

    requested_unique = list(dict.fromkeys(requested))
    _, metadata = pyreadstat.read_dta(str(file_path), metadataonly=True)
    lookup = {str(column).lower(): column for column in metadata.column_names}
    missing = [column for column in requested_unique if column.lower() not in lookup]
    if missing:
        raise ValueError(f"Required DHS columns are missing: {missing}")

    usecols = [lookup[column.lower()] for column in requested_unique]
    frame, _ = pyreadstat.read_dta(
        str(file_path),
        usecols=usecols,
        apply_value_formats=False,
        formats_as_category=False,
        user_missing=False,
        disable_datetime_conversion=True,
    )
    frame.rename(columns={source: requested_unique[i] for i, source in enumerate(usecols)}, inplace=True)
    return frame[requested_unique]


def _clean_base(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.rename(columns={key: value for key, value in RAW_TO_CLEAN.items() if key in frame})
    cleaned = renamed.copy()
    for column in cleaned.columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    valid = cleaned["children_ever_born"].notna() & cleaned["current_age"].between(15, 49)
    cleaned = cleaned.loc[valid].copy()
    cleaned["fertility_class"] = (cleaned["children_ever_born"] >= 3).astype("int8")
    return cleaned


def build_primary_india_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build the leakage-aware seven-feature India research dataset."""

    cleaned = _clean_base(frame)
    validate_no_leakage(PRIMARY_INDIA_FEATURES)
    missing = [feature for feature in PRIMARY_INDIA_FEATURES if feature not in cleaned]
    if missing:
        raise ValueError(f"Primary India features are missing after mapping: {missing}")
    return cleaned[PRIMARY_INDIA_FEATURES].copy(), cleaned["fertility_class"].copy()


def build_transport_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build the five-feature harmonised dataset used for geographic transport."""

    cleaned = _clean_base(frame)
    if "marital_status" not in cleaned:
        raise ValueError("Marital status (v501) is required for the transport feature set")
    cleaned["in_union"] = cleaned["marital_status"].isin([1, 2]).astype("int8")
    validate_no_leakage(MODEL_FEATURES)
    return cleaned[MODEL_FEATURES].copy(), cleaned["fertility_class"].copy()
