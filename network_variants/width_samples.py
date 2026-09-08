"""Compatibility helpers for detailed width-sample artifacts.

New compact states may retain ``link_width_samples.parquet`` while legacy
states retain CSV.  Consumers should use these helpers rather than hard-code a
filename so analysis can span both generations of output.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


WIDTH_SAMPLE_FILENAMES = ("link_width_samples.parquet", "link_width_samples.csv")


def resolve_width_samples_path(path_or_dir: str | Path) -> Path:
    """Resolve a Parquet-first detailed-width artifact from a file or directory."""
    candidate = Path(path_or_dir)
    if candidate.is_file() and candidate.suffix.lower() in {".csv", ".parquet"}:
        return candidate
    directory = candidate if candidate.is_dir() else candidate.parent
    for name in WIDTH_SAMPLE_FILENAMES:
        path = directory / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No detailed width samples found in {directory}. Expected one of "
        f"{list(WIDTH_SAMPLE_FILENAMES)}."
    )


def read_width_samples(path_or_dir: str | Path, **read_csv_kwargs: object) -> pd.DataFrame:
    """Read Parquet or legacy CSV width samples with one stable API."""
    path = resolve_width_samples_path(path_or_dir)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, **read_csv_kwargs)
