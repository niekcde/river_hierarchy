"""Small compatibility helpers for robust Parquet exports of lineage metadata."""

from __future__ import annotations

import pandas as pd


def parquet_safe_metadata_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert heterogeneous object metadata columns to stable nullable strings.

    Linkage fields may legitimately contain a mixture of empty numeric cells and
    comma-separated identifiers. CSV tolerates that mixture; Arrow requires one
    concrete type. Numeric routing columns are left unchanged.
    """
    result = frame.copy()
    for column in result.select_dtypes(include=["object"]).columns:
        values = result[column]

        def stringify(value: object) -> object:
            if value is None:
                return pd.NA
            try:
                if bool(pd.isna(value)):
                    return pd.NA
            except (TypeError, ValueError):
                pass
            return str(value)

        result[column] = values.map(stringify).astype("string")
    return result
