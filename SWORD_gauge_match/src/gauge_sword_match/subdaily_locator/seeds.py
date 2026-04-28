from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

REQUIRED_SEED_COLUMNS = ["station_key", "lat", "lon"]
DEFAULT_LAYER = "hierarchy_examples_filtered"
STANDARDIZED_SEED_COLUMNS = [
    "station_key",
    "country",
    "source_station_id",
    "lat",
    "lon",
    "occurrence_count",
    "example_ids",
    "down_values",
]


def load_hierarchy_example_station_seeds(
    path: str | Path,
    *,
    layer: str = DEFAULT_LAYER,
) -> pd.DataFrame:
    frame = _read_seed_layer(path, layer=layer)
    if frame.empty:
        return pd.DataFrame(columns=STANDARDIZED_SEED_COLUMNS)

    working = frame.copy()
    working["station_key"] = working["station_key"].astype("string").str.strip()
    working = working[working["station_key"].notna() & working["station_key"].str.contains(":", regex=False)].copy()
    if working.empty:
        return pd.DataFrame(columns=STANDARDIZED_SEED_COLUMNS)

    working["country"] = working["station_key"].str.split(":", n=1).str[0].str.upper()
    working["source_station_id"] = working["station_key"].str.split(":", n=1).str[1].str.strip()
    working["lat"] = pd.to_numeric(working["lat"], errors="coerce")
    working["lon"] = pd.to_numeric(working["lon"], errors="coerce")
    if "down" not in working.columns:
        working["down"] = pd.NA
    if "example_id" not in working.columns:
        working["example_id"] = pd.NA

    aggregated = (
        working.groupby(["station_key", "country", "source_station_id"], dropna=False, sort=True)
        .agg(
            lat=("lat", "first"),
            lon=("lon", "first"),
            occurrence_count=("station_key", "size"),
            example_ids=("example_id", _join_unique_values),
            down_values=("down", _join_unique_values),
        )
        .reset_index()
    )
    return aggregated[STANDARDIZED_SEED_COLUMNS].copy()


def _read_seed_layer(path: str | Path, *, layer: str) -> pd.DataFrame:
    path = Path(path)
    with sqlite3.connect(path) as connection:
        table_names = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            connection,
            params=[layer],
        )
        if table_names.empty:
            raise ValueError(f"Layer '{layer}' not found in {path}")

        frame = pd.read_sql_query(f'SELECT * FROM "{layer}"', connection)

    missing = [column for column in REQUIRED_SEED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Seed layer '{layer}' is missing required columns: {', '.join(missing)}")
    return frame


def _join_unique_values(values: pd.Series) -> str:
    unique_values: list[str] = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text:
            continue
        if text.endswith(".0"):
            try:
                numeric_value = float(text)
            except ValueError:
                pass
            else:
                if numeric_value.is_integer():
                    text = str(int(numeric_value))
        if text not in unique_values:
            unique_values.append(text)
    return ",".join(unique_values)
