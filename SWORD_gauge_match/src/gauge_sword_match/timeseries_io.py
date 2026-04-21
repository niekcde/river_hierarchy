from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from .utils import list_table_columns, read_table

REQUIRED_TIMESERIES_COLUMNS = ["station_id", "country", "time", "discharge"]
STANDARDIZED_TIMESERIES_COLUMNS = [
    "station_key",
    "station_id",
    "country",
    "time",
    "discharge",
    "variable",
    "source_function",
]
OPTIONAL_RENAMES = {
    "station_id": ["site_no", "site", "site_number", "station_number"],
    "country": ["country_code"],
    "time": ["Date", "date", "datetime", "time", "timestamp", "obs_time"],
    "discharge": ["Q", "q", "flow", "streamflow", "value", "discharge_cms"],
    "variable": ["parameter", "varname"],
    "source_function": ["source", "provider_function"],
    "fetch_error": ["error", "download_error"],
}
TIMESERIES_READ_CANDIDATE_COLUMNS = list(
    dict.fromkeys(
        REQUIRED_TIMESERIES_COLUMNS
        + ["station_key", "variable", "source_function", "fetch_error"]
        + [candidate for candidates in OPTIONAL_RENAMES.values() for candidate in candidates]
    )
)

SUPPORTED_TIMESERIES_SCOPES = {"all_cleaned", "matched_only", "high_medium_matched_only"}


def load_timeseries(
    path: str | Path,
    *,
    columns: list[str] | None = None,
    filters: list[tuple[str, str, object]] | None = None,
) -> pd.DataFrame:
    return read_table(path, columns=columns, filters=filters)


def load_standardized_timeseries_subset(
    path: str | Path,
    *,
    station_keys: list[str] | pd.Index | set[str] | None = None,
    expected_variable: str | None = None,
) -> pd.DataFrame:
    station_key_list = [str(value) for value in station_keys] if station_keys is not None else None
    available_columns = list_table_columns(path)
    read_columns = [column for column in TIMESERIES_READ_CANDIDATE_COLUMNS if column in available_columns]
    filters = _build_timeseries_station_filters(available_columns, station_key_list)
    raw = load_timeseries(path, columns=read_columns or None, filters=filters)
    standardized = standardize_timeseries(raw, expected_variable=expected_variable)
    if station_key_list is not None and not standardized.empty:
        standardized = standardized[standardized["station_key"].isin(station_key_list)].copy()
    return standardized


def standardize_timeseries(frame: pd.DataFrame, expected_variable: str | None = None) -> pd.DataFrame:
    working = frame.copy()
    rename_map: dict[str, str] = {}
    lower_cols = {column.lower(): column for column in working.columns}
    for target, candidates in OPTIONAL_RENAMES.items():
        if target in working.columns:
            continue
        for candidate in candidates:
            source = lower_cols.get(candidate.lower())
            if source:
                rename_map[source] = target
                break
    if rename_map:
        working = working.rename(columns=rename_map)

    missing = [column for column in REQUIRED_TIMESERIES_COLUMNS if column not in working.columns]
    if missing:
        raise ValueError(f"Timeseries table is missing required columns: {', '.join(missing)}")

    if "variable" not in working.columns:
        working["variable"] = pd.NA
    if "source_function" not in working.columns:
        working["source_function"] = pd.NA
    if "fetch_error" not in working.columns:
        working["fetch_error"] = pd.NA

    working["station_id"] = working["station_id"].astype("string").str.strip()
    working["country"] = working["country"].astype("string").str.upper().fillna(pd.NA)
    working["time"] = _parse_time_column(working["time"])
    working["discharge"] = pd.to_numeric(working["discharge"], errors="coerce")
    working["variable"] = working["variable"].astype("string").fillna(pd.NA)
    working["source_function"] = working["source_function"].astype("string").fillna(pd.NA)
    working["fetch_error"] = working["fetch_error"].astype("string").fillna(pd.NA)

    working = working.dropna(subset=["station_id", "country", "time", "discharge"]).copy()
    working = working[working["discharge"] >= 0].copy()
    if expected_variable is not None and "variable" in working.columns:
        variable_mask = working["variable"].isna() | (working["variable"].str.lower() == expected_variable.lower())
        working = working[variable_mask].copy()
    working = working[working["fetch_error"].isna()].copy()

    working["station_key"] = _build_station_key(working)

    working = (
        working.groupby(["station_key", "station_id", "country", "time"], as_index=False)
        .agg(
            discharge=("discharge", "mean"),
            variable=("variable", "first"),
            source_function=("source_function", "first"),
        )
        .sort_values(["station_key", "time"])
        .reset_index(drop=True)
    )

    return working[STANDARDIZED_TIMESERIES_COLUMNS].copy()


def combine_standardized_timeseries(
    frames: list[pd.DataFrame],
    *,
    priorities: list[int] | None = None,
) -> pd.DataFrame:
    if priorities is None:
        priorities = list(range(len(frames)))
    if len(priorities) != len(frames):
        raise ValueError("priorities must be the same length as frames")

    prepared: list[pd.DataFrame] = []
    for frame, priority in zip(frames, priorities):
        if frame.empty:
            continue
        missing = [column for column in STANDARDIZED_TIMESERIES_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(
                "Standardized timeseries frame is missing required columns: " + ", ".join(sorted(missing))
            )
        working = frame[STANDARDIZED_TIMESERIES_COLUMNS].copy()
        working["_provider_priority"] = int(priority)
        prepared.append(working)

    if not prepared:
        return pd.DataFrame(columns=STANDARDIZED_TIMESERIES_COLUMNS)

    merged = pd.concat(prepared, ignore_index=True)
    merged = (
        merged.sort_values(["station_key", "time", "_provider_priority"], kind="mergesort")
        .drop_duplicates(subset=["station_key", "time"], keep="first")
        .sort_values(["station_key", "time"], kind="mergesort")
        .reset_index(drop=True)
    )
    return merged[STANDARDIZED_TIMESERIES_COLUMNS].copy()


def filter_station_table_for_timeseries(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    working = _filter_station_table_for_scope(frame, scope)

    keep_columns = [
        column for column in ["station_id", "country", "source_function", "country_function"] if column in working.columns
    ]
    result = working[keep_columns].drop_duplicates(subset=["station_id", "country"]).reset_index(drop=True)
    return result


def select_station_keys_for_timeseries(frame: pd.DataFrame, scope: str) -> pd.Index:
    working = _filter_station_table_for_scope(frame, scope)
    if "station_key" not in working.columns:
        working = working.copy()
        working["station_key"] = _build_station_key(working)
    station_keys = (
        working["station_key"].astype("string").dropna().drop_duplicates().reset_index(drop=True)
    )
    return pd.Index(station_keys)


def _filter_station_table_for_scope(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    normalized_scope = str(scope).strip().lower()
    if normalized_scope not in SUPPORTED_TIMESERIES_SCOPES:
        raise ValueError(
            f"Unsupported timeseries scope '{scope}'. Expected one of: {', '.join(sorted(SUPPORTED_TIMESERIES_SCOPES))}"
        )

    working = frame.copy()
    required = {"station_id", "country"}
    missing = required - set(working.columns)
    if missing:
        raise ValueError(f"Station table is missing required columns for timeseries fetch: {', '.join(sorted(missing))}")

    if normalized_scope == "all_cleaned":
        filtered = working.copy()
    else:
        if "reach_id" not in working.columns:
            raise ValueError("Matched timeseries scopes require a station table with a 'reach_id' column.")
        mask = working["reach_id"].notna()
        if normalized_scope == "high_medium_matched_only":
            if "confidence_class" not in working.columns:
                raise ValueError(
                    "high_medium_matched_only requires a station table with a 'confidence_class' column."
                )
            mask = mask & working["confidence_class"].isin(["high", "medium"])
        filtered = working[mask].copy()
    return filtered


def _build_station_key(frame: pd.DataFrame) -> pd.Series:
    return frame.apply(
        lambda row: f"{row['country']}:{row['station_id']}" if pd.notna(row["country"]) else str(row["station_id"]),
        axis=1,
    )


def _build_timeseries_station_filters(
    available_columns: list[str],
    station_keys: list[str] | None,
) -> list[tuple[str, str, object]] | None:
    if not station_keys:
        return None

    available = set(available_columns)
    if "station_key" in available:
        return [("station_key", "in", station_keys)]

    if "station_id" in available:
        station_ids = sorted({station_key.split(":", 1)[1] if ":" in station_key else station_key for station_key in station_keys})
        filters: list[tuple[str, str, object]] = [("station_id", "in", station_ids)]
        if "country" in available:
            countries = sorted(
                {station_key.split(":", 1)[0] for station_key in station_keys if ":" in station_key}
            )
            if countries:
                filters.append(("country", "in", countries))
        return filters

    return None


def _parse_time_column(series: pd.Series) -> pd.Series:
    if is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    if is_numeric_dtype(series):
        # R Date columns often round-trip through Parquet as days since 1970-01-01.
        return pd.to_datetime(series, unit="D", origin="1970-01-01", errors="coerce")
    return pd.to_datetime(series, errors="coerce")
