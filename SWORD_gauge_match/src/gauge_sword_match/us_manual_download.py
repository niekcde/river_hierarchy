from __future__ import annotations

import csv
import io
import math
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

US_MANUAL_SERIES_NAME = "manual_usgs_continuous_zip"
US_MANUAL_REQUIRED_SUFFIX = ".zip"
US_MANUAL_SITE_OVERRIDES: dict[str, dict[str, Any]] = {
    "15453500": {
        "lat": 65.875101,
        "lon": -149.720349,
        "station_name": "YUKON R NR STEVENS VILLAGE AK",
        "daily_available_explicit": True,
        "daily_audit_class": "daily_yes_explicit",
    },
    "15564860": {
        "lat": 64.7375,
        "lon": -156.891667,
        "station_name": "YUKON R AT GALENA AK",
        "daily_available_explicit": False,
        "daily_audit_class": "daily_unknown_not_queried",
    },
}


def discover_us_manual_station_ids(archive_dir: str | Path) -> list[str]:
    archive_dir = Path(archive_dir)
    if not archive_dir.exists() or not archive_dir.is_dir():
        return []
    station_ids = {
        station_id
        for path in archive_dir.iterdir()
        for station_id in [_station_id_from_name(path.name)]
        if path.is_file() and station_id is not None
    }
    return sorted(station_ids)


def load_us_manual_archive(
    station_id: str,
    archive_dir: str | Path,
) -> tuple[pd.DataFrame, str | None]:
    normalized_station_id = _normalize_station_id(station_id)
    archive_dir = Path(archive_dir)
    if not archive_dir.exists() or not archive_dir.is_dir():
        return pd.DataFrame(), None

    matched_paths = sorted(
        path
        for path in archive_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() == US_MANUAL_REQUIRED_SUFFIX
        and _station_id_from_name(path.name) == normalized_station_id
    )
    if not matched_paths:
        return pd.DataFrame(), None

    frames: list[pd.DataFrame] = []
    for path in matched_paths:
        frame = _parse_manual_zip(path)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame(), None

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["time", "last_modified", "source_file"]).drop_duplicates(
        subset=["time"],
        keep="last",
    )
    combined = combined.reset_index(drop=True)

    raw_values = pd.to_numeric(combined.get("value"), errors="coerce")
    raw_unit = _first_non_null(combined.get("unit_of_measure"))
    discharge, unit_of_measure, unit_normalized, raw_unit_series = _convert_units(
        raw_values,
        raw_unit=raw_unit,
    )
    time_series_id = _first_non_null(combined.get("time_series_id")) or normalized_station_id
    output = pd.DataFrame(
        {
            "time": pd.to_datetime(combined.get("time"), errors="coerce", utc=True),
            "discharge": discharge,
            "raw_discharge": raw_values,
            "unit_of_measure": unit_of_measure,
            "raw_unit_of_measure": raw_unit_series,
            "unit_normalized": unit_normalized,
            "provider_series_name": US_MANUAL_SERIES_NAME,
            "provider_series_id": str(time_series_id),
        }
    )
    output = output.dropna(subset=["time", "discharge"]).reset_index(drop=True)
    notes = f"USGS manual ZIP archive parsed from {len(matched_paths)} file(s) under `manual_download`."
    return output, notes


def build_manual_us_station_row(
    station_id: str,
    *,
    examples: pd.DataFrame,
    gauges_cleaned_path: str | Path,
) -> dict[str, Any]:
    normalized_station_id = _normalize_station_id(station_id)
    station_key = f"US:{normalized_station_id}"
    lat, lon, station_name = _resolve_manual_station_coordinates(
        normalized_station_id,
        gauges_cleaned_path=gauges_cleaned_path,
    )
    if lat is None or lon is None:
        raise ValueError(f"Could not resolve coordinates for manual US station {normalized_station_id}.")

    example_id = _nearest_example_id(lat=lat, lon=lon, examples=examples)
    if example_id is None:
        raise ValueError(f"Could not assign example_id for manual US station {normalized_station_id}.")

    override = US_MANUAL_SITE_OVERRIDES.get(normalized_station_id, {})
    daily_available_explicit = bool(override.get("daily_available_explicit", False))
    daily_audit_class = override.get("daily_audit_class", "daily_unknown_not_queried")
    reason_summary = (
        "Daily and subdaily are both exposed, or daily is explicit in provider metadata."
        if daily_available_explicit
        else "Manual USGS continuous archive added; subdaily is confirmed, but daily was not re-audited here."
    )
    notes = (
        f"Manual-added US hierarchy example station using reviewed provider station `{normalized_station_id}`."
    )

    return {
        "station_key": station_key,
        "country": "US",
        "source_station_id": normalized_station_id,
        "provider": "usgs",
        "status": "subdaily_found",
        "resolved_site_number": normalized_station_id,
        "resolved_station_name": station_name,
        "resolution_method": "manual_added_example_station",
        "resolution_distance_m": 0.0,
        "daily_series_count": 1 if daily_available_explicit else pd.NA,
        "instantaneous_series_count": 1,
        "daily_begin": pd.NA,
        "daily_end": pd.NA,
        "daily_coverage_type": pd.NA,
        "daily_available_explicit": daily_available_explicit,
        "daily_audit_class": daily_audit_class,
        "reason_summary": reason_summary,
        "manual_option_type": pd.NA,
        "manual_option_url": "https://waterdata.usgs.gov/",
        "manual_option_note": pd.NA,
        "notes": notes,
        "source_file": "manual_us_download",
        "lat": lat,
        "lon": lon,
        "example_id": int(example_id),
        "down": True,
    }


def _parse_manual_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        payload = archive.read("primary-time-series.csv").decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(payload))
    rows: list[dict[str, Any]] = []
    for record in reader:
        if not record:
            continue
        rows.append(
            {
                "time": record.get("time"),
                "value": record.get("value"),
                "unit_of_measure": record.get("unit_of_measure"),
                "time_series_id": record.get("time_series_id"),
                "last_modified": pd.to_datetime(record.get("last_modified"), errors="coerce", utc=True),
                "source_file": path.name,
            }
        )
    return pd.DataFrame.from_records(rows)


def _resolve_manual_station_coordinates(
    station_id: str,
    *,
    gauges_cleaned_path: str | Path,
) -> tuple[float | None, float | None, str | None]:
    gauges_cleaned_path = Path(gauges_cleaned_path)
    station_name = None
    if gauges_cleaned_path.exists():
        gauges = pd.read_parquet(gauges_cleaned_path)
        match = gauges[gauges.get("station_id", pd.Series(dtype="string")).astype(str) == station_id]
        if not match.empty:
            row = match.iloc[0]
            lat = _to_float(row.get("lat") if "lat" in row else row.get("latitude"))
            lon = _to_float(row.get("lon") if "lon" in row else row.get("longitude"))
            station_name = _nullable_str(row.get("station_name"))
            if lat is not None and lon is not None:
                override = US_MANUAL_SITE_OVERRIDES.get(station_id, {})
                return lat, lon, station_name or override.get("station_name")

    override = US_MANUAL_SITE_OVERRIDES.get(station_id, {})
    return _to_float(override.get("lat")), _to_float(override.get("lon")), _nullable_str(override.get("station_name"))


def _nearest_example_id(
    *,
    lat: float,
    lon: float,
    examples: pd.DataFrame,
) -> int | None:
    if examples.empty:
        return None
    working = examples.dropna(subset=["lat", "lon", "example_id"]).copy()
    if working.empty:
        return None
    distances = working.apply(
        lambda row: _haversine_distance_m(lon, lat, _to_float(row["lon"]), _to_float(row["lat"])),
        axis=1,
    )
    if distances.isna().all():
        return None
    idx = distances.astype(float).idxmin()
    value = working.loc[idx, "example_id"]
    try:
        return int(float(value))
    except Exception:
        return None


def _station_id_from_name(filename: str) -> str | None:
    if not filename.lower().endswith(".zip"):
        return None
    prefix = "mlp_continuous_USGS-"
    if not filename.startswith(prefix):
        return None
    remainder = filename[len(prefix) :]
    station_id = remainder.split("_", 1)[0]
    if not station_id:
        return None
    return _normalize_station_id(station_id)


def _convert_units(values: pd.Series, *, raw_unit: str | None) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    raw_unit_text = (raw_unit or "").strip().lower()
    if raw_unit_text in {"ft3/s", "ft^3/s", "cfs", "cubic feet per second"}:
        converted = values * 0.028316846592
        return (
            converted,
            pd.Series(["m3/s"] * len(values), index=values.index, dtype="string"),
            pd.Series([True] * len(values), index=values.index, dtype="boolean"),
            pd.Series([raw_unit] * len(values), index=values.index, dtype="string"),
        )
    normalized = raw_unit if raw_unit is not None else pd.NA
    return (
        values,
        pd.Series([normalized] * len(values), index=values.index, dtype="string"),
        pd.Series([False] * len(values), index=values.index, dtype="boolean"),
        pd.Series([normalized] * len(values), index=values.index, dtype="string"),
    )


def _haversine_distance_m(lon1: float | None, lat1: float | None, lon2: float | None, lat2: float | None) -> float:
    if None in {lon1, lat1, lon2, lat2}:
        return math.inf
    radius_m = 6_371_000.0
    lon1_rad, lat1_rad, lon2_rad, lat2_rad = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0) ** 2
    return 2.0 * radius_m * math.asin(math.sqrt(a))


def _first_non_null(values: Any) -> Any | None:
    if values is None:
        return None
    if isinstance(values, pd.Series):
        for value in values:
            if _nullable_str(value) is not None:
                return value
        return None
    return values


def _normalize_station_id(value: Any) -> str:
    return str(value).strip()


def _nullable_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return None
    return text


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric
