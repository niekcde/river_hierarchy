from __future__ import annotations

import csv
import lzma
import re
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

CANADA_MANUAL_DISCHARGE_UNIT = "m3/s"
CANADA_MANUAL_SERIES_NAME = "manual_aquarius_corrected_signal"
CANADA_MANUAL_REQUIRED_SUFFIX = ".csv.xz"
CANADA_MANUAL_FILENAME_PATTERN = re.compile(
    r"Discharge\.Working@(?P<station_id>[A-Za-z0-9]+)\.(?P<start_date>\d{8})_corrected(?: \(\d+\))?\.csv\.xz$",
    re.IGNORECASE,
)
CANADA_MANUAL_METADATA_PATTERN = re.compile(r"^#\s*(?P<key>[^:]+):\s*(?P<value>.*)$")


def load_canada_manual_archive(
    station_id: str,
    archive_dir: str | Path,
) -> tuple[pd.DataFrame, str | None]:
    normalized_station_id = _normalize_station_id(station_id)
    archive_dir = Path(archive_dir)
    if not archive_dir.exists() or not archive_dir.is_dir():
        return pd.DataFrame(), None

    matched_paths = _select_station_files(normalized_station_id, archive_dir)
    if not matched_paths:
        return pd.DataFrame(), None

    frames: list[pd.DataFrame] = []
    time_series_ids: set[str] = set()
    locations: set[str] = set()
    for path in matched_paths:
        metadata, frame = _parse_canada_manual_file(path)
        if frame.empty:
            continue
        frames.append(frame)
        time_series_id = _nullable_str(metadata.get("Time-series identifier"))
        location = _nullable_str(metadata.get("Location"))
        if time_series_id is not None:
            time_series_ids.add(time_series_id)
        if location is not None:
            locations.add(location)

    if not frames:
        return pd.DataFrame(), None

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["time", "source_start_date", "source_file"]).drop_duplicates(
        subset=["time"],
        keep="last",
    )
    combined = combined.reset_index(drop=True)
    series_id = sorted(time_series_ids)[0] if time_series_ids else normalized_station_id
    output = pd.DataFrame(
        {
            "time": combined["time"],
            "discharge": combined["discharge"],
            "raw_discharge": combined["raw_discharge"],
            "unit_of_measure": CANADA_MANUAL_DISCHARGE_UNIT,
            "raw_unit_of_measure": combined["raw_unit_of_measure"].fillna(CANADA_MANUAL_DISCHARGE_UNIT),
            "unit_normalized": True,
            "provider_series_name": CANADA_MANUAL_SERIES_NAME,
            "provider_series_id": series_id,
        }
    )

    location_note = ""
    if locations:
        location_note = f" for `{sorted(locations)[0]}`"
    notes = (
        f"Canada manual AQUARIUS archive parsed from {len(matched_paths)} file(s) under `manual_download`"
        f"{location_note}."
    )
    return output, notes


def _select_station_files(station_id: str, archive_dir: Path) -> list[Path]:
    chosen: dict[tuple[str, str], Path] = {}
    for path in sorted(archive_dir.iterdir()):
        if not path.is_file() or not path.name.lower().endswith(CANADA_MANUAL_REQUIRED_SUFFIX):
            continue
        match = CANADA_MANUAL_FILENAME_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        candidate_station_id = _normalize_station_id(match.group("station_id"))
        if candidate_station_id != station_id:
            continue
        key = (candidate_station_id, match.group("start_date"))
        existing = chosen.get(key)
        if existing is None or _file_preference_key(path) < _file_preference_key(existing):
            chosen[key] = path
    return [chosen[key] for key in sorted(chosen)]


def _parse_canada_manual_file(path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    metadata: dict[str, Any] = {}
    data_lines: list[str] = []
    with lzma.open(path, mode="rt", encoding="utf-8", newline="") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.startswith("#"):
                match = CANADA_MANUAL_METADATA_PATTERN.match(line)
                if match is not None:
                    metadata[match.group("key").strip()] = match.group("value").strip()
                continue
            data_lines.append(line)

    if not data_lines:
        return metadata, pd.DataFrame()

    reader = csv.reader(StringIO("\n".join(data_lines)))
    try:
        header = next(reader)
    except StopIteration:
        return metadata, pd.DataFrame()

    time_idx = header.index("ISO 8601 UTC") if "ISO 8601 UTC" in header else 0
    value_idx = header.index("Value") if "Value" in header else min(2, max(0, len(header) - 1))
    rows: list[dict[str, Any]] = []
    for row in reader:
        if not row or len(row) <= max(time_idx, value_idx):
            continue
        rows.append(
            {
                "time": row[time_idx],
                "discharge": row[value_idx],
                "raw_discharge": row[value_idx],
                "raw_unit_of_measure": _normalize_unit(_nullable_str(metadata.get("Value units"))) or CANADA_MANUAL_DISCHARGE_UNIT,
                "source_file": path.name,
                "source_start_date": _start_date_from_name(path.name),
            }
        )

    raw_unit = _normalize_unit(_nullable_str(metadata.get("Value units")))
    normalized = pd.DataFrame.from_records(rows)
    if normalized.empty:
        return metadata, normalized

    normalized["time"] = pd.to_datetime(normalized["time"], errors="coerce", utc=True)
    normalized["discharge"] = pd.to_numeric(normalized["discharge"], errors="coerce")
    normalized["raw_discharge"] = pd.to_numeric(normalized["raw_discharge"], errors="coerce")
    normalized["raw_unit_of_measure"] = raw_unit or CANADA_MANUAL_DISCHARGE_UNIT
    normalized = normalized.dropna(subset=["time", "discharge"]).reset_index(drop=True)
    return metadata, normalized


def _start_date_from_name(filename: str) -> pd.Timestamp:
    match = CANADA_MANUAL_FILENAME_PATTERN.fullmatch(filename)
    if match is None:
        return pd.Timestamp.min.tz_localize("UTC")
    parsed = pd.to_datetime(match.group("start_date"), format="%Y%m%d", utc=True, errors="coerce")
    if pd.isna(parsed):
        return pd.Timestamp.min.tz_localize("UTC")
    return parsed


def _file_preference_key(path: Path) -> tuple[int, str]:
    return (len(path.name), path.name)


def _normalize_station_id(value: Any) -> str:
    return str(value).strip().upper()


def _nullable_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return None
    return text


def _normalize_unit(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower().replace("^", "")
    if normalized in {"m3/s", "m3s", "m3 / s"}:
        return CANADA_MANUAL_DISCHARGE_UNIT
    return value.strip()
