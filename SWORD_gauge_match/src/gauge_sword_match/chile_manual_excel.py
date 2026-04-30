from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

CHILE_DGA_LOCAL_TIMEZONE = ZoneInfo("America/Santiago")
CHILE_MANUAL_DISCHARGE_UNIT = "m3/s"
CHILE_MANUAL_SERIES_NAME = "manual_excel_altura_caudal_instantaneo"
CHILE_MANUAL_REQUIRED_SUFFIXES = {".xls", ".xlsx", ".xlsm"}


def load_chile_manual_archive(
    station_id: str,
    archive_dir: str | Path,
) -> tuple[pd.DataFrame, str | None]:
    station_id = str(station_id).strip()
    normalized_station_id = _normalize_chile_station_code(station_id)
    archive_dir = Path(archive_dir)
    if not archive_dir.exists() or not archive_dir.is_dir():
        return pd.DataFrame(), None

    workbook_paths = sorted(
        path for path in archive_dir.iterdir() if path.is_file() and path.suffix.lower() in CHILE_MANUAL_REQUIRED_SUFFIXES
    )
    if not workbook_paths:
        return pd.DataFrame(), None

    frames: list[pd.DataFrame] = []
    matched_workbooks: set[str] = set()
    matched_sheets: list[str] = []
    for workbook_path in workbook_paths:
        engine = _excel_engine_for_path(workbook_path)
        workbook = pd.ExcelFile(workbook_path, engine=engine)
        for sheet_name in workbook.sheet_names:
            sheet = pd.read_excel(workbook_path, sheet_name=sheet_name, header=None, engine=engine)
            metadata, frame = parse_chile_manual_sheet(
                sheet,
                source_file=workbook_path.name,
                source_sheet=sheet_name,
            )
            if _normalize_chile_station_code(metadata.get("station_id")) != normalized_station_id or frame.empty:
                continue
            matched_workbooks.add(workbook_path.name)
            matched_sheets.append(sheet_name)
            frames.append(frame)

    if not frames:
        return pd.DataFrame(), None

    combined = pd.concat(frames, ignore_index=True)
    combined["source_priority"] = combined["source_period_end"].fillna(pd.Timestamp.min.tz_localize("UTC"))
    combined = combined.sort_values(["time", "source_priority", "source_file", "source_sheet"]).drop_duplicates(
        subset=["time"],
        keep="last",
    )
    combined = combined.reset_index(drop=True)
    output = pd.DataFrame(
        {
            "time": combined["time"],
            "discharge": combined["discharge"],
            "raw_discharge": combined["raw_discharge"],
            "unit_of_measure": CHILE_MANUAL_DISCHARGE_UNIT,
            "raw_unit_of_measure": CHILE_MANUAL_DISCHARGE_UNIT,
            "unit_normalized": True,
            "provider_series_name": CHILE_MANUAL_SERIES_NAME,
            "provider_series_id": combined["station_code"].fillna(station_id),
        }
    )
    notes = (
        f"Chile DGA manual Excel archive parsed from {len(matched_workbooks)} workbook(s) "
        f"and {len(matched_sheets)} sheet(s) under `excel_download`."
    )
    return output, notes


def parse_chile_manual_sheet(
    frame: pd.DataFrame,
    *,
    source_file: str,
    source_sheet: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    working = frame.copy()
    metadata = {
        "source_file": source_file,
        "source_sheet": source_sheet,
        "station_name": _extract_label_value(working, "Estación:"),
        "station_id": _extract_label_value(working, "Codigo BNA:"),
        "period_label": _extract_period_label(working),
    }
    period_start, period_end = _parse_period_label(metadata["period_label"])
    metadata["period_start"] = period_start
    metadata["period_end"] = period_end

    records: list[dict[str, Any]] = []
    current_month: tuple[int, int] | None = None
    current_layout: dict[str, int] | None = None

    for _, row in working.iterrows():
        if _row_contains_label(row, "MES:"):
            current_month = _parse_month_label(_row_value_after_label(row, "MES:"))
            current_layout = None
            continue

        if _row_contains_label(row, "DIA") and _row_contains_label(row, "HORA"):
            current_layout = _detect_table_layout(row)
            continue

        if current_month is None or current_layout is None:
            continue

        day = _to_int(_safe_get(row, current_layout["day_col"]))
        time_text = _normalize_time_text(_safe_get(row, current_layout["hour_col"]))
        discharge = _to_float(_safe_get(row, current_layout["discharge_col"]))
        if day is None or time_text is None or discharge is None:
            continue

        timestamp = _build_chile_timestamp_utc(
            year=current_month[1],
            month=current_month[0],
            day=day,
            time_text=time_text,
        )
        if timestamp is None:
            continue
        records.append(
            {
                "time": timestamp,
                "discharge": discharge,
                "raw_discharge": discharge,
                "source_file": source_file,
                "source_sheet": source_sheet,
                "source_period_label": metadata["period_label"],
                "source_period_end": period_end,
                "station_code": metadata["station_id"],
            }
        )

    return metadata, pd.DataFrame.from_records(records)


def _excel_engine_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".xls":
        try:
            import xlrd  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Chile manual Excel parsing requires `xlrd>=2.0.1` for .xls workbooks."
            ) from exc
        return "xlrd"
    return "openpyxl"


def _extract_label_value(frame: pd.DataFrame, label: str) -> str | None:
    for _, row in frame.iterrows():
        value = _row_value_after_label(row, label)
        if value is not None:
            return value
    return None


def _extract_period_label(frame: pd.DataFrame) -> str | None:
    for _, row in frame.iterrows():
        for value in row.tolist():
            text = _nullable_str(value)
            if text and "PERIODO:" in text.upper():
                return text
    return None


def _parse_period_label(value: str | None) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if value is None:
        return None, None
    match = re.search(r"(\d{2}/\d{2}/\d{4})\s*-\s*(\d{2}/\d{2}/\d{4})", value)
    if not match:
        return None, None
    start = pd.to_datetime(match.group(1), dayfirst=True, utc=True, errors="coerce")
    end = pd.to_datetime(match.group(2), dayfirst=True, utc=True, errors="coerce")
    if pd.isna(start):
        start = None
    if pd.isna(end):
        end = None
    return start, end


def _parse_month_label(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    match = re.search(r"(\d{2})/(\d{4})", value)
    if not match:
        return None
    month = int(match.group(1))
    year = int(match.group(2))
    if month < 1 or month > 12:
        return None
    return month, year


def _detect_table_layout(row: pd.Series) -> dict[str, int]:
    day_col = hour_col = discharge_col = None
    for idx, value in enumerate(row.tolist()):
        text = _nullable_str(value)
        if text is None:
            continue
        normalized = text.strip().upper()
        if normalized == "DIA":
            day_col = idx
        elif normalized == "HORA":
            hour_col = idx
        elif "CAUDAL" in normalized:
            discharge_col = idx
    if day_col is None or hour_col is None or discharge_col is None:
        raise ValueError("Could not detect Chile manual Excel table layout from header row.")
    return {
        "day_col": day_col,
        "hour_col": hour_col,
        "discharge_col": discharge_col,
    }


def _row_contains_label(row: pd.Series, label: str) -> bool:
    target = label.strip().upper()
    for value in row.tolist():
        text = _nullable_str(value)
        if text is not None and text.strip().upper() == target:
            return True
    return False


def _row_value_after_label(row: pd.Series, label: str) -> str | None:
    target = label.strip().upper()
    values = row.tolist()
    for idx, value in enumerate(values):
        text = _nullable_str(value)
        if text is None or text.strip().upper() != target:
            continue
        for next_value in values[idx + 1 :]:
            candidate = _nullable_str(next_value)
            if candidate is not None:
                return candidate
    return None


def _safe_get(row: pd.Series, idx: int) -> Any:
    if idx < 0 or idx >= len(row):
        return None
    return row.iloc[idx]


def _normalize_time_text(value: Any) -> str | None:
    text = _nullable_str(value)
    if text is None:
        return None
    match = re.search(r"(\d{1,2}):(\d{2})", text)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return None
    return f"{hour:02d}:{minute:02d}"


def _build_chile_timestamp_utc(
    *,
    year: int,
    month: int,
    day: int,
    time_text: str,
) -> pd.Timestamp | None:
    try:
        hour, minute = [int(part) for part in time_text.split(":", 1)]
        local_dt = datetime(year, month, day, hour, minute, tzinfo=CHILE_DGA_LOCAL_TIMEZONE)
    except Exception:
        return None
    return pd.Timestamp(local_dt.astimezone(ZoneInfo("UTC")))


def _nullable_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _normalize_chile_station_code(value: Any) -> str | None:
    text = _nullable_str(value)
    if text is None:
        return None
    return text.split("-", 1)[0].strip()


def _to_int(value: Any) -> int | None:
    numeric = _to_float(value)
    if numeric is None:
        return None
    rounded = int(round(numeric))
    if abs(numeric - rounded) > 1e-9:
        return None
    return rounded


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if pd.isna(numeric):
        return None
    return numeric
