from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any, Callable

import pandas as pd

LOGGER = logging.getLogger("grdc_convert")

FILE_PATTERN = "*_Q_Day.Cmd.txt"
DEFAULT_INPUT_DIR = Path("/Volumes/PhD/river_hierarchy/data/GRDC")
DEFAULT_METADATA_OUT = Path("outputs/grdc_station_metadata.parquet")
DEFAULT_TIMESERIES_OUT = Path("outputs/grdc_timeseries.parquet")
MISSING_SENTINEL = -999.0

TIMESERIES_COLUMNS = [
    "station_key",
    "station_id",
    "country",
    "time",
    "discharge",
    "variable",
    "source_function",
    "grdc_no",
]

METADATA_COLUMNS = [
    "station_key",
    "station_id",
    "country",
    "agency",
    "source_function",
    "grdc_no",
    "station_name",
    "river_name",
    "lat",
    "lon",
    "drainage_area",
    "altitude_m_asl",
    "next_downstream_station",
    "owner_of_original_data",
    "data_set_content",
    "unit_of_measure",
    "time_series_label",
    "record_start_month",
    "record_end_month",
    "record_years",
    "last_update",
    "file_generation_date",
    "data_lines",
    "raw_file",
]

HEADER_FIELD_MAP: dict[str, tuple[str, Callable[[str], object]]] = {
    "file generation date": ("file_generation_date", lambda value: _parse_date(value)),
    "grdc no": ("grdc_no", lambda value: _parse_station_id(value)),
    "river": ("river_name", lambda value: _clean_string(value)),
    "station": ("station_name", lambda value: _clean_string(value)),
    "country": ("country", lambda value: _clean_string(value).upper()),
    "latitude dd": ("lat", lambda value: _parse_float(value)),
    "longitude dd": ("lon", lambda value: _parse_float(value)),
    "catchment area km": ("drainage_area", lambda value: _parse_float(value)),
    "catchment area km2": ("drainage_area", lambda value: _parse_float(value)),
    "altitude m asl": ("altitude_m_asl", lambda value: _parse_float(value)),
    "next downstream station": ("next_downstream_station", lambda value: _parse_station_id(value, allow_missing=True)),
    "owner of original data": ("owner_of_original_data", lambda value: _clean_string(value)),
    "data set content": ("data_set_content", lambda value: _clean_string(value)),
    "unit of measure": ("unit_of_measure", lambda value: _clean_string(value)),
    "time series": ("time_series_label", lambda value: _clean_string(value)),
    "no of years": ("record_years", lambda value: _parse_float(value)),
    "last update": ("last_update", lambda value: _parse_date(value)),
    "data lines": ("data_lines", lambda value: _parse_float(value)),
}


def parse_grdc_station_file(path: str | Path) -> tuple[dict[str, object], pd.DataFrame]:
    path = Path(path)
    header_values: dict[str, object] = {}
    data_rows: list[dict[str, object]] = []
    in_data = False
    data_header_seen = False

    with path.open("r", encoding="latin-1", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")

            if not in_data:
                if line.strip() == "# DATA":
                    in_data = True
                    continue
                parsed = _parse_header_line(line)
                if parsed is not None:
                    field_name, field_value = parsed
                    header_values[field_name] = field_value
                continue

            if not data_header_seen:
                if line.strip() and not line.startswith("#"):
                    data_header_seen = True
                continue

            if not line.strip() or line.startswith("#"):
                continue

            parsed_row = _parse_data_line(line)
            if parsed_row is None:
                continue
            data_rows.append(parsed_row)

    metadata = _build_metadata_record(header_values, path)
    timeseries = _build_timeseries_frame(data_rows, metadata)
    return metadata, timeseries


def convert_grdc_download(
    input_dir: str | Path,
    metadata_out: str | Path,
    timeseries_out: str | Path,
    *,
    overwrite: bool = False,
    batch_rows: int = 250_000,
) -> tuple[Path, Path]:
    input_dir = Path(input_dir)
    metadata_out = Path(metadata_out)
    timeseries_out = Path(timeseries_out)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not overwrite:
        for path in (metadata_out, timeseries_out):
            if path.exists():
                raise FileExistsError(f"Output already exists: {path}. Pass --overwrite to replace it.")

    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    timeseries_out.parent.mkdir(parents=True, exist_ok=True)

    station_files = discover_station_files(input_dir)
    if not station_files:
        raise FileNotFoundError(f"No GRDC station files matching {FILE_PATTERN} were found under {input_dir}")

    LOGGER.info("Discovered %s GRDC station files in %s", len(station_files), input_dir)

    metadata_records: list[dict[str, object]] = []
    pending_frames: list[pd.DataFrame] = []
    pending_rows = 0
    writer: Any | None = None
    total_rows = 0

    for index, station_file in enumerate(station_files, start=1):
        metadata, timeseries = parse_grdc_station_file(station_file)
        metadata_records.append(metadata)

        if not timeseries.empty:
            pending_frames.append(timeseries)
            pending_rows += len(timeseries)
            total_rows += len(timeseries)

        if pending_rows >= batch_rows:
            writer = _flush_timeseries_batch(pending_frames, writer, timeseries_out)
            pending_frames = []
            pending_rows = 0

        if index % 250 == 0 or index == len(station_files):
            LOGGER.info("Parsed %s/%s station files", index, len(station_files))

    writer = _flush_timeseries_batch(pending_frames, writer, timeseries_out)
    if writer is not None:
        writer.close()

    metadata_frame = pd.DataFrame.from_records(metadata_records)
    metadata_frame = metadata_frame.reindex(columns=METADATA_COLUMNS)
    metadata_frame.to_parquet(metadata_out, index=False)

    LOGGER.info("Wrote %s metadata rows to %s", len(metadata_frame), metadata_out)
    LOGGER.info("Wrote %s timeseries rows to %s", total_rows, timeseries_out)
    return metadata_out, timeseries_out


def discover_station_files(input_dir: str | Path) -> list[Path]:
    root = Path(input_dir)
    return sorted(
        path
        for path in root.rglob(FILE_PATTERN)
        if path.is_file() and not path.name.startswith("._")
    )


def _parse_header_line(line: str) -> tuple[str, object] | None:
    if not line.startswith("#"):
        return None
    content = line[1:].strip()
    if not content or ":" not in content:
        return None

    raw_key, raw_value = content.split(":", 1)
    normalized_key = _normalize_header_key(raw_key)
    mapped = HEADER_FIELD_MAP.get(normalized_key)
    if mapped is None:
        return None

    field_name, parser = mapped
    return field_name, parser(raw_value.strip())


def _parse_data_line(line: str) -> tuple[pd.Timestamp, float] | None:
    parts = [part.strip() for part in line.split(";")]
    if len(parts) < 3:
        return None

    timestamp = _parse_timestamp(parts[0], parts[1])
    discharge = _parse_float(parts[2], allow_missing=True)
    if timestamp is None or discharge is None or discharge == MISSING_SENTINEL or discharge < 0:
        return None
    return timestamp, discharge


def _build_metadata_record(header_values: dict[str, object], path: Path) -> dict[str, object]:
    station_id = _parse_station_id(header_values.get("grdc_no"), allow_missing=False)
    if not station_id:
        station_id = path.name.split("_", 1)[0]

    country = _clean_string(header_values.get("country")).upper()
    station_key = f"{country}:{station_id}" if country else str(station_id)

    time_series_label = _clean_string(header_values.get("time_series_label"))
    record_start_month, record_end_month = _parse_time_series_range(time_series_label)

    record = {
        "station_key": station_key,
        "station_id": station_id,
        "country": country or pd.NA,
        "agency": "GRDC",
        "source_function": "grdc",
        "grdc_no": station_id,
        "station_name": _clean_string(header_values.get("station_name")) or pd.NA,
        "river_name": _clean_string(header_values.get("river_name")) or pd.NA,
        "lat": header_values.get("lat", pd.NA),
        "lon": header_values.get("lon", pd.NA),
        "drainage_area": header_values.get("drainage_area", pd.NA),
        "altitude_m_asl": header_values.get("altitude_m_asl", pd.NA),
        "next_downstream_station": header_values.get("next_downstream_station", pd.NA),
        "owner_of_original_data": _clean_string(header_values.get("owner_of_original_data")) or pd.NA,
        "data_set_content": _clean_string(header_values.get("data_set_content")) or pd.NA,
        "unit_of_measure": _clean_string(header_values.get("unit_of_measure")) or pd.NA,
        "time_series_label": time_series_label or pd.NA,
        "record_start_month": record_start_month,
        "record_end_month": record_end_month,
        "record_years": header_values.get("record_years", pd.NA),
        "last_update": header_values.get("last_update", pd.NA),
        "file_generation_date": header_values.get("file_generation_date", pd.NA),
        "data_lines": header_values.get("data_lines", pd.NA),
        "raw_file": str(path),
    }
    return record


def _build_timeseries_frame(data_rows: list[tuple[pd.Timestamp, float]], metadata: dict[str, object]) -> pd.DataFrame:
    if not data_rows:
        return pd.DataFrame(columns=TIMESERIES_COLUMNS)

    station_key = metadata["station_key"]
    station_id = metadata["station_id"]
    country = metadata["country"]
    grdc_no = metadata["grdc_no"]

    frame = pd.DataFrame(
        {
            "station_key": [station_key] * len(data_rows),
            "station_id": [station_id] * len(data_rows),
            "country": [country] * len(data_rows),
            "time": [item[0] for item in data_rows],
            "discharge": [item[1] for item in data_rows],
            "variable": ["discharge"] * len(data_rows),
            "source_function": ["grdc"] * len(data_rows),
            "grdc_no": [grdc_no] * len(data_rows),
        }
    )
    frame = frame.sort_values("time", kind="mergesort").drop_duplicates(subset="time", keep="last").reset_index(drop=True)
    return frame[TIMESERIES_COLUMNS]


def _flush_timeseries_batch(
    frames: list[pd.DataFrame],
    writer: Any | None,
    output_path: Path,
) -> Any | None:
    if not frames:
        return writer

    import pyarrow as pa
    import pyarrow.parquet as pq

    batch = pd.concat(frames, ignore_index=True)
    batch = batch.reindex(columns=TIMESERIES_COLUMNS)
    batch["station_id"] = batch["station_id"].astype("string")
    batch["country"] = batch["country"].astype("string")
    batch["grdc_no"] = batch["grdc_no"].astype("string")
    batch["variable"] = batch["variable"].astype("string")
    batch["source_function"] = batch["source_function"].astype("string")
    batch["station_key"] = batch["station_key"].astype("string")
    batch["time"] = pd.to_datetime(batch["time"], errors="coerce")
    batch["discharge"] = pd.to_numeric(batch["discharge"], errors="coerce")
    batch = batch.dropna(subset=["station_id", "country", "time", "discharge"]).copy()
    batch = batch[batch["discharge"] >= 0].copy()
    batch = batch.sort_values(["station_key", "time"], kind="mergesort").reset_index(drop=True)

    if batch.empty:
        return writer

    table = pa.Table.from_pandas(batch, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
    writer.write_table(table)
    return writer


def _normalize_header_key(value: str) -> str:
    lowered = str(value).strip().lower()
    lowered = lowered.replace("�", "")
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _clean_string(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text in {"-", "", "nan", "None", "<NA>"}:
        return ""
    return text


def _parse_float(value: object, allow_missing: bool = False) -> float | None:
    text = _clean_string(value)
    if not text:
        return None if allow_missing else pd.NA
    try:
        return float(text)
    except ValueError:
        return None if allow_missing else pd.NA


def _parse_date(value: object) -> pd.Timestamp | object:
    text = _clean_string(value)
    if not text:
        return pd.NA
    parsed = pd.to_datetime(text, errors="coerce")
    return parsed if pd.notna(parsed) else pd.NA


def _parse_timestamp(date_value: str, time_value: str) -> pd.Timestamp | None:
    date_text = _clean_string(date_value)
    if not date_text:
        return None
    time_text = _clean_string(time_value)
    if not time_text or time_text == "--:--":
        parsed = pd.to_datetime(date_text, errors="coerce")
    else:
        parsed = pd.to_datetime(f"{date_text} {time_text}", errors="coerce")
    return parsed if pd.notna(parsed) else None


def _parse_station_id(value: object, allow_missing: bool = False) -> str | object:
    text = _clean_string(value)
    if not text:
        return pd.NA if allow_missing else ""
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def _parse_time_series_range(value: object) -> tuple[object, object]:
    text = _clean_string(value)
    if not text:
        return pd.NA, pd.NA
    match = re.match(r"^\s*(\d{4}-\d{2})\s*-\s*(\d{4}-\d{2})\s*$", text)
    if not match:
        return pd.NA, pd.NA
    return match.group(1), match.group(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert GRDC station text downloads into pipeline-friendly Parquet files.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory containing GRDC *_Q_Day.Cmd.txt files.")
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=DEFAULT_METADATA_OUT,
        help="Output Parquet path for one-row-per-station metadata.",
    )
    parser.add_argument(
        "--timeseries-out",
        type=Path,
        default=DEFAULT_TIMESERIES_OUT,
        help="Output Parquet path for pipeline-ready long timeseries rows.",
    )
    parser.add_argument("--batch-rows", type=int, default=250_000, help="Approximate row count to buffer before writing a timeseries batch.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--log-level", default="INFO", help="Logging level, for example INFO or DEBUG.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        force=True,
    )
    metadata_out, timeseries_out = convert_grdc_download(
        input_dir=args.input_dir,
        metadata_out=args.metadata_out,
        timeseries_out=args.timeseries_out,
        overwrite=bool(args.overwrite),
        batch_rows=max(1, int(args.batch_rows)),
    )
    LOGGER.info("Metadata parquet: %s", metadata_out)
    LOGGER.info("Timeseries parquet: %s", timeseries_out)


if __name__ == "__main__":
    main()
