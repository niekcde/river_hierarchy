from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

Bbox = tuple[float, float, float, float]
DEFAULT_CRS = "EPSG:4326"
LOGGER_NAME = "gauge_sword_match"


def get_logger(name: str | None = None) -> logging.Logger:
    if name:
        return logging.getLogger(f"{LOGGER_NAME}.{name}")
    return logging.getLogger(LOGGER_NAME)


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        force=True,
    )


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_path(path: str | Path | None, base_dir: str | Path) -> Path | None:
    if path is None:
        return None
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return Path(base_dir).joinpath(candidate).resolve()


def expand_point_bbox(lon: float, lat: float, radius_m: float) -> Bbox:
    lat_delta = radius_m / 111_320.0
    cos_lat = max(math.cos(math.radians(lat)), 0.01)
    lon_delta = radius_m / (111_320.0 * cos_lat)
    return (lon - lon_delta, lat - lat_delta, lon + lon_delta, lat + lat_delta)


def merge_bboxes(bboxes: Iterable[Bbox]) -> Bbox | None:
    items = list(bboxes)
    if not items:
        return None
    min_x = min(item[0] for item in items)
    min_y = min(item[1] for item in items)
    max_x = max(item[2] for item in items)
    max_y = max(item[3] for item in items)
    return (min_x, min_y, max_x, max_y)


def bbox_intersects(left: Bbox | None, right: Bbox | None) -> bool:
    if left is None or right is None:
        return True
    return not (
        left[2] < right[0]
        or left[0] > right[2]
        or left[3] < right[1]
        or left[1] > right[3]
    )


def intersect_bboxes(left: Bbox | None, right: Bbox | None) -> Bbox | None:
    if left is None:
        return right
    if right is None:
        return left
    if not bbox_intersects(left, right):
        return None
    return (
        max(left[0], right[0]),
        max(left[1], right[1]),
        min(left[2], right[2]),
        min(left[3], right[3]),
    )


def normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def read_table(
    path: str | Path,
    *,
    columns: list[str] | None = None,
    filters: list[tuple[str, str, object]] | None = None,
) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        kwargs: dict[str, Any] = {}
        if columns is not None:
            kwargs["columns"] = columns
        if filters is not None:
            kwargs["filters"] = filters
        return pd.read_parquet(path, **kwargs)
    if path.suffix.lower() == ".csv":
        kwargs = {}
        if columns is not None:
            kwargs["usecols"] = columns
        return pd.read_csv(path, **kwargs)
    raise ValueError(f"Unsupported file format for {path}")


def list_table_columns(path: str | Path) -> list[str]:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError:
            return list(pd.read_parquet(path).columns)
        return pq.read_schema(path).names
    if path.suffix.lower() == ".csv":
        return list(pd.read_csv(path, nrows=0).columns)
    raise ValueError(f"Unsupported file format for {path}")


def write_table(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    ensure_directory(path.parent)
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
        return path
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
        return path
    raise ValueError(f"Unsupported file format for {path}")


def write_json(payload: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def first_present(mapping: dict[str, Any], candidates: list[str]) -> Any | None:
    lower_map = {key.lower(): key for key in mapping}
    for candidate in candidates:
        source_key = lower_map.get(candidate.lower())
        if source_key is not None:
            return mapping[source_key]
    return None
