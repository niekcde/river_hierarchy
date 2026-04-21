from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from math import isfinite
from pathlib import Path
from typing import Iterable, Sequence

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import shapely
from shapely.geometry.base import BaseGeometry

from .utils import Bbox, DEFAULT_CRS, bbox_intersects, get_logger

LOGGER = get_logger("sword_io")


@dataclass(frozen=True, slots=True)
class SwordParquetFile:
    path: Path
    kind: str
    region: str
    bbox: Bbox | None
    geometry_column: str
    geometry_encoding: str | None
    columns: tuple[str, ...]


@dataclass(slots=True)
class SwordFileCatalog:
    parquet_dir: Path
    reach_files: list[SwordParquetFile]
    node_files: list[SwordParquetFile]

    @property
    def reach_map(self) -> dict[str, SwordParquetFile]:
        return {item.region: item for item in self.reach_files}

    @property
    def node_map(self) -> dict[str, SwordParquetFile]:
        return {item.region: item for item in self.node_files}

    def select_reach_files(
        self,
        bbox: Bbox | None = None,
        continent: Sequence[str] | None = None,
        regions: Sequence[str] | None = None,
    ) -> list[SwordParquetFile]:
        return _select_files(self.reach_files, bbox=bbox, continent=continent, regions=regions)

    def select_node_files(
        self,
        bbox: Bbox | None = None,
        continent: Sequence[str] | None = None,
        regions: Sequence[str] | None = None,
    ) -> list[SwordParquetFile]:
        return _select_files(self.node_files, bbox=bbox, continent=continent, regions=regions)


def scan_sword_parquet_dir(parquet_dir: str | Path) -> SwordFileCatalog:
    root = Path(parquet_dir).expanduser().resolve()
    reach_files: list[SwordParquetFile] = []
    node_files: list[SwordParquetFile] = []

    for path in sorted(root.rglob("*.parquet")):
        if path.name.startswith("._"):
            continue
        if path.name.endswith("_reaches.parquet"):
            reach_files.append(_inspect_sword_file(path, kind="reach"))
        elif path.name.endswith("_nodes.parquet"):
            node_files.append(_inspect_sword_file(path, kind="node"))

    LOGGER.info("Discovered %s reach files and %s node files in %s", len(reach_files), len(node_files), root)
    return SwordFileCatalog(parquet_dir=root, reach_files=reach_files, node_files=node_files)


def load_reaches(
    catalog: SwordFileCatalog,
    bbox: Bbox | None = None,
    continent: Sequence[str] | None = None,
    regions: Sequence[str] | None = None,
    columns: Sequence[str] | None = None,
    reach_ids: Sequence[int | str] | None = None,
) -> gpd.GeoDataFrame:
    files = catalog.select_reach_files(bbox=bbox, continent=continent, regions=regions)
    return _load_files(files, bbox=bbox, columns=columns, reach_ids=reach_ids)


def load_nodes(
    catalog: SwordFileCatalog,
    bbox: Bbox | None = None,
    continent: Sequence[str] | None = None,
    regions: Sequence[str] | None = None,
    columns: Sequence[str] | None = None,
    reach_ids: Sequence[int | str] | None = None,
) -> gpd.GeoDataFrame:
    files = catalog.select_node_files(bbox=bbox, continent=continent, regions=regions)
    return _load_files(files, bbox=bbox, columns=columns, reach_ids=reach_ids)


def _select_files(
    files: Iterable[SwordParquetFile],
    bbox: Bbox | None = None,
    continent: Sequence[str] | None = None,
    regions: Sequence[str] | None = None,
) -> list[SwordParquetFile]:
    allowed_regions = {item.lower() for item in regions or []} or None
    allowed_continents = {item.lower() for item in continent or []} or None

    selected: list[SwordParquetFile] = []
    for item in files:
        if allowed_regions and item.region.lower() not in allowed_regions:
            continue
        if allowed_continents and item.region.lower() not in allowed_continents:
            continue
        if bbox and item.bbox and not bbox_intersects(item.bbox, bbox):
            continue
        selected.append(item)
    return selected


@lru_cache(maxsize=128)
def _inspect_sword_file(path: Path, kind: str) -> SwordParquetFile:
    parquet_file = pq.ParquetFile(path)
    metadata = parquet_file.schema_arrow.metadata or {}
    geo_meta = json.loads(metadata.get(b"geo", b"{}").decode("utf-8") or "{}")
    geometry_column = geo_meta.get("primary_column") or _guess_geometry_column(parquet_file.schema.names)
    geometry_encoding = None
    geometry_columns = geo_meta.get("columns", {})
    if geometry_column in geometry_columns:
        geometry_encoding = geometry_columns[geometry_column].get("encoding")
    bbox = None
    if geometry_column in geometry_columns:
        raw_bbox = geometry_columns[geometry_column].get("bbox")
        if raw_bbox and len(raw_bbox) == 4:
            bbox = (float(raw_bbox[0]), float(raw_bbox[1]), float(raw_bbox[2]), float(raw_bbox[3]))
    region = _infer_sword_region(path.name)
    return SwordParquetFile(
        path=path,
        kind=kind,
        region=region,
        bbox=bbox,
        geometry_column=geometry_column,
        geometry_encoding=geometry_encoding,
        columns=tuple(parquet_file.schema.names),
    )


def _infer_sword_region(filename: str) -> str:
    parts = filename.lower().split("_")
    if len(parts) >= 2 and parts[0] == "sword":
        return parts[1]
    return parts[0]


def _guess_geometry_column(columns: Sequence[str]) -> str:
    for candidate in ("geometry", "geom", "wkb_geometry"):
        if candidate in columns:
            return candidate
    raise ValueError(f"Could not infer geometry column from columns: {columns}")


def _load_files(
    files: Sequence[SwordParquetFile],
    bbox: Bbox | None = None,
    columns: Sequence[str] | None = None,
    reach_ids: Sequence[int | str] | None = None,
) -> gpd.GeoDataFrame:
    frames: list[gpd.GeoDataFrame] = []
    for item in files:
        frame = _load_single_file(item, bbox=bbox, columns=columns, reach_ids=reach_ids)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return _empty_geodataframe(columns)
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=DEFAULT_CRS)


def _load_single_file(
    item: SwordParquetFile,
    bbox: Bbox | None = None,
    columns: Sequence[str] | None = None,
    reach_ids: Sequence[int | str] | None = None,
) -> gpd.GeoDataFrame:
    requested_columns = [
        column
        for column in dict.fromkeys([*(columns or []), item.geometry_column])
        if column in item.columns
    ]
    filter_expr = _build_filter(item.columns, bbox=bbox, reach_ids=reach_ids)
    dataset = ds.dataset(item.path, format="parquet")
    table = dataset.to_table(columns=requested_columns, filter=filter_expr)
    if table.num_rows == 0:
        return _empty_geodataframe(columns)

    frame = table.to_pandas()
    geo_frame = _to_geodataframe(frame, geometry_column=item.geometry_column, geometry_encoding=item.geometry_encoding)
    geo_frame["sword_region"] = item.region
    geo_frame["source_file"] = str(item.path)
    if bbox is not None and not geo_frame.empty:
        bbox_polygon = shapely.box(*bbox)
        geo_frame = geo_frame[geo_frame.geometry.intersects(bbox_polygon)].copy()
    return geo_frame.reset_index(drop=True)


def _build_filter(
    columns: Sequence[str],
    bbox: Bbox | None = None,
    reach_ids: Sequence[int | str] | None = None,
):
    expr = None
    column_set = set(columns)
    if bbox is not None:
        min_x, min_y, max_x, max_y = bbox
        if {"x_min", "x_max", "y_min", "y_max"}.issubset(column_set):
            expr = (
                (pc.field("x_max") >= min_x)
                & (pc.field("x_min") <= max_x)
                & (pc.field("y_max") >= min_y)
                & (pc.field("y_min") <= max_y)
            )
        elif {"x", "y"}.issubset(column_set):
            expr = (
                (pc.field("x") >= min_x)
                & (pc.field("x") <= max_x)
                & (pc.field("y") >= min_y)
                & (pc.field("y") <= max_y)
            )

    if reach_ids:
        ids = pa.array(_normalize_reach_ids(reach_ids))
        reach_expr = pc.field("reach_id").isin(ids)
        expr = reach_expr if expr is None else expr & reach_expr
    return expr


def _normalize_reach_ids(reach_ids: Sequence[int | str]) -> list[int | str]:
    normalized: list[int | str] = []
    for value in reach_ids:
        if value is None or pd.isna(value):
            continue
        if isinstance(value, str):
            normalized.append(value)
            continue
        if isinstance(value, bool):
            normalized.append(int(value))
            continue
        if isinstance(value, int):
            normalized.append(value)
            continue
        if isinstance(value, float):
            if isfinite(value) and value.is_integer():
                normalized.append(int(value))
            else:
                normalized.append(str(value))
            continue
        normalized.append(str(value))
    return normalized


def _to_geodataframe(
    frame: pd.DataFrame,
    geometry_column: str,
    geometry_encoding: str | None,
) -> gpd.GeoDataFrame:
    series = frame[geometry_column]
    sample = next((value for value in series.tolist() if value is not None), None)
    geometry = None

    if geometry_encoding == "WKB" or isinstance(sample, (bytes, bytearray, memoryview)):
        values = [bytes(value) if value is not None else b"" for value in series.tolist()]
        geometry = shapely.from_wkb(values, on_invalid="ignore")
    elif isinstance(sample, BaseGeometry) or sample is None:
        geometry = series
    else:
        geometry = gpd.GeoSeries(series, crs=DEFAULT_CRS)

    result = frame.drop(columns=[geometry_column]).copy()
    return gpd.GeoDataFrame(result, geometry=geometry, crs=DEFAULT_CRS)


def _empty_geodataframe(columns: Sequence[str] | None) -> gpd.GeoDataFrame:
    data = {column: pd.Series(dtype="object") for column in columns or []}
    return gpd.GeoDataFrame(pd.DataFrame(data), geometry=gpd.GeoSeries([], crs=DEFAULT_CRS), crs=DEFAULT_CRS)
