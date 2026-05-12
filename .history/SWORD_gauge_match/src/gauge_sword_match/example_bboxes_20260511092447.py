from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from .sword_io import load_reaches, scan_sword_parquet_dir


DEFAULT_SWORD_PARQUET_DIR = Path("/Volumes/PhD/SWORD/v17c/beta/parquet")
DEFAULT_WIDTH_FIELD = "width_obs_p50"


def build_example_bbox_layer(
    station_summary_path: str | Path,
    *,
    input_layer: str = "subdaily_station_summary",
    reach_lists_path: str | Path | None = None,
    reach_summary_path: str | Path | None = None,
    sword_parquet_dir: str | Path = DEFAULT_SWORD_PARQUET_DIR,
    width_field: str = DEFAULT_WIDTH_FIELD,
    buffer_multiplier: float = 1.0,
    fallback_buffer_m: float = 0.0,
    ) -> gpd.GeoDataFrame:
    station_summary_path = Path(station_summary_path).expanduser().resolve()
    reach_lists_path = _resolve_default_path(
        reach_lists_path,
        station_summary_path.with_name("hierarchy_example_reach_lists.parquet"),
    )
    reach_summary_path = _resolve_default_path(
        reach_summary_path,
        station_summary_path.with_name("hierarchy_example_reach_summary.csv"),
    )






    stations = gpd.read_file(station_summary_path, layer=input_layer)
    if stations.empty:
        return gpd.GeoDataFrame({"example_id": []}, geometry=gpd.GeoSeries([], crs=stations.crs), crs=stations.crs)
    if "example_ids" not in stations.columns:
        raise ValueError(f"Layer '{input_layer}' in {station_summary_path} does not contain an 'example_ids' column.")

    station_examples = _explode_station_examples(stations)
    route_table      = _load_example_route_table(reach_lists_path, reach_summary_path)
    reaches          = _load_example_reaches(
        route_table,
        sword_parquet_dir=sword_parquet_dir,
        width_field=width_field,
        crs=stations.crs,
    )

    rows: list[dict[str, Any]] = []
    geometry_name = stations.geometry.name
    route_by_example = route_table.set_index("example_id") if not route_table.empty else pd.DataFrame().set_index(pd.Index([], name="example_id"))

    for example_id, station_group in station_examples.groupby("example_id", sort=True):
        route_row = route_by_example.loc[example_id] if example_id in route_by_example.index else None
        route_status = str(route_row["route_status"]) if route_row is not None and "route_status" in route_row else ""
        route_found = bool(route_row["route_found"]) if route_row is not None and pd.notna(route_row["route_found"]) else False
        sword_region = str(route_row["sword_region"]) if route_row is not None and pd.notna(route_row["sword_region"]) else ""
        route_reach_ids = route_row["reach_ids"] if route_row is not None and isinstance(route_row["reach_ids"], list) else []

        example_reaches = reaches.loc[reaches["example_id"].eq(int(example_id))].copy() if not reaches.empty else reaches.iloc[0:0].copy()
        max_width_m = pd.to_numeric(example_reaches.get(width_field, pd.Series(dtype=float)), errors="coerce").max()
        if pd.isna(max_width_m):
            max_width_m = float("nan")

        buffer_m = float(fallback_buffer_m)
        if np.isfinite(max_width_m):
            buffer_m = float(max_width_m) * float(buffer_multiplier)

        combined = pd.concat(
            [
                station_group[[geometry_name]].rename(columns={geometry_name: "geometry"}),
                example_reaches[[geometry_name]].rename(columns={geometry_name: "geometry"}),
            ],
            ignore_index=True,
        )
        combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=stations.crs)
        combined = combined.loc[combined.geometry.notna()].copy()
        if combined.empty:
            continue

        metric_crs = combined.estimate_utm_crs()
        if metric_crs is None:
            metric_crs = "EPSG:3857"
        combined_metric = combined.to_crs(metric_crs)

        minx, miny, maxx, maxy = combined_metric.total_bounds
        safeguard_m = max(buffer_m, 1.0)
        if minx == maxx:
            minx -= safeguard_m
            maxx += safeguard_m
        if miny == maxy:
            miny -= safeguard_m
            maxy += safeguard_m
        bounds_geom = box(minx - buffer_m, miny - buffer_m, maxx + buffer_m, maxy + buffer_m)
        bounds_geo = gpd.GeoSeries([bounds_geom], crs=metric_crs).to_crs(stations.crs)

        rows.append(
            {
                "example_id": int(example_id),
                "station_count": int(len(station_group)),
                "station_keys": ";".join(sorted(station_group["station_key"].astype(str).unique().tolist()))
                if "station_key" in station_group.columns
                else "",
                "route_found": route_found,
                "route_status": route_status,
                "sword_region": sword_region,
                "route_reach_count": int(len(route_reach_ids)),
                "loaded_reach_count": int(len(example_reaches)),
                "width_field": width_field,
                "max_width_m": float(max_width_m) if np.isfinite(max_width_m) else np.nan,
                "buffer_multiplier": float(buffer_multiplier),
                "buffer_m": float(buffer_m),
                "geometry_source": "stations_and_reaches" if not example_reaches.empty else "stations_only",
                "geometry": bounds_geo.iloc[0],
            }
        )

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=stations.crs)


def write_example_bbox_layer(
    station_summary_path: str | Path,
    output_path: str | Path,
    *,
    input_layer: str = "subdaily_station_summary",
    output_layer: str = "example_bboxes",
    reach_lists_path: str | Path | None = None,
    reach_summary_path: str | Path | None = None,
    sword_parquet_dir: str | Path = DEFAULT_SWORD_PARQUET_DIR,
    width_field: str = DEFAULT_WIDTH_FIELD,
    buffer_multiplier: float = 1.0,
    fallback_buffer_m: float = 0.0,
    ) -> gpd.GeoDataFrame:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    layer = build_example_bbox_layer(
        station_summary_path,
        input_layer=input_layer,
        reach_lists_path=reach_lists_path,
        reach_summary_path=reach_summary_path,
        sword_parquet_dir=sword_parquet_dir,
        width_field=width_field,
        buffer_multiplier=buffer_multiplier,
        fallback_buffer_m=fallback_buffer_m,
    )
    layer.to_file(output_path, layer=output_layer, driver="GPKG")
    return layer


def _resolve_default_path(value: str | Path | None, default_path: Path) -> Path:
    return default_path if value is None else Path(value).expanduser().resolve()


def _explode_station_examples(stations: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    records: list[pd.Series] = []
    for _, row in stations.iterrows():
        example_ids = _parse_int_list(row.get("example_ids"))
        if not example_ids:
            continue
        for example_id in example_ids:
            record = row.copy()
            record["example_id"] = int(example_id)
            records.append(record)
    if not records:
        return gpd.GeoDataFrame({"example_id": []}, geometry=gpd.GeoSeries([], crs=stations.crs), crs=stations.crs)
    return gpd.GeoDataFrame(records, geometry=stations.geometry.name, crs=stations.crs).reset_index(drop=True)


def _load_example_route_table(reach_lists_path: Path, reach_summary_path: Path) -> pd.DataFrame:
    if not reach_lists_path.exists():
        return pd.DataFrame(columns=["example_id", "reach_ids", "route_found", "route_status", "sword_region"])

    reach_lists = pd.read_parquet(reach_lists_path).copy()
    keep_columns = [column for column in ("example_id", "reaches_between", "route_found") if column in reach_lists.columns]
    reach_lists = reach_lists[keep_columns].copy()
    reach_lists["reach_ids"] = reach_lists["reaches_between"].map(_parse_int_list)
    reach_lists = reach_lists.drop(columns=["reaches_between"], errors="ignore")

    if reach_summary_path.exists():
        reach_summary = pd.read_csv(reach_summary_path)
        summary_keep = [column for column in ("example_id", "sword_region", "route_status", "route_found") if column in reach_summary.columns]
        reach_summary = reach_summary[summary_keep].copy()
        reach_summary = reach_summary.rename(columns={"route_found": "route_found_summary"})
        reach_lists = reach_lists.merge(
            reach_summary,
            on="example_id",
            how="left",
        )
    else:
        reach_lists["sword_region"] = pd.NA
        reach_lists["route_status"] = ""

    if "route_found" not in reach_lists.columns:
        reach_lists["route_found"] = reach_lists["reach_ids"].map(bool)
    reach_lists["route_found"] = reach_lists["route_found"].fillna(reach_lists["reach_ids"].map(bool)).astype(bool)
    return reach_lists


def _load_example_reaches(
    route_table: pd.DataFrame,
    *,
    sword_parquet_dir: str | Path,
    width_field: str,
    crs,
    ) -> gpd.GeoDataFrame:
    if route_table.empty:
        return gpd.GeoDataFrame({"example_id": []}, geometry=gpd.GeoSeries([], crs=crs), crs=crs)

    route_candidates = route_table.loc[
        route_table["route_found"] & route_table["reach_ids"].map(bool),
        ["example_id", "sword_region", "reach_ids"],
    ].copy()
    if route_candidates.empty:
        return gpd.GeoDataFrame({"example_id": []}, geometry=gpd.GeoSeries([], crs=crs), crs=crs)

    catalog = scan_sword_parquet_dir(sword_parquet_dir)
    frames: list[gpd.GeoDataFrame] = []

    for region, group in route_candidates.groupby("sword_region", dropna=False):
        reach_ids = sorted({reach_id for values in group["reach_ids"] for reach_id in values})
        if not reach_ids:
            continue
        regions = [str(region)] if pd.notna(region) and str(region).strip() else None
        reaches = load_reaches(
            catalog,
            regions=regions,
            reach_ids=reach_ids,
            columns=["reach_id", width_field],
        )
        if reaches.empty:
            continue
        reaches = reaches.drop_duplicates(subset=["sword_region", "reach_id"]).copy()
        frame = group.explode("reach_ids").rename(columns={"reach_ids": "reach_id"}).copy()
        frame["reach_id"] = pd.to_numeric(frame["reach_id"], errors="coerce").astype("Int64")
        merged = frame.merge(
            reaches,
            on=["sword_region", "reach_id"],
            how="left",
        )
        merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=reaches.crs)
        frames.append(merged)

    if not frames:
        return gpd.GeoDataFrame({"example_id": []}, geometry=gpd.GeoSeries([], crs=crs), crs=crs)
    result = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=frames[0].crs)
    if crs is not None and result.crs != crs:
        result = result.to_crs(crs)
    return result


def _parse_int_list(value: Any) -> list[int]:
    if value is None or value is pd.NA:
        return []
    if isinstance(value, (list, tuple, set, np.ndarray)):
        result: list[int] = []
        for item in value:
            result.extend(_parse_int_list(item))
        return result
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "<na>", "none"}:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                parsed = None
            if parsed is not None:
                return _parse_int_list(parsed)
        matches = re.findall(r"\d+", text)
        return [int(match) for match in matches]
    try:
        numeric = pd.to_numeric([value], errors="coerce")[0]
    except Exception:
        return []
    if pd.isna(numeric):
        return []
    return [int(numeric)]
