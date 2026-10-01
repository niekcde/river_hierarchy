"""Observed peak-propagation reference speeds along single-channel geometry."""
from pathlib import Path
import json

import numpy as np
import pandas as pd


def projected_gauge_length(line, upstream, downstream):
    """Distance between gauge projections on one continuous metric centreline."""
    if line.geom_type != "LineString" or line.is_empty or not line.is_simple:
        raise ValueError("Reference geometry must be one simple, continuous LineString")
    length = abs(line.project(downstream) - line.project(upstream))
    if not np.isfinite(length) or length <= 0:
        raise ValueError("Gauge projections must define a positive reference distance")
    return length, line.distance(upstream), line.distance(downstream)


def absolute_speed_change(lag_change_pct, reference_length_m, observed_lag_hours):
    """Signed m/s change at fixed distance; invalid reference times remain NaN."""
    r, length, hours = np.broadcast_arrays(
        np.asarray(lag_change_pct, dtype=float) / 100.,
        np.asarray(reference_length_m, dtype=float), np.asarray(observed_lag_hours, dtype=float))
    valid = np.isfinite(r) & np.isfinite(length) & np.isfinite(hours) & (r > -1) & (length > 0) & (hours > 0)
    result = np.full(r.shape, np.nan)
    result[valid] = -r[valid] / (1 + r[valid]) * length[valid] / (hours[valid] * 3600.)
    return result


def build_speed_reference_frame(example_ids, *, output_root, gauge_root=None):
    """Read existing reference geometry and observed gauge peaks; write nothing.

    Gauge coordinates prefer the manually updated source used by experiments.
    Each example uses its independent experiment's fixed single-thread reference.
    Invalid/missing inputs receive an explicit status, never a surrogate speed.
    """
    import geopandas as gpd
    from shapely.ops import linemerge, unary_union
    from .analysis import _load_gauge_quality_sheet, _observed_gauge_peak_shift_hours

    output_root = Path(output_root)
    gauge_root = Path(gauge_root) if gauge_root else Path(__file__).resolve().parents[3] / "SWORD_gauge_match/outputs"
    quality = _load_gauge_quality_sheet()
    if quality is None:
        raise FileNotFoundError("Observed gauge quality table is unavailable")
    station_frames = []
    for name in ("hierarchy_examples_filtered_subdaily_manual_updates_final.gpkg",
                 "selected_event_stations_same_main_path.gpkg", "gauges_cleaned.parquet"):
        path = gauge_root / name
        if not path.exists():
            continue
        if path.suffix == ".parquet":
            data = pd.read_parquet(path).dropna(subset=["lon", "lat"])
            stations = gpd.GeoDataFrame(data[["station_key"]], geometry=gpd.points_from_xy(data.lon, data.lat), crs=4326)
        else:
            stations = gpd.read_file(path).to_crs(4326)
        stations = stations[["station_key", "geometry"]].copy()
        stations["coordinate_source"] = name
        station_frames.append(stations)
    if not station_frames:
        raise FileNotFoundError("No gauge coordinate sources are available")
    stations = gpd.GeoDataFrame(pd.concat(station_frames, ignore_index=True), crs=4326)
    stations = stations.dropna(subset=["geometry"]).drop_duplicates("station_key").set_index("station_key")
    rows = []
    for value in example_ids:
        example = str(value).zfill(2)
        row = {"example_id": example, "reference_length_m": np.nan,
               "observed_lag_hours": np.nan, "reference_speed_mps": np.nan, "status": "unavailable"}
        try:
            match = quality.loc[pd.to_numeric(quality.Example, errors="coerce").eq(int(example))]
            if len(match) != 1:
                raise ValueError("Expected one observed-gauge configuration")
            up, down = str(match.iloc[0].Up).strip(), str(match.iloc[0].Down).strip()
            row.update(up_station=up, down_station=down)
            ref = output_root / f"{example}_indep__anchored/reference_section"
            summary = json.loads((ref / "reference_section_summary.json").read_text())
            if not summary.get("single_thread"):
                raise ValueError("Reference geometry is not marked single-thread")
            path = ref / "variant/directed" / Path(summary["paths"]["directed_links_path"]).name
            links = gpd.read_file(path)
            if links.crs is None:
                raise ValueError("Reference geometry has no CRS")
            if not links.crs.is_projected or links.crs.axis_info[0].unit_conversion_factor != 1:
                links = links.to_crs(links.estimate_utm_crs())
            union = unary_union(links.geometry)
            line = union if union.geom_type == "LineString" else linemerge(union)
            points = stations.loc[[up, down]].to_crs(links.crs)
            length, up_offset, down_offset = projected_gauge_length(line, *points.geometry)
            hours = _observed_gauge_peak_shift_hours(example)
            row.update(reference_length_m=length, observed_lag_hours=hours,
                       up_snap_distance_m=up_offset, down_snap_distance_m=down_offset,
                       up_coordinate_source=points.iloc[0].coordinate_source,
                       down_coordinate_source=points.iloc[1].coordinate_source,
                       reference_geometry=str(path), length_method="between_gauge_projections_on_single_thread_reference")
            if not np.isfinite(hours) or hours <= 0:
                raise ValueError("Observed downstream-minus-upstream peak lag is not positive")
            row.update(reference_speed_mps=length/(hours*3600.), status="ok")
        except (OSError, ValueError, KeyError, IndexError) as exc:
            row["status"] = str(exc)
        rows.append(row)
    return pd.DataFrame(rows)
