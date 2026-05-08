from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from .sword_io import SwordFileCatalog, load_reaches
from .utils import read_table, write_json, write_table


def compute_summary_metrics(best_matches: pd.DataFrame) -> dict[str, int | float | None]:
    matched_mask = best_matches["reach_id"].notna()
    matched_distances = pd.to_numeric(best_matches.loc[matched_mask, "distance_m"], errors="coerce")
    return {
        "number_of_gauges": int(len(best_matches)),
        "matched_gauges": int(matched_mask.sum()),
        "unmatched_gauges": int((~matched_mask).sum()),
        "median_distance_m": None if matched_distances.empty else float(matched_distances.median()),
        "low_confidence_count": int((best_matches["confidence_class"] == "low").sum()),
        "review_flag_count": int(best_matches["review_flag"].sum()),
        "node_refined_count": int(best_matches["sword_node_id"].notna().sum()),
    }


def export_summary_metrics(best_matches: pd.DataFrame, path) -> None:
    write_json(compute_summary_metrics(best_matches), path)


def export_review_queue(best_matches: pd.DataFrame, path) -> None:
    review_queue = best_matches[best_matches["review_flag"]].copy()
    review_queue["review_reason"] = review_queue.apply(_review_reason, axis=1)
    write_table(review_queue, path)


def export_qgis_package(
    best_matches: pd.DataFrame,
    gauges_gdf: gpd.GeoDataFrame,
    catalog: SwordFileCatalog,
    output_path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    matched_gauges = gauges_gdf.merge(
        best_matches[
            [
                "station_key",
                "reach_id",
                "sword_region",
                "sword_node_id",
                "distance_m",
                "total_score",
                "second_best_score",
                "score_gap",
                "confidence_class",
                "review_flag",
                "node_distance_m",
            ]
        ],
        on="station_key",
        how="left",
    )
    matched_gauges = matched_gauges[matched_gauges["reach_id"].notna()].copy()
    matched_gauges.to_file(output_path, layer="matched_gauges", driver="GPKG")

    matched_reaches = _build_matched_reaches_layer(best_matches, catalog)
    matched_reaches.to_file(output_path, layer="matched_reaches", driver="GPKG")


def export_subdaily_hierarchy_package(
    hierarchy_examples_path,
    audit_path,
    output_path,
    *,
    layer: str = "hierarchy_examples_filtered",
    manifests_dir=None,
) -> None:
    hierarchy_examples_path = Path(hierarchy_examples_path)
    audit_path = Path(audit_path)
    output_path = Path(output_path)
    manifests_dir = None if manifests_dir is None else Path(manifests_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    examples = gpd.read_file(hierarchy_examples_path, layer=layer)
    audit = read_table(audit_path)
    audit_prepared = _prepare_subdaily_audit_for_export(audit)
    manifest_prepared = _prepare_download_manifests_for_export(manifests_dir)
    if manifest_prepared is not None:
        audit_prepared = audit_prepared.merge(manifest_prepared, on="station_key", how="left")

    points_layer = examples.merge(audit_prepared, on="station_key", how="left")
    points_layer.to_file(output_path, layer=layer, driver="GPKG")

    summary_layer = _build_subdaily_station_summary(points_layer)
    summary_layer.to_file(output_path, layer="subdaily_station_summary", driver="GPKG")


def _build_matched_reaches_layer(best_matches: pd.DataFrame, catalog: SwordFileCatalog) -> gpd.GeoDataFrame:
    matched = best_matches[best_matches["reach_id"].notna()].copy()
    if matched.empty:
        return gpd.GeoDataFrame({"reach_id": []}, geometry=gpd.GeoSeries([], crs="EPSG:4326"), crs="EPSG:4326")

    reach_layers: list[gpd.GeoDataFrame] = []
    for region, group in matched.groupby("sword_region"):
        reach_ids = group["reach_id"].dropna().unique().tolist()
        reaches = load_reaches(
            catalog,
            regions=[region],
            columns=["reach_id", "river_name", "facc", "stream_order", "reach_length"],
            reach_ids=reach_ids,
        )
        if reaches.empty:
            continue

        summary = (
            group.groupby(["sword_region", "reach_id"], as_index=False)
            .agg(
                matched_gauge_count=("station_key", "count"),
                best_score_max=("total_score", "max"),
                best_score_min=("total_score", "min"),
                confidence_modes=("confidence_class", lambda values: ",".join(sorted(set(values.astype(str))))),
            )
        )
        reaches = reaches.merge(summary, on=["sword_region", "reach_id"], how="left")
        reach_layers.append(reaches)

    if not reach_layers:
        return gpd.GeoDataFrame({"reach_id": []}, geometry=gpd.GeoSeries([], crs="EPSG:4326"), crs="EPSG:4326")
    return gpd.GeoDataFrame(pd.concat(reach_layers, ignore_index=True), geometry="geometry", crs="EPSG:4326")


def _prepare_subdaily_audit_for_export(audit: pd.DataFrame) -> pd.DataFrame:
    if "station_key" not in audit.columns:
        raise ValueError("Subdaily audit table must include a 'station_key' column.")
    if "status" not in audit.columns:
        raise ValueError("Subdaily audit table must include a 'status' column.")

    working = audit.copy()
    if "daily_available_explicit" not in working.columns:
        working["daily_available_explicit"] = False
    working["subdaily_found"] = working["status"].astype("string").fillna(pd.NA).str.lower().eq("subdaily_found")

    keep_columns = [
        column
        for column in [
            "station_key",
            "country",
            "source_station_id",
            "provider",
            "status",
            "subdaily_found",
            "daily_available_explicit",
            "daily_audit_class",
            "resolved_site_number",
            "resolved_station_name",
            "resolution_method",
            "resolution_distance_m",
            "daily_begin",
            "daily_end",
            "manual_option_type",
            "manual_option_url",
            "manual_option_note",
            "reason_summary",
            "notes",
        ]
        if column in working.columns
    ]
    prepared = working[keep_columns].copy()
    prepared = prepared.drop_duplicates(subset=["station_key"], keep="first").reset_index(drop=True)
    return prepared


def _prepare_download_manifests_for_export(manifests_dir: Path | None) -> pd.DataFrame | None:
    if manifests_dir is None:
        return None
    if not manifests_dir.exists() or not manifests_dir.is_dir():
        raise ValueError(f"Subdaily manifest directory does not exist: {manifests_dir}")

    manifest_paths = sorted(manifests_dir.glob("*/subdaily_download_manifest.csv"))
    if not manifest_paths:
        return None

    frames: list[pd.DataFrame] = []
    for path in manifest_paths:
        manifest = pd.read_csv(path)
        if "station_key" not in manifest.columns:
            continue
        manifest["manifest_country"] = path.parent.name.upper()
        frames.append(manifest)

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True, sort=False)
    keep_columns = [
        column
        for column in [
            "station_key",
            "provider_station_id",
            "download_status",
            "window_strategy",
            "requested_start",
            "requested_end",
            "raw_returned_start",
            "raw_returned_end",
            "raw_row_count",
            "selected_start",
            "selected_end",
            "selected_row_count",
            "median_timestep_hours",
            "completeness_ratio",
            "max_gap_days",
            "provider_series_name",
            "provider_series_id",
            "notes",
            "manifest_country",
        ]
        if column in combined.columns
    ]
    prepared = combined[keep_columns].copy()
    rename_map = {
        "provider_station_id": "download_provider_station_id",
        "download_status": "download_status",
        "window_strategy": "download_window_strategy",
        "requested_start": "download_requested_start",
        "requested_end": "download_requested_end",
        "raw_returned_start": "download_raw_returned_start",
        "raw_returned_end": "download_raw_returned_end",
        "raw_row_count": "download_raw_row_count",
        "selected_start": "download_selected_start",
        "selected_end": "download_selected_end",
        "selected_row_count": "download_selected_row_count",
        "median_timestep_hours": "download_median_timestep_hours",
        "completeness_ratio": "download_completeness_ratio",
        "max_gap_days": "download_max_gap_days",
        "provider_series_name": "download_provider_series_name",
        "provider_series_id": "download_provider_series_id",
        "notes": "download_notes",
        "manifest_country": "download_manifest_country",
    }
    prepared = prepared.rename(columns=rename_map)
    prepared = prepared.drop_duplicates(subset=["station_key"], keep="first").reset_index(drop=True)
    return prepared


def _build_subdaily_station_summary(points_layer: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if points_layer.empty:
        return gpd.GeoDataFrame({"station_key": []}, geometry=gpd.GeoSeries([], crs=points_layer.crs), crs=points_layer.crs)

    non_geometry_columns = [column for column in points_layer.columns if column != points_layer.geometry.name]
    summary_rows: list[dict[str, object]] = []
    for station_key, group in points_layer.groupby("station_key", dropna=False, sort=True):
        summary: dict[str, object] = {"station_key": station_key}
        for column in non_geometry_columns:
            if column == "station_key":
                continue
            if column == "example_id":
                numeric_values = pd.to_numeric(group[column], errors="coerce").dropna().tolist()
                summary["example_ids"] = ";".join(
                    str(int(value)) if float(value).is_integer() else str(value)
                    for value in sorted(set(numeric_values))
                )
                summary["example_count"] = len(set(numeric_values))
                continue
            if column == "down":
                values = sorted({str(value).strip() for value in group[column].dropna() if str(value).strip()})
                summary["down_values"] = ";".join(values)
                continue
            if column == points_layer.geometry.name:
                continue
            non_null = group[column].dropna()
            summary[column] = non_null.iloc[0] if not non_null.empty else pd.NA

        if "example_count" not in summary:
            summary["example_count"] = int(group["example_id"].notna().sum()) if "example_id" in group.columns else len(group)
        if "example_ids" not in summary:
            summary["example_ids"] = ""
        if "down_values" not in summary:
            summary["down_values"] = ""
        summary["occurrence_count"] = int(len(group))
        summary["geometry"] = group.geometry.iloc[0]
        summary_rows.append(summary)

    summary_frame = pd.DataFrame(summary_rows)
    return gpd.GeoDataFrame(summary_frame, geometry="geometry", crs=points_layer.crs)


def _review_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if pd.isna(row.get("reach_id")):
        reasons.append("no_reach_within_radius")
    if row.get("confidence_class") == "low":
        reasons.append("low_score")
    if pd.notna(row.get("score_gap")) and float(row["score_gap"]) < 0.1:
        reasons.append("small_score_gap")
    if pd.notna(row.get("distance_m")) and float(row["distance_m"]) > 2_500:
        reasons.append("long_distance")
    if pd.isna(row.get("sword_node_id")) and pd.notna(row.get("reach_id")):
        reasons.append("missing_node_refinement")
    return ",".join(reasons) if reasons else "manual_check"
