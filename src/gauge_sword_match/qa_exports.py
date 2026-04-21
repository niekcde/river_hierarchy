from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from .sword_io import SwordFileCatalog, load_reaches
from .utils import write_json, write_table


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
