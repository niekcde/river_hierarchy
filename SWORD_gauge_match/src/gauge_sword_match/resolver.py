from __future__ import annotations

import pandas as pd

from .config import MatchingConfig
from .spatial_index import distance_to_geometries
from .sword_io import SwordFileCatalog, load_nodes
from .utils import expand_point_bbox, merge_bboxes


def resolve_best_matches(
    gauges: pd.DataFrame,
    scored_candidates: pd.DataFrame,
    matching_config: MatchingConfig,
) -> pd.DataFrame:
    base = gauges.copy()
    if scored_candidates.empty:
        return _unmatched_frame(base)

    ordered = scored_candidates.sort_values(
        ["station_key", "total_score", "distance_m", "reach_id"],
        ascending=[True, False, True, True],
    )
    best = ordered.groupby("station_key", as_index=False).head(1).copy()
    second = (
        ordered.groupby("station_key")["total_score"]
        .apply(lambda series: series.iloc[1] if len(series) > 1 else pd.NA)
        .rename("second_best_score")
        .reset_index()
    )
    candidate_counts = (
        ordered.groupby("station_key")
        .size()
        .rename("candidate_count")
        .reset_index()
    )
    resolved = base.merge(
        best[
            [
                "station_key",
                "reach_id",
                "sword_region",
                "source_file",
                "distance_m",
                "reach_river_name",
                "reach_drainage_proxy",
                "candidate_rank",
                "raw_score",
                "ambiguity_penalty",
                "total_score",
                "distance_score",
                "river_name_score",
                "drainage_area_score",
            ]
        ],
        on="station_key",
        how="left",
    )
    resolved = resolved.merge(second, on="station_key", how="left")
    resolved = resolved.merge(candidate_counts, on="station_key", how="left")
    resolved["score_gap"] = resolved["total_score"] - resolved["second_best_score"].fillna(0.0)
    resolved["candidate_count"] = resolved["candidate_count"].fillna(0).astype(int)
    resolved["confidence_class"] = resolved.apply(
        lambda row: _classify_confidence(row, matching_config),
        axis=1,
    )
    resolved["review_flag"] = resolved.apply(
        lambda row: _needs_review(row, matching_config),
        axis=1,
    )
    resolved["sword_node_id"] = pd.NA
    resolved["node_distance_m"] = pd.NA
    return resolved


def refine_best_matches_with_nodes(
    best_matches: pd.DataFrame,
    gauges_gdf,
    catalog: SwordFileCatalog,
    search_radius_m: float,
) -> pd.DataFrame:
    result = best_matches.copy()
    matched = result[result["reach_id"].notna()].copy()
    if matched.empty:
        return result

    gauge_lookup = gauges_gdf.set_index("station_key")
    for region, group in matched.groupby("sword_region"):
        reach_ids = group["reach_id"].dropna().unique().tolist()
        gauge_boxes = [
            expand_point_bbox(float(gauge_lookup.loc[row.station_key, "lon"]), float(gauge_lookup.loc[row.station_key, "lat"]), search_radius_m)
            for row in group.itertuples()
        ]
        nodes = load_nodes(
            catalog,
            bbox=merge_bboxes(gauge_boxes),
            regions=[region],
            columns=["node_id", "reach_id", "river_name"],
            reach_ids=reach_ids,
        )
        if nodes.empty:
            continue

        nodes_by_reach = {reach_id: frame for reach_id, frame in nodes.groupby("reach_id")}
        for row in group.itertuples():
            node_frame = nodes_by_reach.get(row.reach_id)
            if node_frame is None or node_frame.empty:
                continue
            gauge_geometry = gauge_lookup.loc[row.station_key, "geometry"]
            distances = distance_to_geometries(gauge_geometry, node_frame.geometry.tolist())
            best_idx = min(range(len(distances)), key=distances.__getitem__)
            node_row = node_frame.iloc[best_idx]
            result.loc[result["station_key"] == row.station_key, "sword_node_id"] = node_row["node_id"]
            result.loc[result["station_key"] == row.station_key, "node_distance_m"] = float(distances[best_idx])
    return result


def _classify_confidence(row: pd.Series, config: MatchingConfig) -> str:
    if pd.isna(row.get("reach_id")):
        return "unmatched"
    total_score = _safe_float(row.get("total_score"))
    score_gap = _safe_float(row.get("score_gap"))
    distance_m = _safe_float(row.get("distance_m"))
    if (
        total_score >= config.high_confidence_score
        and score_gap >= config.min_score_gap
        and distance_m <= config.review_distance_m
    ):
        return "high"
    if total_score >= config.medium_confidence_score and score_gap >= (config.min_score_gap / 2.0):
        return "medium"
    return "low"


def _needs_review(row: pd.Series, config: MatchingConfig) -> bool:
    if row["confidence_class"] in {"unmatched", "low"}:
        return True
    if pd.notna(row.get("distance_m")) and float(row["distance_m"]) > config.review_distance_m:
        return True
    if pd.notna(row.get("score_gap")) and float(row["score_gap"]) < config.min_score_gap:
        return True
    return False


def _unmatched_frame(gauges: pd.DataFrame) -> pd.DataFrame:
    result = gauges.copy()
    for column in [
        "reach_id",
        "sword_region",
        "source_file",
        "distance_m",
        "reach_river_name",
        "reach_drainage_proxy",
        "candidate_rank",
        "raw_score",
        "ambiguity_penalty",
        "total_score",
        "distance_score",
        "river_name_score",
        "drainage_area_score",
        "second_best_score",
        "score_gap",
    ]:
        result[column] = pd.NA
    result["candidate_count"] = 0
    result["confidence_class"] = "unmatched"
    result["review_flag"] = True
    result["sword_node_id"] = pd.NA
    result["node_distance_m"] = pd.NA
    return result


def _safe_float(value) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(value)
