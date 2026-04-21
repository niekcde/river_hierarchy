from __future__ import annotations

from collections import defaultdict

import geopandas as gpd
import pandas as pd

from .spatial_index import GeometrySpatialIndex
from .sword_io import SwordFileCatalog, load_reaches
from .utils import Bbox, bbox_intersects, expand_point_bbox, get_logger, intersect_bboxes, merge_bboxes

LOGGER = get_logger("candidate_search")

REACH_COLUMNS = [
    "reach_id",
    "river_name",
    "facc",
    "stream_order",
    "reach_length",
]


def search_reach_candidates(
    gauges: gpd.GeoDataFrame,
    catalog: SwordFileCatalog,
    search_radius_m: float,
    max_candidates: int,
    continent: list[str] | None = None,
    bbox: Bbox | None = None,
) -> pd.DataFrame:
    if gauges.empty:
        return _empty_candidates()

    working = gauges.reset_index(drop=True).copy()
    working["gauge_row_id"] = working.index
    gauge_boxes: dict[int, Bbox] = {}
    region_to_gauges: dict[str, set[int]] = defaultdict(set)

    for row in working.itertuples():
        gauge_bbox = expand_point_bbox(float(row.lon), float(row.lat), search_radius_m)
        if bbox is not None and not bbox_intersects(gauge_bbox, bbox):
            continue
        gauge_boxes[row.gauge_row_id] = gauge_bbox
        candidate_files = catalog.select_reach_files(bbox=gauge_bbox, continent=continent)
        for info in candidate_files:
            region_to_gauges[info.region].add(row.gauge_row_id)

    rows: list[dict[str, object]] = []
    for region, gauge_ids in sorted(region_to_gauges.items()):
        region_bbox = merge_bboxes(gauge_boxes[gauge_id] for gauge_id in gauge_ids)
        region_bbox = intersect_bboxes(region_bbox, bbox)
        reach_frame = load_reaches(
            catalog,
            bbox=region_bbox,
            regions=[region],
            columns=REACH_COLUMNS,
        )
        if reach_frame.empty:
            LOGGER.warning("No reaches loaded for region %s and bbox %s", region, region_bbox)
            continue

        spatial_index = GeometrySpatialIndex(reach_frame)
        LOGGER.info("Searching %s gauges against %s candidate reaches in region %s", len(gauge_ids), len(reach_frame), region)

        for gauge_id in gauge_ids:
            gauge_row = working.loc[working["gauge_row_id"] == gauge_id].iloc[0]
            matches = spatial_index.query(gauge_row.geometry, search_radius_m=search_radius_m, max_results=max_candidates)
            for item in matches:
                reach_row = reach_frame.iloc[item.row_index]
                rows.append(
                    {
                        "station_key": gauge_row["station_key"],
                        "station_id": gauge_row["station_id"],
                        "country": gauge_row["country"],
                        "gauge_name": gauge_row["station_name"],
                        "gauge_river_name": gauge_row["river_name"],
                        "gauge_drainage_area": gauge_row["drainage_area"],
                        "gauge_lat": gauge_row["lat"],
                        "gauge_lon": gauge_row["lon"],
                        "reach_id": reach_row.get("reach_id"),
                        "sword_region": reach_row.get("sword_region"),
                        "source_file": reach_row.get("source_file"),
                        "reach_river_name": reach_row.get("river_name"),
                        "reach_drainage_proxy": reach_row.get("facc"),
                        "reach_length": reach_row.get("reach_length"),
                        "stream_order": reach_row.get("stream_order"),
                        "distance_m": item.distance_m,
                    }
                )

    if not rows:
        return _empty_candidates()

    candidates = pd.DataFrame(rows)
    candidates = (
        candidates.sort_values(["station_key", "distance_m", "reach_id"], ascending=[True, True, True])
        .drop_duplicates(subset=["station_key", "reach_id"], keep="first")
        .reset_index(drop=True)
    )
    candidates["candidate_rank"] = (
        candidates.groupby("station_key")["distance_m"].rank(method="first", ascending=True).astype(int)
    )
    candidates["candidate_count"] = candidates.groupby("station_key")["reach_id"].transform("count").astype(int)
    return candidates


def _empty_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "station_key",
            "station_id",
            "country",
            "gauge_name",
            "gauge_river_name",
            "gauge_drainage_area",
            "gauge_lat",
            "gauge_lon",
            "reach_id",
            "sword_region",
            "source_file",
            "reach_river_name",
            "reach_drainage_proxy",
            "reach_length",
            "stream_order",
            "distance_m",
            "candidate_rank",
            "candidate_count",
        ]
    )
