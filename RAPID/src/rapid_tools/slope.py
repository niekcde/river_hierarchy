from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class SlopeConfig:
    min_slope: float = 1e-6
    preferred_length_field: str = "len"
    interpolate_along_corridor: bool = True
    min_anchor_nodes_for_profile: int = 2
    fill_from_nearest_valid_link: bool = True
    max_slope_for_k: float | None = None
    section_slope_ratio_min: float | None = None
    section_slope_ratio_max: float | None = None
    use_section_slope_fallback: bool = True


REQUIRED_NODE_COLUMNS = {
    "id_node",
    "sword_wse",
    "sword_wse_field",
    "sword_wse_fill_method",
}


def _select_variant_lengths(
    links: gpd.GeoDataFrame,
    *,
    preferred_length_field: str,
) -> tuple[pd.Series, pd.Series]:
    geometry_length = pd.Series(np.nan, index=links.index, dtype=float)
    if links.geometry.name in links.columns and links.geometry.notna().any():
        if links.crs is not None and not links.crs.is_geographic:
            geometry_length = links.geometry.length.astype(float)

    field_length = pd.Series(np.nan, index=links.index, dtype=float)
    if preferred_length_field in links.columns:
        field_length = pd.to_numeric(links[preferred_length_field], errors="coerce")

    use_geometry = geometry_length.gt(0)
    lengths = geometry_length.where(use_geometry, field_length)
    length_source = pd.Series(
        np.where(use_geometry, "geometry", preferred_length_field),
        index=links.index,
        dtype="object",
    )

    if lengths.isna().any() or (lengths <= 0).any():
        bad = links.loc[lengths.isna() | lengths.le(0), "id_link"].tolist() if "id_link" in links.columns else []
        raise ValueError(
            "Could not resolve a positive variant length for all links."
            f" Problem id_link values: {bad[:10]}"
        )
    return lengths.astype(float), length_source


def _resolve_rapid_wse_for_slope(
    nodes: gpd.GeoDataFrame,
    *,
    config: SlopeConfig,
) -> pd.DataFrame:
    optional_columns = [
        "sword_wse_fallback_used",
        "sword_node_id",
        "sword_reach_id",
        "sword_dist_out",
    ]
    columns = [column for column in ["id_node", "sword_wse", "sword_wse_field", "sword_wse_fill_method"] + optional_columns if column in nodes.columns]
    node_attrs = (
        nodes[columns]
        .copy()
        .rename(
            columns={
                "sword_wse": "node_wse_raw",
                "sword_wse_field": "node_wse_field",
                "sword_wse_fill_method": "node_wse_fill_method",
                "sword_wse_fallback_used": "node_wse_fallback_used",
                "sword_node_id": "node_sword_node_id",
                "sword_reach_id": "node_sword_reach_id",
                "sword_dist_out": "node_sword_dist_out",
            }
        )
    )
    node_attrs["id_node"] = pd.to_numeric(node_attrs["id_node"], errors="coerce").astype("Int64")
    node_attrs["node_wse_raw"] = pd.to_numeric(node_attrs["node_wse_raw"], errors="coerce")
    if "node_sword_dist_out" in node_attrs.columns:
        node_attrs["node_sword_dist_out"] = pd.to_numeric(node_attrs["node_sword_dist_out"], errors="coerce")

    node_attrs["rapid_wse_for_slope"] = node_attrs["node_wse_raw"]
    node_attrs["rapid_wse_source_method"] = node_attrs["node_wse_fill_method"].fillna("unresolved").astype("object")
    node_attrs["rapid_wse_interpolated"] = False
    node_attrs["rapid_wse_anchor_count"] = 0

    if not config.interpolate_along_corridor or "node_sword_dist_out" not in node_attrs.columns:
        return node_attrs

    anchor_mask = node_attrs["node_wse_fill_method"].eq("requested_field") & node_attrs["node_wse_raw"].notna()
    anchors = node_attrs.loc[anchor_mask & node_attrs["node_sword_dist_out"].notna()].copy()
    anchors = anchors.sort_values("node_sword_dist_out")
    xp = anchors["node_sword_dist_out"].to_numpy(dtype=float)
    yp = anchors["node_wse_raw"].to_numpy(dtype=float)
    xp_unique, unique_index = np.unique(xp, return_index=True)
    yp_unique = yp[unique_index]
    node_attrs["rapid_wse_anchor_count"] = int(len(xp_unique))
    if len(xp_unique) < int(config.min_anchor_nodes_for_profile):
        return node_attrs

    target_mask = ~anchor_mask & node_attrs["node_sword_dist_out"].notna()
    targets = node_attrs.loc[target_mask].copy()
    if targets.empty:
        return node_attrs

    targets = targets.sort_values("node_sword_dist_out")
    interpolated = np.interp(
        targets["node_sword_dist_out"].to_numpy(dtype=float),
        xp_unique,
        yp_unique,
    )
    node_attrs.loc[targets.index, "rapid_wse_for_slope"] = interpolated
    node_attrs.loc[targets.index, "rapid_wse_source_method"] = "interpolated_along_corridor"
    node_attrs.loc[targets.index, "rapid_wse_interpolated"] = True

    return node_attrs


def _fill_invalid_slopes_from_neighbors(
    frame: pd.DataFrame,
    links: gpd.GeoDataFrame,
    *,
    config: SlopeConfig,
    invalid_mask: pd.Series,
    slope_column: str = "raw_slope",
) -> pd.DataFrame:
    frame = frame.copy()
    frame["slope_neighbor_source_link_id"] = pd.NA
    frame["slope_neighbor_distance"] = pd.NA
    frame["slope_neighbor_direction"] = ""
    frame["slope_neighbor_value"] = np.nan
    if not config.fill_from_nearest_valid_link or not invalid_mask.any():
        return frame

    link_ids = pd.to_numeric(frame["id_link"], errors="coerce").astype(int)
    us_nodes = pd.to_numeric(frame["id_us_node"], errors="coerce").astype(int)
    ds_nodes = pd.to_numeric(frame["id_ds_node"], errors="coerce").astype(int)
    frame_index = frame.index

    outgoing_by_node: dict[int, list[int]] = {}
    incoming_by_node: dict[int, list[int]] = {}
    for link_id, us_node, ds_node in zip(link_ids, us_nodes, ds_nodes):
        outgoing_by_node.setdefault(int(us_node), []).append(int(link_id))
        incoming_by_node.setdefault(int(ds_node), []).append(int(link_id))

    successors: dict[int, list[int]] = {}
    predecessors: dict[int, list[int]] = {}
    for link_id, us_node, ds_node in zip(link_ids, us_nodes, ds_nodes):
        successors[int(link_id)] = [candidate for candidate in outgoing_by_node.get(int(ds_node), []) if candidate != int(link_id)]
        predecessors[int(link_id)] = [candidate for candidate in incoming_by_node.get(int(us_node), []) if candidate != int(link_id)]

    row_index_by_link = dict(zip(link_ids, frame_index))
    slope_by_link = dict(zip(link_ids, pd.to_numeric(frame[slope_column], errors="coerce")))
    valid_links = {
        int(link_id)
        for link_id, raw_slope in slope_by_link.items()
        if pd.notna(raw_slope) and not bool(invalid_mask.loc[row_index_by_link[int(link_id)]])
    }

    centroid_by_link: dict[int, object] = {}
    if links.geometry.name in links.columns and links.geometry.notna().any():
        centroids = links.geometry.centroid
        centroid_by_link = dict(zip(pd.to_numeric(links["id_link"], errors="coerce").astype(int), centroids))

    def _nearest_valid(start_link: int, direction: str) -> tuple[int | None, int | None]:
        graph = successors if direction == "downstream" else predecessors
        visited = {int(start_link)}
        frontier = [int(start_link)]
        distance = 0
        while frontier:
            distance += 1
            next_frontier: list[int] = []
            candidates: list[int] = []
            for current in frontier:
                for neighbor in graph.get(int(current), []):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    if neighbor in valid_links:
                        candidates.append(neighbor)
                    next_frontier.append(neighbor)
            if candidates:
                return distance, min(candidates)
            frontier = next_frontier
        return None, None

    def _choose_candidate(start_link: int, upstream_result: tuple[int | None, int | None], downstream_result: tuple[int | None, int | None]) -> tuple[int | None, int | None, str]:
        candidate_rows: list[tuple[int, int, str]] = []
        up_distance, up_link = upstream_result
        if up_distance is not None and up_link is not None:
            candidate_rows.append((int(up_distance), int(up_link), "upstream"))
        down_distance, down_link = downstream_result
        if down_distance is not None and down_link is not None:
            candidate_rows.append((int(down_distance), int(down_link), "downstream"))
        if not candidate_rows:
            return None, None, ""
        min_distance = min(row[0] for row in candidate_rows)
        candidate_rows = [row for row in candidate_rows if row[0] == min_distance]
        if len(candidate_rows) == 1:
            distance, link_id, direction = candidate_rows[0]
            return link_id, distance, direction

        start_centroid = centroid_by_link.get(int(start_link))
        if start_centroid is not None:
            candidate_rows = sorted(
                candidate_rows,
                key=lambda row: (
                    float(start_centroid.distance(centroid_by_link.get(int(row[1]), start_centroid))),
                    row[1],
                ),
            )
        else:
            candidate_rows = sorted(candidate_rows, key=lambda row: row[1])
        distance, link_id, direction = candidate_rows[0]
        return link_id, distance, direction

    for link_id in link_ids[invalid_mask.to_numpy()]:
        upstream_result = _nearest_valid(int(link_id), "upstream")
        downstream_result = _nearest_valid(int(link_id), "downstream")
        neighbor_link, neighbor_distance, neighbor_direction = _choose_candidate(
            int(link_id),
            upstream_result,
            downstream_result,
        )
        if neighbor_link is None:
            continue
        neighbor_row_index = row_index_by_link[int(neighbor_link)]
        row_index = row_index_by_link[int(link_id)]
        frame.at[row_index, "slope_neighbor_source_link_id"] = int(neighbor_link)
        frame.at[row_index, "slope_neighbor_distance"] = int(neighbor_distance)
        frame.at[row_index, "slope_neighbor_direction"] = neighbor_direction
        frame.at[row_index, "slope_neighbor_value"] = float(frame.at[neighbor_row_index, slope_column])
    return frame


def _compute_section_slope_reference(
    node_attrs: pd.DataFrame,
    *,
    config: SlopeConfig,
) -> tuple[float, str, int]:
    if "node_sword_dist_out" not in node_attrs.columns:
        return float("nan"), "missing_dist_out", 0

    anchors = node_attrs.loc[
        node_attrs["node_wse_raw"].notna() & node_attrs["node_sword_dist_out"].notna()
    ].copy()
    if anchors.empty:
        return float("nan"), "missing_anchor_wse", 0

    requested = anchors.loc[anchors["node_wse_fill_method"].eq("requested_field")].copy()
    if len(requested) >= int(config.min_anchor_nodes_for_profile):
        anchors = requested
        source_method = "requested_field_nodes"
    else:
        source_method = "all_available_nodes"

    anchors = (
        anchors.groupby("node_sword_dist_out", as_index=False)["node_wse_raw"]
        .mean()
        .sort_values("node_sword_dist_out")
    )
    if len(anchors) < int(config.min_anchor_nodes_for_profile):
        return float("nan"), "insufficient_anchor_nodes", int(len(anchors))

    downstream = anchors.iloc[0]
    upstream = anchors.iloc[-1]
    delta_dist = float(upstream["node_sword_dist_out"] - downstream["node_sword_dist_out"])
    delta_wse = float(upstream["node_wse_raw"] - downstream["node_wse_raw"])
    if delta_dist <= 0 or delta_wse <= 0:
        return float("nan"), "invalid_anchor_profile", int(len(anchors))

    return float(delta_wse / delta_dist), source_method, int(len(anchors))


def compute_link_slopes(
    links: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    *,
    config: SlopeConfig | None = None,
) -> pd.DataFrame:
    config = config or SlopeConfig()
    if config.max_slope_for_k is not None and config.max_slope_for_k <= 0:
        raise ValueError("Slope outlier upper bound must be positive when provided.")
    if config.section_slope_ratio_min is not None and config.section_slope_ratio_min <= 0:
        raise ValueError("Section-slope ratio minimum must be positive when provided.")
    if config.section_slope_ratio_max is not None and config.section_slope_ratio_max <= 0:
        raise ValueError("Section-slope ratio maximum must be positive when provided.")
    if (
        config.section_slope_ratio_min is not None
        and config.section_slope_ratio_max is not None
        and config.section_slope_ratio_min > config.section_slope_ratio_max
    ):
        raise ValueError("Section-slope ratio minimum cannot exceed the maximum.")
    missing = sorted(REQUIRED_NODE_COLUMNS.difference(nodes.columns))
    if missing:
        raise ValueError(
            "Directed nodes are missing required SWORD WSE columns: "
            + ", ".join(missing)
        )
    for column in ("id_link", "id_us_node", "id_ds_node"):
        if column not in links.columns:
            raise ValueError(f"Directed links are missing required column '{column}'.")

    lengths, length_source = _select_variant_lengths(
        links,
        preferred_length_field=config.preferred_length_field,
    )

    node_attrs = _resolve_rapid_wse_for_slope(nodes, config=config)
    section_slope_ref, section_slope_source_method, section_slope_anchor_count = _compute_section_slope_reference(
        node_attrs,
        config=config,
    )

    frame = links[["id_link", "id_us_node", "id_ds_node"]].copy()
    frame["link_length_m"] = lengths
    frame["link_length_source"] = length_source
    frame["id_us_node"] = pd.to_numeric(frame["id_us_node"], errors="coerce").astype("Int64")
    frame["id_ds_node"] = pd.to_numeric(frame["id_ds_node"], errors="coerce").astype("Int64")

    us = node_attrs.rename(
        columns={
            "id_node": "id_us_node",
            "node_wse_raw": "wse_us_raw",
            "rapid_wse_for_slope": "wse_us",
            "node_wse_field": "wse_us_field",
            "node_wse_fill_method": "wse_us_fill_method",
            "node_wse_fallback_used": "wse_us_fallback_used",
            "node_sword_node_id": "sword_us_node_id",
            "node_sword_reach_id": "sword_us_reach_id",
            "node_sword_dist_out": "sword_us_dist_out",
            "rapid_wse_source_method": "wse_us_source_method",
            "rapid_wse_interpolated": "wse_us_interpolated",
            "rapid_wse_anchor_count": "wse_us_anchor_count",
        }
    )
    ds = node_attrs.rename(
        columns={
            "id_node": "id_ds_node",
            "node_wse_raw": "wse_ds_raw",
            "rapid_wse_for_slope": "wse_ds",
            "node_wse_field": "wse_ds_field",
            "node_wse_fill_method": "wse_ds_fill_method",
            "node_wse_fallback_used": "wse_ds_fallback_used",
            "node_sword_node_id": "sword_ds_node_id",
            "node_sword_reach_id": "sword_ds_reach_id",
            "node_sword_dist_out": "sword_ds_dist_out",
            "rapid_wse_source_method": "wse_ds_source_method",
            "rapid_wse_interpolated": "wse_ds_interpolated",
            "rapid_wse_anchor_count": "wse_ds_anchor_count",
        }
    )

    frame = frame.merge(us, on="id_us_node", how="left").merge(ds, on="id_ds_node", how="left")
    frame["raw_slope"] = (pd.to_numeric(frame["wse_us"], errors="coerce") - pd.to_numeric(frame["wse_ds"], errors="coerce")) / frame["link_length_m"]
    frame["slope_local_raw"] = pd.to_numeric(frame["raw_slope"], errors="coerce")
    frame["slope_section_ref"] = float(section_slope_ref)
    frame["slope_section_source_method"] = section_slope_source_method
    frame["slope_section_anchor_count"] = int(section_slope_anchor_count)

    missing_wse = frame["wse_us"].isna() | frame["wse_ds"].isna()
    invalid_raw = frame["slope_local_raw"].isna() | frame["slope_local_raw"].lt(float(config.min_slope))
    above_max = (
        frame["slope_local_raw"].gt(float(config.max_slope_for_k))
        if config.max_slope_for_k is not None
        else pd.Series(False, index=frame.index)
    )
    section_slope_available = np.isfinite(section_slope_ref) and section_slope_ref >= float(config.min_slope)
    slope_ratio = (
        frame["slope_local_raw"] / float(section_slope_ref)
        if section_slope_available
        else pd.Series(np.nan, index=frame.index)
    )
    below_ratio = (
        slope_ratio.lt(float(config.section_slope_ratio_min))
        if section_slope_available and config.section_slope_ratio_min is not None
        else pd.Series(False, index=frame.index)
    )
    above_ratio = (
        slope_ratio.gt(float(config.section_slope_ratio_max))
        if section_slope_available and config.section_slope_ratio_max is not None
        else pd.Series(False, index=frame.index)
    )

    frame["slope_outlier_reason"] = np.select(
        [
            missing_wse,
            invalid_raw,
            above_max,
            below_ratio,
            above_ratio,
        ],
        [
            "missing_wse",
            "below_minimum_raw",
            "above_maximum_raw",
            "below_section_ratio",
            "above_section_ratio",
        ],
        default="ok",
    )
    frame["slope_outlier_flag"] = frame["slope_outlier_reason"].ne("ok")
    frame["slope_section_ratio"] = slope_ratio.astype(float)

    frame = _fill_invalid_slopes_from_neighbors(
        frame,
        links,
        config=config,
        invalid_mask=frame["slope_outlier_flag"],
        slope_column="slope_local_raw",
    )

    use_neighbor = frame["slope_outlier_flag"] & frame["slope_neighbor_value"].notna()
    use_section = (
        frame["slope_outlier_flag"]
        & ~use_neighbor
        & bool(config.use_section_slope_fallback)
        & section_slope_available
    )
    use_minimum = frame["slope_outlier_flag"] & ~use_neighbor & ~use_section

    frame["slope_source_method"] = "raw_wse"
    frame["slope_used"] = frame["slope_local_raw"].astype(float)
    frame.loc[use_neighbor, "slope_source_method"] = "nearest_valid_link"
    frame.loc[use_neighbor, "slope_used"] = frame.loc[use_neighbor, "slope_neighbor_value"].astype(float)
    frame.loc[use_section, "slope_source_method"] = "section_slope"
    frame.loc[use_section, "slope_used"] = float(section_slope_ref)
    frame.loc[use_minimum, "slope_source_method"] = "minimum_slope"
    frame.loc[use_minimum, "slope_used"] = float(config.min_slope)

    frame["slope_adjusted"] = frame["slope_source_method"].ne("raw_wse")
    frame["slope_reason"] = np.select(
        [
            use_neighbor,
            use_section,
            use_minimum,
        ],
        [
            "filled_from_neighbor_link",
            "filled_from_section_slope",
            frame["slope_outlier_reason"],
        ],
        default="ok",
    )
    frame["slope_minimum_applied"] = frame["slope_used"].eq(config.min_slope)
    return frame


def compute_section_slope_reference(
    nodes: gpd.GeoDataFrame,
    *,
    config: SlopeConfig | None = None,
) -> dict[str, object]:
    config = config or SlopeConfig()
    missing = sorted(REQUIRED_NODE_COLUMNS.difference(nodes.columns))
    if missing:
        raise ValueError(
            "Directed nodes are missing required SWORD WSE columns: "
            + ", ".join(missing)
        )
    node_attrs = _resolve_rapid_wse_for_slope(nodes, config=config)
    slope_ref, source_method, anchor_count = _compute_section_slope_reference(
        node_attrs,
        config=config,
    )
    return {
        "slope": float(slope_ref),
        "source_method": str(source_method),
        "anchor_count": int(anchor_count),
    }
