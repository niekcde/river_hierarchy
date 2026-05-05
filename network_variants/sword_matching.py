from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import geopandas as gpd
import pandas as pd

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

SWORD_NODE_ID_CANDIDATES = ("node_id", "sword_node_id", "id_node", "id")
SWORD_REACH_ID_CANDIDATES = ("reach_id", "sword_reach_id")
SWORD_REGION_CANDIDATES = ("sword_region", "region")
SWORD_DIST_OUT_CANDIDATES = ("dist_out", "dist_out_m")
SWORD_WSE_CANDIDATES = (
    "wse",
    "wse_m",
    "wse_obs",
    "wse_obs_p50",
    "wse_var",
    "water_surface_elevation",
)
STANDARD_MATCH_COLUMNS = [
    "id_node",
    "matched_parent_node_id",
    "sword_node_id",
    "sword_reach_id",
    "sword_region",
    "sword_dist_out",
    "sword_wse",
    "sword_wse_field",
    "sword_match_distance",
    "sword_match_method",
    "sword_match_from_parent",
    "sword_match_within_tolerance",
    "sword_source_file",
]


def _first_existing(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def _empty_match_frame(directed_nodes: gpd.GeoDataFrame, *, matched_parent_lookup: Mapping[int, int | None]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for node_id in directed_nodes["id_node"].astype(int).tolist():
        rows.append(
            {
                "id_node": int(node_id),
                "matched_parent_node_id": matched_parent_lookup.get(int(node_id), pd.NA),
                "sword_node_id": pd.NA,
                "sword_reach_id": pd.NA,
                "sword_region": pd.NA,
                "sword_dist_out": float("nan"),
                "sword_wse": float("nan"),
                "sword_wse_field": "",
                "sword_match_distance": float("nan"),
                "sword_match_method": "unmatched",
                "sword_match_from_parent": False,
                "sword_match_within_tolerance": pd.NA,
                "sword_source_file": "",
            }
        )
    return pd.DataFrame.from_records(rows, columns=STANDARD_MATCH_COLUMNS)


def _maybe_import_sword_io():
    try:
        from gauge_sword_match.sword_io import load_nodes, scan_sword_parquet_dir
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[1]
        sword_src = repo_root / "SWORD_gauge_match" / "src"
        if str(sword_src) not in sys.path:
            sys.path.insert(0, str(sword_src))
        from gauge_sword_match.sword_io import load_nodes, scan_sword_parquet_dir
    return load_nodes, scan_sword_parquet_dir


def _bbox_in_wgs84(frame: gpd.GeoDataFrame) -> tuple[float, float, float, float] | None:
    if frame.empty:
        return None
    if frame.crs is not None:
        try:
            bounds_frame = frame.to_crs("EPSG:4326")
            bounds = bounds_frame.total_bounds
            return tuple(float(value) for value in bounds)
        except Exception:
            pass
    bounds = frame.total_bounds
    if len(bounds) != 4:
        return None
    return tuple(float(value) for value in bounds)


def load_sword_node_reference(
    sword_node_source_path: str | Path,
    *,
    bbox_wgs84: tuple[float, float, float, float] | None = None,
    sword_wse_field: str | None = None,
) -> gpd.GeoDataFrame:
    source_path = Path(sword_node_source_path).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"SWORD node source was not found: {source_path}")

    if source_path.is_dir():
        load_nodes, scan_sword_parquet_dir = _maybe_import_sword_io()
        catalog = scan_sword_parquet_dir(source_path)
        requested_columns = [
            column
            for column in [
                "node_id",
                "reach_id",
                "sword_region",
                "dist_out",
                sword_wse_field,
                *SWORD_WSE_CANDIDATES,
            ]
            if column
        ]
        source = load_nodes(catalog, bbox=bbox_wgs84, columns=requested_columns)
    elif source_path.suffix.lower() == ".parquet":
        source = gpd.read_parquet(source_path)
    else:
        source = gpd.read_file(source_path)

    if source.empty:
        return source

    node_id_col = _first_existing(source.columns, SWORD_NODE_ID_CANDIDATES)
    if node_id_col is None:
        raise ValueError(
            f"SWORD node source {source_path} does not contain a recognizable node-id column. "
            f"Tried {SWORD_NODE_ID_CANDIDATES}."
        )
    reach_id_col = _first_existing(source.columns, SWORD_REACH_ID_CANDIDATES)
    region_col = _first_existing(source.columns, SWORD_REGION_CANDIDATES)
    dist_out_col = _first_existing(source.columns, SWORD_DIST_OUT_CANDIDATES)
    wse_col = sword_wse_field if sword_wse_field in source.columns else _first_existing(source.columns, SWORD_WSE_CANDIDATES)
    source_file_col = "source_file" if "source_file" in source.columns else None

    standardized = source.copy()
    standardized["sword_node_id"] = pd.to_numeric(standardized[node_id_col], errors="coerce").astype("Int64")
    standardized["sword_reach_id"] = (
        pd.to_numeric(standardized[reach_id_col], errors="coerce").astype("Int64")
        if reach_id_col is not None
        else pd.Series(pd.NA, index=standardized.index, dtype="Int64")
    )
    standardized["sword_region"] = standardized[region_col].astype("string") if region_col is not None else pd.Series(pd.NA, index=standardized.index, dtype="string")
    standardized["sword_dist_out"] = pd.to_numeric(standardized[dist_out_col], errors="coerce") if dist_out_col is not None else float("nan")
    standardized["sword_wse"] = pd.to_numeric(standardized[wse_col], errors="coerce") if wse_col is not None else float("nan")
    standardized["sword_wse_field"] = str(wse_col) if wse_col is not None else ""
    standardized["sword_source_file"] = (
        standardized[source_file_col].astype("string")
        if source_file_col is not None
        else str(source_path.resolve())
    )
    keep_columns = [
        "sword_node_id",
        "sword_reach_id",
        "sword_region",
        "sword_dist_out",
        "sword_wse",
        "sword_wse_field",
        "sword_source_file",
        standardized.geometry.name,
    ]
    standardized = standardized[keep_columns].dropna(subset=["sword_node_id"]).copy()
    standardized["sword_node_id"] = standardized["sword_node_id"].astype(int)
    return standardized.reset_index(drop=True)


def _metric_crs_for_matching(left: gpd.GeoDataFrame, right: gpd.GeoDataFrame):
    if left.crs is not None:
        try:
            if left.crs.is_projected:
                return left.crs
        except Exception:
            return left.crs
        try:
            utm = left.estimate_utm_crs()
            if utm is not None:
                return utm
        except Exception:
            return left.crs
        return left.crs
    if right.crs is not None:
        try:
            if right.crs.is_projected:
                return right.crs
        except Exception:
            return right.crs
        try:
            utm = right.estimate_utm_crs()
            if utm is not None:
                return utm
        except Exception:
            return right.crs
        return right.crs
    return None


def _nearest_source_rows(
    nodes: gpd.GeoDataFrame,
    sword_nodes: gpd.GeoDataFrame,
    *,
    max_distance: float | None,
) -> pd.DataFrame:
    if nodes.empty or sword_nodes.empty:
        return pd.DataFrame(columns=["id_node"])

    target_crs = _metric_crs_for_matching(nodes, sword_nodes)
    nodes_metric = nodes.to_crs(target_crs) if target_crs is not None and nodes.crs != target_crs else nodes.copy()
    sword_metric = sword_nodes.to_crs(target_crs) if target_crs is not None and sword_nodes.crs != target_crs else sword_nodes.copy()

    join_columns = [
        "sword_node_id",
        "sword_reach_id",
        "sword_region",
        "sword_dist_out",
        "sword_wse",
        "sword_wse_field",
        "sword_source_file",
        sword_metric.geometry.name,
    ]
    joined = nodes_metric[["id_node", nodes_metric.geometry.name]].sjoin_nearest(
        sword_metric[join_columns],
        how="left",
        distance_col="sword_match_distance",
        max_distance=max_distance,
    )
    joined = joined.drop(columns=[column for column in ("index_right",) if column in joined.columns], errors="ignore")
    return pd.DataFrame(joined.drop(columns=nodes_metric.geometry.name, errors="ignore"))


def match_variant_nodes_to_sword(
    *,
    directed_nodes: gpd.GeoDataFrame,
    parent_nodes: gpd.GeoDataFrame,
    node_match: pd.DataFrame,
    sword_node_source_path: str | Path | None = None,
    sword_wse_field: str | None = None,
    sword_match_tolerance: float | None = None,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    child_nodes = directed_nodes.copy()
    child_nodes["id_node"] = child_nodes["id_node"].astype(int)

    matched_parent_lookup: dict[int, int | None] = {}
    for row in node_match.itertuples(index=False):
        if row.matched_parent_node_id is pd.NA or pd.isna(row.matched_parent_node_id):
            matched_parent_lookup[int(row.child_id_node)] = None
        else:
            matched_parent_lookup[int(row.child_id_node)] = int(row.matched_parent_node_id)

    match_frame = _empty_match_frame(child_nodes, matched_parent_lookup=matched_parent_lookup)

    parent_lookup: dict[int, dict[str, Any]] = {}
    if "id_node" in parent_nodes.columns and "sword_node_id" in parent_nodes.columns:
        for row in parent_nodes.itertuples(index=False):
            sword_node_id = getattr(row, "sword_node_id", pd.NA)
            if sword_node_id is pd.NA or pd.isna(sword_node_id):
                continue
            parent_lookup[int(row.id_node)] = {
                "sword_node_id": int(sword_node_id),
                "sword_reach_id": getattr(row, "sword_reach_id", pd.NA),
                "sword_region": getattr(row, "sword_region", pd.NA),
                "sword_dist_out": getattr(row, "sword_dist_out", float("nan")),
                "sword_wse": getattr(row, "sword_wse", float("nan")),
                "sword_wse_field": getattr(row, "sword_wse_field", ""),
                "sword_source_file": getattr(row, "sword_source_file", ""),
            }

    if parent_lookup:
        for index, row in match_frame.iterrows():
            parent_node_id = row["matched_parent_node_id"]
            if parent_node_id is pd.NA or pd.isna(parent_node_id):
                continue
            parent_attrs = parent_lookup.get(int(parent_node_id))
            if not parent_attrs:
                continue
            for key, value in parent_attrs.items():
                match_frame.at[index, key] = value
            match_frame.at[index, "sword_match_method"] = "propagated_parent"
            match_frame.at[index, "sword_match_from_parent"] = True

    sword_nodes = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs=child_nodes.crs), crs=child_nodes.crs)
    if sword_node_source_path is not None:
        sword_nodes = load_sword_node_reference(
            sword_node_source_path,
            bbox_wgs84=_bbox_in_wgs84(child_nodes),
            sword_wse_field=sword_wse_field,
        )

    if not sword_nodes.empty:
        source_by_id: dict[int, dict[str, Any]] = {
            int(row.sword_node_id): row._asdict()
            for row in sword_nodes.itertuples(index=False)
            if row.sword_node_id is not pd.NA and not pd.isna(row.sword_node_id)
        }

        propagated_rows = match_frame.loc[match_frame["sword_match_method"] == "propagated_parent", ["id_node", "sword_node_id"]].copy()
        if not propagated_rows.empty:
            propagated_join = propagated_rows.merge(
                sword_nodes.drop(columns=sword_nodes.geometry.name).drop_duplicates(subset=["sword_node_id"]),
                on="sword_node_id",
                how="left",
                suffixes=("", "_source"),
            )
            if not propagated_join.empty:
                distance_inputs = child_nodes[["id_node", child_nodes.geometry.name]].merge(
                    sword_nodes[["sword_node_id", sword_nodes.geometry.name]],
                    on=None,
                    how="cross",
                )
                distance_inputs = distance_inputs.merge(propagated_rows, on=["id_node", "sword_node_id"], how="inner")
                if not distance_inputs.empty:
                    target_crs = _metric_crs_for_matching(
                        child_nodes[["id_node", child_nodes.geometry.name]],
                        sword_nodes[["sword_node_id", sword_nodes.geometry.name]],
                    )
                    child_metric = child_nodes[["id_node", child_nodes.geometry.name]].to_crs(target_crs) if target_crs is not None and child_nodes.crs != target_crs else child_nodes[["id_node", child_nodes.geometry.name]].copy()
                    sword_metric = sword_nodes[["sword_node_id", sword_nodes.geometry.name]].to_crs(target_crs) if target_crs is not None and sword_nodes.crs != target_crs else sword_nodes[["sword_node_id", sword_nodes.geometry.name]].copy()
                    child_metric = child_metric.merge(propagated_rows, on="id_node", how="inner")
                    propagated_metric = child_metric.merge(sword_metric, on="sword_node_id", how="left", suffixes=("_child", "_sword"))
                    propagated_metric["distance"] = propagated_metric[f"{child_nodes.geometry.name}_child"].distance(
                        propagated_metric[f"{sword_nodes.geometry.name}_sword"]
                    )
                    for row in propagated_metric.itertuples(index=False):
                        mask = match_frame["id_node"].eq(int(row.id_node))
                        match_frame.loc[mask, "sword_match_distance"] = float(row.distance)
                        if sword_match_tolerance is None:
                            match_frame.loc[mask, "sword_match_within_tolerance"] = True
                        else:
                            match_frame.loc[mask, "sword_match_within_tolerance"] = bool(float(row.distance) <= float(sword_match_tolerance))

        unmatched_nodes = child_nodes.loc[
            match_frame["sword_node_id"].isna() | match_frame["sword_node_id"].eq(pd.NA),
            ["id_node", child_nodes.geometry.name],
        ].copy()
        if not unmatched_nodes.empty:
            nearest = _nearest_source_rows(
                unmatched_nodes,
                sword_nodes,
                max_distance=sword_match_tolerance,
            )
            if not nearest.empty:
                nearest = nearest.dropna(subset=["sword_node_id"]).copy()
                if not nearest.empty:
                    nearest["sword_node_id"] = pd.to_numeric(nearest["sword_node_id"], errors="coerce").astype("Int64")
                    nearest["sword_reach_id"] = pd.to_numeric(nearest["sword_reach_id"], errors="coerce").astype("Int64")
                    for row in nearest.itertuples(index=False):
                        mask = match_frame["id_node"].eq(int(row.id_node))
                        match_frame.loc[mask, "sword_node_id"] = int(row.sword_node_id)
                        match_frame.loc[mask, "sword_reach_id"] = row.sword_reach_id
                        match_frame.loc[mask, "sword_region"] = row.sword_region
                        match_frame.loc[mask, "sword_dist_out"] = row.sword_dist_out
                        match_frame.loc[mask, "sword_wse"] = row.sword_wse
                        match_frame.loc[mask, "sword_wse_field"] = row.sword_wse_field
                        match_frame.loc[mask, "sword_match_distance"] = row.sword_match_distance
                        match_frame.loc[mask, "sword_match_method"] = "nearest_sword_node"
                        match_frame.loc[mask, "sword_match_from_parent"] = False
                        match_frame.loc[mask, "sword_match_within_tolerance"] = (
                            True if sword_match_tolerance is None else bool(float(row.sword_match_distance) <= float(sword_match_tolerance))
                        )
                        match_frame.loc[mask, "sword_source_file"] = row.sword_source_file

    merge_columns = [
        "id_node",
        "sword_node_id",
        "sword_reach_id",
        "sword_region",
        "sword_dist_out",
        "sword_wse",
        "sword_wse_field",
        "sword_match_distance",
        "sword_match_method",
        "sword_match_from_parent",
        "sword_match_within_tolerance",
        "sword_source_file",
    ]
    child_nodes = child_nodes.drop(columns=[column for column in merge_columns if column in child_nodes.columns and column != "id_node"], errors="ignore")
    child_nodes = child_nodes.merge(match_frame[merge_columns], on="id_node", how="left", validate="1:1")
    return child_nodes, match_frame
