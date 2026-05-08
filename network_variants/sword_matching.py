from __future__ import annotations

import re
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
SWORD_NODE_ORDER_CANDIDATES = ("node_order",)
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
    "sword_wse_fallback_used",
    "sword_wse_fill_method",
    "sword_wse_source_node_id",
    "sword_match_distance",
    "sword_match_method",
    "sword_match_from_parent",
    "sword_match_within_tolerance",
    "sword_source_file",
]
DEFAULT_REACH_BUFFER_STEPS = 2


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_example_station_source_path() -> Path:
    return _repo_root() / "SWORD_gauge_match" / "outputs" / "hierarchy_examples_filtered_subdaily_manual_updates_final.gpkg"


def _default_station_match_source_path() -> Path:
    return _repo_root() / "SWORD_gauge_match" / "outputs" / "selected_event_stations_same_main_path.gpkg"


def _first_existing(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def _coerce_optional_int_scalar(value: Any) -> int | None:
    if value is None or value is pd.NA or pd.isna(value):
        return None
    numeric = pd.to_numeric([value], errors="coerce")[0]
    if pd.isna(numeric):
        return None
    return int(numeric)


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
                "sword_wse_fallback_used": False,
                "sword_wse_fill_method": "unresolved",
                "sword_wse_source_node_id": pd.NA,
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
        try:
            from gauge_sword_match.sword_io import load_nodes, scan_sword_parquet_dir
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Reading a SWORD parquet directory requires the SWORD_gauge_match loaders and their parquet "
                "dependencies (for example pyarrow) to be importable in the active environment."
            ) from exc
    return load_nodes, scan_sword_parquet_dir


def _maybe_import_reach_tools():
    try:
        from gauge_sword_match.reach_paths import (
            build_downstream_adjacency,
            find_reaches_between,
            load_reach_topology,
            normalize_reach_id,
        )
        from gauge_sword_match.sword_io import scan_sword_parquet_dir
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[1]
        sword_src = repo_root / "SWORD_gauge_match" / "src"
        if str(sword_src) not in sys.path:
            sys.path.insert(0, str(sword_src))
        try:
            from gauge_sword_match.reach_paths import (
                build_downstream_adjacency,
                find_reaches_between,
                load_reach_topology,
                normalize_reach_id,
            )
            from gauge_sword_match.sword_io import scan_sword_parquet_dir
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Building example-specific SWORD reach corridors requires the SWORD_gauge_match parquet loaders "
                "and their parquet dependencies (for example pyarrow) to be importable in the active environment."
            ) from exc
    return build_downstream_adjacency, find_reaches_between, load_reach_topology, normalize_reach_id, scan_sword_parquet_dir


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


def _fill_missing_wse_from_same_reach_neighbor(frame: gpd.GeoDataFrame, *, source_field_name: str) -> gpd.GeoDataFrame:
    required = {"sword_reach_id", "sword_node_order", "sword_wse", "sword_node_id", "sword_wse_field"}
    if not required.issubset(frame.columns):
        return frame

    for reach_id, group in frame.groupby("sword_reach_id", dropna=True, sort=False):
        candidates = group.loc[
            group["sword_wse"].notna()
            & group["sword_node_order"].notna()
            & group["sword_wse_field"].eq(source_field_name)
        ].copy()
        if candidates.empty:
            continue
        unresolved = group.loc[group["sword_wse"].isna() & group["sword_node_order"].notna()].copy()
        if unresolved.empty:
            continue
        for idx, row in unresolved.iterrows():
            candidate_rank = candidates.copy()
            candidate_rank["order_delta"] = (candidate_rank["sword_node_order"] - row["sword_node_order"]).abs()
            if "sword_dist_out" in candidate_rank.columns and pd.notna(row.get("sword_dist_out", pd.NA)):
                candidate_rank["dist_out_delta"] = (candidate_rank["sword_dist_out"] - row["sword_dist_out"]).abs()
            else:
                candidate_rank["dist_out_delta"] = float("inf")
            candidate_rank = candidate_rank.sort_values(
                ["order_delta", "dist_out_delta", "sword_node_id"],
                ascending=[True, True, True],
                kind="mergesort",
            )
            if candidate_rank.empty:
                continue
            source_row = candidate_rank.iloc[0]
            frame.at[idx, "sword_wse"] = source_row["sword_wse"]
            frame.at[idx, "sword_wse_field"] = source_field_name
            frame.at[idx, "sword_wse_fallback_used"] = True
            frame.at[idx, "sword_wse_fill_method"] = "nearest_reach_node"
            frame.at[idx, "sword_wse_source_node_id"] = int(source_row["sword_node_id"])
    return frame


def _parse_example_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        result: list[int] = []
        for item in value:
            result.extend(_parse_example_ids(item))
        return result
    text = str(value).strip()
    if not text or text.lower() in {"nan", "<na>", "none"}:
        return []
    return [int(match.group(0)) for match in re.finditer(r"\d+", text)]


def _infer_numeric_example_id(example_id: str | int | None) -> int | None:
    if example_id is None:
        return None
    if isinstance(example_id, int):
        return int(example_id)
    matches = re.findall(r"\d+", str(example_id))
    if not matches:
        return None
    return int(matches[-1])


def _read_example_station_rows(source_path: str | Path, *, example_numeric_id: int) -> gpd.GeoDataFrame:
    frame = gpd.read_file(source_path)
    if frame.empty:
        return frame
    if "example_ids" in frame.columns:
        mask = frame["example_ids"].apply(lambda value: example_numeric_id in _parse_example_ids(value))
        return frame.loc[mask].copy()
    if "example_id" in frame.columns:
        example_ids = pd.to_numeric(frame["example_id"], errors="coerce")
        return frame.loc[example_ids.eq(example_numeric_id)].copy()
    raise ValueError(
        f"Example station source {source_path} does not contain 'example_ids' or 'example_id'."
    )


def _normalize_reach_series(values: pd.Series, normalize_reach_id) -> pd.Series:
    return values.map(normalize_reach_id).astype("Int64")


def _limited_traverse(adjacency: Mapping[int, set[int]], starts: Sequence[int], *, max_steps: int) -> set[int]:
    seen: set[int] = set()
    frontier = {int(value) for value in starts}
    for _ in range(max(0, int(max_steps)) + 1):
        next_frontier: set[int] = set()
        for node in frontier:
            if node in seen:
                continue
            seen.add(node)
            next_frontier.update(int(value) for value in adjacency.get(int(node), set()) if int(value) not in seen)
        frontier = next_frontier
        if not frontier:
            break
    return seen


def _reverse_adjacency(adjacency: Mapping[int, set[int]]) -> dict[int, set[int]]:
    reverse: dict[int, set[int]] = {}
    for upstream_id, downstream_ids in adjacency.items():
        reverse.setdefault(int(upstream_id), set())
        for downstream_id in downstream_ids:
            reverse.setdefault(int(downstream_id), set()).add(int(upstream_id))
    return reverse


def _resolve_example_reach_filter(
    *,
    example_id: str | int | None,
    sword_node_source_path: str | Path,
    example_station_source_path: str | Path | None,
    station_match_source_path: str | Path | None,
    reach_buffer_steps: int,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "scope": "bbox_only",
        "reason": "",
        "example_numeric_id": None,
        "candidate_region": None,
        "candidate_reach_ids": [],
        "candidate_reach_count": 0,
        "upstream_station_key": None,
        "downstream_station_key": None,
        "upstream_reach_id": None,
        "downstream_reach_id": None,
        "reach_buffer_steps": int(reach_buffer_steps),
        "example_station_source": None,
        "station_match_source": None,
    }

    example_numeric_id = _infer_numeric_example_id(example_id)
    if example_numeric_id is None:
        metadata["reason"] = "example_id_not_numeric"
        return metadata
    metadata["example_numeric_id"] = int(example_numeric_id)

    node_source_path = Path(sword_node_source_path).expanduser()
    if not node_source_path.exists():
        metadata["reason"] = "sword_node_source_missing"
        return metadata
    if not node_source_path.is_dir():
        metadata["reason"] = "sword_node_source_not_parquet_dir"
        return metadata

    example_station_path = Path(example_station_source_path) if example_station_source_path is not None else _default_example_station_source_path()
    station_match_path = Path(station_match_source_path) if station_match_source_path is not None else _default_station_match_source_path()
    metadata["example_station_source"] = str(example_station_path.resolve())
    metadata["station_match_source"] = str(station_match_path.resolve())
    if not example_station_path.exists():
        metadata["reason"] = "example_station_source_missing"
        return metadata
    if not station_match_path.exists():
        metadata["reason"] = "station_match_source_missing"
        return metadata

    example_stations = _read_example_station_rows(example_station_path, example_numeric_id=example_numeric_id)
    if example_stations.empty or "station_key" not in example_stations.columns:
        metadata["reason"] = "example_stations_not_found"
        return metadata
    station_keys = sorted(
        dict.fromkeys(
            str(value).strip()
            for value in example_stations["station_key"].dropna().astype("string").tolist()
            if str(value).strip()
        )
    )
    if len(station_keys) < 2:
        metadata["reason"] = "insufficient_example_stations"
        return metadata

    station_matches = gpd.read_file(station_match_path)
    required_columns = {"station_key", "reach_id", "sword_region"}
    missing_columns = required_columns - set(station_matches.columns)
    if missing_columns:
        metadata["reason"] = f"station_match_missing_columns:{','.join(sorted(missing_columns))}"
        return metadata
    station_matches = station_matches.loc[station_matches["station_key"].isin(station_keys)].copy()
    if station_matches.empty:
        metadata["reason"] = "station_matches_not_found"
        return metadata

    try:
        build_downstream_adjacency, find_reaches_between, load_reach_topology, normalize_reach_id, scan_sword_parquet_dir = _maybe_import_reach_tools()
    except ModuleNotFoundError as exc:
        metadata["reason"] = str(exc)
        return metadata
    station_matches["reach_id"] = _normalize_reach_series(station_matches["reach_id"], normalize_reach_id)
    station_matches["sword_region"] = station_matches["sword_region"].astype("string").str.lower()
    station_matches = station_matches.dropna(subset=["reach_id", "sword_region"]).copy()
    if station_matches.empty:
        metadata["reason"] = "station_matches_invalid"
        return metadata

    unique_regions = sorted(
        dict.fromkeys(
            str(value).strip().lower()
            for value in station_matches["sword_region"].dropna().astype("string").tolist()
            if str(value).strip()
        )
    )
    if len(unique_regions) != 1:
        metadata["reason"] = "multiple_station_match_regions"
        return metadata
    region = unique_regions[0]
    metadata["candidate_region"] = region

    catalog = scan_sword_parquet_dir(node_source_path)
    topology = load_reach_topology(
        catalog,
        region,
        columns=("reach_id", "rch_id_dn", "rch_id_dn_main", "rch_id_dn_1", "rch_id_dn_2", "rch_id_dn_3", "rch_id_dn_4", "dist_out"),
    )
    if topology.empty:
        metadata["reason"] = "region_topology_empty"
        return metadata

    topology = topology.drop_duplicates(subset=["reach_id"]).copy()
    topology["reach_id"] = _normalize_reach_series(topology["reach_id"], normalize_reach_id)
    topology["dist_out"] = pd.to_numeric(topology["dist_out"], errors="coerce")

    matched = station_matches.merge(
        topology[["reach_id", "dist_out"]],
        on="reach_id",
        how="left",
        validate="many_to_one",
    ).dropna(subset=["dist_out"]).copy()
    if matched.empty:
        metadata["reason"] = "station_match_dist_out_missing"
        return metadata

    matched = matched.sort_values(["dist_out", "station_key"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    upstream = matched.iloc[0]
    downstream = matched.iloc[-1]
    upstream_reach_id = int(upstream["reach_id"])
    downstream_reach_id = int(downstream["reach_id"])
    metadata["upstream_station_key"] = str(upstream["station_key"])
    metadata["downstream_station_key"] = str(downstream["station_key"])
    metadata["upstream_reach_id"] = upstream_reach_id
    metadata["downstream_reach_id"] = downstream_reach_id

    selected_reaches = find_reaches_between(
        topology,
        [upstream_reach_id],
        [downstream_reach_id],
        reach_id_col="reach_id",
        downstream_col=None,
    )
    if not selected_reaches:
        selected_reaches = {upstream_reach_id, downstream_reach_id}
        metadata["reason"] = "no_directed_route_fallback_to_station_reaches"
    adjacency = build_downstream_adjacency(topology, reach_id_col="reach_id", downstream_col=None)
    reverse_adjacency = _reverse_adjacency(adjacency)
    upstream_buffer = _limited_traverse(reverse_adjacency, [upstream_reach_id], max_steps=reach_buffer_steps)
    downstream_buffer = _limited_traverse(adjacency, [downstream_reach_id], max_steps=reach_buffer_steps)
    candidate_reach_ids = sorted(int(value) for value in (set(selected_reaches) | upstream_buffer | downstream_buffer))
    metadata["candidate_reach_ids"] = candidate_reach_ids
    metadata["candidate_reach_count"] = int(len(candidate_reach_ids))
    metadata["scope"] = "example_reach_corridor"
    return metadata


def load_sword_node_reference(
    sword_node_source_path: str | Path,
    *,
    bbox_wgs84: tuple[float, float, float, float] | None = None,
    sword_wse_field: str | None = None,
    regions: Sequence[str] | None = None,
    reach_ids: Sequence[int] | None = None,
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
        source = load_nodes(catalog, bbox=bbox_wgs84, columns=requested_columns, regions=regions, reach_ids=reach_ids)
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
    node_order_col = _first_existing(source.columns, SWORD_NODE_ORDER_CANDIDATES)
    wse_col = sword_wse_field if sword_wse_field in source.columns else _first_existing(source.columns, SWORD_WSE_CANDIDATES)
    fallback_wse_col = None
    if sword_wse_field is not None and sword_wse_field != "wse" and "wse" in source.columns:
        fallback_wse_col = "wse"
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
    standardized["sword_node_order"] = pd.to_numeric(standardized[node_order_col], errors="coerce") if node_order_col is not None else float("nan")
    if wse_col is not None:
        primary_wse = pd.to_numeric(standardized[wse_col], errors="coerce")
        standardized["sword_wse"] = primary_wse
        standardized["sword_wse_field"] = str(wse_col)
        standardized["sword_wse_fallback_used"] = False
        standardized["sword_wse_fill_method"] = "requested_field"
        standardized["sword_wse_source_node_id"] = standardized["sword_node_id"].astype("Int64")
        unresolved_mask = standardized["sword_wse"].isna()
        if unresolved_mask.any():
            standardized.loc[unresolved_mask, "sword_wse_fill_method"] = "unresolved"
            standardized = _fill_missing_wse_from_same_reach_neighbor(standardized, source_field_name=str(wse_col))
        if fallback_wse_col is not None:
            fallback_wse = pd.to_numeric(standardized[fallback_wse_col], errors="coerce")
            use_fallback = standardized["sword_wse"].isna() & fallback_wse.notna()
            if use_fallback.any():
                standardized.loc[use_fallback, "sword_wse"] = fallback_wse.loc[use_fallback]
                standardized.loc[use_fallback, "sword_wse_field"] = str(fallback_wse_col)
                standardized.loc[use_fallback, "sword_wse_fallback_used"] = True
                standardized.loc[use_fallback, "sword_wse_fill_method"] = "same_node_wse"
    else:
        standardized["sword_wse"] = float("nan")
        standardized["sword_wse_field"] = ""
        standardized["sword_wse_fallback_used"] = False
        standardized["sword_wse_fill_method"] = "unresolved"
        standardized["sword_wse_source_node_id"] = standardized["sword_node_id"].astype("Int64")
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
        "sword_wse_fallback_used",
        "sword_wse_fill_method",
        "sword_wse_source_node_id",
        "sword_source_file",
        standardized.geometry.name,
    ]
    standardized = standardized[keep_columns].dropna(subset=["sword_node_id"]).copy()
    if reach_ids and "sword_reach_id" in standardized.columns:
        standardized = standardized.loc[standardized["sword_reach_id"].isin(list(reach_ids))].copy()
    if regions and "sword_region" in standardized.columns:
        allowed_regions = {str(value).strip().lower() for value in regions if str(value).strip()}
        standardized = standardized.loc[
            standardized["sword_region"].astype("string").str.lower().isin(allowed_regions)
        ].copy()
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
        "sword_wse_fallback_used",
        "sword_wse_fill_method",
        "sword_wse_source_node_id",
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
    example_id: str | int | None = None,
    sword_example_station_source_path: str | Path | None = None,
    sword_station_match_source_path: str | Path | None = None,
    sword_reach_buffer_steps: int = DEFAULT_REACH_BUFFER_STEPS,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, dict[str, Any]]:
    child_nodes = directed_nodes.copy()
    child_nodes["id_node"] = child_nodes["id_node"].astype(int)
    match_metadata: dict[str, Any] = {
        "scope": "bbox_only",
        "reason": "",
        "example_numeric_id": _infer_numeric_example_id(example_id),
        "candidate_region": None,
        "candidate_reach_count": 0,
        "candidate_reach_ids": [],
        "upstream_station_key": None,
        "downstream_station_key": None,
        "upstream_reach_id": None,
        "downstream_reach_id": None,
        "reach_buffer_steps": int(sword_reach_buffer_steps),
    }

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
            sword_node_id = _coerce_optional_int_scalar(getattr(row, "sword_node_id", pd.NA))
            if sword_node_id is None:
                continue
            fallback_used = bool(getattr(row, "sword_wse_fallback_used", False))
            fill_method = getattr(row, "sword_wse_fill_method", None)
            if fill_method is None or pd.isna(fill_method) or fill_method == "":
                fill_method = "same_node_wse" if fallback_used else "requested_field"
            sword_wse_source_node_id = _coerce_optional_int_scalar(
                getattr(row, "sword_wse_source_node_id", pd.NA)
            )
            parent_lookup[int(row.id_node)] = {
                "sword_node_id": sword_node_id,
                "sword_reach_id": getattr(row, "sword_reach_id", pd.NA),
                "sword_region": getattr(row, "sword_region", pd.NA),
                "sword_dist_out": getattr(row, "sword_dist_out", float("nan")),
                "sword_wse": getattr(row, "sword_wse", float("nan")),
                "sword_wse_field": getattr(row, "sword_wse_field", ""),
                "sword_wse_fallback_used": fallback_used,
                "sword_wse_fill_method": fill_method,
                "sword_wse_source_node_id": (
                    sword_wse_source_node_id if sword_wse_source_node_id is not None else sword_node_id
                ),
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
        reach_filter_metadata = _resolve_example_reach_filter(
            example_id=example_id,
            sword_node_source_path=sword_node_source_path,
            example_station_source_path=sword_example_station_source_path,
            station_match_source_path=sword_station_match_source_path,
            reach_buffer_steps=sword_reach_buffer_steps,
        )
        match_metadata.update(reach_filter_metadata)
        sword_nodes = load_sword_node_reference(
            sword_node_source_path,
            bbox_wgs84=_bbox_in_wgs84(child_nodes),
            sword_wse_field=sword_wse_field,
            regions=([str(reach_filter_metadata["candidate_region"])] if reach_filter_metadata.get("candidate_region") else None),
            reach_ids=(reach_filter_metadata.get("candidate_reach_ids") or None),
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
                        match_frame.loc[mask, "sword_wse_fallback_used"] = bool(row.sword_wse_fallback_used)
                        match_frame.loc[mask, "sword_wse_fill_method"] = row.sword_wse_fill_method
                        match_frame.loc[mask, "sword_wse_source_node_id"] = row.sword_wse_source_node_id
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
        "sword_wse_fallback_used",
        "sword_wse_fill_method",
        "sword_wse_source_node_id",
        "sword_match_distance",
        "sword_match_method",
        "sword_match_from_parent",
        "sword_match_within_tolerance",
        "sword_source_file",
    ]
    child_nodes = child_nodes.drop(columns=[column for column in merge_columns if column in child_nodes.columns and column != "id_node"], errors="ignore")
    child_nodes = child_nodes.merge(match_frame[merge_columns], on="id_node", how="left", validate="1:1")
    return child_nodes, match_frame, match_metadata
