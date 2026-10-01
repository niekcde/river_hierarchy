"""Build reviewed reference-profile inputs directly from one base experiment.

This is the production counterpart of the accepted notebook corridor logic.
It deliberately uses the realised network endpoints, follows the SWORD main
path between their projected reaches, and limits SWORD nodes to that endpoint
bounded corridor.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from network_variants.hydraulic_context import _load_reference_section_support
from network_variants.sword_matching import load_sword_node_reference

from rapid_tools.reference_profile import _pava_nondecreasing


RAW_RIVERSP_FILTER_DESCRIPTION = (
    "wse_sm_u <= 1 AND dark_frac <= 0.5 AND "
    "ABS(xtrk_dist) BETWEEN 10000 AND 60000 AND wse_sm > 0 AND all QC fields finite"
)


@dataclass(frozen=True, slots=True)
class ReferenceProfileInputs:
    static_profile: pd.DataFrame
    observations: pd.DataFrame
    endpoint_distances_m: tuple[float, float]
    metadata: dict[str, Any]


def _one_path(directory: Path, pattern: str) -> Path:
    matches = sorted(path for path in directory.glob(pattern) if not path.name.startswith("._"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {pattern!r} in {directory}; found {matches}")
    return matches[0]


def _valid_positive_int(value: object) -> int | None:
    numeric = pd.to_numeric([value], errors="coerce")[0]
    return int(numeric) if pd.notna(numeric) and int(numeric) > 0 else None


def _orient_upstream_at_start(reach_geometry: object, reach_nodes: gpd.GeoDataFrame) -> bool:
    valid = reach_nodes.loc[pd.to_numeric(reach_nodes["sword_dist_out"], errors="coerce").notna()]
    if valid.empty:
        return True
    upstream = valid.loc[pd.to_numeric(valid["sword_dist_out"], errors="coerce").idxmax()].geometry
    return float(reach_geometry.interpolate(0).distance(upstream)) <= float(
        reach_geometry.interpolate(reach_geometry.length).distance(upstream)
    )


def _endpoint_nodes(links: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame, *, crs: object) -> gpd.GeoDataFrame:
    degree = pd.concat(
        [
            links.groupby("id_us_node").size().rename("out_degree"),
            links.groupby("id_ds_node").size().rename("in_degree"),
        ],
        axis=1,
    ).fillna(0)
    result = nodes.to_crs(crs)[["id_node", nodes.geometry.name]].join(degree, on="id_node").fillna(
        {"in_degree": 0, "out_degree": 0}
    )
    result["anchor_role"] = np.select(
        [result["in_degree"].eq(0), result["out_degree"].eq(0)],
        ["network_source", "network_sink"],
        default="",
    )
    result = result.loc[result["anchor_role"].ne("")].copy()
    if len(result) != 2 or set(result["anchor_role"]) != {"network_source", "network_sink"}:
        raise ValueError("Expected exactly one realised network source and one realised network sink.")
    return result


def _prepare_raw_observations(
    raw_observations: pd.DataFrame,
    profile_nodes: pd.DataFrame,
) -> pd.DataFrame:
    required = {"node_id", "wse_sm", "wse_sm_u", "dark_frac", "xtrk_dist", "source_file", "version"}
    missing = sorted(required - set(raw_observations.columns))
    if missing:
        raise ValueError(f"Raw RiverSP observation source is missing fields: {missing}")
    raw = raw_observations.copy()
    for column in ("node_id", "version"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    for column in ("wse_sm", "wse_sm_u", "dark_frac", "xtrk_dist"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    valid = (
        raw["wse_sm_u"].le(1.0)
        & raw["dark_frac"].le(0.5)
        & raw["xtrk_dist"].abs().between(10_000, 60_000)
        & raw["wse_sm"].gt(0.0)
        & np.isfinite(raw[["wse_sm", "wse_sm_u", "dark_frac", "xtrk_dist"]]).all(axis=1)
    )
    raw = raw.loc[valid].copy()
    raw["acquisition_id"] = raw["source_file"].astype(str).str.replace(
        r"_[A-Z]+\d+_\d+\.parquet$", "", regex=True
    )
    newest = raw.groupby("acquisition_id")["version"].transform("max")
    raw = raw.loc[raw["version"].eq(newest)].copy()
    raw["node_id"] = raw["node_id"].astype("Int64")
    node_lookup = profile_nodes[["sword_node_id", "reference_section_dist_out_m"]].copy()
    node_lookup["sword_node_id"] = pd.to_numeric(node_lookup["sword_node_id"], errors="coerce").astype("Int64")
    result = raw.merge(
        node_lookup,
        left_on="node_id",
        right_on="sword_node_id",
        how="inner",
        validate="m:1",
    ).drop_duplicates(["acquisition_id", "node_id"]).copy()
    result = result.rename(columns={"wse_sm": "wse_m", "wse_sm_u": "wse_uncertainty_m"})
    if "time_str" in result.columns:
        result["time"] = result["time_str"].astype(str)
    return result.sort_values(["reference_section_dist_out_m", "acquisition_id", "node_id"]).reset_index(drop=True)


def build_reference_profile_inputs(
    experiment_dir: str | Path,
    *,
    sword_node_source: str | Path,
    raw_observations_path: str | Path,
    max_centerline_offset_m: float = 350.0,
    endpoint_buffer_m: float = 1_000.0,
) -> ReferenceProfileInputs:
    """Build static SWORD and filtered raw RiverSP inputs for one base example."""
    experiment = Path(experiment_dir).expanduser().resolve()
    state = experiment / "states" / "S000_base"
    summary_path = experiment / "reference_section" / "reference_section_summary.json"
    if not state.exists() or not summary_path.exists():
        raise FileNotFoundError("Expected a complete base state and reference_section_summary.json.")
    directed = state / "variant" / "directed"
    links = gpd.read_file(_one_path(directed, "*directed_links.gpkg"))
    nodes = gpd.read_file(_one_path(directed, "*directed_nodes.gpkg"))
    support = _load_reference_section_support(str(summary_path))
    line = support["line"]
    centerline_crs = support["centerline_crs"]
    endpoint_nodes = _endpoint_nodes(links, nodes, crs=centerline_crs)

    variant_manifest = json.loads((state / "variant" / "summary" / "variant_manifest.json").read_text())
    matching = variant_manifest.get("sword_matching", {}) or {}
    region = str(matching.get("candidate_region", "")).strip().lower()
    reach_ids = [int(value) for value in matching.get("candidate_reach_ids", [])]
    if not region or not reach_ids:
        raise ValueError("Base variant manifest does not contain a reviewed SWORD candidate region/reach corridor.")
    sword_root = Path(sword_node_source).expanduser().resolve()
    reach_path = sword_root / f"sword_{region.upper()}_v17c_reaches.parquet"
    if not reach_path.exists():
        raise FileNotFoundError(f"SWORD reach source was not found: {reach_path}")
    reaches = gpd.read_parquet(reach_path, columns=["reach_id", "rch_id_up_main", "rch_id_dn_main", "geometry"])
    reaches["reach_id"] = pd.to_numeric(reaches["reach_id"], errors="coerce").astype("Int64")
    reaches = reaches.loc[reaches["reach_id"].isin(reach_ids)].drop_duplicates("reach_id").to_crs(centerline_crs).copy()
    if reaches.empty:
        raise ValueError("No reviewed candidate reaches could be loaded.")
    reaches["reach_id"] = reaches["reach_id"].astype(int)
    lookup = reaches.set_index("reach_id")

    projections: list[dict[str, object]] = []
    for endpoint in endpoint_nodes.itertuples(index=False):
        distance = reaches.geometry.distance(endpoint.geometry)
        reach = reaches.loc[distance.idxmin()]
        projections.append(
            {
                "anchor_role": endpoint.anchor_role,
                "id_node": int(endpoint.id_node),
                "sword_reach_id": int(reach.reach_id),
                "projection_m": float(reach.geometry.project(endpoint.geometry)),
                "projection_distance_m": float(distance.loc[reach.name]),
            }
        )
    projections_table = pd.DataFrame(projections).set_index("anchor_role", drop=False)
    source_reach = int(projections_table.loc["network_source", "sword_reach_id"])
    sink_reach = int(projections_table.loc["network_sink", "sword_reach_id"])
    corridor_ids = [source_reach]
    while corridor_ids[-1] != sink_reach:
        next_reach = _valid_positive_int(lookup.loc[corridor_ids[-1], "rch_id_dn_main"])
        if next_reach is None or next_reach not in lookup.index or next_reach in corridor_ids:
            raise ValueError("Could not trace the SWORD downstream main path between realised network endpoints.")
        corridor_ids.append(next_reach)

    endpoint_sword = load_sword_node_reference(
        sword_root, sword_wse_field="wse_obs_p50", regions=[region], reach_ids=[source_reach, sink_reach]
    ).to_crs(centerline_crs)
    for role, neighbour_field, prepend in (
        ("network_source", "rch_id_up_main", True),
        ("network_sink", "rch_id_dn_main", False),
    ):
        projected = projections_table.loc[role]
        reach_id = int(projected.sword_reach_id)
        reach = lookup.loc[reach_id]
        reach_nodes = endpoint_sword.loc[endpoint_sword["sword_reach_id"].eq(reach_id)]
        upstream_at_start = _orient_upstream_at_start(reach.geometry, reach_nodes)
        outward_end = 0.0 if ((role == "network_source") == upstream_at_start) else float(reach.geometry.length)
        neighbour = _valid_positive_int(reach[neighbour_field])
        if abs(float(projected.projection_m) - outward_end) < endpoint_buffer_m and neighbour is not None and neighbour in lookup.index:
            corridor_ids = [neighbour, *corridor_ids] if prepend else [*corridor_ids, neighbour]

    corridor_reaches = reaches.set_index("reach_id").loc[corridor_ids].reset_index()
    corridor_lookup = corridor_reaches.set_index("reach_id")
    all_sword = load_sword_node_reference(
        sword_root, sword_wse_field="wse_obs_p50", regions=[region], reach_ids=corridor_ids
    ).to_crs(centerline_crs).drop_duplicates("sword_node_id").copy()
    offsets: dict[int, float] = {}
    orientations: dict[int, bool] = {}
    chainage = 0.0
    for reach_id in corridor_ids:
        reach = corridor_lookup.loc[int(reach_id)]
        reach_nodes = all_sword.loc[all_sword["sword_reach_id"].eq(int(reach_id))]
        offsets[int(reach_id)] = chainage
        orientations[int(reach_id)] = _orient_upstream_at_start(reach.geometry, reach_nodes)
        chainage += float(reach.geometry.length)

    def path_chainage(reach_id: int, position_m: float) -> float:
        reach = corridor_lookup.loc[int(reach_id)]
        local = float(position_m) if orientations[int(reach_id)] else float(reach.geometry.length - position_m)
        return offsets[int(reach_id)] + local

    all_sword["_on_reach_m"] = [
        float(corridor_lookup.loc[int(row.sword_reach_id)].geometry.project(row.geometry))
        for row in all_sword.itertuples(index=False)
    ]
    all_sword["_path_chainage_m"] = [
        path_chainage(int(reach_id), float(position))
        for reach_id, position in zip(all_sword["sword_reach_id"], all_sword["_on_reach_m"], strict=True)
    ]
    source_chainage = path_chainage(source_reach, float(projections_table.loc["network_source", "projection_m"]))
    sink_chainage = path_chainage(sink_reach, float(projections_table.loc["network_sink", "projection_m"]))
    if source_chainage > sink_chainage:
        raise ValueError("SWORD endpoint projections are inconsistent with the traced corridor direction.")
    all_sword = all_sword.loc[all_sword["_path_chainage_m"].between(source_chainage, sink_chainage)].copy()
    all_sword["centerline_offset_m"] = all_sword.geometry.distance(line)
    all_sword["centerline_station_m"] = all_sword.geometry.map(line.project)
    all_sword["reference_section_dist_out_m"] = np.where(
        bool(support["outlet_at_start"]),
        all_sword["centerline_station_m"],
        float(line.length) - all_sword["centerline_station_m"],
    )
    candidates = all_sword.loc[
        all_sword["sword_wse_fill_method"].astype(str).eq("requested_field")
        & pd.to_numeric(all_sword["sword_wse"], errors="coerce").notna()
        & all_sword["centerline_offset_m"].le(max_centerline_offset_m)
    ].drop_duplicates("sword_node_id").copy()
    if len(candidates) < 2:
        raise RuntimeError("Fewer than two valid endpoint-bounded SWORD profile nodes survive screening.")
    static = candidates.groupby("reference_section_dist_out_m", as_index=False).agg(
        static_wse_m=("sword_wse", "mean"),
        n_sword_nodes=("sword_node_id", "size"),
    ).sort_values("reference_section_dist_out_m")
    static["static_wse_m"] = _pava_nondecreasing(static["static_wse_m"].to_numpy(float))

    endpoints = endpoint_nodes.geometry.map(line.project).to_numpy(float)
    if not bool(support["outlet_at_start"]):
        endpoints = float(line.length) - endpoints
    endpoints = tuple(float(value) for value in np.sort(endpoints))
    raw_path = Path(raw_observations_path).expanduser().resolve()
    raw = pd.read_parquet(raw_path)
    observations = _prepare_raw_observations(raw, candidates)
    if observations.empty:
        raise RuntimeError("No raw RiverSP observations survive the accepted filter inside the endpoint-bounded SWORD corridor.")
    metadata = {
        "experiment_dir": str(experiment),
        "reference_section_summary": str(summary_path),
        "sword_node_source": str(sword_root),
        "raw_observations_path": str(raw_path),
        "sword_region": region,
        "corridor_reach_ids": [int(value) for value in corridor_ids],
        "n_static_profile_nodes": int(len(static)),
        "n_raw_observations": int(len(observations)),
        "n_raw_acquisitions": int(observations["acquisition_id"].nunique()),
        "raw_filter": RAW_RIVERSP_FILTER_DESCRIPTION,
        "max_centerline_offset_m": float(max_centerline_offset_m),
        "endpoint_buffer_m": float(endpoint_buffer_m),
        "endpoint_projection": projections,
    }
    return ReferenceProfileInputs(
        static_profile=static[["reference_section_dist_out_m", "static_wse_m"]],
        observations=observations,
        endpoint_distances_m=endpoints,
        metadata=metadata,
    )
