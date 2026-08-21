"""Read-only diagnostics for saved local-patch fallback attempts.

The helpers deliberately consume only artifacts already written beneath a
failed child's ``variant/local_patch_attempt`` directory.  They never rerun a
collapse or modify an experiment, which makes them safe for interactive use in
the local-patch investigation notebook.
"""

from __future__ import annotations

from collections import Counter
import ast
import json
import math
from pathlib import Path
import re
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shapely.geometry import LineString, Point, box

from hierarchy_level_definition.graph_building.directed_network_checks import parse_id_nodes


FAILURE_MODE_CATALOG: dict[str, dict[str, str]] = {
    "shared_upstream_downstream_boundary": {
        "title": "One parent node has both upstream and downstream interface roles",
        "stage": "exact anchored-boundary allocation before parent/local merge",
        "description": (
            "The selected unit has an external incoming and an external outgoing "
            "continuation at the same parent node. That node is therefore a valid "
            "shared interface, but the current allocator requires the upstream and "
            "downstream boundary-ID sets to be disjoint."
        ),
        "review": (
            "Verify that the shared node is incident to selected links on both sides "
            "and to retained external links. It should be allocated once and retain "
            "both interface roles rather than being duplicated."
        ),
    },
    "directed_anchored_interval": {
        "title": "Directed anchored interval omits a selected interface",
        "stage": "direction assignment followed by selected-interval extraction",
        "description": (
            "All exact boundary locations exist, but the provisional local direction "
            "does not place every prescribed interface on a directed path through the "
            "selected replacement. This commonly indicates that a boundary-adjacent "
            "regenerated link was directed without using its forced parent binding."
        ),
        "review": (
            "Compare the parent selected-link directions, exact boundary nodes, raw "
            "local graph, and full-raster changed geometry. Do not accept the case by "
            "weakening connectivity; determine which boundary-adjacent direction is inconsistent."
        ),
    },
    "anchored_link_distance_tie": {
        "title": "Two local links are equally close to an anchored boundary",
        "stage": "exact anchored-boundary allocation before parent/local merge",
        "description": (
            "The exact parent boundary lies at or beside a local junction where two "
            "regenerated links have the same minimum distance. Distance alone cannot "
            "identify which link should be split or snapped."
        ),
        "review": (
            "Inspect both candidate links, their common local node, parent direction, "
            "and the full-raster reference. Prefer an existing junction match when it "
            "preserves topology; otherwise use parent-link lineage to choose the segment."
        ),
    },
    "anchored_boundary_allocation_collision": {
        "title": "One local node is reused for multiple anchored boundaries",
        "stage": "exact anchored-boundary allocation before parent/local merge",
        "description": (
            "A compound-member replacement has several exact parent interfaces. "
            "Two nearby parent boundaries selected the same raw local node, so one "
            "parent interface was left as an artificial internal source or sink in "
            "the merged graph. The corrected allocator reserves each selected local "
            "boundary node and splits a local link at every additional exact parent "
            "location, producing a one-to-one boundary allocation."
        ),
        "review": (
            "Inspect parent boundary IDs that share a nearest raw node, their mutual "
            "distance, and the local link passing through both locations. The fix is "
            "valid when distinct zero-distance boundary nodes can be inserted in "
            "directed order without inventing a new corridor."
        ),
    },
    "raw_cut_terminal_count_or_tolerance": {
        "title": "Raw cut-terminal count or tolerance failure",
        "stage": "raw local RivGraph validation, before anchored trimming",
        "description": (
            "A safe local mask and raw RivGraph network were created, but the raw "
            "graph could not assign compatible upstream and downstream terminals "
            "from their RivGraph roles, relative distances to both computational "
            "cuts, and cut-to-cut connectivity. Absolute cut distance is diagnostic; "
            "the scientific splice remains the later exact anchored boundary."
        ),
        "review": (
            "Compare the cut with every labelled raw terminal and the blue water "
            "mask. For a zero-terminal case, decide whether the gap is a harmless "
            "pixel-scale skeleton offset or a real cut/corridor mismatch. For a "
            "multiple-terminal case, determine whether one candidate is a removable "
            "dangling spur or both represent real flow paths."
        ),
    },
    "duplicate_parent_node_match": {
        "title": "Multiple local nodes map to one parent node",
        "stage": "anchored child-to-parent node matching",
        "description": (
            "The local graph reached and was trimmed toward the selected scientific "
            "interfaces, but two or more distinct local nodes selected the same "
            "parent node as their spatial/topological match. A conservative one-to-one "
            "node map cannot accept both because that could silently merge a real "
            "local branch, create a zero-length link, or change connectivity."
        ),
        "review": (
            "Inspect the labelled local candidates around the red parent node. Check "
            "whether one candidate is the exact anchored interface and the other is "
            "an internal node that should remain new, or whether a short duplicated "
            "segment should be collapsed before matching."
        ),
    },
    "anchored_boundary_reach": {
        "title": "Exact anchored boundary is not reached",
        "stage": "trimming raw local topology to the selected-unit boundary",
        "description": (
            "The cut-to-cut raw network exists, but neither a raw local node nor a "
            "raw local link passes within the fixed tolerance of an exact parent "
            "boundary node that defines the selected anchored replacement. The merge "
            "therefore cannot split or snap the local graph at the required scientific "
            "interface without inventing unsupported geometry."
        ),
        "review": (
            "Compare the red exact boundary, the nearest black raw point, the green "
            "raw graph, and the purple full-raster reference. A small gap may be a "
            "consistent raster-resolution offset; a missing green corridor indicates "
            "that crop, collapse, or trimming topology must change."
        ),
    },
    "anchored_flow_connectivity": {
        "title": "Anchored boundaries are not flow-connected",
        "stage": "directed anchored-subgraph validation",
        "description": (
            "The required anchored boundary locations were found, but the retained "
            "directed local subgraph contains no upstream-to-downstream path between "
            "them. Accepting it would break network routing continuity."
        ),
        "review": "Inspect raw direction, removed spurs, and the retained path between boundaries.",
    },
    "cut_terminal_parent_match": {
        "title": "Cut terminal is absent from parent-node matches",
        "stage": "provisional parent-guided direction matching",
        "description": (
            "A unique cut terminal exists, but it was not retained in the provisional "
            "child-to-parent node-match table needed to guide local direction."
        ),
        "review": "Check forced cut bindings and whether the terminal should remain computational only.",
    },
    "raster_index_mismatch": {
        "title": "Full/local raster pixel-index mismatch",
        "stage": "changed-link width measurement",
        "description": (
            "A flat pixel index from the parent full raster was interpreted in the "
            "bounded local-raster shape. These index domains are not interchangeable."
        ),
        "review": "Confirm every measured link received regenerated local-raster indices.",
    },
    "anchored_projection_order": {
        "title": "Anchored projections occur in the wrong directed order",
        "stage": "anchored local-link geometry trimming",
        "description": (
            "The intended upstream and downstream parent boundaries both reach a "
            "raw local line, but their projections occur in the opposite order to "
            "that line's assigned flow direction."
        ),
        "review": "Check local direction, boundary roles, and whether an alternate exit-side run is appropriate.",
    },
    "collar_projection_collision": {
        "title": "Collar endpoints share one reference-centerline projection",
        "stage": "computational cut placement",
        "description": (
            "The collar's inner and outer nodes project to the same location on the "
            "single reference centerline, so the outward direction for the 1,000 m "
            "context cut cannot be inferred from those two projections."
        ),
        "review": "Inspect the collar, selected domain, and nearby reference-centerline trajectory.",
    },
    "context_cut_beyond_centerline": {
        "title": "Required context cut lies beyond the reference centerline",
        "stage": "computational cut placement at a global network end",
        "description": (
            "Moving 1,000 m away from the anchored boundary exceeds the available "
            "reference-centerline length, normally near a global source or sink."
        ),
        "review": "Check whether this is a true global boundary where the cut should clamp to the available endpoint.",
    },
    "unsafe_crop": {
        "title": "No safe local crop was found",
        "stage": "adaptive local-raster planning",
        "description": (
            "Even after halo expansion, retained main water touched the crop edge "
            "outside the two authorized bank cuts."
        ),
        "review": "Inspect the first and final crop attempts, water contacts, and whether another connected channel enters the window.",
    },
    "filename_too_long": {
        "title": "Compound RivGraph filename exceeds filesystem limits",
        "stage": "raw local RivGraph output creation",
        "description": (
            "The run name embedded every member-unit ID from a large compound and "
            "therefore exceeded the filesystem filename limit. A short deterministic "
            "compound name plus hash now replaces that filename; full IDs remain in "
            "the manifest."
        ),
        "review": "Confirm the shortened name is deterministic and the manifest retains the complete selected-unit list.",
    },
    "other": {
        "title": "Unclassified local-patch fallback",
        "stage": "unknown",
        "description": "The fallback reason is not yet represented in the diagnostic catalog.",
        "review": "Read the exact saved exception and add a new explicit failure class before fixing it.",
    },
}


def _failure_subtype(reason: str, failure_class: str) -> str:
    if failure_class == "anchored_boundary_allocation_collision":
        if "inlet/source" in reason and "outlet/sink" in reason:
            return "extra_internal_sources_and_sinks"
        if "inlet/source" in reason:
            return "extra_internal_source"
        if "outlet/sink" in reason:
            return "extra_internal_sink"
        return "merged_graph_connectivity_failure"
    if failure_class == "raw_cut_terminal_count_or_tolerance":
        match = re.search(
            r"has (\d+) (inlet|outlet) terminals .* the (upstream|downstream) cut",
            reason,
        )
        if match:
            count, terminal_role, cut_role = match.groups()
            count_label = "missing" if int(count) == 0 else "multiple"
            return f"{count_label}_{cut_role}_{terminal_role}_terminals"
        match = re.search(
            r"no unambiguous (inlet|outlet) terminal for the (upstream|downstream) cut",
            reason,
        )
        if match:
            terminal_role, cut_role = match.groups()
            return f"ambiguous_relative_{cut_role}_{terminal_role}_assignment"
    if failure_class == "duplicate_parent_node_match":
        return "nonunique_child_to_parent_node_assignment"
    if failure_class == "anchored_boundary_reach":
        role_match = re.search(r"the (upstream|downstream) anchored boundary", reason)
        role = role_match.group(1) if role_match else "unknown"
        return f"{role}_anchored_boundary_outside_tolerance"
    return failure_class


def _classify_failure_reason(reason: str) -> str:
    if "Anchored upstream and downstream parent boundary IDs must be unique" in reason:
        return "shared_upstream_downstream_boundary"
    if "No directed anchored interval connects every selected interface" in reason:
        return "directed_anchored_interval"
    if "Multiple local links are equally close" in reason:
        return "anchored_link_distance_tie"
    if "Merged local-patch graph failed validation" in reason:
        return "anchored_boundary_allocation_collision"
    if "Anchored boundary-node projections do not occur" in reason:
        return "anchored_projection_order"
    if "same reference-centerline projection" in reason:
        return "collar_projection_collision"
    if "falls beyond the reference centerline" in reason:
        return "context_cut_beyond_centerline"
    if "unauthorized_main_water_crop_edge_contact" in reason:
        return "unsafe_crop"
    if "File name too long" in reason:
        return "filename_too_long"
    if "not connected in flow direction" in reason:
        return "anchored_flow_connectivity"
    if re.search(r"Raw local RivGraph has [02] (?:inlet|outlet) terminals", reason):
        return "raw_cut_terminal_count_or_tolerance"
    if re.search(r"Raw local RivGraph has no unambiguous (?:inlet|outlet) terminal", reason):
        return "raw_cut_terminal_count_or_tolerance"
    if re.search(r"Cut terminal \d+ is absent from node matches", reason):
        return "cut_terminal_parent_match"
    if "Multiple local nodes map to the same parent node" in reason:
        return "duplicate_parent_node_match"
    if "anchored boundary node" in reason:
        return "anchored_boundary_reach"
    if "out of bounds for array with size" in reason:
        return "raster_index_mismatch"
    return "other"


def find_repo_root(start: str | Path | None = None) -> Path:
    """Find the repository whether the notebook starts at root or below it."""
    origin = Path(start or Path.cwd()).expanduser().resolve()
    for candidate in (origin, *origin.parents):
        if (candidate / "network_variants").is_dir() and (candidate / "RAPID").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find the river-hierarchy repository above {origin}.")


def _one_path(directory: Path, pattern: str) -> Path:
    paths = sorted(
        path for path in directory.glob(pattern) if not path.name.startswith("._")
    )
    if len(paths) != 1:
        raise FileNotFoundError(
            f"Expected one {pattern!r} beneath {directory}, found {len(paths)}."
        )
    return paths[0]


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def load_fallback_case(
    example_id: str | int,
    mode: str,
    state_id: str,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load one saved fallback and every available partial local artifact."""
    repo = find_repo_root(repo_root)
    example = f"{int(example_id):02d}" if str(example_id).isdigit() else str(example_id)
    mode_key = {"independent-units": "indep", "compound-bubbles": "compound_bubbles"}.get(
        str(mode), str(mode)
    )
    experiment = repo / "network_variants" / "output" / f"{example}_{mode_key}__anchored"
    state = experiment / "states" / str(state_id)
    attempt = state / "variant" / "local_patch_attempt"
    fallback_path = attempt / "fallback_to_full_raster.json"
    if not fallback_path.exists():
        raise FileNotFoundError(f"Saved fallback audit not found: {fallback_path}")

    base_directed = experiment / "states" / "S000_base" / "variant" / "directed"
    base_links = gpd.read_file(_one_path(base_directed, "*directed_links.gpkg"))
    base_nodes = gpd.read_file(_one_path(base_directed, "*directed_nodes.gpkg"))
    planning = _read_json(attempt / "planning" / "local_patch_planning_result.json")
    raster_plan = _read_json(attempt / "raster" / "local_patch_raster_plan.json")

    case: dict[str, Any] = {
        "repo_root": repo,
        "example_id": example,
        "mode": mode_key,
        "state_id": str(state_id),
        "experiment_dir": experiment,
        "state_dir": state,
        "attempt_dir": attempt,
        "fallback": _read_json(fallback_path),
        "planning": planning,
        "raster_plan": raster_plan,
        "base_links": base_links,
        "base_nodes": base_nodes,
        "cuts": None,
        "crop": None,
        "local_links": None,
        "local_nodes": None,
        "merged_links": None,
        "merged_nodes": None,
        "local_mask_path": None,
        "final_child_links": None,
        "final_child_nodes": None,
    }
    cuts_path = attempt / "raster" / "main_bank_cuts.gpkg"
    crop_path = attempt / "raster" / "local_patch_crop.gpkg"
    if cuts_path.exists():
        case["cuts"] = gpd.read_file(cuts_path)
    if crop_path.exists():
        case["crop"] = gpd.read_file(crop_path)
    rivgraph_dir = attempt / "variant" / "rivgraph"
    link_paths = sorted(path for path in rivgraph_dir.glob("*_links.gpkg") if not path.name.startswith("._"))
    node_paths = sorted(path for path in rivgraph_dir.glob("*_nodes.gpkg") if not path.name.startswith("._"))
    if len(link_paths) == 1 and len(node_paths) == 1:
        case["local_links"] = gpd.read_file(link_paths[0])
        case["local_nodes"] = gpd.read_file(node_paths[0])
    merged_links_path = attempt / "merge" / "merged_links.gpkg"
    merged_nodes_path = attempt / "merge" / "merged_nodes.gpkg"
    if merged_links_path.exists():
        case["merged_links"] = gpd.read_file(merged_links_path)
    if merged_nodes_path.exists():
        case["merged_nodes"] = gpd.read_file(merged_nodes_path)
    local_mask_path = attempt / "variant" / "mask" / "local_collapsed_mask.tif"
    if local_mask_path.exists():
        case["local_mask_path"] = local_mask_path
    final_directed = state / "variant" / "directed"
    final_link_paths = sorted(
        path for path in final_directed.glob("*directed_links.gpkg")
        if not path.name.startswith("._")
    )
    final_node_paths = sorted(
        path for path in final_directed.glob("*directed_nodes.gpkg")
        if not path.name.startswith("._")
    )
    if len(final_link_paths) == 1 and len(final_node_paths) == 1:
        case["final_child_links"] = gpd.read_file(final_link_paths[0])
        case["final_child_nodes"] = gpd.read_file(final_node_paths[0])
    return case


def _plot_local_mask(case: dict[str, Any], ax, *, alpha: float = 0.45) -> bool:
    """Draw the saved collapsed local water mask and return whether it existed."""
    path = case.get("local_mask_path")
    if path is None or not Path(path).exists():
        return False
    with rasterio.open(path) as source:
        water = source.read(1, masked=True)
        bounds = source.bounds
    visible = np.ma.masked_where(np.asarray(water).astype(bool) == 0, water)
    ax.imshow(
        visible,
        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
        origin="upper",
        cmap=ListedColormap(["#4c78a8"]),
        interpolation="nearest",
        alpha=float(alpha),
        zorder=0,
    )
    return True


def _local_mask_bounds(case: dict[str, Any]) -> np.ndarray | None:
    path = case.get("local_mask_path")
    if path is None or not Path(path).exists():
        return None
    with rasterio.open(path) as source:
        bounds = source.bounds
    return np.asarray([bounds.left, bounds.bottom, bounds.right, bounds.top], dtype=float)


def fallback_summary(case: dict[str, Any]) -> pd.Series:
    fallback = case["fallback"]
    planning = case.get("planning") or {}
    return pd.Series(
        {
            "example_id": case["example_id"],
            "mode": case["mode"],
            "state_id": case["state_id"],
            "fallback_stage": fallback.get("fallback_stage"),
            "fallback_code": fallback.get("fallback_code"),
            "fallback_reason": fallback.get("fallback_reason"),
            "planning_status": planning.get("status"),
            "local_artifacts_available": case.get("local_links") is not None,
        },
        name="value",
    )


def failure_mode_summary(case: dict[str, Any]) -> pd.Series:
    """Return the explicit type and interpretation of one saved fallback."""
    reason = str((case.get("fallback") or {}).get("fallback_reason", ""))
    failure_class = _classify_failure_reason(reason)
    metadata = FAILURE_MODE_CATALOG.get(
        failure_class, FAILURE_MODE_CATALOG["other"]
    )
    return pd.Series(
        {
            "failure_class": failure_class,
            "failure_subtype": _failure_subtype(reason, failure_class),
            "failure_title": metadata["title"],
            "failure_stage": metadata["stage"],
            "exact_saved_reason": reason,
            "what_this_failure_means": metadata["description"],
            "what_to_review_in_the_figure": metadata["review"],
        },
        name="value",
    )


def failure_mode_catalog() -> pd.DataFrame:
    """Return the documented diagnostic catalog as a notebook-friendly table."""
    return pd.DataFrame.from_records(
        [
            {
                "failure_class": failure_class,
                "failure_title": metadata["title"],
                "failure_stage": metadata["stage"],
                "description": metadata["description"],
                "review_guidance": metadata["review"],
            }
            for failure_class, metadata in FAILURE_MODE_CATALOG.items()
        ]
    )


def raw_geometry_quality_audit(case: dict[str, Any]) -> pd.DataFrame:
    """Expose self-crossing or implausibly long persisted raw-local geometries."""
    links = case.get("local_links")
    if links is None:
        raise FileNotFoundError("Saved raw local links are unavailable for this case.")
    records: list[dict[str, Any]] = []
    for feature in links.itertuples(index=False):
        geometry = feature.geometry
        min_x, min_y, max_x, max_y = geometry.bounds
        bbox_diagonal = float(math.hypot(max_x - min_x, max_y - min_y))
        if geometry.geom_type == "LineString":
            vertex_count = len(geometry.coords)
        elif hasattr(geometry, "geoms"):
            vertex_count = sum(len(part.coords) for part in geometry.geoms if hasattr(part, "coords"))
        else:
            vertex_count = 0
        records.append(
            {
                "id_link": int(feature.id_link),
                "id_nodes": feature.id_nodes,
                "length_m": float(geometry.length),
                "bbox_diagonal_m": bbox_diagonal,
                "length_to_bbox_ratio": (
                    float(geometry.length) / bbox_diagonal if bbox_diagonal > 0 else np.inf
                ),
                "vertex_count": int(vertex_count),
                "is_simple": bool(geometry.is_simple),
                "is_valid": bool(geometry.is_valid),
            }
        )
    return pd.DataFrame.from_records(records).sort_values(
        ["is_simple", "length_to_bbox_ratio", "length_m"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def anchored_link_distance_tie_audit(case: dict[str, Any]) -> pd.DataFrame:
    """List tied local links and whether they meet at a reusable local junction."""
    reason = str((case.get("fallback") or {}).get("fallback_reason", ""))
    match = re.search(
        r"Multiple local links are equally close to the (upstream|downstream) "
        r"anchored boundary node (\d+)",
        reason,
    )
    if not match:
        return pd.DataFrame()
    role, parent_id_text = match.groups()
    parent_id = int(parent_id_text)
    target = case["base_nodes"].loc[
        case["base_nodes"]["id_node"].astype(int).eq(parent_id)
    ].geometry.iloc[0]
    links = case["local_links"].copy()
    links["distance_m"] = links.geometry.distance(target)
    minimum = float(links["distance_m"].min())
    tied = links.loc[links["distance_m"].sub(minimum).abs().le(1e-6)].copy()
    endpoint_sets = [set(parse_id_nodes(value)) for value in tied["id_nodes"]]
    shared_nodes = set.intersection(*endpoint_sets) if endpoint_sets else set()
    records = []
    for feature in tied.itertuples(index=False):
        endpoint_ids = parse_id_nodes(feature.id_nodes)
        records.append(
            {
                "boundary_role": role,
                "parent_boundary_node_id": parent_id,
                "raw_local_link_id": int(feature.id_link),
                "raw_local_endpoint_ids": ", ".join(str(value) for value in endpoint_ids),
                "distance_m": float(feature.distance_m),
                "shared_local_junction_ids": ", ".join(
                    str(value) for value in sorted(shared_nodes)
                ),
                "shared_junction_available": bool(shared_nodes),
            }
        )
    return pd.DataFrame.from_records(records)


def fallback_inventory(
    example_ids: tuple[str | int, ...] = (2, 3, 5),
    *,
    repo_root: str | Path | None = None,
) -> pd.DataFrame:
    """Inventory the current anchored local-patch fallbacks without loading rasters."""
    repo = find_repo_root(repo_root)
    output_root = repo / "network_variants" / "output"
    records: list[dict[str, Any]] = []
    for example_id in example_ids:
        example = f"{int(example_id):02d}" if str(example_id).isdigit() else str(example_id)
        for mode in ("indep", "compound_bubbles"):
            experiment = output_root / f"{example}_{mode}__anchored"
            for path in sorted(
                experiment.glob(
                    "states/*/variant/local_patch_attempt/fallback_to_full_raster.json"
                )
            ):
                fallback = _read_json(path) or {}
                state_id = path.parents[2].name
                planning = _read_json(
                    path.parent / "planning" / "local_patch_planning_result.json"
                ) or {}
                contract = planning.get("contract") or {}
                reason = str(fallback.get("fallback_reason", ""))
                failure_class = _classify_failure_reason(reason)
                failure_metadata = FAILURE_MODE_CATALOG.get(
                    failure_class, FAILURE_MODE_CATALOG["other"]
                )
                boundary_match = re.search(
                    r"within ([0-9.]+) m of the (upstream|downstream) "
                    r"anchored boundary node (\d+)",
                    reason,
                )
                records.append(
                    {
                        "example_id": example,
                        "mode": mode,
                        "state_id": state_id,
                        "selected_unit_ids": ", ".join(
                            str(int(value)) for value in contract.get("selected_unit_ids", [])
                        ),
                        "fallback_stage": fallback.get("fallback_stage"),
                        "fallback_code": fallback.get("fallback_code"),
                        "fallback_reason": reason,
                        "failure_class": failure_class,
                        "failure_subtype": _failure_subtype(reason, failure_class),
                        "failure_title": failure_metadata["title"],
                        "failure_stage_description": failure_metadata["stage"],
                        "failure_mode_description": failure_metadata["description"],
                        "failure_review_guidance": failure_metadata["review"],
                        "boundary_role": boundary_match.group(2) if boundary_match else "",
                        "boundary_node_id": (
                            int(boundary_match.group(3)) if boundary_match else pd.NA
                        ),
                        "merge_tolerance_m": (
                            float(boundary_match.group(1)) if boundary_match else np.nan
                        ),
                        "raw_local_vectors_available": (
                            len(list((path.parent / "variant" / "rivgraph").glob("*_links.gpkg")))
                            == 1
                        ),
                    }
                )
    return pd.DataFrame.from_records(records)


def state_failure_inventory(
    example_ids: tuple[str | int, ...] = (11, 13, 15),
    *,
    repo_root: str | Path | None = None,
) -> pd.DataFrame:
    """Inventory child-state failures separately from local-patch fallbacks."""
    repo = find_repo_root(repo_root)
    output_root = repo / "network_variants" / "output"
    records: list[dict[str, Any]] = []
    for example_id in example_ids:
        example = f"{int(example_id):02d}" if str(example_id).isdigit() else str(example_id)
        for mode in ("indep", "compound_bubbles"):
            experiment = output_root / f"{example}_{mode}__anchored"
            for path in sorted(experiment.glob("states/*/state_failure.json")):
                failure = _read_json(path) or {}
                records.append(
                    {
                        "example_id": example,
                        "mode": mode,
                        "state_id": path.parent.name,
                        "selection_type": failure.get("selection_type"),
                        "selected_unit_ids": ", ".join(
                            str(int(value))
                            for value in failure.get("selected_unit_ids", [])
                        ),
                        "error_stage": failure.get("error_stage"),
                        "error": failure.get("error"),
                    }
                )
    return pd.DataFrame.from_records(records)


def load_state_failure_case(
    example_id: str | int,
    mode: str,
    state_id: str,
    *,
    repo_root: str | Path | None = None,
    max_path_cutoff: int = 100,
    max_paths: int = 5000,
) -> dict[str, Any]:
    """Rebuild the detected-unit context for one saved hierarchy failure.

    This is a read-only diagnostic reconstruction.  It deliberately uses the
    persisted directed child network rather than rerunning variant generation.
    """
    from hierarchy_level_definition.unit_detection.bifurcation_confluence_units import (
        annotate_unit_context,
        build_graph,
        build_structural_units,
        load_network,
    )

    repo = find_repo_root(repo_root)
    example = f"{int(example_id):02d}" if str(example_id).isdigit() else str(example_id)
    mode_key = {"independent-units": "indep", "compound-bubbles": "compound_bubbles"}.get(
        str(mode), str(mode)
    )
    experiment = repo / "network_variants" / "output" / f"{example}_{mode_key}__anchored"
    state = experiment / "states" / str(state_id)
    failure_path = state / "state_failure.json"
    if not failure_path.exists():
        raise FileNotFoundError(f"Saved state failure not found: {failure_path}")

    base_directed = experiment / "states" / "S000_base" / "variant" / "directed"
    child_directed = state / "variant" / "directed"
    base_links_path = _one_path(base_directed, "*directed_links.gpkg")
    base_nodes_path = _one_path(base_directed, "*directed_nodes.gpkg")
    child_links_path = _one_path(child_directed, "*directed_links.gpkg")
    child_nodes_path = _one_path(child_directed, "*directed_nodes.gpkg")

    child_links, child_nodes = load_network(child_links_path, child_nodes_path)
    child_graph = build_graph(child_links, child_nodes)
    units = build_structural_units(
        child_graph,
        max_path_cutoff=int(max_path_cutoff),
        max_paths=int(max_paths),
    )
    annotate_unit_context(units, graph=child_graph)

    failure = _read_json(failure_path) or {}
    bubble_match = re.search(r"Assembled bubble (\d+)", str(failure.get("error", "")))
    bubble_id = int(bubble_match.group(1)) if bubble_match else None
    bubble_units = [
        unit for unit in units if bubble_id is not None and unit.compound_bubble_id == bubble_id
    ]
    edge_membership: dict[tuple[int, int, int], list[int]] = {}
    for unit in bubble_units:
        for edge in unit.edge_set:
            normalized = tuple(int(value) for value in edge)
            edge_membership.setdefault(normalized, []).append(int(unit.unit_id))

    assembled = nx.MultiDiGraph()
    for edge in edge_membership:
        assembled.add_edge(edge[0], edge[1], key=edge[2])
    cyclic_components = [
        set(int(node) for node in component)
        for component in nx.strongly_connected_components(assembled)
        if len(component) > 1
        or any(assembled.has_edge(int(node), int(node)) for node in component)
    ]
    cyclic_node_ids = set().union(*cyclic_components) if cyclic_components else set()
    cyclic_edge_ids = {
        int(key)
        for u, v, key in assembled.edges(keys=True)
        if any(int(u) in component and int(v) in component for component in cyclic_components)
    }
    local_rivgraph = state / "variant" / "local_patch_attempt" / "variant" / "rivgraph"
    local_link_paths = sorted(
        path for path in local_rivgraph.glob("*_links.gpkg") if not path.name.startswith("._")
    )
    local_node_paths = sorted(
        path for path in local_rivgraph.glob("*_nodes.gpkg") if not path.name.startswith("._")
    )
    local_mask_path = (
        state
        / "variant"
        / "local_patch_attempt"
        / "variant"
        / "mask"
        / "local_collapsed_mask.tif"
    )
    lineage_path = state / "variant" / "matching" / "link_lineage.csv"
    direction_report_path = child_directed / "direction_validation_report.json"

    return {
        "repo_root": repo,
        "example_id": example,
        "mode": mode_key,
        "state_id": str(state_id),
        "experiment_dir": experiment,
        "state_dir": state,
        "failure": failure,
        "base_links": gpd.read_file(base_links_path),
        "base_nodes": gpd.read_file(base_nodes_path),
        "child_links": child_links,
        "child_nodes": child_nodes,
        "child_graph": child_graph,
        "whole_child_network_is_dag": nx.is_directed_acyclic_graph(nx.DiGraph(child_graph)),
        "direction_validation_report": _read_json(direction_report_path) or {},
        "link_lineage": pd.read_csv(lineage_path) if lineage_path.exists() else None,
        "units": units,
        "bubble_id": bubble_id,
        "bubble_units": bubble_units,
        "bubble_edge_membership": edge_membership,
        "assembled_bubble_graph": assembled,
        "cyclic_components": cyclic_components,
        "cyclic_node_ids": cyclic_node_ids,
        "cyclic_edge_ids": cyclic_edge_ids,
        "local_links": gpd.read_file(local_link_paths[0]) if len(local_link_paths) == 1 else None,
        "local_nodes": gpd.read_file(local_node_paths[0]) if len(local_node_paths) == 1 else None,
        "local_mask_path": local_mask_path if local_mask_path.exists() else None,
    }


def bubble_dag_audit(case: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Return member-unit, edge, and strongly-connected-component tables."""
    unit_rows = [
        {
            "unit_id": int(unit.unit_id),
            "bifurcation": int(unit.bifurcation),
            "confluence": int(unit.confluence),
            "compound_bubble_role": str(unit.compound_bubble_role),
            "n_nodes": len(unit.node_set),
            "n_edges": len(unit.edge_set),
            "contains_cycle_node": bool(set(unit.node_set) & case["cyclic_node_ids"]),
        }
        for unit in case["bubble_units"]
    ]
    edge_rows = []
    child_link_index = case["child_links"].set_index(
        case["child_links"]["id_link"].astype(int), drop=False
    )
    lineage = case.get("link_lineage")
    lineage_index = (
        lineage.set_index(lineage["id_link"].astype(int), drop=False)
        if lineage is not None and not lineage.empty
        else None
    )
    for (u, v, key), member_ids in sorted(case["bubble_edge_membership"].items()):
        child_row = child_link_index.loc[int(key)]
        lineage_row = (
            lineage_index.loc[int(key)]
            if lineage_index is not None and int(key) in lineage_index.index
            else None
        )
        edge_rows.append(
            {
                "id_link": int(key),
                "id_us_node": int(u),
                "id_ds_node": int(v),
                "member_unit_ids": ", ".join(str(value) for value in sorted(member_ids)),
                "membership_count": len(member_ids),
                "in_cycle": int(key) in case["cyclic_edge_ids"],
                "lineage_type": child_row.get("lineage_type", pd.NA),
                "dominant_parent_link_id": (
                    lineage_row.get("dominant_parent_link_id", pd.NA)
                    if lineage_row is not None
                    else pd.NA
                ),
                "lineage_method": (
                    lineage_row.get("lineage_method", pd.NA)
                    if lineage_row is not None
                    else pd.NA
                ),
            }
        )
    component_rows = [
        {
            "component_index": index,
            "node_ids": ", ".join(str(value) for value in sorted(component)),
            "node_count": len(component),
            "edge_ids": ", ".join(
                str(key)
                for u, v, key in case["assembled_bubble_graph"].edges(keys=True)
                if int(u) in component and int(v) in component
            ),
        }
        for index, component in enumerate(case["cyclic_components"], start=1)
    ]
    return {
        "member_units": pd.DataFrame.from_records(unit_rows),
        "bubble_edges": pd.DataFrame.from_records(edge_rows),
        "cyclic_components": pd.DataFrame.from_records(component_rows),
    }


def plotly_bubble_dag_failure(case: dict[str, Any]) -> go.Figure:
    """Plot the valid child network and the cycle introduced by bubble assembly."""
    child_links = case["child_links"]
    bubble_ids = set(case["bubble_edge_membership"])
    bubble_link_ids = {int(edge[2]) for edge in bubble_ids}
    bubble_links = child_links.loc[child_links["id_link"].astype(int).isin(bubble_link_ids)]
    if bubble_links.empty:
        raise ValueError("The reconstructed bubble contains no persisted child-link geometry.")
    bounds = np.asarray(bubble_links.total_bounds, dtype=float)
    span = max(float(bounds[2] - bounds[0]), float(bounds[3] - bounds[1]), 1.0)
    area = box(
        bounds[0] - 0.12 * span,
        bounds[1] - 0.12 * span,
        bounds[2] + 0.12 * span,
        bounds[3] + 0.12 * span,
    )
    bounds = np.asarray(area.bounds, dtype=float)
    figure = go.Figure()
    _add_plotly_local_mask(figure, case)
    visible_base = case["base_links"].loc[case["base_links"].geometry.intersects(area)]
    _add_plotly_lines(
        figure,
        visible_base,
        name="base/reference network",
        color="#777777",
        width=1.6,
        dash="dash",
        opacity=0.65,
        id_column="id_link",
    )
    local_links = case.get("local_links")
    if local_links is not None:
        _add_plotly_lines(
            figure,
            local_links.loc[local_links.geometry.intersects(area)],
            name="raw local RivGraph",
            color="#2ca02c",
            width=2.5,
            id_column="id_link",
        )
    changed = child_links.loc[child_links.geometry.intersects(area)]
    if "lineage_type" in changed.columns:
        changed = changed.loc[changed["lineage_type"].astype(str).ne("unchanged_1to1")]
    _add_plotly_lines(
        figure,
        changed,
        name="final anchored changed links",
        color="#9467bd",
        width=3.2,
        id_column="id_link",
    )
    _add_plotly_lines(
        figure,
        bubble_links,
        name=f"assembled bubble {case['bubble_id']}",
        color="#d62728",
        width=4.0,
        id_column="id_link",
    )
    cycle_links = bubble_links.loc[
        bubble_links["id_link"].astype(int).isin(case["cyclic_edge_ids"])
    ]
    _add_plotly_lines(
        figure,
        cycle_links,
        name="cycle-forming links",
        color="#ff7f0e",
        width=8.0,
        id_column="id_link",
    )
    _add_plotly_direction_arrows(
        figure,
        cycle_links,
        case["child_nodes"],
        color="#9c3f00",
        position=0.52,
        half_span=0.09,
    )
    cycle_nodes = case["child_nodes"].loc[
        case["child_nodes"]["id_node"].astype(int).isin(case["cyclic_node_ids"])
    ]
    if not cycle_nodes.empty:
        figure.add_trace(
            go.Scattergl(
                x=cycle_nodes.geometry.x,
                y=cycle_nodes.geometry.y,
                mode="markers+text",
                marker={"size": 13, "color": "#111111", "symbol": "x"},
                text=[f"cycle node {int(value)}" for value in cycle_nodes["id_node"]],
                textposition="top right",
                name="cycle nodes",
                hovertemplate="%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
            )
        )
    return _finish_plotly_map(
        figure,
        title=(
            f"Example {case['example_id']} {case['state_id']} — assembled bubble "
            f"{case['bubble_id']} DAG failure"
        ),
        bounds=bounds,
    )


def raw_endpoint_column_audit(case: dict[str, Any]) -> pd.DataFrame:
    """Show persisted endpoint columns beside endpoints parsed from ``id_nodes``."""
    links = case.get("local_links")
    if links is None:
        raise FileNotFoundError("Saved raw local links are unavailable for this case.")
    records: list[dict[str, Any]] = []
    for row in links.itertuples(index=False):
        parsed_us, parsed_ds = parse_id_nodes(row.id_nodes)
        stored_us = getattr(row, "id_us_node", None)
        stored_ds = getattr(row, "id_ds_node", None)
        records.append(
            {
                "id_link": int(row.id_link),
                "id_nodes": row.id_nodes,
                "stored_id_us_node": stored_us,
                "stored_id_ds_node": stored_ds,
                "stored_us_is_null": bool(pd.isna(stored_us)),
                "stored_ds_is_null": bool(pd.isna(stored_ds)),
                "parsed_id_us_node": int(parsed_us),
                "parsed_id_ds_node": int(parsed_ds),
                "normalization_required": bool(pd.isna(stored_us) or pd.isna(stored_ds)),
            }
        )
    return pd.DataFrame.from_records(records).sort_values("id_link").reset_index(drop=True)


def anchored_boundary_audit(case: dict[str, Any]) -> pd.DataFrame:
    """Measure exact selected boundaries against raw and completed child geometry."""
    local_links = case.get("local_links")
    local_nodes = case.get("local_nodes")
    if local_links is None or local_nodes is None:
        raise FileNotFoundError("Saved raw local vectors are unavailable for this case.")
    contract = (case.get("planning") or {}).get("contract") or {}
    reason = str(case["fallback"].get("fallback_reason", ""))
    failed_match = re.search(
        r"within ([0-9.]+) m of the (upstream|downstream) "
        r"anchored boundary node (\d+)",
        reason,
    )
    tolerance = float(failed_match.group(1)) if failed_match else np.nan
    failed_role = failed_match.group(2) if failed_match else ""
    failed_node = int(failed_match.group(3)) if failed_match else None
    specifications = [
        ("upstream", value)
        for value in contract.get("anchored_upstream_boundary_node_ids", [])
    ] + [
        ("downstream", value)
        for value in contract.get("anchored_downstream_boundary_node_ids", [])
    ]
    base_nodes = case["base_nodes"].set_index(
        case["base_nodes"]["id_node"].astype(int), drop=False
    )
    cuts = case.get("cuts")
    final_links = case.get("final_child_links")
    final_nodes = case.get("final_child_nodes")
    records: list[dict[str, Any]] = []
    for role, value in specifications:
        parent_id = int(value)
        target = base_nodes.loc[parent_id].geometry
        node_distances = local_nodes.geometry.distance(target)
        link_distances = local_links.geometry.distance(target)
        node_index = node_distances.idxmin()
        link_index = link_distances.idxmin()
        nearest_link = local_links.loc[link_index]
        measure = float(nearest_link.geometry.project(target))
        nearest_point = nearest_link.geometry.interpolate(measure)
        cut_distance = np.nan
        if cuts is not None:
            role_cuts = cuts.loc[cuts["role"].astype(str).eq(role)]
            if len(role_cuts) == 1:
                cut_distance = float(target.distance(role_cuts.iloc[0].geometry))
        final_node_distance = np.nan
        final_link_distance = np.nan
        if final_nodes is not None and not final_nodes.empty:
            final_node_distance = float(final_nodes.geometry.distance(target).min())
        if final_links is not None and not final_links.empty:
            final_link_distance = float(final_links.geometry.distance(target).min())
        minimum_raw_distance = min(
            float(node_distances.loc[node_index]), float(link_distances.loc[link_index])
        )
        records.append(
            {
                "role": role,
                "parent_boundary_node_id": parent_id,
                "is_reported_failure": bool(role == failed_role and parent_id == failed_node),
                "merge_tolerance_m": tolerance,
                "nearest_raw_node_id": int(local_nodes.loc[node_index, "id_node"]),
                "nearest_raw_node_distance_m": float(node_distances.loc[node_index]),
                "nearest_raw_link_id": int(nearest_link["id_link"]),
                "nearest_raw_link_distance_m": float(link_distances.loc[link_index]),
                "minimum_raw_distance_m": minimum_raw_distance,
                "distance_beyond_tolerance_m": (
                    minimum_raw_distance - tolerance if np.isfinite(tolerance) else np.nan
                ),
                "boundary_to_computational_cut_m": cut_distance,
                "nearest_raw_point_x": float(nearest_point.x),
                "nearest_raw_point_y": float(nearest_point.y),
                "final_child_nearest_node_distance_m": final_node_distance,
                "final_child_nearest_link_distance_m": final_link_distance,
            }
        )
    return pd.DataFrame.from_records(records)


def anchored_boundary_allocation_audit(case: dict[str, Any]) -> pd.DataFrame:
    """Identify parent boundaries competing for the same raw local node."""
    audit = anchored_boundary_audit(case).copy()
    if audit.empty:
        return audit
    base_nodes = case["base_nodes"].set_index(
        case["base_nodes"]["id_node"].astype(int), drop=False
    )
    audit["boundaries_sharing_nearest_raw_node"] = audit.groupby(
        "nearest_raw_node_id"
    )["parent_boundary_node_id"].transform("count")
    audit["nearest_raw_node_reused"] = (
        audit["boundaries_sharing_nearest_raw_node"] > 1
    )
    minimum_other_distances: list[float] = []
    for row in audit.itertuples(index=False):
        target = base_nodes.loc[int(row.parent_boundary_node_id)].geometry
        other_ids = audit.loc[
            audit["parent_boundary_node_id"].astype(int).ne(
                int(row.parent_boundary_node_id)
            ),
            "parent_boundary_node_id",
        ].astype(int)
        minimum_other_distances.append(
            min(
                (
                    float(target.distance(base_nodes.loc[int(other_id)].geometry))
                    for other_id in other_ids
                ),
                default=np.nan,
            )
        )
    audit["nearest_other_parent_boundary_distance_m"] = minimum_other_distances
    audit["corrected_allocation_rule"] = np.where(
        audit["nearest_raw_node_reused"],
        "reserve first child; split local link at each additional exact parent boundary",
        "unique nearest child candidate",
    )
    return audit


def _terminal_nodes(case: dict[str, Any]) -> gpd.GeoDataFrame:
    links = case.get("local_links")
    nodes = case.get("local_nodes")
    if links is None or nodes is None:
        raise FileNotFoundError("This fallback stopped before local RivGraph outputs were written.")
    degree: Counter[int] = Counter()
    for value in links["id_nodes"]:
        node_a, node_b = parse_id_nodes(value)
        degree[int(node_a)] += 1
        degree[int(node_b)] += 1
    terminals = nodes.loc[nodes["id_node"].astype(int).map(degree).eq(1)].copy()
    terminals["degree"] = 1
    return terminals


def terminal_audit(case: dict[str, Any]) -> pd.DataFrame:
    """Measure every degree-one local node against both intended cuts."""
    cuts = case.get("cuts")
    if cuts is None or cuts.empty:
        raise FileNotFoundError("Saved main-bank cut geometry is unavailable for this case.")
    terminals = _terminal_nodes(case)
    records: list[dict[str, Any]] = []
    for terminal in terminals.itertuples(index=False):
        record = {
            "child_id_node": int(terminal.id_node),
            "is_inlet": bool(getattr(terminal, "is_inlet", False)),
            "is_outlet": bool(getattr(terminal, "is_outlet", False)),
            "x": float(terminal.geometry.x),
            "y": float(terminal.geometry.y),
        }
        for cut in cuts.itertuples(index=False):
            record[f"distance_to_{cut.role}_cut_m"] = float(terminal.geometry.distance(cut.geometry))
        records.append(record)
    result = pd.DataFrame.from_records(records)
    reason = str(case["fallback"].get("fallback_reason", ""))
    match = re.search(r"within ([0-9.]+) m", reason)
    result["fixed_merge_tolerance_m"] = float(match.group(1)) if match else np.nan
    return result


def raster_index_audit(case: dict[str, Any]) -> pd.DataFrame:
    """Compare saved link pixel-index domains with the local raster domain."""
    path = case.get("local_mask_path")
    if path is None or not Path(path).exists():
        raise FileNotFoundError("Saved local collapsed mask is unavailable for this case.")
    with rasterio.open(path) as source:
        raster_size = int(source.width * source.height)
        raster_width = int(source.width)
        raster_height = int(source.height)
    records: list[dict[str, Any]] = []
    for product_name in ("local_links", "merged_links", "final_child_links"):
        links = case.get(product_name)
        if links is None:
            continue
        for column in ("idx_link", "idx"):
            if column not in links.columns:
                continue
            values = [
                item
                for value in links[column]
                for item in _parse_id_list(value)
            ]
            records.append(
                {
                    "product": product_name,
                    "index_column": column,
                    "n_links": int(len(links)),
                    "n_pixel_indices": int(len(values)),
                    "minimum_index": min(values) if values else pd.NA,
                    "maximum_index": max(values) if values else pd.NA,
                    "local_raster_width": raster_width,
                    "local_raster_height": raster_height,
                    "local_raster_size": raster_size,
                    "indices_fit_local_raster": bool(values and min(values) >= 0 and max(values) < raster_size),
                }
            )
    return pd.DataFrame.from_records(records)


def final_changed_link_audit(case: dict[str, Any]) -> pd.DataFrame:
    """Describe the completed anchored child links that replaced parent topology."""
    links = case.get("final_child_links")
    if links is None:
        raise FileNotFoundError("The completed fallback child network is unavailable.")
    if "lineage_type" not in links.columns:
        raise ValueError("The completed child links do not contain lineage_type.")
    changed = links.loc[links["lineage_type"].astype(str).ne("unchanged_1to1")].copy()
    columns = [
        column for column in (
            "id_link",
            "id_us_node",
            "id_ds_node",
            "matched_parent_us_node",
            "matched_parent_ds_node",
            "lineage_type",
            "len",
        )
        if column in changed.columns
    ]
    return pd.DataFrame(changed[columns]).reset_index(drop=True)


def plot_terminal_case(case: dict[str, Any], *, pad_fraction: float = 0.05):
    """Plot the local mask, cut lines, raw links, and role-labelled terminals."""
    links = case.get("local_links")
    cuts = case.get("cuts")
    if links is None or cuts is None:
        raise FileNotFoundError("Terminal plots require saved local links and cut geometries.")
    terminals = _terminal_nodes(case)
    fig, ax = plt.subplots(figsize=(10, 8))
    mask_drawn = _plot_local_mask(case, ax)
    crop = case.get("crop")
    mask_bounds = _local_mask_bounds(case)
    if crop is not None and not crop.empty:
        display_bounds = np.asarray(crop.total_bounds, dtype=float)
    elif mask_bounds is not None:
        display_bounds = mask_bounds
    else:
        display_bounds = np.asarray(links.total_bounds, dtype=float)
    local_extent = box(*display_bounds)
    base_context = case["base_links"].loc[
        case["base_links"].geometry.intersects(local_extent)
    ]
    base_context.plot(
        ax=ax,
        color="0.35",
        linewidth=2.2,
        linestyle="--",
        alpha=0.85,
        zorder=1,
        label="base/reference network",
    )
    if crop is not None:
        crop.boundary.plot(ax=ax, color="black", linewidth=1.4, zorder=2, label="crop boundary")
    links.plot(ax=ax, color="#2ca02c", linewidth=2.2, zorder=3, label="raw local RivGraph")
    final_links = case.get("final_child_links")
    if final_links is not None and "lineage_type" in final_links.columns:
        final_changed = final_links.loc[
            final_links["lineage_type"].astype(str).ne("unchanged_1to1")
            & final_links.geometry.intersects(local_extent)
        ].copy()
        if not final_changed.empty:
            final_changed.plot(
                ax=ax,
                color="#9467bd",
                linewidth=3.2,
                alpha=0.9,
                zorder=5,
                label="final anchored changed link",
            )
            final_nodes = case.get("final_child_nodes")
            if final_nodes is not None:
                endpoint_ids: set[int] = set()
                for value in final_changed["id_nodes"]:
                    endpoint_ids.update(int(node) for node in parse_id_nodes(value))
                endpoints = final_nodes.loc[
                    final_nodes["id_node"].astype(int).isin(endpoint_ids)
                ]
                endpoints.plot(
                    ax=ax,
                    color="#9467bd",
                    marker="o",
                    edgecolor="white",
                    markersize=75,
                    zorder=6,
                    label="final anchored endpoints",
                )
    colors = {"upstream": "#1f77b4", "downstream": "#ff7f0e"}
    for cut in cuts.itertuples(index=False):
        gpd.GeoSeries([cut.geometry], crs=cuts.crs).plot(
            ax=ax, color=colors[str(cut.role)], linewidth=3, zorder=4, label=f"{cut.role} cut"
        )
    for row in terminals.itertuples(index=False):
        marker = "^" if bool(getattr(row, "is_inlet", False)) else "s"
        color = "#1f77b4" if bool(getattr(row, "is_inlet", False)) else "#ff7f0e"
        ax.scatter(row.geometry.x, row.geometry.y, marker=marker, s=110, color=color, edgecolor="black", zorder=5)
        ax.annotate(f"local node {int(row.id_node)}", (row.geometry.x, row.geometry.y), xytext=(5, 5), textcoords="offset points")
    bounds = display_bounds
    pad = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * float(pad_fraction)
    ax.set_xlim(bounds[0] - pad, bounds[2] + pad)
    ax.set_ylim(bounds[1] - pad, bounds[3] + pad)
    ax.set_aspect("equal")
    ax.set_title(f"Example {case['example_id']} {case['state_id']} — cut/terminal audit")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if mask_drawn:
        from matplotlib.patches import Patch

        unique = {"collapsed local water mask": Patch(facecolor="#4c78a8", alpha=0.45), **unique}
    ax.legend(unique.values(), unique.keys(), loc="best")
    return fig, ax


def _duplicate_parent_ids(case: dict[str, Any]) -> list[int]:
    reason = str(case["fallback"].get("fallback_reason", ""))
    match = re.search(r"same parent node: \[([^]]+)\]", reason)
    return [int(value) for value in re.findall(r"\d+", match.group(1))] if match else []


def node_collision_audit(case: dict[str, Any]) -> pd.DataFrame:
    """List all local nodes near each duplicated parent match."""
    local_nodes = case.get("local_nodes")
    if local_nodes is None:
        raise FileNotFoundError("Saved local nodes are unavailable for this case.")
    parent_nodes = case["base_nodes"]
    reason = str(case["fallback"].get("fallback_reason", ""))
    tolerance_match = re.search(r"within ([0-9.]+) m", reason)
    tolerance = float(tolerance_match.group(1)) if tolerance_match else np.nan
    if not np.isfinite(tolerance):
        cuts = case.get("cuts")
        if cuts is not None:
            # Merge matching uses five pixels; infer it from the fallback's
            # terminal tolerance when available, otherwise show a broad 100 m neighborhood.
            tolerance = 100.0
    records: list[dict[str, Any]] = []
    terminals = _terminal_nodes(case)
    cuts = case.get("cuts")
    contract = (case.get("planning") or {}).get("contract") or {}
    for parent_id in _duplicate_parent_ids(case):
        parent = parent_nodes.loc[parent_nodes["id_node"].astype(int).eq(parent_id)].iloc[0]
        distances = local_nodes.geometry.distance(parent.geometry)
        nearby = local_nodes.loc[distances <= tolerance].copy()
        if nearby.empty:
            nearby = local_nodes.loc[distances.nsmallest(4).index].copy()
        for index, child in nearby.iterrows():
            records.append(
                {
                    "parent_id_node": parent_id,
                    "child_id_node": int(child["id_node"]),
                    "distance_m": float(distances.loc[index]),
                    "is_inlet": bool(child.get("is_inlet", False)),
                    "is_outlet": bool(child.get("is_outlet", False)),
                    "match_source": "spatial_parent_match",
                }
            )
        expected_roles = []
        if int(contract.get("source_splice_anchor_node_id", -1)) == parent_id:
            expected_roles.append(("upstream", "is_inlet"))
        if int(contract.get("sink_splice_anchor_node_id", -1)) == parent_id:
            expected_roles.append(("downstream", "is_outlet"))
        for role, role_column in expected_roles:
            if cuts is None:
                continue
            role_cuts = cuts.loc[cuts["role"].astype(str).eq(role)]
            compatible = terminals.loc[terminals[role_column].fillna(False).astype(bool)]
            if len(role_cuts) != 1 or compatible.empty:
                continue
            cut = role_cuts.iloc[0].geometry
            selected_index = compatible.geometry.distance(cut).idxmin()
            child = compatible.loc[selected_index]
            child_id = int(child["id_node"])
            if any(
                record["parent_id_node"] == parent_id
                and record["child_id_node"] == child_id
                and record["match_source"] == f"forced_{role}_cut"
                for record in records
            ):
                continue
            records.append(
                {
                    "parent_id_node": parent_id,
                    "child_id_node": child_id,
                    "distance_m": float(child.geometry.distance(parent.geometry)),
                    "is_inlet": bool(child.get("is_inlet", False)),
                    "is_outlet": bool(child.get("is_outlet", False)),
                    "match_source": f"forced_{role}_cut",
                }
            )
    return (
        pd.DataFrame.from_records(records)
        .drop_duplicates(["parent_id_node", "child_id_node", "match_source"])
        .sort_values(["parent_id_node", "distance_m", "match_source"])
    )


def plot_node_collision_case(case: dict[str, Any], *, radius_m: float = 250.0):
    local_links = case.get("local_links")
    local_nodes = case.get("local_nodes")
    if local_links is None or local_nodes is None:
        raise FileNotFoundError("Node-collision plots require saved local vectors.")
    parent_ids = _duplicate_parent_ids(case)
    parent_nodes = case["base_nodes"].loc[
        case["base_nodes"]["id_node"].astype(int).isin(parent_ids)
    ].copy()
    audit = node_collision_audit(case)
    candidate_ids = set(audit["child_id_node"].astype(int))
    candidates = local_nodes.loc[local_nodes["id_node"].astype(int).isin(candidate_ids)].copy()
    focus = pd.concat(
        [parent_nodes[[parent_nodes.geometry.name]], candidates[[candidates.geometry.name]]],
        ignore_index=True,
    )
    area = gpd.GeoSeries(focus[parent_nodes.geometry.name], crs=parent_nodes.crs).union_all().convex_hull.buffer(float(radius_m))
    fig, ax = plt.subplots(figsize=(9, 8))
    mask_drawn = _plot_local_mask(case, ax)
    case["base_links"].loc[case["base_links"].geometry.intersects(area)].plot(
        ax=ax, color="0.75", linewidth=2, label="parent links"
    )
    local_links.loc[local_links.geometry.intersects(area)].plot(
        ax=ax, color="#2ca02c", linewidth=2.5, label="local links"
    )
    nearby = local_nodes.loc[local_nodes.geometry.intersects(area)]
    nearby.plot(ax=ax, color="#1f77b4", markersize=45, label="local nodes")
    parent_nodes.plot(ax=ax, color="red", marker="*", markersize=180, label="duplicated parent match")
    for row in nearby.itertuples(index=False):
        ax.annotate(str(int(row.id_node)), (row.geometry.x, row.geometry.y), xytext=(4, 4), textcoords="offset points")
    cuts = case.get("cuts")
    if cuts is not None:
        cuts.loc[cuts.geometry.intersects(area)].plot(
            ax=ax, color="#17becf", linewidth=3, label="interface cuts"
        )
    minx, miny, maxx, maxy = area.bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    ax.set_title(f"Example {case['example_id']} {case['state_id']} — child/parent node collision")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if mask_drawn:
        from matplotlib.patches import Patch

        unique = {"collapsed local water mask": Patch(facecolor="#4c78a8", alpha=0.45), **unique}
    ax.legend(unique.values(), unique.keys(), loc="best")
    return fig, ax


def _parse_id_list(value: Any) -> list[int]:
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [int(item) for item in value]
    if pd.isna(value):
        return []
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = re.findall(r"\d+", text)
    if isinstance(parsed, (int, float, str)):
        parsed = [parsed]
    return [int(item) for item in parsed]


def planning_interface_audit(case: dict[str, Any]) -> dict[str, Any]:
    reason = str(case["fallback"].get("fallback_reason", ""))
    node_match = re.search(r"at node (\d+)", reason)
    links_match = re.search(r"links \[([^]]+)\]", reason)
    problem_node = int(node_match.group(1)) if node_match else None
    continuation_links = [int(value) for value in re.findall(r"\d+", links_match.group(1))] if links_match else []
    selected_units = (case.get("planning") or {}).get("audit", {}).get("selected_unit_ids", [])
    path_metrics = pd.read_csv(
        case["experiment_dir"] / "states" / "S000_base" / "hierarchy" / "path_metrics.csv"
    )
    core_links: set[int] = set()
    for value in path_metrics.loc[path_metrics["unit_id"].astype(int).isin(selected_units), "id_links"]:
        core_links.update(_parse_id_list(value))
    return {
        "problem_node_id": problem_node,
        "continuation_link_ids": continuation_links,
        "selected_unit_ids": [int(value) for value in selected_units],
        "core_link_ids": sorted(core_links),
    }


def plot_planning_interface_case(case: dict[str, Any], *, radius_m: float = 2500.0):
    audit = planning_interface_audit(case)
    node_id = audit["problem_node_id"]
    if node_id is None:
        raise ValueError("Could not parse the ambiguous interface node from the fallback reason.")
    nodes = case["base_nodes"]
    links = case["base_links"]
    node = nodes.loc[nodes["id_node"].astype(int).eq(node_id)].iloc[0]
    area = node.geometry.buffer(float(radius_m))
    visible = links.loc[links.geometry.intersects(area)].copy()
    fig, ax = plt.subplots(figsize=(9, 8))
    visible.plot(ax=ax, color="0.75", linewidth=2, label="base context")
    visible.loc[visible["id_link"].astype(int).isin(audit["core_link_ids"])].plot(
        ax=ax, color="crimson", linewidth=3, label="selected unit links"
    )
    visible.loc[visible["id_link"].astype(int).isin(audit["continuation_link_ids"])].plot(
        ax=ax, color="darkorange", linewidth=4, label="competing external continuations"
    )
    ax.scatter(node.geometry.x, node.geometry.y, marker="x", s=150, color="black", zorder=5, label=f"problem node {node_id}")
    for row in visible.itertuples(index=False):
        if int(row.id_link) in set(audit["core_link_ids"]) | set(audit["continuation_link_ids"]):
            point = row.geometry.interpolate(0.5, normalized=True)
            ax.annotate(str(int(row.id_link)), (point.x, point.y), xytext=(4, 4), textcoords="offset points")
    minx, miny, maxx, maxy = area.bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    ax.set_title(f"Example {case['example_id']} {case['state_id']} — ambiguous planning collar")
    ax.legend(loc="best")
    return fig, ax


def crop_attempt_table(case: dict[str, Any]) -> pd.DataFrame:
    plan = case.get("raster_plan")
    if not plan:
        raise FileNotFoundError("Saved raster-plan attempts are unavailable for this case.")
    return pd.DataFrame(
        [
            {
                "attempt": index,
                "halo_m": attempt.get("halo_m"),
                "full_raster_fraction": attempt.get("full_raster_fraction"),
                "upstream_cut_contained": attempt.get("cuts_contained_in_crop", [None, None])[0],
                "downstream_cut_contained": attempt.get("cuts_contained_in_crop", [None, None])[1],
                "reason_codes": ", ".join(attempt.get("reason_codes", [])),
            }
            for index, attempt in enumerate(plan.get("attempts", []))
        ]
    )


def _serialized_cut_geometry(record: dict[str, Any]) -> LineString:
    return LineString(
        [
            (record["first_endpoint_x"], record["first_endpoint_y"]),
            (record["anchor_x"], record["anchor_y"]),
            (record["second_endpoint_x"], record["second_endpoint_y"]),
        ]
    )


def plot_crop_containment_case(case: dict[str, Any]):
    plan = case.get("raster_plan")
    if not plan or not plan.get("attempts"):
        raise FileNotFoundError("Saved raster attempts are unavailable for this case.")
    attempts = plan["attempts"]
    selected = [(0, attempts[0])]
    if len(attempts) > 1:
        selected.append((len(attempts) - 1, attempts[-1]))
    fig, axes = plt.subplots(1, len(selected), figsize=(8 * len(selected), 7), squeeze=False)
    colors = {"upstream": "#1f77b4", "downstream": "#ff7f0e"}
    for ax, (index, attempt) in zip(axes[0], selected):
        bounds = attempt["bounds"]
        crop_geometry = box(*bounds)
        area = crop_geometry.buffer(max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.05)
        case["base_links"].loc[case["base_links"].geometry.intersects(area)].plot(
            ax=ax, color="0.75", linewidth=1.5, label="base links"
        )
        gpd.GeoSeries([crop_geometry], crs=case["base_links"].crs).boundary.plot(
            ax=ax, color="black", linewidth=2, label="crop bounds"
        )
        for cut_record in attempt.get("bank_cuts", []):
            geometry = _serialized_cut_geometry(cut_record)
            role = str(cut_record["role"])
            gpd.GeoSeries([geometry], crs=case["base_links"].crs).plot(
                ax=ax, color=colors[role], linewidth=3, label=f"{role} cut"
            )
        minx, miny, maxx, maxy = area.bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_aspect("equal")
        ax.set_title(
            f"attempt {index}: halo={attempt['halo_m']:.0f} m\n"
            f"contained={attempt['cuts_contained_in_crop']}"
        )
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="best")
    fig.suptitle(f"Example {case['example_id']} {case['state_id']} — cut containment")
    fig.tight_layout()
    return fig, axes


def _plotly_line_coordinates(geometry) -> tuple[list[float | None], list[float | None]]:
    """Flatten line or polygon boundaries into Plotly coordinates."""
    if geometry is None or geometry.is_empty:
        return [], []
    if geometry.geom_type == "Polygon":
        geometry = geometry.boundary
    if geometry.geom_type in {"MultiLineString", "GeometryCollection"}:
        xs: list[float | None] = []
        ys: list[float | None] = []
        for part in geometry.geoms:
            part_x, part_y = _plotly_line_coordinates(part)
            if part_x:
                xs.extend(part_x + [None])
                ys.extend(part_y + [None])
        return xs, ys
    if not hasattr(geometry, "coords"):
        return [], []
    coordinates = list(geometry.coords)
    return [float(value[0]) for value in coordinates], [float(value[1]) for value in coordinates]


def _add_plotly_lines(
    figure: go.Figure,
    frame: gpd.GeoDataFrame,
    *,
    name: str,
    color: str,
    width: float,
    dash: str = "solid",
    opacity: float = 1.0,
    id_column: str | None = None,
    row: int | None = None,
    col: int | None = None,
) -> None:
    if frame is None or frame.empty:
        return
    first = True
    for feature in frame.itertuples(index=False):
        xs, ys = _plotly_line_coordinates(feature.geometry)
        if not xs:
            continue
        identifier = getattr(feature, id_column) if id_column and hasattr(feature, id_column) else None
        label = f"{name} {identifier}" if identifier is not None else name
        trace = go.Scattergl(
            x=xs,
            y=ys,
            mode="lines",
            line={"color": color, "width": width, "dash": dash},
            opacity=opacity,
            name=name,
            legendgroup=name,
            showlegend=first,
            text=[label] * len(xs),
            hovertemplate="%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
        )
        if row is None:
            figure.add_trace(trace)
        else:
            figure.add_trace(trace, row=row, col=col)
        first = False


def _add_plotly_direction_arrows(
    figure: go.Figure,
    frame: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    *,
    color: str,
    position: float = 0.55,
    half_span: float = 0.055,
) -> None:
    """Add one data-coordinate arrow near the middle of each directed link."""
    if frame is None or frame.empty or nodes is None or nodes.empty:
        return
    node_geometry = nodes.set_index(nodes["id_node"].astype(int)).geometry
    for feature in frame.itertuples(index=False):
        geometry = feature.geometry
        if geometry is None or geometry.is_empty:
            continue
        if geometry.geom_type == "MultiLineString":
            geometry = max(geometry.geoms, key=lambda part: part.length)
        if geometry.geom_type != "LineString" or geometry.length <= 0:
            continue
        try:
            upstream, downstream = parse_id_nodes(feature.id_nodes)
        except (AttributeError, TypeError, ValueError):
            continue
        if upstream not in node_geometry.index or downstream not in node_geometry.index:
            continue
        start = Point(geometry.coords[0])
        end = Point(geometry.coords[-1])
        if (
            end.distance(node_geometry.loc[upstream])
            + start.distance(node_geometry.loc[downstream])
            < start.distance(node_geometry.loc[upstream])
            + end.distance(node_geometry.loc[downstream])
        ):
            geometry = LineString(list(geometry.coords)[::-1])
        tail = geometry.interpolate(max(0.0, position - half_span), normalized=True)
        head = geometry.interpolate(min(1.0, position + half_span), normalized=True)
        figure.add_annotation(
            x=float(head.x),
            y=float(head.y),
            ax=float(tail.x),
            ay=float(tail.y),
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.15,
            arrowwidth=2.2,
            arrowcolor=color,
            opacity=0.95,
        )


def _links_near_anchored_boundaries(
    case: dict[str, Any],
    frame: gpd.GeoDataFrame,
    *,
    radius_m: float = 1000.0,
    max_links: int = 60,
) -> gpd.GeoDataFrame:
    """Keep direction arrows legible by focusing them on scientific interfaces."""
    if frame is None or frame.empty:
        return frame
    contract = (case.get("planning") or {}).get("contract") or {}
    boundary_ids = {
        int(value)
        for key in (
            "anchored_upstream_boundary_node_ids",
            "anchored_downstream_boundary_node_ids",
        )
        for value in contract.get(key, [])
    }
    if not boundary_ids:
        return frame.head(int(max_links))
    targets = case["base_nodes"].loc[
        case["base_nodes"]["id_node"].astype(int).isin(boundary_ids)
    ].geometry.tolist()
    if not targets:
        return frame.head(int(max_links))
    distances = frame.geometry.apply(
        lambda geometry: min(float(geometry.distance(target)) for target in targets)
    )
    nearby = frame.loc[distances.le(float(radius_m))]
    if nearby.empty:
        nearby = frame.loc[distances.nsmallest(min(int(max_links), len(frame))).index]
    elif len(nearby) > int(max_links):
        nearby = nearby.loc[distances.loc[nearby.index].nsmallest(int(max_links)).index]
    return nearby


def _add_plotly_local_mask(
    figure: go.Figure,
    case: dict[str, Any],
    *,
    row: int | None = None,
    col: int | None = None,
) -> bool:
    path = case.get("local_mask_path")
    if path is None or not Path(path).exists():
        return False
    with rasterio.open(path) as source:
        bounds = source.bounds
        source_height, source_width = source.height, source.width
        source_water = source.read(1).astype(bool)
        stride = max(
            1,
            int(math.ceil(math.sqrt((source_height * source_width) / 400_000.0))),
            int(math.ceil(max(source_height, source_width) / 900.0)),
        )
        if stride == 1:
            water = source_water
        else:
            pad_height = (-source_height) % stride
            pad_width = (-source_width) % stride
            padded = np.pad(
                source_water,
                ((0, pad_height), (0, pad_width)),
                mode="constant",
                constant_values=False,
            )
            water = padded.reshape(
                padded.shape[0] // stride,
                stride,
                padded.shape[1] // stride,
                stride,
            ).max(axis=(1, 3))
        height, width = water.shape
    z = np.where(water, 1.0, np.nan)
    x = np.linspace(bounds.left, bounds.right, width, endpoint=False) + (
        bounds.right - bounds.left
    ) / (2.0 * width)
    y = np.linspace(bounds.top, bounds.bottom, height, endpoint=False) - (
        bounds.top - bounds.bottom
    ) / (2.0 * height)
    trace = go.Heatmap(
        x=x,
        y=y,
        z=z,
        colorscale=[[0.0, "rgba(76,120,168,0.48)"], [1.0, "rgba(76,120,168,0.48)"]],
        showscale=False,
        name="collapsed local water mask",
        hovertemplate="local water mask<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
    )
    if row is None:
        figure.add_trace(trace)
    else:
        figure.add_trace(trace, row=row, col=col)
    return True


def _finish_plotly_map(
    figure: go.Figure,
    *,
    title: str,
    bounds: np.ndarray | None = None,
) -> go.Figure:
    figure.update_layout(
        title=title,
        template="plotly_white",
        hovermode="closest",
        legend={"groupclick": "togglegroup"},
        height=760,
        margin={"l": 45, "r": 25, "t": 65, "b": 45},
    )
    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    if bounds is not None:
        pad = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.03
        figure.update_xaxes(range=[bounds[0] - pad, bounds[2] + pad])
        figure.update_yaxes(range=[bounds[1] - pad, bounds[3] + pad])
    return figure


def plotly_terminal_case(case: dict[str, Any]) -> go.Figure:
    """Interactive mask, raw-network, final-child, cut, and terminal audit."""
    links = case.get("local_links")
    cuts = case.get("cuts")
    if links is None or cuts is None:
        raise FileNotFoundError("Terminal plots require saved local links and cut geometries.")
    terminals = _terminal_nodes(case)
    crop = case.get("crop")
    mask_bounds = _local_mask_bounds(case)
    if crop is not None and not crop.empty:
        bounds = np.asarray(crop.total_bounds, dtype=float)
    elif mask_bounds is not None:
        bounds = mask_bounds
    else:
        bounds = np.asarray(links.total_bounds, dtype=float)
    extent = box(*bounds)
    figure = go.Figure()
    _add_plotly_local_mask(figure, case)
    base = case["base_links"].loc[case["base_links"].geometry.intersects(extent)]
    _add_plotly_lines(
        figure, base, name="base/reference network", color="#666666", width=2,
        dash="dash", opacity=0.8, id_column="id_link",
    )
    if crop is not None:
        _add_plotly_lines(figure, crop, name="crop boundary", color="#111111", width=2)
    _add_plotly_lines(
        figure, links, name="raw local RivGraph", color="#2ca02c", width=2.5,
        id_column="id_link",
    )
    _add_plotly_direction_arrows(
        figure,
        _links_near_anchored_boundaries(case, links),
        case["local_nodes"],
        color="#146b1d",
    )
    final_links = case.get("final_child_links")
    if final_links is not None and "lineage_type" in final_links.columns:
        changed = final_links.loc[
            final_links["lineage_type"].astype(str).ne("unchanged_1to1")
            & final_links.geometry.intersects(extent)
        ]
        _add_plotly_lines(
            figure, changed, name="final anchored changed link", color="#9467bd",
            width=4, id_column="id_link",
        )
        final_nodes = case.get("final_child_nodes")
        if final_nodes is not None:
            _add_plotly_direction_arrows(
                figure,
                _links_near_anchored_boundaries(case, changed),
                final_nodes,
                color="#5f3b8c",
            )
        if final_nodes is not None and not changed.empty:
            endpoint_ids: set[int] = set()
            for value in changed["id_nodes"]:
                endpoint_ids.update(int(node) for node in parse_id_nodes(value))
            endpoints = final_nodes.loc[final_nodes["id_node"].astype(int).isin(endpoint_ids)]
            figure.add_trace(go.Scattergl(
                x=endpoints.geometry.x,
                y=endpoints.geometry.y,
                mode="markers",
                marker={"color": "#9467bd", "size": 11, "line": {"color": "white", "width": 1}},
                name="final anchored endpoints",
                text=[f"final child node {int(value)}" for value in endpoints["id_node"]],
                hovertemplate="%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
            ))
    cut_colors = {"upstream": "#1f77b4", "downstream": "#ff7f0e"}
    for cut in cuts.itertuples(index=False):
        one = gpd.GeoDataFrame([{"geometry": cut.geometry}], crs=cuts.crs)
        _add_plotly_lines(
            figure, one, name=f"{cut.role} cut", color=cut_colors[str(cut.role)], width=4,
        )
    roles = np.where(terminals.get("is_inlet", False), "inlet", "outlet")
    colors = np.where(terminals.get("is_inlet", False), "#1f77b4", "#ff7f0e")
    symbols = np.where(terminals.get("is_inlet", False), "triangle-up", "square")
    figure.add_trace(go.Scattergl(
        x=terminals.geometry.x,
        y=terminals.geometry.y,
        mode="markers+text",
        marker={"color": colors, "symbol": symbols, "size": 12, "line": {"color": "#111111", "width": 1}},
        text=[f"local node {int(value)}" for value in terminals["id_node"]],
        textposition="top right",
        name="raw local terminals",
        customdata=np.column_stack([terminals["id_node"].astype(int), roles]),
        hovertemplate="local node %{customdata[0]}<br>role=%{customdata[1]}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
    ))
    return _finish_plotly_map(
        figure,
        title=f"Example {case['example_id']} {case['state_id']} — cut/terminal audit",
        bounds=bounds,
    )


def plotly_anchored_boundary_case(case: dict[str, Any]) -> go.Figure:
    """Add exact selected boundaries and nearest raw points to the common map."""
    figure = plotly_terminal_case(case)
    audit = anchored_boundary_audit(case)
    base_nodes = case["base_nodes"].set_index(
        case["base_nodes"]["id_node"].astype(int), drop=False
    )
    boundary_x: list[float] = []
    boundary_y: list[float] = []
    boundary_text: list[str] = []
    nearest_x: list[float] = []
    nearest_y: list[float] = []
    nearest_text: list[str] = []
    connector_x: list[float | None] = []
    connector_y: list[float | None] = []
    for row in audit.itertuples(index=False):
        target = base_nodes.loc[int(row.parent_boundary_node_id)].geometry
        boundary_x.append(float(target.x))
        boundary_y.append(float(target.y))
        boundary_text.append(
            f"{row.role} parent boundary node {int(row.parent_boundary_node_id)}"
            f"<br>reported failure={bool(row.is_reported_failure)}"
            f"<br>merge tolerance={row.merge_tolerance_m:.3f} m"
        )
        nearest_x.append(float(row.nearest_raw_point_x))
        nearest_y.append(float(row.nearest_raw_point_y))
        nearest_text.append(
            f"nearest raw point on link {int(row.nearest_raw_link_id)}"
            f"<br>link distance={row.nearest_raw_link_distance_m:.3f} m"
            f"<br>nearest node {int(row.nearest_raw_node_id)}: "
            f"{row.nearest_raw_node_distance_m:.3f} m"
            f"<br>beyond tolerance={row.distance_beyond_tolerance_m:.3f} m"
        )
        connector_x.extend([float(target.x), float(row.nearest_raw_point_x), None])
        connector_y.extend([float(target.y), float(row.nearest_raw_point_y), None])
    figure.add_trace(
        go.Scattergl(
            x=connector_x,
            y=connector_y,
            mode="lines",
            line={"color": "#d62728", "width": 2, "dash": "dot"},
            name="boundary-to-nearest-raw distance",
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scattergl(
            x=boundary_x,
            y=boundary_y,
            mode="markers+text",
            marker={
                "color": "#d62728",
                "size": 14,
                "symbol": "diamond",
                "line": {"color": "white", "width": 1},
            },
            name="exact anchored boundary",
            text=[
                f"{str(row.role)[0].upper()} parent {int(row.parent_boundary_node_id)}"
                for row in audit.itertuples(index=False)
            ],
            textposition="top right",
            customdata=np.asarray(boundary_text, dtype=object),
            hovertemplate="%{customdata}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
        )
    )
    contract = (case.get("planning") or {}).get("contract") or {}
    shared_boundary_ids = sorted(
        set(map(int, contract.get("anchored_upstream_boundary_node_ids", [])))
        & set(map(int, contract.get("anchored_downstream_boundary_node_ids", [])))
    )
    if shared_boundary_ids:
        shared = base_nodes.loc[shared_boundary_ids]
        figure.add_trace(
            go.Scattergl(
                x=shared.geometry.x,
                y=shared.geometry.y,
                mode="markers+text",
                marker={
                    "color": "#ffbf00",
                    "size": 20,
                    "symbol": "star",
                    "line": {"color": "#111111", "width": 1.5},
                },
                text=[f"SHARED U+D parent {value}" for value in shared_boundary_ids],
                textposition="bottom right",
                name="shared upstream/downstream boundary",
                hovertemplate="%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
            )
        )
    figure.add_trace(
        go.Scattergl(
            x=nearest_x,
            y=nearest_y,
            mode="markers",
            marker={
                "color": "#111111",
                "size": 11,
                "symbol": "circle-open",
                "line": {"width": 2},
            },
            name="nearest raw local point",
            text=nearest_text,
            hovertemplate="%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=(
            f"Example {case['example_id']} {case['state_id']} — "
            "anchored-boundary miss audit"
        )
    )
    return figure


def plotly_anchored_boundary_allocation_case(case: dict[str, Any]) -> go.Figure:
    """Show exact parent boundaries that selected the same raw local node."""
    figure = plotly_anchored_boundary_case(case)
    audit = anchored_boundary_allocation_audit(case)
    local_nodes = case["local_nodes"].set_index(
        case["local_nodes"]["id_node"].astype(int), drop=False
    )
    base_nodes = case["base_nodes"].set_index(
        case["base_nodes"]["id_node"].astype(int), drop=False
    )
    connector_x: list[float | None] = []
    connector_y: list[float | None] = []
    for row in audit.itertuples(index=False):
        parent = base_nodes.loc[int(row.parent_boundary_node_id)].geometry
        child = local_nodes.loc[int(row.nearest_raw_node_id)].geometry
        connector_x.extend([float(parent.x), float(child.x), None])
        connector_y.extend([float(parent.y), float(child.y), None])
    figure.add_trace(
        go.Scattergl(
            x=connector_x,
            y=connector_y,
            mode="lines",
            line={"color": "#e377c2", "width": 2, "dash": "dash"},
            name="parent-to-nearest-raw-node allocation",
            hoverinfo="skip",
        )
    )
    candidate_ids = sorted(set(audit["nearest_raw_node_id"].astype(int)))
    candidates = local_nodes.loc[candidate_ids]
    counts = audit.groupby("nearest_raw_node_id").size().to_dict()
    figure.add_trace(
        go.Scattergl(
            x=candidates.geometry.x,
            y=candidates.geometry.y,
            mode="markers+text",
            marker={
                "color": [
                    "#d62728" if counts.get(int(node_id), 0) > 1 else "#1f77b4"
                    for node_id in candidates["id_node"]
                ],
                "size": 13,
                "line": {"color": "white", "width": 1},
            },
            text=[
                f"raw node {int(node_id)} ({counts.get(int(node_id), 0)} boundaries)"
                for node_id in candidates["id_node"]
            ],
            textposition="top right",
            name="raw boundary-node candidates",
            hovertemplate="%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=(
            f"Example {case['example_id']} {case['state_id']} — "
            "anchored-boundary allocation collision"
        )
    )
    return figure


def plotly_contract_case(case: dict[str, Any]) -> go.Figure:
    """Plot planning-stage failures that stopped before a local raster existed."""
    contract = (case.get("planning") or {}).get("contract") or {}
    links = case["base_links"]
    nodes = case["base_nodes"]
    selected_ids = {
        int(value)
        for key in (
            "selected_link_ids",
            "core_link_ids",
            "anchored_replacement_link_ids",
        )
        for value in contract.get(key, [])
    }
    collar_ids = {int(value) for value in contract.get("collar_link_ids", [])}
    context_ids = {
        int(value)
        for key in (
            "computational_domain_link_ids",
            "local_context_link_ids",
            "upstream_connector_link_ids",
            "downstream_connector_link_ids",
        )
        for value in contract.get(key, [])
    }
    focus_ids = selected_ids | collar_ids | context_ids
    focus = links.loc[links["id_link"].astype(int).isin(focus_ids)]
    if focus.empty:
        focus = links
    span = max(
        float(focus.total_bounds[2] - focus.total_bounds[0]),
        float(focus.total_bounds[3] - focus.total_bounds[1]),
        1.0,
    )
    area = box(*focus.total_bounds).buffer(max(250.0, span * 0.08))
    visible = links.loc[links.geometry.intersects(area)]
    figure = go.Figure()
    _add_plotly_lines(
        figure, visible, name="base/reference network", color="#999999",
        width=2, dash="dash", id_column="id_link",
    )
    _add_plotly_lines(
        figure,
        visible.loc[visible["id_link"].astype(int).isin(context_ids)],
        name="planned local context", color="#1f77b4", width=3,
        id_column="id_link",
    )
    _add_plotly_lines(
        figure,
        visible.loc[visible["id_link"].astype(int).isin(collar_ids)],
        name="source/sink collar", color="#ff7f0e", width=5,
        id_column="id_link",
    )
    _add_plotly_lines(
        figure,
        visible.loc[visible["id_link"].astype(int).isin(selected_ids)],
        name="selected anchored replacement", color="#d62728", width=5,
        id_column="id_link",
    )
    boundary_specs = [
        ("upstream", value)
        for value in contract.get("anchored_upstream_boundary_node_ids", [])
    ] + [
        ("downstream", value)
        for value in contract.get("anchored_downstream_boundary_node_ids", [])
    ]
    boundary_ids = {int(value) for _, value in boundary_specs}
    boundaries = nodes.loc[nodes["id_node"].astype(int).isin(boundary_ids)]
    if not boundaries.empty:
        role_by_id = {int(value): role for role, value in boundary_specs}
        figure.add_trace(
            go.Scattergl(
                x=boundaries.geometry.x,
                y=boundaries.geometry.y,
                mode="markers+text",
                marker={"color": "#9467bd", "size": 14, "symbol": "diamond"},
                text=[
                    f"{role_by_id.get(int(value), 'boundary')} node {int(value)}"
                    for value in boundaries["id_node"]
                ],
                textposition="top right",
                name="exact anchored boundaries",
                hovertemplate="%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
            )
        )
    return _finish_plotly_map(
        figure,
        title=f"Example {case['example_id']} {case['state_id']} — planning contract",
        bounds=np.asarray(area.bounds, dtype=float),
    )


def plotly_failure_case(case: dict[str, Any]) -> go.Figure:
    """Render one current fallback with the common audit layers and type label."""
    summary = failure_mode_summary(case)
    failure_class = str(summary["failure_class"])
    subtype = str(summary["failure_subtype"])
    if case.get("local_links") is None:
        if (case.get("raster_plan") or {}).get("attempts"):
            figure = plotly_crop_containment_case(case)
        else:
            figure = plotly_contract_case(case)
    elif failure_class == "anchored_boundary_allocation_collision":
        figure = plotly_anchored_boundary_allocation_case(case)
    else:
        figure = plotly_anchored_boundary_case(case)

    if failure_class == "duplicate_parent_node_match":
        parent_ids = _duplicate_parent_ids(case)
        parent_nodes = case["base_nodes"].loc[
            case["base_nodes"]["id_node"].astype(int).isin(parent_ids)
        ]
        collision = node_collision_audit(case)
        local_nodes = case.get("local_nodes")
        if local_nodes is not None and not collision.empty:
            candidate_ids = set(collision["child_id_node"].astype(int))
            candidates = local_nodes.loc[
                local_nodes["id_node"].astype(int).isin(candidate_ids)
            ]
            figure.add_trace(
                go.Scattergl(
                    x=candidates.geometry.x,
                    y=candidates.geometry.y,
                    mode="markers+text",
                    marker={
                        "color": "#1f77b4",
                        "size": 12,
                        "line": {"color": "white", "width": 1},
                    },
                    text=[f"local node {int(value)}" for value in candidates["id_node"]],
                    textposition="top right",
                    name="competing local node match",
                    hovertemplate="%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
                )
            )
        if not parent_nodes.empty:
            figure.add_trace(
                go.Scattergl(
                    x=parent_nodes.geometry.x,
                    y=parent_nodes.geometry.y,
                    mode="markers+text",
                    marker={"color": "#d62728", "size": 17, "symbol": "star"},
                    text=[f"parent node {int(value)}" for value in parent_nodes["id_node"]],
                    textposition="bottom right",
                    name="duplicated parent-node target",
                    hovertemplate="%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
                )
            )

    figure.update_layout(
        title=(
            f"Example {case['example_id']} {case['state_id']} — "
            f"{summary['failure_title']}<br>"
            f"<sup>type: {failure_class} | subtype: {subtype}</sup>"
        )
    )
    return figure


def plotly_node_collision_case(case: dict[str, Any], *, radius_m: float = 250.0) -> go.Figure:
    local_links = case.get("local_links")
    local_nodes = case.get("local_nodes")
    if local_links is None or local_nodes is None:
        raise FileNotFoundError("Node-collision plots require saved local vectors.")
    parent_ids = _duplicate_parent_ids(case)
    parent_nodes = case["base_nodes"].loc[case["base_nodes"]["id_node"].astype(int).isin(parent_ids)]
    audit = node_collision_audit(case)
    candidates = local_nodes.loc[local_nodes["id_node"].astype(int).isin(set(audit["child_id_node"].astype(int)))]
    focus = gpd.GeoSeries(
        list(parent_nodes.geometry) + list(candidates.geometry), crs=parent_nodes.crs
    ).union_all().convex_hull.buffer(float(radius_m))
    bounds = np.asarray(focus.bounds, dtype=float)
    figure = go.Figure()
    _add_plotly_local_mask(figure, case)
    _add_plotly_lines(
        figure, case["base_links"].loc[case["base_links"].geometry.intersects(focus)],
        name="parent links", color="#999999", width=2, id_column="id_link",
    )
    _add_plotly_lines(
        figure, local_links.loc[local_links.geometry.intersects(focus)],
        name="local links", color="#2ca02c", width=3, id_column="id_link",
    )
    nearby = local_nodes.loc[local_nodes.geometry.intersects(focus)]
    figure.add_trace(go.Scattergl(
        x=nearby.geometry.x, y=nearby.geometry.y, mode="markers+text",
        marker={"color": "#1f77b4", "size": 10},
        text=nearby["id_node"].astype(int).astype(str), textposition="top right",
        name="local nodes",
        hovertemplate="local node %{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
    ))
    figure.add_trace(go.Scattergl(
        x=parent_nodes.geometry.x, y=parent_nodes.geometry.y, mode="markers",
        marker={"color": "red", "size": 16, "symbol": "star"},
        text=[f"parent node {int(value)}" for value in parent_nodes["id_node"]],
        name="duplicated parent match",
        hovertemplate="%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
    ))
    cuts = case.get("cuts")
    if cuts is not None:
        _add_plotly_lines(
            figure, cuts.loc[cuts.geometry.intersects(focus)], name="interface cuts",
            color="#17becf", width=4,
        )
    return _finish_plotly_map(
        figure,
        title=f"Example {case['example_id']} {case['state_id']} — child/parent node collision",
        bounds=bounds,
    )


def plotly_planning_interface_case(case: dict[str, Any], *, radius_m: float = 2500.0) -> go.Figure:
    audit = planning_interface_audit(case)
    node_id = audit["problem_node_id"]
    if node_id is None:
        raise ValueError("Could not parse the ambiguous interface node from the fallback reason.")
    nodes = case["base_nodes"]
    links = case["base_links"]
    node = nodes.loc[nodes["id_node"].astype(int).eq(node_id)].iloc[0]
    area = node.geometry.buffer(float(radius_m))
    visible = links.loc[links.geometry.intersects(area)]
    figure = go.Figure()
    _add_plotly_lines(figure, visible, name="base context", color="#aaaaaa", width=2, id_column="id_link")
    _add_plotly_lines(
        figure, visible.loc[visible["id_link"].astype(int).isin(audit["core_link_ids"])],
        name="selected unit links", color="crimson", width=4, id_column="id_link",
    )
    _add_plotly_lines(
        figure, visible.loc[visible["id_link"].astype(int).isin(audit["continuation_link_ids"])],
        name="competing external continuations", color="darkorange", width=5, id_column="id_link",
    )
    figure.add_trace(go.Scattergl(
        x=[node.geometry.x], y=[node.geometry.y], mode="markers",
        marker={"color": "#111111", "size": 14, "symbol": "x"},
        name=f"problem node {node_id}",
        hovertemplate=f"problem node {node_id}<br>x=%{{x:.2f}}<br>y=%{{y:.2f}}<extra></extra>",
    ))
    return _finish_plotly_map(
        figure,
        title=f"Example {case['example_id']} {case['state_id']} — ambiguous planning collar",
        bounds=np.asarray(area.bounds, dtype=float),
    )


def plotly_crop_containment_case(case: dict[str, Any]) -> go.Figure:
    plan = case.get("raster_plan")
    if not plan or not plan.get("attempts"):
        raise FileNotFoundError("Saved raster attempts are unavailable for this case.")
    attempts = plan["attempts"]
    selected = [(0, attempts[0])]
    if len(attempts) > 1:
        selected.append((len(attempts) - 1, attempts[-1]))
    titles = [
        f"attempt {index}: halo={attempt['halo_m']:.0f} m, contained={attempt['cuts_contained_in_crop']}"
        for index, attempt in selected
    ]
    figure = make_subplots(rows=1, cols=len(selected), subplot_titles=titles)
    cut_colors = {"upstream": "#1f77b4", "downstream": "#ff7f0e"}
    for column, (_, attempt) in enumerate(selected, start=1):
        bounds = np.asarray(attempt["bounds"], dtype=float)
        crop_geometry = box(*bounds)
        area = crop_geometry.buffer(max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.05)
        visible = case["base_links"].loc[case["base_links"].geometry.intersects(area)]
        _add_plotly_lines(
            figure, visible, name="base links", color="#999999", width=2,
            id_column="id_link", row=1, col=column,
        )
        crop_frame = gpd.GeoDataFrame([{"geometry": crop_geometry}], crs=case["base_links"].crs)
        _add_plotly_lines(
            figure, crop_frame, name="crop bounds", color="#111111", width=3,
            row=1, col=column,
        )
        for cut_record in attempt.get("bank_cuts", []):
            role = str(cut_record["role"])
            cut_frame = gpd.GeoDataFrame(
                [{"geometry": _serialized_cut_geometry(cut_record)}], crs=case["base_links"].crs
            )
            _add_plotly_lines(
                figure, cut_frame, name=f"{role} cut", color=cut_colors[role], width=4,
                row=1, col=column,
            )
        figure.update_xaxes(range=[area.bounds[0], area.bounds[2]], row=1, col=column)
        figure.update_yaxes(
            range=[area.bounds[1], area.bounds[3]], scaleanchor=f"x{column if column > 1 else ''}",
            scaleratio=1, row=1, col=column,
        )
    figure.update_layout(
        title=f"Example {case['example_id']} {case['state_id']} — cut containment",
        template="plotly_white",
        hovermode="closest",
        legend={"groupclick": "togglegroup"},
        height=650,
        margin={"l": 45, "r": 25, "t": 90, "b": 45},
    )
    return figure


def unit_overlap_audit(
    case: dict[str, Any],
    *,
    selected_unit_id: int,
    overlapping_unit_id: int,
) -> dict[str, pd.DataFrame]:
    """Describe shared links, interfaces, and raw/full contacts for two units."""
    membership_path = (
        Path(case["experiment_dir"])
        / "states"
        / "S000_base"
        / "variant"
        / "summary"
        / "unit_link_membership.csv"
    )
    membership = pd.read_csv(membership_path)
    membership["unit_id"] = pd.to_numeric(membership["unit_id"], errors="coerce").astype("Int64")
    membership["id_link"] = pd.to_numeric(membership["id_link"], errors="coerce").astype("Int64")
    selected_ids = set(
        membership.loc[membership["unit_id"].eq(int(selected_unit_id)), "id_link"]
        .dropna().astype(int)
    )
    overlap_ids = set(
        membership.loc[membership["unit_id"].eq(int(overlapping_unit_id)), "id_link"]
        .dropna().astype(int)
    )
    union_ids = selected_ids | overlap_ids
    members_by_link = (
        membership.loc[membership["id_link"].isin(union_ids)]
        .groupby("id_link")["unit_id"]
        .apply(lambda values: tuple(sorted(set(values.dropna().astype(int)))))
        .to_dict()
    )
    base_links = case["base_links"]
    unit_links = base_links.loc[base_links["id_link"].astype(int).isin(union_ids)].copy()
    unit_links["unit_memberships"] = unit_links["id_link"].astype(int).map(members_by_link)
    unit_links["overlap_role"] = [
        "shared"
        if int(link_id) in selected_ids & overlap_ids
        else "selected_only"
        if int(link_id) in selected_ids
        else "overlapping_only"
        for link_id in unit_links["id_link"]
    ]
    link_table = unit_links[
        ["id_link", "id_us_node", "id_ds_node", "overlap_role", "unit_memberships"]
    ].sort_values(["overlap_role", "id_link"]).reset_index(drop=True)

    selected_links = base_links.loc[
        base_links["id_link"].astype(int).isin(selected_ids)
    ]
    selected_nodes = set(selected_links["id_us_node"].astype(int)).union(
        selected_links["id_ds_node"].astype(int)
    )
    external = base_links.loc[
        ~base_links["id_link"].astype(int).isin(selected_ids)
        & (
            base_links["id_us_node"].astype(int).isin(selected_nodes)
            | base_links["id_ds_node"].astype(int).isin(selected_nodes)
        )
    ].copy()
    external["belongs_to_overlapping_unit"] = external["id_link"].astype(int).isin(overlap_ids)
    external["selected_contact_node_id"] = [
        int(row.id_us_node)
        if int(row.id_us_node) in selected_nodes
        else int(row.id_ds_node)
        for row in external.itertuples(index=False)
    ]
    external_table = external[
        [
            "id_link",
            "id_us_node",
            "id_ds_node",
            "selected_contact_node_id",
            "belongs_to_overlapping_unit",
        ]
    ].sort_values("id_link").reset_index(drop=True)
    return {
        "link_membership": link_table,
        "external_contacts": external_table,
        "anchored_boundaries": anchored_boundary_audit(case),
    }


def plotly_unit_overlap_case(
    case: dict[str, Any],
    *,
    selected_unit_id: int,
    overlapping_unit_id: int,
) -> go.Figure:
    """Compare base overlap topology with local and full-raster collapse products."""
    audit = unit_overlap_audit(
        case,
        selected_unit_id=selected_unit_id,
        overlapping_unit_id=overlapping_unit_id,
    )
    link_table = audit["link_membership"]
    base_links = case["base_links"]
    base_nodes = case["base_nodes"]
    role_by_link = link_table.set_index("id_link")["overlap_role"].to_dict()
    union_ids = set(int(value) for value in link_table["id_link"])
    unit_links = base_links.loc[base_links["id_link"].astype(int).isin(union_ids)]
    span = max(
        float(unit_links.total_bounds[2] - unit_links.total_bounds[0]),
        float(unit_links.total_bounds[3] - unit_links.total_bounds[1]),
        1.0,
    )
    area = unit_links.geometry.union_all().convex_hull.buffer(max(600.0, span * 0.35))
    bounds = np.asarray(area.bounds, dtype=float)
    visible_base = base_links.loc[base_links.geometry.intersects(area)]
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Base topology: units {selected_unit_id} and {overlapping_unit_id}",
            "Collapsed mask, raw local RivGraph, and full-raster fallback",
        ),
        horizontal_spacing=0.06,
    )
    _add_plotly_lines(
        figure, visible_base, name="base/reference context", color="#888888",
        width=1.5, dash="dash", opacity=0.65, id_column="id_link", row=1, col=1,
    )
    role_styles = {
        "selected_only": (f"unit {selected_unit_id} only", "#d62728", 5.0),
        "overlapping_only": (f"unit {overlapping_unit_id} only", "#ff7f0e", 5.0),
        "shared": ("shared unit link", "#7b2cbf", 7.0),
    }
    for role, (name, color, width) in role_styles.items():
        ids = {int(link_id) for link_id, value in role_by_link.items() if value == role}
        _add_plotly_lines(
            figure,
            unit_links.loc[unit_links["id_link"].astype(int).isin(ids)],
            name=name,
            color=color,
            width=width,
            id_column="id_link",
            row=1,
            col=1,
        )

    _add_plotly_local_mask(figure, case, row=1, col=2)
    _add_plotly_lines(
        figure, visible_base, name="base/reference network", color="#777777",
        width=1.5, dash="dash", opacity=0.65, id_column="id_link", row=1, col=2,
    )
    local_links = case.get("local_links")
    if local_links is not None:
        _add_plotly_lines(
            figure,
            local_links.loc[local_links.geometry.intersects(area)],
            name="raw local RivGraph",
            color="#2ca02c",
            width=3.0,
            id_column="id_link",
            row=1,
            col=2,
        )
    final_links = case.get("final_child_links")
    if final_links is not None:
        if "lineage_type" in final_links.columns:
            changed = final_links.loc[
                final_links["lineage_type"].astype(str).ne("unchanged_1to1")
                & final_links.geometry.intersects(area)
            ]
        else:
            changed = final_links.loc[final_links.geometry.intersects(area)]
        _add_plotly_lines(
            figure,
            changed,
            name="full-raster changed result",
            color="#9467bd",
            width=4.5,
            id_column="id_link",
            row=1,
            col=2,
        )
    edit_path = Path(case["attempt_dir"]) / "variant" / "mask" / "collapse_edit_geometries.gpkg"
    if edit_path.exists():
        edits = gpd.read_file(edit_path)
        _add_plotly_lines(
            figure,
            edits,
            name="selected collapse footprint",
            color="#e41a1c",
            width=2.5,
            dash="dot",
            row=1,
            col=2,
        )

    involved_ids = set(unit_links["id_us_node"].astype(int)).union(
        unit_links["id_ds_node"].astype(int)
    )
    involved_nodes = base_nodes.loc[base_nodes["id_node"].astype(int).isin(involved_ids)]
    contract = (case.get("planning") or {}).get("contract") or {}
    upstream = {int(value) for value in contract.get("anchored_upstream_boundary_node_ids", [])}
    downstream = {int(value) for value in contract.get("anchored_downstream_boundary_node_ids", [])}
    boundary_ids = upstream | downstream
    for column in (1, 2):
        figure.add_trace(
            go.Scattergl(
                x=involved_nodes.geometry.x,
                y=involved_nodes.geometry.y,
                mode="markers+text",
                marker={
                    "color": [
                        "#1f77b4" if int(value) in upstream
                        else "#ff7f0e" if int(value) in downstream
                        else "#222222"
                        for value in involved_nodes["id_node"]
                    ],
                    "size": [13 if int(value) in boundary_ids else 8 for value in involved_nodes["id_node"]],
                    "symbol": ["diamond" if int(value) in boundary_ids else "circle" for value in involved_nodes["id_node"]],
                    "line": {"color": "white", "width": 1},
                },
                text=[f"parent node {int(value)}" for value in involved_nodes["id_node"]],
                textposition="top right",
                name="unit nodes and exact interfaces" if column == 1 else "exact interface reference",
                hovertemplate="%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=column,
        )

    boundary_audit = audit["anchored_boundaries"]
    base_by_id = base_nodes.set_index(base_nodes["id_node"].astype(int), drop=False)
    connector_x: list[float | None] = []
    connector_y: list[float | None] = []
    connector_text: list[str | None] = []
    for row in boundary_audit.itertuples(index=False):
        target = base_by_id.loc[int(row.parent_boundary_node_id)].geometry
        connector_x.extend([float(target.x), float(row.nearest_raw_point_x), None])
        connector_y.extend([float(target.y), float(row.nearest_raw_point_y), None])
        connector_text.extend([
            f"parent node {int(row.parent_boundary_node_id)} to raw local: "
            f"{float(row.minimum_raw_distance_m):.1f} m",
            f"nearest raw point for parent node {int(row.parent_boundary_node_id)}",
            None,
        ])
    figure.add_trace(
        go.Scattergl(
            x=connector_x,
            y=connector_y,
            mode="lines+markers",
            line={"color": "#d62728", "width": 2, "dash": "dot"},
            marker={"size": 5, "color": "#d62728"},
            text=connector_text,
            name="exact-interface to nearest raw distance",
            hovertemplate="%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    for column in (1, 2):
        figure.update_xaxes(range=[bounds[0], bounds[2]], row=1, col=column)
        figure.update_yaxes(
            range=[bounds[1], bounds[3]],
            scaleanchor="x" if column == 1 else "x2",
            scaleratio=1,
            row=1,
            col=column,
        )
    figure.update_layout(
        title=(
            f"Example {case['example_id']} {case['state_id']} — overlap audit for "
            f"units {selected_unit_id} and {overlapping_unit_id}"
        ),
        template="plotly_white",
        hovermode="closest",
        legend={"groupclick": "togglegroup"},
        height=780,
        width=1450,
        margin={"l": 40, "r": 25, "t": 90, "b": 45},
    )
    return figure


def plotly_unit_overlap_interface_case(
    case: dict[str, Any],
    *,
    selected_unit_id: int,
    overlapping_unit_id: int,
    boundary_node_id: int,
    radius_m: float = 900.0,
) -> go.Figure:
    """Zoom into one shared interface and compare raw and full-raster contacts."""
    audit = unit_overlap_audit(
        case,
        selected_unit_id=selected_unit_id,
        overlapping_unit_id=overlapping_unit_id,
    )
    boundary = audit["anchored_boundaries"].loc[
        audit["anchored_boundaries"]["parent_boundary_node_id"]
        .astype(int).eq(int(boundary_node_id))
    ]
    if len(boundary) != 1:
        raise ValueError(
            f"Expected one anchored-boundary audit row for node {boundary_node_id}; "
            f"found {len(boundary)}."
        )
    boundary_row = boundary.iloc[0]
    base_nodes = case["base_nodes"].set_index(
        case["base_nodes"]["id_node"].astype(int), drop=False
    )
    target = base_nodes.loc[int(boundary_node_id)].geometry
    area = target.buffer(float(radius_m))
    bounds = np.asarray(area.bounds, dtype=float)
    figure = go.Figure()
    _add_plotly_local_mask(figure, case)
    visible_base = case["base_links"].loc[case["base_links"].geometry.intersects(area)]
    _add_plotly_lines(
        figure, visible_base, name="base/reference context", color="#777777",
        width=1.5, dash="dash", opacity=0.65, id_column="id_link",
    )
    link_table = audit["link_membership"]
    styles = {
        "selected_only": (f"unit {selected_unit_id} only", "#d62728", 5.0),
        "overlapping_only": (f"unit {overlapping_unit_id} only", "#ff7f0e", 5.0),
        "shared": ("shared unit link", "#7b2cbf", 7.0),
    }
    for role, (name, color, width) in styles.items():
        ids = set(
            link_table.loc[link_table["overlap_role"].eq(role), "id_link"].astype(int)
        )
        _add_plotly_lines(
            figure,
            visible_base.loc[visible_base["id_link"].astype(int).isin(ids)],
            name=name,
            color=color,
            width=width,
            id_column="id_link",
        )
    local_links = case.get("local_links")
    if local_links is not None:
        _add_plotly_lines(
            figure,
            local_links.loc[local_links.geometry.intersects(area)],
            name="raw local RivGraph",
            color="#2ca02c",
            width=3.0,
            id_column="id_link",
        )
    final_links = case.get("final_child_links")
    nearest_final_point = None
    if final_links is not None:
        visible_final = final_links.loc[final_links.geometry.intersects(area)]
        if "lineage_type" in visible_final.columns:
            visible_final = visible_final.loc[
                visible_final["lineage_type"].astype(str).ne("unchanged_1to1")
            ]
        _add_plotly_lines(
            figure,
            visible_final,
            name="full-raster changed result",
            color="#9467bd",
            width=4.5,
            id_column="id_link",
        )
        if not final_links.empty:
            final_distances = final_links.geometry.distance(target)
            nearest = final_links.loc[final_distances.idxmin()].geometry
            nearest_final_point = nearest.interpolate(float(nearest.project(target)))
    raw_point = Point(
        float(boundary_row["nearest_raw_point_x"]),
        float(boundary_row["nearest_raw_point_y"]),
    )
    connector_x = [float(target.x), float(raw_point.x), None]
    connector_y = [float(target.y), float(raw_point.y), None]
    connector_text = [
        f"exact node {boundary_node_id}",
        f"nearest raw point: {float(boundary_row['minimum_raw_distance_m']):.1f} m",
        None,
    ]
    if nearest_final_point is not None:
        connector_x.extend([float(target.x), float(nearest_final_point.x), None])
        connector_y.extend([float(target.y), float(nearest_final_point.y), None])
        connector_text.extend([
            f"exact node {boundary_node_id}",
            f"nearest full-raster point: "
            f"{float(boundary_row['final_child_nearest_link_distance_m']):.1f} m",
            None,
        ])
    figure.add_trace(
        go.Scattergl(
            x=connector_x,
            y=connector_y,
            mode="lines+markers",
            line={"color": "#d62728", "width": 2.5, "dash": "dot"},
            marker={"size": 7, "color": "#d62728"},
            text=connector_text,
            name="interface-contact distances",
            hovertemplate="%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scattergl(
            x=[float(target.x)],
            y=[float(target.y)],
            mode="markers+text",
            marker={
                "color": "#111111", "size": 16, "symbol": "diamond",
                "line": {"color": "white", "width": 1},
            },
            text=[f"exact shared interface node {boundary_node_id}"],
            textposition="top right",
            name="exact shared interface",
            hovertemplate="%{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
        )
    )
    return _finish_plotly_map(
        figure,
        title=(
            f"Example {case['example_id']} {case['state_id']} — detailed contact "
            f"audit at parent node {boundary_node_id}"
        ),
        bounds=bounds,
    )
