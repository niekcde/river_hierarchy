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
from pathlib import Path
import re
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, box

from hierarchy_level_definition.graph_building.directed_network_checks import parse_id_nodes


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
    return case


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


def plot_terminal_case(case: dict[str, Any], *, pad_fraction: float = 0.05):
    """Plot cut lines, raw local links, and role-labelled terminals."""
    links = case.get("local_links")
    cuts = case.get("cuts")
    if links is None or cuts is None:
        raise FileNotFoundError("Terminal plots require saved local links and cut geometries.")
    terminals = _terminal_nodes(case)
    fig, ax = plt.subplots(figsize=(10, 8))
    local_extent = box(*links.total_bounds)
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
    crop = case.get("crop")
    if crop is not None:
        crop.boundary.plot(ax=ax, color="black", linewidth=1.4, zorder=2, label="crop boundary")
    links.plot(ax=ax, color="#2ca02c", linewidth=2.2, zorder=3, label="raw local RivGraph")
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
    bounds = links.total_bounds
    pad = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * float(pad_fraction)
    ax.set_xlim(bounds[0] - pad, bounds[2] + pad)
    ax.set_ylim(bounds[1] - pad, bounds[3] + pad)
    ax.set_aspect("equal")
    ax.set_title(f"Example {case['example_id']} {case['state_id']} — cut/terminal audit")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
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
    ax.legend(loc="best")
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
