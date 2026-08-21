"""Compare selected local-patch child states with full-raster references.

The comparison is deliberately state- and artifact-level.  It verifies that
the local run did not fall back, validates both directed networks, matches the
changed link geometries, and compares the width, length, slope, hierarchy, and
equivalent-resistance products that downstream analyses consume.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from hierarchy_level_definition.graph_building.directed_network_checks import (
    build_directed_graph,
    validate_single_inlet_single_outlet,
)


DEFAULT_FIELDS = ("len", "len_adj", "wid_adj_wet", "wid_adj_routing", "slope")


def _cycle_rank(links: gpd.GeoDataFrame) -> int:
    graph = nx.MultiGraph()
    for row in links.itertuples(index=False):
        graph.add_edge(int(row.id_us_node), int(row.id_ds_node), key=int(row.id_link))
    if graph.number_of_edges() == 0:
        return 0
    return int(
        graph.number_of_edges()
        - graph.number_of_nodes()
        + nx.number_connected_components(graph)
    )


def _one_path(directory: Path, pattern: str) -> Path:
    paths = sorted(
        path for path in directory.glob(pattern) if not path.name.startswith("._")
    )
    if len(paths) != 1:
        raise FileNotFoundError(
            f"Expected one {pattern!r} below {directory}, found {len(paths)}."
        )
    return paths[0]


def _state_for_unit(experiment: Path, unit_id: int) -> Path:
    states = sorted((experiment / "states").glob(f"S*_unit_{int(unit_id)}"))
    if len(states) != 1:
        raise FileNotFoundError(
            f"Expected one state for unit {unit_id} in {experiment}, found {len(states)}."
        )
    return states[0]


def _load_state(state: Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, Any]]:
    directed = state / "variant" / "directed"
    widths = state / "variant" / "widths"
    links = gpd.read_file(_one_path(widths, "*links_with_width_families.gpkg"))
    nodes = gpd.read_file(_one_path(directed, "*directed_nodes.gpkg"))
    manifest_path = state / "variant" / "summary" / "variant_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    return links, nodes, manifest


def _load_base_links(experiment: Path) -> gpd.GeoDataFrame:
    return gpd.read_file(
        _one_path(
            experiment / "states" / "S000_base" / "variant" / "widths",
            "*links_with_width_families.gpkg",
        )
    )


def _local_product_contract(manifest: dict[str, Any]) -> tuple[set[int], dict[int, int]]:
    summary = (
        manifest.get("local_patch", {})
        .get("products", {})
        .get("summary", {})
    )
    width = summary.get("width_product_inheritance", {})
    measured = {
        int(value)
        for value in summary.get("recalculated_link_ids", ())
    }
    parent_map = {
        int(child_id): int(parent_id)
        for child_id, parent_id in dict(
            width.get("parent_link_by_child_link", {})
        ).items()
        if int(child_id) not in measured
    }
    if not parent_map:
        raise ValueError("Local variant manifest has no unchanged-link inheritance map.")
    return measured, parent_map


def _full_changed_ids(state: Path) -> set[int]:
    lineage_path = state / "variant" / "matching" / "link_lineage.csv"
    if not lineage_path.exists():
        raise FileNotFoundError(f"Full-raster lineage is missing: {lineage_path}")
    lineage = pd.read_csv(lineage_path)
    return set(
        pd.to_numeric(
            lineage.loc[
                ~lineage["lineage_type"].astype(str).eq("unchanged_1to1"),
                "id_link",
            ],
            errors="coerce",
        ).dropna().astype(int)
    )


def _relative_difference(first: float, second: float) -> float:
    scale = max(abs(float(first)), abs(float(second)), 1e-12)
    return abs(float(first) - float(second)) / scale


def _exact_common_ids(
    local: gpd.GeoDataFrame,
    full: gpd.GeoDataFrame,
) -> set[int]:
    local_by_id = local.set_index(local["id_link"].astype(int), drop=False)
    full_by_id = full.set_index(full["id_link"].astype(int), drop=False)
    exact: set[int] = set()
    for link_id in set(local_by_id.index).intersection(full_by_id.index):
        local_row = local_by_id.loc[int(link_id)]
        full_row = full_by_id.loc[int(link_id)]
        if (
            int(local_row["id_us_node"]) == int(full_row["id_us_node"])
            and int(local_row["id_ds_node"]) == int(full_row["id_ds_node"])
            and float(local_row.geometry.hausdorff_distance(full_row.geometry)) <= 1e-7
        ):
            exact.add(int(link_id))
    return exact


def _match_changed_links(
    local: gpd.GeoDataFrame,
    full: gpd.GeoDataFrame,
    exact_ids: set[int],
) -> tuple[list[tuple[int, int, float]], int, int]:
    local_changed = local.loc[~local["id_link"].astype(int).isin(exact_ids)].copy()
    full_changed = full.loc[~full["id_link"].astype(int).isin(exact_ids)].copy()
    if local_changed.empty or full_changed.empty:
        return [], len(local_changed), len(full_changed)
    matrix = np.asarray(
        [
            [float(left.hausdorff_distance(right)) for right in full_changed.geometry]
            for left in local_changed.geometry
        ],
        dtype=float,
    )
    rows, columns = linear_sum_assignment(matrix)
    matches = [
        (
            int(local_changed.iloc[row]["id_link"]),
            int(full_changed.iloc[column]["id_link"]),
            float(matrix[row, column]),
        )
        for row, column in zip(rows, columns)
    ]
    return matches, len(local_changed), len(full_changed)


def compare_unit(
    local_experiment: Path,
    full_experiment: Path,
    unit_id: int,
    *,
    geometry_tolerance_m: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    local_state = _state_for_unit(local_experiment, unit_id)
    full_state = _state_for_unit(full_experiment, unit_id)
    local_links, local_nodes, local_manifest = _load_state(local_state)
    full_links, full_nodes, _ = _load_state(full_state)
    parent_links = _load_base_links(local_experiment)
    if local_links.crs != full_links.crs:
        full_links = full_links.to_crs(local_links.crs)
        full_nodes = full_nodes.to_crs(local_nodes.crs)

    local_validation = validate_single_inlet_single_outlet(
        build_directed_graph(local_links, local_nodes)
    )
    full_validation = validate_single_inlet_single_outlet(
        build_directed_graph(full_links, full_nodes)
    )
    execution_mode = str(local_manifest.get("execution_mode_used", ""))
    no_local_fallback = execution_mode == "local_patch"

    measured_ids, inherited_parent_map = _local_product_contract(local_manifest)
    local_changed = local_links.loc[local_links["id_link"].astype(int).isin(measured_ids)]
    full_changed_ids = _full_changed_ids(full_state)
    full_changed = full_links.loc[full_links["id_link"].astype(int).isin(full_changed_ids)]
    matches, local_changed_count, full_changed_count = _match_changed_links(
        local_changed, full_changed, set()
    )
    max_hausdorff = max((value for _, _, value in matches), default=0.0)
    changed_counts_equal = local_changed_count == full_changed_count
    local_cycle_rank = _cycle_rank(local_links)
    full_cycle_rank = _cycle_rank(full_links)
    cycle_ranks_equal = local_cycle_rank == full_cycle_rank

    local_by_id = local_links.set_index(local_links["id_link"].astype(int), drop=False)
    full_by_id = full_links.set_index(full_links["id_link"].astype(int), drop=False)
    parent_by_id = parent_links.set_index(parent_links["id_link"].astype(int), drop=False)
    inherited_geometry_differences = []
    inherited_field_differences = []
    for child_id, parent_id in inherited_parent_map.items():
        inherited_geometry_differences.append(
            float(local_by_id.loc[child_id].geometry.hausdorff_distance(parent_by_id.loc[parent_id].geometry))
        )
        for field in DEFAULT_FIELDS:
            if field not in local_links.columns or field not in parent_links.columns:
                continue
            child_value = pd.to_numeric(pd.Series([local_by_id.loc[child_id, field]]), errors="coerce").iloc[0]
            parent_value = pd.to_numeric(pd.Series([parent_by_id.loc[parent_id, field]]), errors="coerce").iloc[0]
            if pd.notna(child_value) and pd.notna(parent_value):
                inherited_field_differences.append(_relative_difference(child_value, parent_value))
    max_inherited_geometry_difference = max(inherited_geometry_differences, default=0.0)
    max_inherited_field_difference = max(inherited_field_differences, default=0.0)
    field_differences: dict[str, float] = {}
    for field in DEFAULT_FIELDS:
        differences = []
        if field in local_links.columns and field in full_links.columns:
            for local_id, full_id, _ in matches:
                local_value = pd.to_numeric(
                    pd.Series([local_by_id.loc[local_id, field]]), errors="coerce"
                ).iloc[0]
                full_value = pd.to_numeric(
                    pd.Series([full_by_id.loc[full_id, field]]), errors="coerce"
                ).iloc[0]
                if pd.notna(local_value) and pd.notna(full_value):
                    differences.append(_relative_difference(local_value, full_value))
        field_differences[field] = max(differences, default=0.0)

    hierarchy_files = (
        "unit_summary.csv",
        "path_metrics.csv",
        "bubble_edge_resistance.csv",
    )
    hierarchy_counts_equal = True
    hierarchy_count_details: dict[str, list[int]] = {}
    for filename in hierarchy_files:
        local_path = local_state / "hierarchy" / filename
        full_path = full_state / "hierarchy" / filename
        if not local_path.exists() or not full_path.exists():
            hierarchy_counts_equal = False
            hierarchy_count_details[filename] = [-1, -1]
            continue
        counts = [len(pd.read_csv(local_path)), len(pd.read_csv(full_path))]
        hierarchy_count_details[filename] = counts
        hierarchy_counts_equal &= counts[0] == counts[1]

    equivalent_fields = ("K_after", "G_after", "Lambda_K", "Lambda_G")
    equivalent_differences: dict[str, float] = {}
    local_equivalent = (
        local_state / "equivalent_resistance" / "bubble_equivalent_state_comparison.csv"
    )
    full_equivalent = (
        full_state / "equivalent_resistance" / "bubble_equivalent_state_comparison.csv"
    )
    equivalent_products_available = local_equivalent.exists() and full_equivalent.exists()
    if equivalent_products_available:
        local_equivalent_frame = pd.read_csv(local_equivalent)
        full_equivalent_frame = pd.read_csv(full_equivalent)
        equivalent_products_available = bool(
            not local_equivalent_frame.empty and not full_equivalent_frame.empty
        )
        if equivalent_products_available:
            for field in equivalent_fields:
                if (
                    field in local_equivalent_frame.columns
                    and field in full_equivalent_frame.columns
                ):
                    local_value = pd.to_numeric(
                        local_equivalent_frame[field], errors="coerce"
                    ).dropna()
                    full_value = pd.to_numeric(
                        full_equivalent_frame[field], errors="coerce"
                    ).dropna()
                    if not local_value.empty and not full_value.empty:
                        equivalent_differences[field] = _relative_difference(
                            float(local_value.iloc[-1]), float(full_value.iloc[-1])
                        )
    equivalent_within_tolerance = bool(
        equivalent_products_available
        and equivalent_differences
        and all(
            value <= float(relative_tolerance)
            for value in equivalent_differences.values()
        )
    )

    fields_within_tolerance = all(
        value <= float(relative_tolerance) for value in field_differences.values()
    )
    passed = bool(
        no_local_fallback
        and local_validation.is_valid
        and full_validation.is_valid
        and cycle_ranks_equal
        and max_inherited_geometry_difference <= 1e-7
        and max_inherited_field_difference <= 1e-10
        and max_hausdorff <= float(geometry_tolerance_m)
        and fields_within_tolerance
        and hierarchy_counts_equal
        and equivalent_within_tolerance
    )
    return {
        "unit_id": int(unit_id),
        "passed": passed,
        "local_state_id": local_state.name,
        "full_state_id": full_state.name,
        "local_execution_mode_used": execution_mode,
        "local_used_without_fallback": no_local_fallback,
        "local_network_valid": bool(local_validation.is_valid),
        "full_network_valid": bool(full_validation.is_valid),
        "local_link_count": int(len(local_links)),
        "full_link_count": int(len(full_links)),
        "inherited_link_count": int(len(inherited_parent_map)),
        "max_inherited_geometry_hausdorff_m": float(max_inherited_geometry_difference),
        "max_inherited_field_relative_difference": float(max_inherited_field_difference),
        "local_changed_link_count": int(local_changed_count),
        "full_changed_link_count": int(full_changed_count),
        "changed_link_counts_equal": bool(changed_counts_equal),
        "local_cycle_rank": int(local_cycle_rank),
        "full_cycle_rank": int(full_cycle_rank),
        "cycle_ranks_equal": bool(cycle_ranks_equal),
        "max_changed_geometry_hausdorff_m": float(max_hausdorff),
        "geometry_within_tolerance": bool(
            max_hausdorff <= float(geometry_tolerance_m)
        ),
        **{
            f"max_changed_{field}_relative_difference": float(value)
            for field, value in field_differences.items()
        },
        "changed_fields_within_tolerance": bool(fields_within_tolerance),
        "hierarchy_counts_equal": bool(hierarchy_counts_equal),
        "hierarchy_count_details": json.dumps(hierarchy_count_details, sort_keys=True),
        "equivalent_products_available": bool(equivalent_products_available),
        **{
            f"equivalent_{field}_relative_difference": float(value)
            for field, value in equivalent_differences.items()
        },
        "equivalent_resistance_within_tolerance": equivalent_within_tolerance,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-experiment", required=True)
    parser.add_argument("--full-experiment", required=True)
    parser.add_argument("--unit-id", action="append", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--geometry-tolerance-m", type=float, default=25.0)
    parser.add_argument("--relative-tolerance", type=float, default=0.02)
    parser.add_argument("--fail-on-difference", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    local_experiment = Path(args.local_experiment).resolve()
    full_experiment = Path(args.full_experiment).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        compare_unit(
            local_experiment,
            full_experiment,
            unit_id,
            geometry_tolerance_m=args.geometry_tolerance_m,
            relative_tolerance=args.relative_tolerance,
        )
        for unit_id in args.unit_id
    ]
    frame = pd.DataFrame(rows)
    csv_path = output_dir / "local_vs_full_unit_comparison.csv"
    json_path = output_dir / "local_vs_full_unit_comparison_summary.json"
    frame.to_csv(csv_path, index=False)
    summary = {
        "local_experiment": str(local_experiment),
        "full_experiment": str(full_experiment),
        "geometry_tolerance_m": float(args.geometry_tolerance_m),
        "relative_tolerance": float(args.relative_tolerance),
        "unit_count": int(len(frame)),
        "passed_unit_count": int(frame["passed"].sum()),
        "all_passed": bool(frame["passed"].all()),
        "comparison_csv": str(csv_path.resolve()),
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(frame.to_string(index=False))
    print(f"Wrote comparison to {csv_path}")
    if args.fail_on_difference and not summary["all_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
