"""Paired validation of full-raster and local-patch island observations.

The island workflow deliberately has two execution backends for a non-aliased
observation.  This tool checks that the local backend reproduced the frozen
full-raster edit and records where its final network or hydraulic products
differ.  It is intentionally a *paired report*, rather than a replacement for
either run's own topology and routing validation.
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
import rasterio
from scipy.optimize import linear_sum_assignment

from hierarchy_level_definition.graph_building.directed_network_checks import (
    build_directed_graph,
    validate_single_inlet_single_outlet,
)


DEFAULT_MEASUREMENT_FIELDS = (
    "len",
    "len_adj",
    "wid_adj_total",
    "wid_adj_wet",
    "wid_adj_dry",
    "wid_adj_routing",
)

# ``len`` and ``len_adj`` are retained as paired diagnostics, but they are not
# width products.  In particular, a local vector crop can retain equivalent
# topology and two-bank measurements while its re-traced centreline has a
# slightly different adjusted length.  Keep that distinction explicit in the
# validation gates.
FINAL_WIDTH_FIELDS = (
    "wid_adj_total",
    "wid_adj_wet",
    "wid_adj_dry",
    "wid_adj_routing",
)
LENGTH_FIELDS = ("len", "len_adj")


def _cycle_rank(links: gpd.GeoDataFrame) -> int:
    graph = nx.MultiGraph()
    for row in links.itertuples(index=False):
        graph.add_edge(int(row.id_us_node), int(row.id_ds_node), key=int(row.id_link))
    return int(
        graph.number_of_edges()
        - graph.number_of_nodes()
        + nx.number_connected_components(graph)
    ) if graph.number_of_edges() else 0


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_difference(first: float, second: float) -> float:
    return abs(float(first) - float(second)) / max(
        abs(float(first)), abs(float(second)), 1e-12
    )


def _match_links(
    full: gpd.GeoDataFrame,
    local: gpd.GeoDataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(full) != len(local):
        return np.asarray([], dtype=int), np.asarray([], dtype=int), np.empty((0, 0))
    distances = np.asarray(
        [
            [float(full_geometry.hausdorff_distance(local_geometry)) for local_geometry in local.geometry]
            for full_geometry in full.geometry
        ],
        dtype=float,
    )
    full_rows, local_rows = linear_sum_assignment(distances)
    return full_rows, local_rows, distances


def _lineage_signature(path: Path) -> dict[str, int]:
    lineage = pd.read_csv(path)
    return {
        str(key): int(value)
        for key, value in lineage["lineage_type"].value_counts().sort_index().items()
    }


def _width_differences(
    full: gpd.GeoDataFrame,
    local: gpd.GeoDataFrame,
    full_rows: np.ndarray,
    local_rows: np.ndarray,
    *,
    pair_mask: np.ndarray | None = None,
) -> dict[str, dict[str, float | int]]:
    if pair_mask is not None:
        full_rows = full_rows[pair_mask]
        local_rows = local_rows[pair_mask]
    results: dict[str, dict[str, float | int]] = {}
    for field in DEFAULT_MEASUREMENT_FIELDS:
        if field not in full.columns or field not in local.columns:
            results[field] = {"n_compared": 0, "max": float("nan"), "p95": float("nan")}
            continue
        first = pd.to_numeric(full.iloc[full_rows][field], errors="coerce").to_numpy()
        second = pd.to_numeric(local.iloc[local_rows][field], errors="coerce").to_numpy()
        valid = np.isfinite(first) & np.isfinite(second)
        if not valid.any():
            results[field] = {"n_compared": 0, "max": float("nan"), "p95": float("nan")}
            continue
        differences = np.asarray(
            [_relative_difference(left, right) for left, right in zip(first[valid], second[valid])],
            dtype=float,
        )
        results[field] = {
            "n_compared": int(len(differences)),
            "max": float(differences.max()),
            "p95": float(np.quantile(differences, 0.95)),
            "n_above_tolerance": int((differences > 0.02).sum()),
        }
    return results


def _same_mask(full_path: Path, local_path: Path) -> tuple[bool, int]:
    with rasterio.open(full_path) as full, rasterio.open(local_path) as local:
        same_grid = (
            full.crs == local.crs
            and full.transform == local.transform
            and full.width == local.width
            and full.height == local.height
        )
        if not same_grid:
            return False, -1
        return bool(np.array_equal(full.read(1), local.read(1))), int(
            np.count_nonzero(full.read(1) != local.read(1))
        )


def compare_island(
    run_dir: Path,
    *,
    geometry_tolerance_m: float,
    width_relative_tolerance: float,
    full_width_refresh_dirname: str | None = None,
) -> dict[str, Any]:
    full_manifest = _load_json(run_dir / "island_run_manifest.json")
    local_manifest = _load_json(run_dir / "local_patch" / "island_local_patch_run_manifest.json")
    full_paths = {key: Path(value) for key, value in full_manifest["final_paths"].items()}
    local_paths = {key: Path(value) for key, value in local_manifest["final_paths"].items()}
    full_width_product_source = "original_full_raster_widths"
    if full_width_refresh_dirname:
        refresh_manifest_path = (
            run_dir
            / "variant"
            / str(full_width_refresh_dirname)
            / "island_width_refresh_manifest.json"
        )
        refresh_manifest = _load_json(refresh_manifest_path)
        if str(refresh_manifest.get("island_id")) != str(full_manifest.get("island_id")):
            raise ValueError(f"Refresh manifest belongs to another island: {refresh_manifest_path}")
        full_paths["routing_width_links"] = Path(
            refresh_manifest["final_paths"]["routing_width_links"]
        )
        full_width_product_source = str(refresh_manifest_path)
    elif full_manifest.get("width_refresh_provenance", {}).get("promoted_refresh_manifest"):
        # The manifest's final paths already point at the promoted candidate.
        # Record that fact rather than incorrectly labelling it as an original
        # full-raster product.
        full_width_product_source = str(
            full_manifest["width_refresh_provenance"]["promoted_refresh_manifest"]
        )

    mask_equal, mask_different_pixels = _same_mask(
        full_paths["collapsed_mask"], local_paths["collapsed_mask"]
    )
    full_links = gpd.read_file(full_paths["routing_width_links"])
    local_links = gpd.read_file(local_paths["routing_width_links"])
    full_nodes = gpd.read_file(full_paths["directed_nodes"])
    local_nodes = gpd.read_file(local_paths["directed_nodes"])
    if local_links.crs != full_links.crs:
        local_links = local_links.to_crs(full_links.crs)
        local_nodes = local_nodes.to_crs(full_nodes.crs)

    full_validation = validate_single_inlet_single_outlet(
        build_directed_graph(full_links, full_nodes)
    )
    local_validation = validate_single_inlet_single_outlet(
        build_directed_graph(local_links, local_nodes)
    )
    full_rows, local_rows, distances = _match_links(full_links, local_links)
    matched_distances = (
        distances[full_rows, local_rows] if len(full_rows) else np.asarray([], dtype=float)
    )
    local_variant = _load_json(local_paths["variant_manifest"])
    recalculated_link_ids = {
        int(link_id)
        for link_id in (
            local_variant.get("local_patch", {})
            .get("products", {})
            .get("summary", {})
            .get("recalculated_link_ids", [])
        )
    }
    local_pair_ids = local_links.iloc[local_rows]["id_link"].astype(int).to_numpy()
    measured_pair_mask = np.isin(local_pair_ids, list(recalculated_link_ids))
    widths = _width_differences(
        full_links,
        local_links,
        full_rows,
        local_rows,
        pair_mask=measured_pair_mask,
    )
    all_pair_widths = _width_differences(full_links, local_links, full_rows, local_rows)

    full_lineage_path = full_paths["directed_links"].parent / "link_lineage.csv"
    local_lineage_path = local_paths["directed_links"].parent.parent / "matching" / "link_lineage.csv"
    full_lineage = _lineage_signature(full_lineage_path)
    local_lineage = _lineage_signature(local_lineage_path)
    topology_equal = (
        len(full_links) == len(local_links)
        and len(full_nodes) == len(local_nodes)
        and _cycle_rank(full_links) == _cycle_rank(local_links)
        and full_lineage == local_lineage
        and len(matched_distances) == len(full_links)
        and (not len(matched_distances) or matched_distances.max() <= geometry_tolerance_m)
    )
    anchor_equal = all(
        full_manifest.get(field) == local_manifest.get(field)
        for field in ("geometry_sha256", "anchor_unit_id", "anchor_bifurcation", "anchor_confluence", "source_unit_ids")
    )
    local_topology_passed = bool(
        local_variant.get("local_patch", {})
        .get("collapse_topology_validation", {})
        .get("passed", False)
    )
    width_passed = bool(
        all(
            details["n_compared"] > 0 and details["max"] <= width_relative_tolerance
            for field, details in widths.items()
            if field in FINAL_WIDTH_FIELDS
        )
    )
    lengths_passed = bool(
        all(
            details["n_compared"] > 0 and details["max"] <= width_relative_tolerance
            for field, details in widths.items()
            if field in LENGTH_FIELDS
        )
    )
    routing_ready = bool(
        pd.to_numeric(full_links["wid_adj_routing"], errors="coerce").gt(0).all()
        and pd.to_numeric(local_links["wid_adj_routing"], errors="coerce").gt(0).all()
    )
    passed = bool(
        mask_equal
        and anchor_equal
        and full_validation.is_valid
        and local_validation.is_valid
        and local_topology_passed
        and topology_equal
        and routing_ready
        and width_passed
    )
    result: dict[str, Any] = {
        "island_id": str(full_manifest["island_id"]),
        "passed": passed,
        "mask_equal": mask_equal,
        "mask_different_pixels": mask_different_pixels,
        "anchor_provenance_equal": anchor_equal,
        "full_network_valid": bool(full_validation.is_valid),
        "local_network_valid": bool(local_validation.is_valid),
        "local_collapse_topology_passed": local_topology_passed,
        "full_link_count": int(len(full_links)),
        "local_link_count": int(len(local_links)),
        "full_node_count": int(len(full_nodes)),
        "local_node_count": int(len(local_nodes)),
        "full_cycle_rank": _cycle_rank(full_links),
        "local_cycle_rank": _cycle_rank(local_links),
        "lineage_signature_equal": full_lineage == local_lineage,
        "full_lineage_signature": json.dumps(full_lineage, sort_keys=True),
        "local_lineage_signature": json.dumps(local_lineage, sort_keys=True),
        "max_link_hausdorff_m": float(matched_distances.max()) if len(matched_distances) else float("nan"),
        "p95_link_hausdorff_m": float(np.quantile(matched_distances, 0.95)) if len(matched_distances) else float("nan"),
        "topology_equal_within_geometry_tolerance": topology_equal,
        "routing_products_ready": routing_ready,
        "widths_within_relative_tolerance": width_passed,
        "lengths_within_relative_tolerance": lengths_passed,
        "full_width_product_source": full_width_product_source,
        "locally_recalculated_link_count": int(len(recalculated_link_ids)),
        "locally_recalculated_matched_link_count": int(measured_pair_mask.sum()),
    }
    for field, details in widths.items():
        result[f"{field}_n_compared"] = details["n_compared"]
        result[f"{field}_max_relative_difference"] = details["max"]
        result[f"{field}_p95_relative_difference"] = details["p95"]
        result[f"{field}_n_above_tolerance"] = details.get("n_above_tolerance", 0)
        all_pair_details = all_pair_widths[field]
        result[f"all_pair_{field}_max_relative_difference"] = all_pair_details["max"]
        result[f"all_pair_{field}_n_above_tolerance"] = all_pair_details.get(
            "n_above_tolerance", 0
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("setup_dir", help="8-connected island setup directory")
    parser.add_argument("--island-id", action="append", help="Rerun island ID; repeat as needed.")
    parser.add_argument("--output-dir", help="Defaults to the setup directory.")
    parser.add_argument("--geometry-tolerance-m", type=float, default=35.0)
    parser.add_argument("--width-relative-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--full-width-refresh-dirname",
        help="Use a non-destructive full-raster width-refresh candidate below each island run.",
    )
    parser.add_argument("--fail-on-difference", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_dir = Path(args.setup_dir).resolve()
    reruns = setup_dir / "island_reruns"
    run_dirs = [reruns / island_id for island_id in args.island_id] if args.island_id else sorted(
        path for path in reruns.iterdir() if (path / "island_run_manifest.json").exists()
    )
    if not run_dirs:
        raise FileNotFoundError(f"No completed island reruns below {reruns}")
    results = [
        compare_island(
            run_dir,
            geometry_tolerance_m=float(args.geometry_tolerance_m),
            width_relative_tolerance=float(args.width_relative_tolerance),
            full_width_refresh_dirname=args.full_width_refresh_dirname,
        )
        for run_dir in run_dirs
    ]
    report = pd.DataFrame(results).sort_values("island_id")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else setup_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "island_local_patch_paired_comparison.csv"
    summary_path = output_dir / "island_local_patch_paired_comparison_summary.json"
    report.to_csv(report_path, index=False)
    summary = {
        "schema_version": "island-local-patch-paired-comparison-v1",
        "setup_dir": str(setup_dir),
        "geometry_tolerance_m": float(args.geometry_tolerance_m),
        "width_relative_tolerance": float(args.width_relative_tolerance),
        "full_width_refresh_dirname": args.full_width_refresh_dirname,
        "n_islands": int(len(report)),
        "n_passed": int(report["passed"].sum()),
        "all_passed": bool(report["passed"].all()),
        "report_csv": str(report_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(report.to_string(index=False))
    print(f"Wrote paired comparison: {report_path}")
    if args.fail_on_difference and not summary["all_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
