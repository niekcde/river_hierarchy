from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import geopandas as gpd
import pandas as pd

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from hierarchy_level_definition.unit_detection.bifurcation_confluence_units import (
    StructuralUnit,
    analyze_network,
    edge_length_from_attrs,
    load_network,
)


WIDTH_COLUMNS = ("wid_adj", "wid", "wid_med")


def infer_network_name(path: str | Path) -> str:
    stem = Path(path).stem
    for suffix in ("_links", "_nodes"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def default_output_dir_for_links(links_path: str | Path) -> Path:
    return Path(__file__).resolve().parent / "outputs" / infer_network_name(links_path)


def edge_width_from_attrs(attrs: dict[str, Any]) -> float:
    for key in WIDTH_COLUMNS:
        value = attrs.get(key)
        if value is not None and not pd.isna(value):
            value = float(value)
            if value > 0:
                return value
    return float("nan")


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator is None or pd.isna(denominator) or denominator == 0:
        return float("nan")
    return float(numerator) / float(denominator)


def _weighted_harmonic_mean_width(lengths: list[float], widths: list[float]) -> float:
    if not lengths or not widths or len(lengths) != len(widths):
        return float("nan")
    if any(pd.isna(length) or length < 0 for length in lengths):
        return float("nan")
    if any(pd.isna(width) or width <= 0 for width in widths):
        return float("nan")
    total_length = sum(lengths)
    if total_length <= 0:
        return float("nan")
    return total_length / sum(length / width for length, width in zip(lengths, widths, strict=True))


def _normalized_entropy(weights: list[float]) -> float:
    valid_weights = [float(weight) for weight in weights if not pd.isna(weight) and weight > 0]
    if len(valid_weights) <= 1:
        return 1.0
    total = sum(valid_weights)
    shares = [weight / total for weight in valid_weights]
    entropy = -sum(share * math.log(share) for share in shares if share > 0)
    return entropy / math.log(len(shares))


def _count_descendants(children_map: dict[int, list[int]], unit_id: int) -> int:
    children = children_map.get(unit_id, [])
    return len(children) + sum(_count_descendants(children_map, child_id) for child_id in children)


def _collapse_level(children_map: dict[int, list[int]], unit_id: int) -> int:
    children = children_map.get(unit_id, [])
    if not children:
        return 0
    return 1 + max(_collapse_level(children_map, child_id) for child_id in children)


def _build_primary_tree_metadata(units: list[StructuralUnit]) -> pd.DataFrame:
    units_by_id = {unit.unit_id: unit for unit in units}
    primary_parent: dict[int, int | None] = {}
    children_map: dict[int, list[int]] = {unit.unit_id: [] for unit in units}

    for unit in units:
        if not unit.parents:
            primary_parent[unit.unit_id] = None
            continue
        chosen_parent = min(
            unit.parents,
            key=lambda parent_id: (
                len(units_by_id[parent_id].edge_set),
                len(units_by_id[parent_id].node_set),
                parent_id,
            ),
        )
        primary_parent[unit.unit_id] = chosen_parent
        children_map[chosen_parent].append(unit.unit_id)

    for child_ids in children_map.values():
        child_ids.sort()

    records: list[dict[str, Any]] = []

    def walk(unit_id: int, *, root_unit_id: int, depth_from_root: int) -> None:
        child_ids = children_map[unit_id]
        records.append(
            {
                "unit_id": unit_id,
                "primary_parent_id": primary_parent[unit_id],
                "root_unit_id": root_unit_id,
                "depth_from_root": depth_from_root,
                "collapse_level": _collapse_level(children_map, unit_id),
                "n_children": len(child_ids),
                "n_descendants": _count_descendants(children_map, unit_id),
                "is_compound": len(child_ids) > 0,
                "compound_unit_id": unit_id if child_ids else pd.NA,
            }
        )
        for child_id in child_ids:
            walk(child_id, root_unit_id=root_unit_id, depth_from_root=depth_from_root + 1)

    root_ids = sorted(unit.unit_id for unit in units if primary_parent[unit.unit_id] is None)
    for root_id in root_ids:
        walk(root_id, root_unit_id=root_id, depth_from_root=0)

    return pd.DataFrame.from_records(records).sort_values("unit_id").reset_index(drop=True)


def _path_metrics_for_unit(
    unit: StructuralUnit,
    link_lookup: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in unit.paths:
        lengths: list[float] = []
        widths: list[float] = []
        for link_id in path.id_links:
            attrs = link_lookup[int(link_id)]
            lengths.append(edge_length_from_attrs(attrs))
            widths.append(edge_width_from_attrs(attrs))

        path_length = sum(lengths)
        path_width_eq = _weighted_harmonic_mean_width(lengths, widths)
        records.append(
            {
                "unit_id": unit.unit_id,
                "path_id": path.path_id,
                "n_links": len(path.id_links),
                "path_length": path_length,
                "path_width_eq": path_width_eq,
                "path_width_min": min(widths) if widths else float("nan"),
                "path_width_max": max(widths) if widths else float("nan"),
                "id_links": ",".join(str(link_id) for link_id in path.id_links),
            }
        )
    return records


def compute_unit_metrics_from_units(
    links: gpd.GeoDataFrame,
    units: list[StructuralUnit],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    link_lookup = {int(row.id_link): row._asdict() for row in links.itertuples(index=False)}

    path_records: list[dict[str, Any]] = []
    for unit in units:
        path_records.extend(_path_metrics_for_unit(unit, link_lookup))
    path_metrics = pd.DataFrame.from_records(path_records)

    tree_metadata = _build_primary_tree_metadata(units)
    unit_records: list[dict[str, Any]] = []

    for unit in units:
        unit_paths = path_metrics.loc[path_metrics["unit_id"] == unit.unit_id].copy()
        path_lengths = unit_paths["path_length"].tolist()
        path_widths = unit_paths["path_width_eq"].tolist()
        valid_widths = [width for width in path_widths if not pd.isna(width) and width > 0]

        equivalent_width = sum(valid_widths) if valid_widths else float("nan")
        equivalent_length = (
            sum(width * length for width, length in zip(path_widths, path_lengths, strict=True) if not pd.isna(width) and width > 0)
            / equivalent_width
            if valid_widths and equivalent_width > 0
            else float("nan")
        )

        length_min = min(path_lengths) if path_lengths else float("nan")
        length_max = max(path_lengths) if path_lengths else float("nan")
        width_min = min(valid_widths) if valid_widths else float("nan")
        width_max = max(valid_widths) if valid_widths else float("nan")

        width_ratio_2 = float("nan")
        smaller_width_fraction_2 = float("nan")
        length_ratio_2 = float("nan")
        if len(valid_widths) == 2 and width_max > 0:
            width_ratio_2 = width_min / width_max
            smaller_width_fraction_2 = width_min / equivalent_width
        if len(path_lengths) == 2 and length_max > 0:
            length_ratio_2 = length_min / length_max

        unit_records.append(
            {
                "unit_id": unit.unit_id,
                "bifurcation": unit.bifurcation,
                "confluence": unit.confluence,
                "class": unit.unit_class,
                "n_paths": unit.n_paths,
                "equivalent_length": equivalent_length,
                "equivalent_width": equivalent_width,
                "elongation": _safe_divide(equivalent_length, equivalent_width),
                "path_length_min": length_min,
                "path_length_max": length_max,
                "path_length_range_norm": _safe_divide(length_max - length_min, equivalent_length),
                "path_width_eq_min": width_min,
                "path_width_eq_max": width_max,
                "path_width_range_norm": _safe_divide(width_max - width_min, equivalent_width),
                "largest_path_width_fraction": _safe_divide(width_max, equivalent_width),
                "width_evenness": _normalized_entropy(valid_widths) if valid_widths else float("nan"),
                "width_ratio_2": width_ratio_2,
                "smaller_width_fraction_2": smaller_width_fraction_2,
                "length_ratio_2": length_ratio_2,
                "internal_bifurcation_count": len(unit.internal_bifurcations),
                "internal_confluence_count": len(unit.internal_confluences),
            }
        )

    unit_metrics = pd.DataFrame.from_records(unit_records).merge(tree_metadata, on="unit_id", how="left")
    unit_metrics = unit_metrics.sort_values(
        ["root_unit_id", "depth_from_root", "collapse_level", "bifurcation", "confluence", "unit_id"]
    ).reset_index(drop=True)
    path_metrics = path_metrics.sort_values(["unit_id", "path_id"]).reset_index(drop=True)
    return unit_metrics, path_metrics


def compute_unit_metrics(
    links_path: str | Path,
    nodes_path: str | Path,
    *,
    max_path_cutoff: int = 100,
    max_paths: int = 5000,
    debug: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    links, nodes = load_network(links_path, nodes_path)
    summary, units, _ = analyze_network(
        links_path,
        nodes_path,
        max_path_cutoff=max_path_cutoff,
        max_paths=max_paths,
        debug=debug,
    )
    unit_metrics, path_metrics = compute_unit_metrics_from_units(links, units)
    return summary, unit_metrics, path_metrics


def write_metrics_outputs(
    output_dir: str | Path,
    summary: pd.DataFrame,
    unit_metrics: pd.DataFrame,
    path_metrics: pd.DataFrame,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path / "unit_summary.csv", index=False)
    unit_metrics.to_csv(output_path / "unit_metrics.csv", index=False)
    path_metrics.to_csv(output_path / "path_metrics.csv", index=False)

    manifest = {
        "files": [
            "unit_summary.csv",
            "unit_metrics.csv",
            "path_metrics.csv",
        ],
        "n_units": int(len(unit_metrics)),
        "n_paths": int(len(path_metrics)),
    }
    with (output_path / "metrics_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute geometry-based metrics for detected units. The output includes a unit-level "
            "metrics table and a path-level metrics table so the same code can be used from a notebook or batch scripts."
        )
    )
    parser.add_argument("links_gpkg", help="Reviewed links GeoPackage.")
    parser.add_argument("nodes_gpkg", help="Reviewed nodes GeoPackage.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory for CSV outputs.")
    parser.add_argument("--max-path-cutoff", type=int, default=100, help="Maximum edge-count cutoff for unit path enumeration.")
    parser.add_argument("--max-paths", type=int, default=5000, help="Maximum number of simple paths per unit.")
    parser.add_argument("--debug", action="store_true", help="Print debug information during unit detection.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary, unit_metrics, path_metrics = compute_unit_metrics(
        args.links_gpkg,
        args.nodes_gpkg,
        max_path_cutoff=args.max_path_cutoff,
        max_paths=args.max_paths,
        debug=args.debug,
    )

    print(unit_metrics.to_string(index=False))
    print()
    print(path_metrics.to_string(index=False))

    output_dir = args.output_dir if args.output_dir is not None else default_output_dir_for_links(args.links_gpkg)
    write_metrics_outputs(output_dir, summary, unit_metrics, path_metrics)
    print()
    print(f"Wrote outputs to {Path(output_dir).resolve()}")


if __name__ == "__main__":
    main()
