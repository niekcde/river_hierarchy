"""Validate a non-destructive base-network direction rebuild.

The command compares a corrected-direction RivGraph result with the current
base-state product.  It verifies invariant undirected topology, recomputes the
normal two-bank width families against the frozen base mask, then applies the
production routing-width model on the corrected directed topology.  It never
modifies either source experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network_variants.routing_width_application import apply_routing_width_model
from network_variants.variant_generation import compute_width_families


def _width_summary(
    old: gpd.GeoDataFrame,
    candidate: gpd.GeoDataFrame,
    field: str,
) -> dict[str, object]:
    old_values = pd.to_numeric(old[field], errors="coerce").to_numpy(dtype=float)
    new_values = pd.to_numeric(candidate[field], errors="coerce").to_numpy(dtype=float)
    absolute = np.abs(old_values - new_values)
    relative = absolute / np.where(np.abs(old_values) > 0, np.abs(old_values), np.nan)
    return {
        "median_relative_difference": float(np.nanmedian(relative)),
        "p95_relative_difference": float(np.nanquantile(relative, 0.95)),
        "max_relative_difference": float(np.nanmax(relative)),
        "max_abs_difference": float(np.nanmax(absolute)),
        "n_gt_1pct": int(np.sum(relative > 0.01)),
        "n_gt_5pct": int(np.sum(relative > 0.05)),
        "n_gt_10pct": int(np.sum(relative > 0.10)),
        "n_gt_25pct": int(np.sum(relative > 0.25)),
        "n_gt_50pct": int(np.sum(relative > 0.50)),
        "gt_5pct_link_ids": candidate.loc[relative > 0.05, "id_link"].astype(int).tolist(),
    }


def validate_direction_rebuild(
    *,
    example_id: str,
    base_variant_dir: str | Path,
    source_raw_links_path: str | Path,
    candidate_links_path: str | Path,
    candidate_nodes_path: str | Path,
    output_dir: str | Path,
    old_exit_sides: str,
    corrected_exit_sides: str,
) -> dict[str, object]:
    """Run the isolated topology and width audit, returning its JSON payload."""
    example = str(example_id).zfill(2)
    base = Path(base_variant_dir).expanduser().resolve()
    source_raw_links = Path(source_raw_links_path).expanduser().resolve()
    candidate_links_path = Path(candidate_links_path).expanduser().resolve()
    candidate_nodes_path = Path(candidate_nodes_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    for label, path in {
        "base variant": base,
        "source raw links": source_raw_links,
        "candidate links": candidate_links_path,
        "candidate nodes": candidate_nodes_path,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    output.mkdir(parents=True, exist_ok=True)

    source_raw = gpd.read_file(source_raw_links).sort_values("id_link").reset_index(drop=True)
    candidate_raw = gpd.read_file(candidate_links_path).sort_values("id_link").reset_index(drop=True)
    current = gpd.read_file(base / "widths" / "links_with_width_families.gpkg").sort_values("id_link").reset_index(drop=True)
    candidate_nodes = gpd.read_file(candidate_nodes_path)
    if source_raw["id_link"].astype(int).tolist() != candidate_raw["id_link"].astype(int).tolist():
        raise ValueError("Corrected raw link IDs do not match the source raw network.")

    source_pairs = np.sort(source_raw[["id_us_node", "id_ds_node"]].astype(int).to_numpy(), axis=1)
    candidate_pairs = np.sort(candidate_raw[["id_us_node", "id_ds_node"]].astype(int).to_numpy(), axis=1)
    geometry_equal = [bool(left.equals(right)) for left, right in zip(source_raw.geometry, candidate_raw.geometry)]
    reversed_endpoints = (
        source_raw["id_us_node"].astype(int).to_numpy() == candidate_raw["id_ds_node"].astype(int).to_numpy()
    ) & (
        source_raw["id_ds_node"].astype(int).to_numpy() == candidate_raw["id_us_node"].astype(int).to_numpy()
    )

    # This is deliberately the same physical measurement route as a normal
    # base state. Manual review is retained as provenance, not used to scale
    # two-bank width observations.
    print(f"{example}: recomputing two-bank width families for {len(candidate_raw)} links", flush=True)
    families, samples = compute_width_families(
        candidate_raw,
        collapsed_mask_path=base / "mask" / f"{example}__S000_base_collapsed.tif",
        wet_reference_mask_path=base / "mask" / f"{example}__S000_base_collapsed.tif",
        origin_mask_path=base / "mask" / "origin_mask.tif",
        bank_labels_path=base / "mask" / "bank_labels.tif",
    )
    metadata_columns = [
        "id_link", "n_units", "unit_ids", "root_unit_ids", "collapse_levels",
        "compound_unit_ids", "compound_bubble_ids", "unit_classes",
        "unit_topodynamic_classes", "width_reviewed_flag", "width_review_decision",
        "manual_width_review_candidate", "manual_width_review_decision",
        "manual_width_review_requires_correction", "manual_width_review_source",
    ]
    families = families.merge(
        current[[column for column in metadata_columns if column in current]],
        on="id_link", how="left", validate="1:1",
    )
    families["example_id"] = example
    families["variant_id"] = f"S000_base_direction_{corrected_exit_sides}_validation"
    print(f"{example}: applying routing-width model to {len(families)} links", flush=True)
    solved, routing_audit, routing_application = apply_routing_width_model(
        families,
        samples,
        candidate_nodes,
        example_id=example,
        variant_id=str(families["variant_id"].iloc[0]),
        apply_known_base_exclusions=True,
    )
    solved = solved.sort_values("id_link").reset_index(drop=True)

    node_wse = pd.read_csv(base / "matching" / "node_sword_match.csv", low_memory=False).set_index("id_node")
    inlets = candidate_nodes.loc[candidate_nodes["is_inlet"].fillna(0).astype(int).eq(1), "id_node"].astype(int).tolist()
    outlets = candidate_nodes.loc[candidate_nodes["is_outlet"].fillna(0).astype(int).eq(1), "id_node"].astype(int).tolist()
    report: dict[str, object] = {
        "schema_version": "direction-rebuild-validation-v1",
        "example_id": example,
        "old_exit_sides": old_exit_sides,
        "candidate_exit_sides": corrected_exit_sides,
        "n_links_source": int(len(source_raw)),
        "n_links_candidate": int(len(candidate_raw)),
        "n_nodes_candidate": int(len(candidate_nodes)),
        "undirected_endpoint_pairs_equal": bool(np.array_equal(source_pairs, candidate_pairs)),
        "all_link_geometries_equal": bool(all(geometry_equal)),
        "geometry_mismatch_link_ids": candidate_raw.loc[[not value for value in geometry_equal], "id_link"].astype(int).tolist(),
        "reversed_endpoint_count": int(reversed_endpoints.sum()),
        "recomputed_sample_count": int(len(samples)),
        "candidate_inlets": inlets,
        "candidate_outlets": outlets,
        "candidate_inlet_wse": {str(node): float(node_wse.loc[node, "sword_wse"]) for node in inlets if node in node_wse.index and pd.notna(node_wse.loc[node, "sword_wse"])},
        "candidate_outlet_wse": {str(node): float(node_wse.loc[node, "sword_wse"]) for node in outlets if node in node_wse.index and pd.notna(node_wse.loc[node, "sword_wse"])},
        "manual_review_status": "Retained as existing review provenance; physical two-bank families are not numerically rescaled.",
        "fields": {field: _width_summary(current, solved, field) for field in ("wid_adj_wet", "wid_adj_total", "wid_adj_routing")},
        "routing_width_application": routing_application,
    }
    (output / "direction_width_validation_production_semantics.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    samples.to_parquet(output / "direction_width_validation_production_samples.parquet", index=False)
    solved.to_file(output / "direction_width_validation_production_widths.gpkg", driver="GPKG")
    routing_audit.to_csv(output / "direction_width_validation_routing_audit_production.csv", index=False)
    print(f"{example}: validation complete: {output}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example-id", required=True)
    parser.add_argument("--base-variant-dir", required=True)
    parser.add_argument("--source-raw-links", required=True)
    parser.add_argument("--candidate-links", required=True)
    parser.add_argument("--candidate-nodes", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--old-exit-sides", required=True)
    parser.add_argument("--corrected-exit-sides", required=True)
    args = parser.parse_args()
    report = validate_direction_rebuild(
        example_id=args.example_id,
        base_variant_dir=args.base_variant_dir,
        source_raw_links_path=args.source_raw_links,
        candidate_links_path=args.candidate_links,
        candidate_nodes_path=args.candidate_nodes,
        output_dir=args.output_dir,
        old_exit_sides=args.old_exit_sides,
        corrected_exit_sides=args.corrected_exit_sides,
    )
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
