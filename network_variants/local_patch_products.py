"""Phase-6 width and length products for a conservatively merged local patch.

Strict unchanged links inherit the complete parent width product.  Only links
inserted by the local merge are sampled against the bounded collapsed raster.
Final geometries are converted back to ordered local raster indices so width
measurement never includes computational collar context removed in Phase 5.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol

from network_variants.local_patch_anchored import LocalPatchAnchoredResult
from network_variants.local_patch_merge import LocalPatchMergeResult
from network_variants.local_patch_variant import LocalPatchVariantResult
from network_variants.routing_width_application import (
    apply_routing_width_model,
    write_routing_width_audit,
)
from network_variants.variant_generation import (
    _apply_segmented_stub_width_inheritance,
    _compute_changed_link_width_products,
    _infer_parent_node_order,
    _line_length_fields_updated,
    _match_child_links_to_parent_links,
    _match_child_nodes_to_parent_nodes,
    _normalize_width_boolean_fields,
    _refresh_segmented_legacy_width_fields,
    _resolve_link_lineage,
    compute_width_families,
)
from network_variants.width_two_bank import build_state_bank_labels
from network_variants.width_product_inheritance import (
    WIDTH_INHERITANCE_FIELDS,
    inherit_unchanged_link_width_products,
)
from network_variants.width_samples import read_width_samples


LOCAL_PATCH_PRODUCTS_VERSION = "topology-local-patch-products-v3"


LENGTH_INHERITANCE_FIELDS = (
    "len",
    "len_adj",
    "link_length_m",
    "parent_link_length_m",
    "sinuosity",
    "n_nodes",
)


@dataclass(frozen=True, slots=True)
class LocalPatchProductResult:
    status: str
    output_paths: Mapping[str, str]
    summary: Mapping[str, Any]
    products_version: str = LOCAL_PATCH_PRODUCTS_VERSION

    def to_manifest(self) -> dict[str, Any]:
        return {
            "products_version": self.products_version,
            "status": self.status,
            "output_paths": dict(self.output_paths),
            "summary": dict(self.summary),
        }


def _ordered_geometry_pixel_indices(
    geometry: Any,
    *,
    transform: Any,
    shape: tuple[int, int],
    water: np.ndarray,
) -> list[int]:
    """Sample one directed geometry into unique, ordered wet raster cells."""
    if geometry is None or geometry.is_empty or float(geometry.length) <= 0:
        return []
    pixel_size = float(max(abs(transform.a), abs(transform.e)))
    spacing = max(pixel_size * 0.4, 1e-9)
    count = max(2, int(math.ceil(float(geometry.length) / spacing)) + 1)
    distances = np.linspace(0.0, float(geometry.length), count)
    coordinates = [geometry.interpolate(float(distance)).coords[0] for distance in distances]
    rows, cols = rowcol(
        transform,
        [float(coordinate[0]) for coordinate in coordinates],
        [float(coordinate[1]) for coordinate in coordinates],
    )
    height, width = shape
    ordered: list[int] = []
    previous: int | None = None
    for row, col in zip(rows, cols):
        row, col = int(row), int(col)
        if not (0 <= row < height and 0 <= col < width):
            continue
        if not bool(water[row, col]):
            continue
        flat_index = int(np.ravel_multi_index((row, col), shape))
        if flat_index != previous:
            ordered.append(flat_index)
            previous = flat_index
    return ordered


def _attach_changed_link_pixels(
    links: gpd.GeoDataFrame,
    *,
    changed_link_ids: set[int],
    collapsed_mask_path: str | Path,
) -> tuple[gpd.GeoDataFrame, dict[int, int]]:
    result = links.copy()
    if "idx_link" not in result.columns:
        result["idx_link"] = None
    with rasterio.open(collapsed_mask_path) as source:
        water = source.read(1) > 0
        transform = source.transform
        shape = source.shape
        raster_crs = source.crs
    if result.crs is None or raster_crs is None:
        raise ValueError("Merged links and local collapsed mask both require a CRS.")
    raster_links = result.to_crs(raster_crs)
    sample_counts: dict[int, int] = {}
    for index, row in raster_links.iterrows():
        link_id = int(row["id_link"])
        if link_id not in changed_link_ids:
            continue
        indices = _ordered_geometry_pixel_indices(
            row.geometry,
            transform=transform,
            shape=shape,
            water=water,
        )
        if len(indices) < 2:
            raise ValueError(
                f"Changed local link {link_id} has fewer than two wet raster samples."
            )
        result.at[index, "idx_link"] = ", ".join(str(value) for value in indices)
        if "wid_pix" in result.columns:
            result.at[index, "wid_pix"] = None
        sample_counts[link_id] = len(indices)
    return result, sample_counts


def _strict_width_measurement_link_ids(
    child_links: gpd.GeoDataFrame,
    *,
    parent_links: gpd.GeoDataFrame,
    parent_samples: pd.DataFrame,
    lineage: pd.DataFrame,
) -> set[int]:
    """Return links rejected by the authoritative inheritance contract."""
    classified_links, _, _ = inherit_unchanged_link_width_products(
        child_links=child_links,
        child_width_samples=pd.DataFrame(),
        parent_links=parent_links,
        parent_width_samples=parent_samples,
        link_lineage=lineage,
    )
    return set(
        classified_links.loc[
            classified_links["width_inheritance_status"].eq(
                "changed_requires_evaluation"
            ),
            "id_link",
        ].astype(int)
    )


def _local_link_lineage(
    child_links: gpd.GeoDataFrame,
    child_nodes: gpd.GeoDataFrame,
    parent_links: gpd.GeoDataFrame,
    parent_nodes: gpd.GeoDataFrame,
    *,
    match_tolerance: float,
) -> pd.DataFrame:
    parent_node_ids = set(parent_links["id_us_node"].astype(int)).union(parent_links["id_ds_node"].astype(int))
    missing_nodes = parent_node_ids.difference(parent_nodes["id_node"].astype(int))
    if missing_nodes:
        raise ValueError(f"Parent node product is missing parent nodes: {sorted(missing_nodes)}")
    parent_order, parent_graph, _ = _infer_parent_node_order(parent_links, parent_nodes)
    node_match = _match_child_nodes_to_parent_nodes(
        parent_nodes,
        child_nodes,
        match_tolerance=float(match_tolerance),
        parent_node_order=parent_order,
    )
    link_match = _match_child_links_to_parent_links(
        parent_links,
        child_links,
        match_tolerance=float(match_tolerance),
    )
    return _resolve_link_lineage(
        parent_graph=parent_graph,
        directed_child_links=child_links,
        node_match=node_match,
        link_match=link_match,
        match_tolerance=float(match_tolerance),
    )


def _write_local_state_bank_labels(
    *,
    base_labels_path: str | Path,
    origin_mask_path: str | Path,
    output_path: str | Path,
) -> tuple[Path, list[int]]:
    with rasterio.open(base_labels_path) as labels_source:
        labels = labels_source.read(1).astype(np.int32)
        profile = labels_source.profile.copy()
        labels_transform = labels_source.transform
    with rasterio.open(origin_mask_path) as origin_source:
        origin = origin_source.read(1)
        if origin.shape != labels.shape or origin_source.transform != labels_transform:
            raise ValueError("Local origin mask and boundary labels are not aligned.")
    state_labels, removed = build_state_bank_labels(labels, origin)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(dtype="int32", count=1, compress="lzw", nodata=0)
    if not profile.get("tiled", False):
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(state_labels.astype(np.int32), 1)
    return path, [int(value) for value in removed]


def _refresh_changed_lengths_and_legacy_widths(
    links: gpd.GeoDataFrame,
    samples: pd.DataFrame,
    changed_link_ids: set[int],
) -> gpd.GeoDataFrame:
    result = links.copy()
    sample_ids = (
        pd.to_numeric(samples.get("id_link", pd.Series(dtype=float)), errors="coerce")
        if not samples.empty
        else pd.Series(dtype=float)
    )
    for index, row in result.iterrows():
        link_id = int(row["id_link"])
        if link_id not in changed_link_ids:
            continue
        updated = _line_length_fields_updated(row.to_dict(), row.geometry)
        link_samples = samples.loc[sample_ids.eq(link_id)]
        if "sample_index" in link_samples.columns:
            link_samples = link_samples.sort_values("sample_index")
        width_values = pd.to_numeric(
            link_samples.get("width_total", pd.Series(dtype=float)), errors="coerce"
        ).dropna().tolist()
        if width_values:
            updated = _refresh_segmented_legacy_width_fields(
                updated,
                geometry=row.geometry,
                width_values=width_values,
            )
        for column, value in updated.items():
            if column in result.columns and column != result.geometry.name:
                result.at[index, column] = value
    return result


def _inherit_unchanged_length_fields(
    links: gpd.GeoDataFrame,
    *,
    parent_links: gpd.GeoDataFrame,
    inheritance: Mapping[str, Any],
    measured_link_ids: set[int],
) -> tuple[gpd.GeoDataFrame, dict[int, int], list[str]]:
    """Copy parent length products for every strictly inherited child link."""
    result = links.copy()
    parent_by_id = parent_links.set_index(parent_links["id_link"].astype(int), drop=False)
    parent_map = {
        int(child_id): int(parent_id)
        for child_id, parent_id in dict(
            inheritance.get("parent_link_by_child_link", {})
        ).items()
    }
    inherited_map = {
        child_id: parent_id
        for child_id, parent_id in parent_map.items()
        if child_id not in measured_link_ids
    }
    copied_fields = [
        field
        for field in LENGTH_INHERITANCE_FIELDS
        if field in result.columns and field in parent_links.columns
    ]
    for child_id, parent_id in inherited_map.items():
        if parent_id not in parent_by_id.index:
            raise ValueError(
                f"Length inheritance references absent parent link {parent_id}."
            )
        child_index = result.index[result["id_link"].astype(int).eq(child_id)]
        if len(child_index) != 1:
            raise ValueError(f"Expected one child link {child_id} for length inheritance.")
        for field in copied_fields:
            result.at[child_index[0], field] = parent_by_id.loc[parent_id, field]
    return result, inherited_map, copied_fields


def materialize_local_patch_width_length_products(
    *,
    merge_result: LocalPatchMergeResult | LocalPatchAnchoredResult,
    variant_result: LocalPatchVariantResult,
    parent_width_links_path: str | Path,
    parent_nodes_path: str | Path,
    parent_width_samples_path: str | Path,
    output_dir: str | Path,
    example_id: str,
    variant_id: str,
    width_runner: Callable[..., tuple[gpd.GeoDataFrame, pd.DataFrame]] = compute_width_families,
    routing_width_runner: Callable[..., tuple[gpd.GeoDataFrame, pd.DataFrame, dict[str, Any]]] | None = apply_routing_width_model,
    collapse_corridor_mode: str = "anchored",
    exclude_closed_two_link_replacement: bool = False,
) -> LocalPatchProductResult:
    """Create complete Phase-6 width/length products for a merged local state."""
    if merge_result.status not in {
        "complete_conservative_merge",
        "complete_shared_anchored_graph",
    }:
        raise ValueError(
            "Phase-6 products require a complete conservative or shared anchored graph."
        )
    if variant_result.status != "complete_raw_local_variant":
        raise ValueError("Phase-6 products require a complete raw local variant.")

    shared_anchored = merge_result.status == "complete_shared_anchored_graph"
    link_path_key = "anchored_links" if shared_anchored else "merged_links"
    node_path_key = "anchored_nodes" if shared_anchored else "merged_nodes"
    child_links = gpd.read_file(merge_result.output_paths[link_path_key])
    child_nodes = gpd.read_file(merge_result.output_paths[node_path_key])
    parent_links = gpd.read_file(parent_width_links_path).to_crs(child_links.crs)
    parent_nodes = gpd.read_file(parent_nodes_path).to_crs(child_nodes.crs)
    parent_samples = read_width_samples(parent_width_samples_path)
    # Phase 5 may carry sparse or stale width columns through the parent-shaped
    # merge schema.  Rebuild the width layer from exactly two authoritative
    # sources: parent inheritance and changed-link measurement.
    child_links = child_links.drop(
        columns=[column for column in WIDTH_INHERITANCE_FIELDS if column in child_links.columns],
        errors="ignore",
    )
    # Use the same scientific lineage tolerance as the full-raster workflow.
    # Phase 5 intentionally uses a wider terminal-binding tolerance, which is
    # unsuitable for distinguishing core overlap from touch-only lineage.
    with rasterio.open(variant_result.output_paths["local_collapsed_mask"]) as source:
        match_tolerance = float(max(abs(source.transform.a), abs(source.transform.e))) * 1.25
    if shared_anchored:
        lineage = pd.read_csv(merge_result.output_paths["anchored_link_lineage"])
    else:
        lineage = _local_link_lineage(
            child_links,
            child_nodes,
            parent_links,
            parent_nodes,
            match_tolerance=match_tolerance,
        )
    child_links = child_links.drop(
        columns=[column for column in lineage.columns if column in child_links.columns and column != "id_link"],
        errors="ignore",
    ).merge(lineage, on="id_link", how="left", validate="1:1")
    # Use the exact same strict inheritance decision that the downstream width
    # assembly uses.  Lineage alone is only a candidate: a nominal unchanged
    # link can still require measurement when its geometry is not exactly the
    # parent geometry.  Every such link must receive local-raster pixel indices
    # before it is passed to the width runner.
    changed_ids = _strict_width_measurement_link_ids(
        child_links,
        parent_links=parent_links,
        parent_samples=parent_samples,
        lineage=lineage,
    )
    child_links, pixel_sample_counts = _attach_changed_link_pixels(
        child_links,
        changed_link_ids=changed_ids,
        collapsed_mask_path=variant_result.output_paths["local_collapsed_mask"],
    )

    output_dir = Path(output_dir)
    width_dir = output_dir / "widths"
    width_dir.mkdir(parents=True, exist_ok=True)
    bank_labels_path, removed_bank_labels = _write_local_state_bank_labels(
        base_labels_path=variant_result.output_paths["boundary_component_labels"],
        origin_mask_path=variant_result.output_paths["local_origin_mask"],
        output_path=width_dir / "local_state_bank_labels.tif",
    )

    enriched, samples, measured_ids, inheritance = _compute_changed_link_width_products(
        child_links,
        parent_links=parent_links,
        parent_width_samples=parent_samples,
        link_lineage=lineage,
        width_runner=width_runner,
        collapsed_mask_path=variant_result.output_paths["local_collapsed_mask"],
        wet_reference_mask_path=variant_result.output_paths["local_base_mask"],
        origin_mask_path=variant_result.output_paths["local_origin_mask"],
        rivgraph_distance_path=variant_result.output_paths.get("raw_rivgraph_distance"),
        bank_labels_path=bank_labels_path,
    )
    measured_set = set(int(value) for value in measured_ids)
    enriched, inherited_parent_map, inherited_length_fields = (
        _inherit_unchanged_length_fields(
            enriched,
            parent_links=parent_links,
            inheritance=inheritance,
            measured_link_ids=measured_set,
        )
    )
    enriched = _refresh_changed_lengths_and_legacy_widths(enriched, samples, measured_set)
    segmented_summary: dict[str, Any] = {"applied_count": 0, "applied_link_ids": []}
    if str(collapse_corridor_mode) == "anchored":
        segmented_links, enriched, segmented_summary = _apply_segmented_stub_width_inheritance(
            directed_links=enriched,
            enriched_links=enriched,
            link_width_samples=samples,
        )
        for column in segmented_links.columns:
            if column.startswith("family_"):
                enriched[column] = segmented_links[column].to_numpy()
    enriched["length_product_status"] = np.where(
        enriched["id_link"].astype(int).isin(measured_set),
        "recalculated_local_geometry",
        "inherited_unchanged_1to1",
    )
    enriched["product_source_parent_link_id"] = pd.array(
        [
            pd.NA
            if int(value) in measured_set
            else inherited_parent_map.get(int(value), pd.NA)
            for value in enriched["id_link"]
        ],
        dtype="Int64",
    )

    routing_audit = pd.DataFrame()
    routing_summary: dict[str, Any] = {"status": "not_run"}
    routing_paths: dict[str, str] = {}
    if routing_width_runner is not None:
        excluded_routing_width_link_ids: list[int] = []
        closed_replacement_audit = pd.DataFrame()
        if exclude_closed_two_link_replacement and "lineage_type" in enriched.columns:
            from network_variants.width_trim_selection import resolve_closed_replacement_exclusions

            excluded_routing_width_link_ids, closed_replacement_audit = resolve_closed_replacement_exclusions(
                enriched,
                parent_links,
                sorted(measured_set),
            )
            if not closed_replacement_audit.empty:
                decision_columns = [
                    column for column in closed_replacement_audit.columns
                    if column != "id_link" and column not in enriched.columns
                ]
                enriched = enriched.merge(
                    closed_replacement_audit[["id_link", *decision_columns]],
                    on="id_link",
                    how="left",
                    validate="1:1",
                )
        enriched, routing_audit, routing_summary = routing_width_runner(
            enriched,
            samples,
            child_nodes,
            example_id=str(example_id),
            variant_id=str(variant_id),
            apply_known_base_exclusions=False,
            evaluation_link_ids=sorted(measured_set),
            excluded_evaluation_link_ids=excluded_routing_width_link_ids,
            exclusion_reason=(
                "closed_two_link_replacement"
                if excluded_routing_width_link_ids
                else ""
            ),
        )
        from network_variants.width_trim_selection import build_parent_child_width_diagnostics

        parent_child_diagnostics = build_parent_child_width_diagnostics(enriched, parent_links)
        if not parent_child_diagnostics.empty:
            diagnostic_columns = [
                column for column in parent_child_diagnostics.columns
                if column != "id_link" and column not in enriched.columns
            ]
            enriched = enriched.merge(
                parent_child_diagnostics[["id_link", *diagnostic_columns]],
                on="id_link",
                how="left",
                validate="1:1",
            )
            if not routing_audit.empty:
                routing_audit = routing_audit.merge(
                    parent_child_diagnostics,
                    on="id_link",
                    how="left",
                    validate="1:1",
                )
        routing_paths = write_routing_width_audit(
            routing_audit,
            routing_summary,
            output_dir=width_dir,
        )
    enriched.loc[
        enriched["id_link"].astype(int).isin(measured_set), "width_inheritance_status"
    ] = "changed_evaluated_local_patch"
    enriched = _normalize_width_boolean_fields(enriched)

    links_path = width_dir / "links_with_width_families.gpkg"
    families_path = width_dir / "link_width_families.csv"
    samples_path = width_dir / "link_width_samples.csv"
    lineage_path = width_dir / "local_patch_product_lineage.csv"
    enriched.to_file(links_path, driver="GPKG")
    pd.DataFrame(enriched.drop(columns=enriched.geometry.name)).to_csv(families_path, index=False)
    samples.to_csv(samples_path, index=False)
    lineage.to_csv(lineage_path, index=False)

    output_paths = {
        "links_with_width_families": str(links_path.resolve()),
        "link_width_families": str(families_path.resolve()),
        "link_width_samples": str(samples_path.resolve()),
        "product_lineage": str(lineage_path.resolve()),
        "local_state_bank_labels": str(bank_labels_path.resolve()),
        **routing_paths,
    }
    manifest_path = output_dir / "local_patch_products_manifest.json"
    output_paths["manifest"] = str(manifest_path.resolve())
    summary = {
        "example_id": str(example_id),
        "variant_id": str(variant_id),
        "link_count": int(len(enriched)),
        "inherited_link_count": int(inheritance["inherited_link_count"]),
        "recalculated_link_count": int(len(measured_set)),
        "recalculated_link_ids": sorted(measured_set),
        "local_pixel_sample_counts": {
            str(key): int(value) for key, value in sorted(pixel_sample_counts.items())
        },
        "removed_bank_component_labels": removed_bank_labels,
        "width_product_inheritance": inheritance,
        "length_product_inheritance": {
            "inherited_link_count": int(len(inherited_parent_map)),
            "inherited_fields": inherited_length_fields,
            "parent_link_by_child_link": {
                str(key): int(value)
                for key, value in sorted(inherited_parent_map.items())
            },
        },
        "segmented_stub_width_inheritance": segmented_summary,
        "routing_width_application": routing_summary,
        "graph_product_stage": (
            "shared_scientific_anchored"
            if shared_anchored
            else "legacy_local_scientific_merge"
        ),
    }
    result = LocalPatchProductResult(
        status="complete_width_length_products",
        output_paths=output_paths,
        summary=summary,
    )
    manifest_path.write_text(
        json.dumps(result.to_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


__all__ = [
    "LOCAL_PATCH_PRODUCTS_VERSION",
    "LocalPatchProductResult",
    "materialize_local_patch_width_length_products",
]
