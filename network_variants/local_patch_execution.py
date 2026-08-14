"""Production adapter for bounded local-patch child-state generation.

The phase-specific local-patch modules intentionally write diagnostic products
to a scratch workspace.  This adapter runs those phases and publishes the
result in the same directory contract as :func:`generate_network_variant`, so
the existing hierarchy and hydraulic stages can consume it without a separate
code path.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window, transform as window_transform

from network_variants.local_patch import (
    plan_local_patch,
    write_local_patch_planning_result,
)
from network_variants.local_patch_merge import merge_local_patch_variant
from network_variants.local_patch_products import materialize_local_patch_width_length_products
from network_variants.local_patch_raster import (
    build_local_patch_base_raster,
    write_local_patch_raster_result,
)
from network_variants.local_patch_variant import materialize_local_patch_variant
from network_variants.variant_generation import (
    NetworkVariantOutputs,
    _apply_collapsed_selection_metadata,
    _infer_parent_node_order,
    _match_child_links_to_parent_links,
    _match_child_nodes_to_parent_nodes,
    _resolve_link_lineage,
)


LOCAL_PATCH_EXECUTION_VERSION = "topology-local-patch-execution-v1"


class LocalPatchFallbackRequested(RuntimeError):
    """A safe, audited request to rerun one child with the full raster."""

    def __init__(self, *, stage: str, code: str, reason: str, audit_path: Path) -> None:
        self.stage = str(stage)
        self.code = str(code)
        self.reason = str(reason)
        self.audit_path = Path(audit_path)
        super().__init__(f"{self.stage}: {self.code}: {self.reason}")


def _variant_dir_from_parent_links(path: str | Path) -> Path:
    path = Path(path).resolve()
    if path.parent.name in {"widths", "directed"}:
        return path.parent.parent
    raise ValueError(
        "Local-patch execution requires a canonical parent variant links path "
        f"under widths/ or directed/: {path}"
    )


def _one_path(directory: Path, pattern: str, *, label: str) -> Path:
    matches = sorted(path for path in directory.glob(pattern) if not path.name.startswith("._"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {label} matching {pattern!r} in {directory}, found {len(matches)}."
        )
    return matches[0]


def _copy_raster_with_local_edit(
    *,
    parent_mask_path: str | Path,
    local_base_mask_path: str | Path,
    local_collapsed_mask_path: str | Path,
    local_origin_mask_path: str | Path,
    cut_barrier_path: str | Path,
    collapsed_output_path: str | Path,
    origin_output_path: str | Path,
) -> tuple[Path, Path, int]:
    """Stream a true collapse edit into a full raster without copying crop cuts."""
    collapsed_output_path = Path(collapsed_output_path)
    origin_output_path = Path(origin_output_path)
    collapsed_output_path.parent.mkdir(parents=True, exist_ok=True)
    origin_output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(parent_mask_path) as parent, rasterio.open(local_base_mask_path) as base, \
        rasterio.open(local_collapsed_mask_path) as collapsed, \
        rasterio.open(local_origin_mask_path) as origin, rasterio.open(cut_barrier_path) as barrier:
        local_sources = (base, collapsed, origin, barrier)
        if any(source.shape != base.shape or source.transform != base.transform for source in local_sources):
            raise ValueError("Local-patch mask products are not aligned.")
        if any(source.crs != parent.crs for source in local_sources):
            raise ValueError("Local-patch and parent masks do not share a CRS.")
        # ``DatasetReader.index`` floors fractional indices. At large projected
        # coordinates, an exactly aligned origin can evaluate as e.g.
        # 832.9999999999975 and be shifted one pixel left. Resolve the affine
        # offset directly, round it, and verify the reconstructed transform.
        col_offset_float, row_offset_float = (~parent.transform) * (
            base.transform.c,
            base.transform.f,
        )
        col_offset = int(round(float(col_offset_float)))
        row_offset = int(round(float(row_offset_float)))
        if (
            abs(float(col_offset_float) - col_offset) > 1e-6
            or abs(float(row_offset_float) - row_offset) > 1e-6
        ):
            raise ValueError(
                "The local-patch raster origin is not aligned to the parent pixel grid."
            )
        window = Window(
            col_off=col_offset,
            row_off=row_offset,
            width=int(base.width),
            height=int(base.height),
        )
        reconstructed_transform = window_transform(window, parent.transform)
        if not np.allclose(
            tuple(reconstructed_transform),
            tuple(base.transform),
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError(
                "The rounded local-patch window does not reproduce its raster transform."
            )
        if (
            window.col_off < 0
            or window.row_off < 0
            or window.col_off + window.width > parent.width
            or window.row_off + window.height > parent.height
        ):
            raise ValueError("The local-patch raster window lies outside the parent mask.")

        profile = parent.profile.copy()
        profile.update(count=1, compress="lzw")
        origin_profile = profile.copy()
        origin_profile.update(dtype="int32", nodata=0)
        with rasterio.open(collapsed_output_path, "w", **profile) as collapsed_out, \
            rasterio.open(origin_output_path, "w", **origin_profile) as origin_out:
            for _, block_window in parent.block_windows(1):
                values = parent.read(1, window=block_window)
                collapsed_out.write(values, 1, window=block_window)
                origin_out.write((values > 0).astype(np.int32), 1, window=block_window)

            base_values = base.read(1)
            collapsed_values = collapsed.read(1)
            origin_values = origin.read(1).astype(np.int32)
            cut_values = barrier.read(1) > 0
            changed = (base_values != collapsed_values) & ~cut_values
            parent_window = parent.read(1, window=window)
            full_collapsed_window = parent_window.copy()
            full_collapsed_window[changed] = collapsed_values[changed]
            full_origin_window = (parent_window > 0).astype(np.int32)
            full_origin_window[changed] = origin_values[changed]
            collapsed_out.write(full_collapsed_window, 1, window=window)
            origin_out.write(full_origin_window, 1, window=window)
    return collapsed_output_path, origin_output_path, int(np.count_nonzero(changed))


def _write_full_bank_labels(
    *,
    boundary_labels_path: str | Path,
    removed_labels: Sequence[int],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    removed = np.asarray(sorted(set(int(value) for value in removed_labels)), dtype=np.int64)
    with rasterio.open(boundary_labels_path) as source:
        profile = source.profile.copy()
        profile.update(dtype="int32", count=1, nodata=0, compress="lzw")
        with rasterio.open(output_path, "w", **profile) as destination:
            for _, window in source.block_windows(1):
                values = source.read(1, window=window).astype(np.int32)
                if removed.size:
                    values[np.isin(values, removed)] = 0
                destination.write(values, 1, window=window)
    return output_path


def _annotate(frame: pd.DataFrame | gpd.GeoDataFrame, unit_ids: Sequence[int]):
    return _apply_collapsed_selection_metadata(frame, unit_ids=unit_ids, group_label=None)


def materialize_local_patch_network_variant(
    *,
    cleaned_mask_path: str | Path,
    reviewed_links_path: str | Path,
    reviewed_nodes_path: str | Path,
    exit_sides: str,
    unit_ids: Sequence[int] | None,
    variant_id: str,
    example_id: str,
    output_dir: str | Path,
    selection_type: str | None = None,
    compound_bubble_id: int | None = None,
    collapse_corridor_mode: str = "anchored",
    boundary_component_labels_path: str | Path | None = None,
    preferred_width_field: str = "wid_adj_routing",
    footprint_buffer_scale: float = 0.5,
    all_touched: bool = True,
    allow_noop: bool = False,
    single_thread: bool = False,
    verbose_rivgraph: bool = False,
    max_path_cutoff: int = 100,
    max_paths: int = 5000,
    local_patch_reference_centerline_path: str | Path | None = None,
    **_: Any,
) -> NetworkVariantOutputs:
    """Build one canonical child state using the validated local-patch phases."""
    if collapse_corridor_mode != "anchored":
        raise ValueError("Local-patch execution currently supports anchored child states only.")
    selected_ids = tuple(int(value) for value in (unit_ids or ()))
    if not selected_ids:
        raise ValueError("Local-patch execution requires explicit unit_ids.")
    if boundary_component_labels_path is None:
        raise ValueError("Local-patch execution requires global boundary-component labels.")

    output_dir = Path(output_dir)
    attempt_dir = output_dir / "local_patch_attempt"
    parent_variant_dir = _variant_dir_from_parent_links(reviewed_links_path)
    parent_state_dir = parent_variant_dir.parent
    hierarchy_dir = parent_state_dir / "hierarchy"
    membership_path = parent_variant_dir / "summary" / "unit_link_membership.csv"
    unit_summary_path = hierarchy_dir / "unit_summary.csv"
    bubble_path = hierarchy_dir / "bubble_edge_resistance.csv"
    parent_samples_path = parent_variant_dir / "widths" / "link_width_samples.csv"
    if local_patch_reference_centerline_path is not None:
        reference_centerline = Path(local_patch_reference_centerline_path).resolve()
        if not reference_centerline.exists():
            raise FileNotFoundError(
                f"Local-patch reference centerline does not exist: {reference_centerline}"
            )
    else:
        reference_centerline = _one_path(
            parent_variant_dir / "rivgraph", "*centerline.gpkg", label="reference centerline"
        )
    required = (membership_path, unit_summary_path, bubble_path, parent_samples_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Local-patch parent artifacts are missing: " + ", ".join(missing))

    parent_links = gpd.read_file(reviewed_links_path)
    parent_nodes = gpd.read_file(reviewed_nodes_path)
    planning = plan_local_patch(
        example_id=str(example_id),
        full_links=parent_links,
        unit_summary=pd.read_csv(unit_summary_path),
        unit_link_membership=pd.read_csv(membership_path),
        bubble_edge_resistance=pd.read_csv(bubble_path),
        selected_unit_ids=selected_ids,
        selection_type=str(selection_type or "unit"),
        selected_compound_bubble_id=compound_bubble_id,
    )
    planning_path = write_local_patch_planning_result(
        planning, attempt_dir / "planning" / "local_patch_planning_result.json"
    )
    if planning.contract is None:
        raise LocalPatchFallbackRequested(
            stage="planning",
            code=str(planning.fallback_code or "unsafe_plan"),
            reason=str(planning.fallback_reason or "The local patch was not safe."),
            audit_path=planning_path,
        )
    contract = planning.contract
    raster_result = build_local_patch_base_raster(
        contract=contract,
        full_links=parent_links,
        full_nodes=parent_nodes,
        cleaned_mask_path=cleaned_mask_path,
        boundary_component_labels_path=boundary_component_labels_path,
        output_dir=attempt_dir / "raster",
        reference_centerline_path=reference_centerline,
    )
    if not raster_result.is_safe:
        audit_path = Path(raster_result.output_paths.get("manifest", attempt_dir / "raster" / "local_patch_raster_plan.json"))
        if "manifest" not in raster_result.output_paths:
            write_local_patch_raster_result(raster_result, audit_path)
        raise LocalPatchFallbackRequested(
            stage="raster",
            code="unsafe_crop",
            reason=str(raster_result.fallback_reason or "No safe local crop was found."),
            audit_path=audit_path,
        )
    raw_variant = materialize_local_patch_variant(
        contract=contract,
        raster_result=raster_result,
        reviewed_links_path=reviewed_links_path,
        reviewed_nodes_path=reviewed_nodes_path,
        exit_sides=exit_sides,
        output_dir=attempt_dir / "variant",
        preferred_width_field=preferred_width_field,
        footprint_buffer_scale=footprint_buffer_scale,
        all_touched=all_touched,
        allow_noop=allow_noop,
        single_thread=single_thread,
        verbose_rivgraph=verbose_rivgraph,
        max_path_cutoff=max_path_cutoff,
        max_paths=max_paths,
    )
    merge = merge_local_patch_variant(
        contract=contract,
        raster_result=raster_result,
        variant_result=raw_variant,
        parent_links_path=reviewed_links_path,
        parent_nodes_path=reviewed_nodes_path,
        output_dir=attempt_dir / "merge",
    )
    products = materialize_local_patch_width_length_products(
        merge_result=merge,
        variant_result=raw_variant,
        parent_width_links_path=reviewed_links_path,
        parent_nodes_path=reviewed_nodes_path,
        parent_width_samples_path=parent_samples_path,
        output_dir=attempt_dir / "products",
        example_id=str(example_id),
        variant_id=str(variant_id),
        collapse_corridor_mode=collapse_corridor_mode,
    )

    summary_dir = output_dir / "summary"
    mask_dir = output_dir / "mask"
    rivgraph_dir = output_dir / "rivgraph"
    matching_dir = output_dir / "matching"
    directed_dir = output_dir / "directed"
    widths_dir = output_dir / "widths"
    for directory in (summary_dir, mask_dir, rivgraph_dir, matching_dir, directed_dir, widths_dir):
        directory.mkdir(parents=True, exist_ok=True)

    name = f"{example_id}__{variant_id}"
    collapsed_path, origin_path, changed_pixels = _copy_raster_with_local_edit(
        parent_mask_path=cleaned_mask_path,
        local_base_mask_path=raw_variant.output_paths["local_base_mask"],
        local_collapsed_mask_path=raw_variant.output_paths["local_collapsed_mask"],
        local_origin_mask_path=raw_variant.output_paths["local_origin_mask"],
        cut_barrier_path=raw_variant.output_paths["main_bank_cut_barrier"],
        collapsed_output_path=mask_dir / f"{name}_collapsed.tif",
        origin_output_path=mask_dir / f"{name}_origin.tif",
    )
    removed_labels = products.summary.get("removed_bank_component_labels", [])
    bank_labels_path = _write_full_bank_labels(
        boundary_labels_path=boundary_component_labels_path,
        removed_labels=removed_labels,
        output_path=mask_dir / f"{name}_bank_labels.tif",
    )

    collapse_components = _annotate(pd.read_csv(raw_variant.output_paths["collapse_components"]), selected_ids)
    collapse_components_path = summary_dir / "collapse_components.csv"
    collapse_components.to_csv(collapse_components_path, index=False)
    edit_geometries = gpd.GeoDataFrame(geometry=[], crs=parent_links.crs)
    if "collapse_edit_geometries" in raw_variant.output_paths:
        edit_geometries = _annotate(gpd.read_file(raw_variant.output_paths["collapse_edit_geometries"]), selected_ids)
        edit_geometries.to_file(mask_dir / "collapse_edit_geometries.gpkg", driver="GPKG")

    enriched_links = _annotate(gpd.read_file(products.output_paths["links_with_width_families"]), selected_ids)
    directed_nodes = _annotate(gpd.read_file(merge.output_paths["merged_nodes"]), selected_ids)
    directed_links = enriched_links.copy()
    directed_links_path = directed_dir / f"{name}_directed_links.gpkg"
    directed_nodes_path = directed_dir / f"{name}_directed_nodes.gpkg"
    enriched_links_path = widths_dir / "links_with_width_families.gpkg"
    directed_links.to_file(directed_links_path, driver="GPKG")
    directed_nodes.to_file(directed_nodes_path, driver="GPKG")
    enriched_links.to_file(enriched_links_path, driver="GPKG")
    samples = _annotate(pd.read_csv(products.output_paths["link_width_samples"]), selected_ids)
    samples.to_csv(widths_dir / "link_width_samples.csv", index=False)
    families = pd.DataFrame(enriched_links.drop(columns=enriched_links.geometry.name))
    families.to_csv(widths_dir / "link_width_families.csv", index=False)
    routing_audit_value = products.output_paths.get("link_audit")
    if routing_audit_value and Path(routing_audit_value).is_file():
        routing_audit_source = Path(routing_audit_value)
        shutil.copy2(routing_audit_source, widths_dir / "routing_width_application.csv")
        routing_manifest_source = routing_audit_source.with_name(
            "routing_width_application_manifest.json"
        )
        if routing_manifest_source.is_file():
            shutil.copy2(
                routing_manifest_source,
                widths_dir / "routing_width_application_manifest.json",
            )

    parent_order, parent_graph, _ = _infer_parent_node_order(parent_links, parent_nodes)
    with rasterio.open(cleaned_mask_path) as parent_mask:
        lineage_tolerance = float(max(abs(parent_mask.transform.a), abs(parent_mask.transform.e))) * 1.25
    node_match = _match_child_nodes_to_parent_nodes(
        parent_nodes,
        directed_nodes,
        match_tolerance=lineage_tolerance,
        parent_node_order=parent_order,
    )
    link_match = _match_child_links_to_parent_links(
        parent_links,
        directed_links,
        match_tolerance=lineage_tolerance,
    )
    link_lineage = _resolve_link_lineage(
        parent_graph=parent_graph,
        directed_child_links=directed_links,
        node_match=node_match,
        link_match=link_match,
        match_tolerance=lineage_tolerance,
    )
    node_match = _annotate(node_match, selected_ids)
    link_match = _annotate(link_match, selected_ids)
    link_lineage = _annotate(link_lineage, selected_ids)
    node_match.to_csv(matching_dir / "node_match.csv", index=False)
    link_match.to_csv(matching_dir / "link_match.csv", index=False)
    link_lineage.to_csv(matching_dir / "link_lineage.csv", index=False)
    node_sword_columns = [column for column in directed_nodes.columns if column != directed_nodes.geometry.name and (column == "id_node" or column.startswith("sword_"))]
    node_sword_match = pd.DataFrame(directed_nodes[node_sword_columns]).rename(columns={"id_node": "child_id_node"})
    node_sword_match.to_csv(matching_dir / "node_sword_match.csv", index=False)

    direction_report = dict(merge.summary.get("validation_report", {}))
    (directed_dir / "direction_validation_report.json").write_text(
        json.dumps(direction_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    raw_links = Path(raw_variant.output_paths["raw_rivgraph_links"])
    raw_nodes = Path(raw_variant.output_paths["raw_rivgraph_nodes"])
    canonical_raw_links = rivgraph_dir / raw_links.name
    canonical_raw_nodes = rivgraph_dir / raw_nodes.name
    shutil.copy2(raw_links, canonical_raw_links)
    shutil.copy2(raw_nodes, canonical_raw_nodes)
    raw_centerline_value = raw_variant.output_paths.get("raw_rivgraph_centerline")
    canonical_centerline = None
    if raw_centerline_value:
        raw_centerline = Path(raw_centerline_value)
        canonical_centerline = rivgraph_dir / raw_centerline.name
        shutil.copy2(raw_centerline, canonical_centerline)

    manifest = {
        "example_id": str(example_id),
        "variant_id": str(variant_id),
        "selection_type": selection_type,
        "compound_bubble_id": compound_bubble_id,
        "selected_unit_ids": list(selected_ids),
        "collapse_corridor_mode": collapse_corridor_mode,
        "execution_mode_requested": "local_patch",
        "execution_mode_used": "local_patch",
        "execution_version": LOCAL_PATCH_EXECUTION_VERSION,
        "ready_for_graph_matching": True,
        "ready_for_rapid": bool(direction_report.get("is_valid", True)),
        "local_patch": {
            "planning": planning.to_manifest(),
            "raster": raster_result.to_manifest(),
            "merge": merge.to_manifest(),
            "products": products.to_manifest(),
            "true_collapsed_pixel_count": changed_pixels,
            "attempt_dir": str(attempt_dir.resolve()),
        },
        "output_paths": {
            "output_dir": str(output_dir.resolve()),
            "collapsed_mask": str(collapsed_path.resolve()),
            "origin_mask": str(origin_path.resolve()),
            "width_scope_mask": str(origin_path.resolve()),
            "bank_labels": str(bank_labels_path.resolve()),
            "boundary_component_labels": str(Path(boundary_component_labels_path).resolve()),
            "collapse_components": str(collapse_components_path.resolve()),
            "link_width_families": str((widths_dir / "link_width_families.csv").resolve()),
            "link_width_samples": str((widths_dir / "link_width_samples.csv").resolve()),
            "links_with_width_families": str(enriched_links_path.resolve()),
            "node_match": str((matching_dir / "node_match.csv").resolve()),
            "node_sword_match": str((matching_dir / "node_sword_match.csv").resolve()),
            "link_match": str((matching_dir / "link_match.csv").resolve()),
            "link_lineage": str((matching_dir / "link_lineage.csv").resolve()),
            "directed_links": str(directed_links_path.resolve()),
            "directed_nodes": str(directed_nodes_path.resolve()),
            "rivgraph_links": str(canonical_raw_links.resolve()),
            "rivgraph_nodes": str(canonical_raw_nodes.resolve()),
            "rivgraph_centerline": str(canonical_centerline.resolve()) if canonical_centerline else None,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (summary_dir / "variant_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return NetworkVariantOutputs(
        collapse_components=collapse_components,
        edit_geometries=edit_geometries,
        node_match=node_match,
        node_sword_match=node_sword_match,
        link_match=link_match,
        link_lineage=link_lineage,
        link_width_families=families,
        link_width_samples=samples,
        enriched_links=enriched_links,
        directed_links=directed_links,
        directed_nodes=directed_nodes,
        output_dir=output_dir,
        collapsed_mask_path=collapsed_path,
        width_scope_mask_path=origin_path,
        bank_labels_path=bank_labels_path,
        boundary_component_labels_path=Path(boundary_component_labels_path),
        rivgraph_links_path=canonical_raw_links,
        rivgraph_nodes_path=canonical_raw_nodes,
        directed_links_path=directed_links_path,
        directed_nodes_path=directed_nodes_path,
        rivgraph_centerline_path=canonical_centerline,
        sword_reaches_path=None,
        sword_nodes_path=None,
        manifest=manifest,
        origin_mask_path=origin_path,
        rivgraph_distance_path=None,
    )


__all__ = [
    "LOCAL_PATCH_EXECUTION_VERSION",
    "LocalPatchFallbackRequested",
    "materialize_local_patch_network_variant",
]
