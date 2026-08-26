"""Production adapter for bounded local-patch child-state generation.

The phase-specific local-patch modules intentionally write diagnostic products
to a scratch workspace.  This adapter runs those phases and publishes the
result in the same directory contract as :func:`generate_network_variant`, so
the existing hierarchy and hydraulic stages can consume it without a separate
code path.
"""

from __future__ import annotations

from dataclasses import replace
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

from hierarchy_level_definition.unit_detection.bifurcation_confluence_units import (
    StructuralUnit,
)
from network_variants.local_patch import (
    plan_local_patch,
    write_local_patch_planning_result,
)
from network_variants.local_patch_anchored import anchor_local_patch_realized_graph
from network_variants.local_patch_merge import (
    _expected_cycle_reduction,
    _undirected_cycle_rank,
)
from network_variants.local_patch_products import materialize_local_patch_width_length_products
from network_variants.local_patch_realized import assemble_local_patch_realized_graph
from network_variants.local_patch_raster import (
    build_local_patch_base_raster,
    write_local_patch_raster_result,
)
from network_variants.local_patch_variant import (
    derive_local_exit_side_candidates,
    materialize_local_patch_variant,
)
from network_variants.virtual_raster import write_overlay_vrt
from network_variants.variant_generation import (
    NetworkVariantOutputs,
    _apply_collapsed_selection_metadata,
    _infer_parent_node_order,
    _match_child_links_to_parent_links,
    _match_child_nodes_to_parent_nodes,
    _resolve_link_lineage,
)


LOCAL_PATCH_EXECUTION_VERSION = "topology-local-patch-execution-v11"

COLLAR_CONTEXT_RETRY_DISTANCES_M = (1500.0, 2000.0)


def _retryable_local_exit_failure(error: Exception) -> bool:
    reason = str(error)
    return any(
        marker in reason
        for marker in (
            "Raw local RivGraph",
            "anchored boundary node",
            "Anchored local boundaries are not connected",
        )
    )


def _retryable_collar_context_failure(error: Exception) -> bool:
    reason = str(error)
    return any(
        marker in reason
        for marker in (
            "No local link or node lies within",
            "Raw local RivGraph has no unambiguous",
        )
    )


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


def _write_virtual_rasters_with_local_edit(
    *,
    parent_mask_path: str | Path,
    local_base_mask_path: str | Path,
    local_collapsed_mask_path: str | Path,
    local_origin_mask_path: str | Path,
    cut_barrier_path: str | Path,
    output_dir: str | Path,
    name: str,
) -> tuple[Path, Path, int, Path]:
    """Publish full-extent VRT masks backed by one parent and one local patch."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    collapsed_patch_path = output_dir / f"{name}_collapsed_patch.tif"
    origin_patch_path = output_dir / f"{name}_origin_patch.tif"
    collapsed_vrt_path = output_dir / f"{name}_collapsed.vrt"
    origin_vrt_path = output_dir / f"{name}_origin.vrt"

    with rasterio.open(parent_mask_path) as parent, rasterio.open(local_base_mask_path) as base, \
        rasterio.open(local_collapsed_mask_path) as collapsed, \
        rasterio.open(local_origin_mask_path) as origin, rasterio.open(cut_barrier_path) as barrier:
        local_sources = (base, collapsed, origin, barrier)
        if any(source.shape != base.shape or source.transform != base.transform for source in local_sources):
            raise ValueError("Local-patch mask products are not aligned.")
        if any(source.crs != parent.crs for source in local_sources):
            raise ValueError("Local-patch and parent masks do not share a CRS.")
        col_float, row_float = (~parent.transform) * (base.transform.c, base.transform.f)
        col_off = int(round(float(col_float)))
        row_off = int(round(float(row_float)))
        if abs(float(col_float) - col_off) > 1e-6 or abs(float(row_float) - row_off) > 1e-6:
            raise ValueError("The local-patch raster origin is not aligned to the parent grid.")
        window = Window(col_off, row_off, int(base.width), int(base.height))
        if (
            window.col_off < 0
            or window.row_off < 0
            or window.col_off + window.width > parent.width
            or window.row_off + window.height > parent.height
        ):
            raise ValueError("The local-patch raster window lies outside the parent mask.")
        if not np.allclose(
            tuple(window_transform(window, parent.transform)),
            tuple(base.transform),
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError("The local-patch window does not reproduce its raster transform.")

        base_values = base.read(1)
        collapsed_values = collapsed.read(1)
        origin_values = origin.read(1).astype(np.int32)
        cut_values = barrier.read(1) > 0
        changed = (base_values != collapsed_values) & ~cut_values
        parent_window = parent.read(1, window=window)
        collapsed_patch = parent_window.copy()
        collapsed_patch[changed] = collapsed_values[changed]
        origin_patch = (parent_window > 0).astype(np.int32)
        origin_patch[changed] = origin_values[changed]

        patch_profile = parent.profile.copy()
        for key in ("blockxsize", "blockysize", "tiled"):
            patch_profile.pop(key, None)
        patch_profile.update(
            width=int(base.width),
            height=int(base.height),
            transform=base.transform,
            count=1,
            compress="lzw",
            tiled=False,
            # Zero is a real collapse edit, not transparency.  The overlay is
            # already bounded by its VRT destination window, so inheriting the
            # parent mask's zero nodata value would silently reveal the parent
            # water pixel wherever the patch deliberately writes zero.
            nodata=None,
        )
        with rasterio.open(collapsed_patch_path, "w", **patch_profile) as destination:
            destination.write(collapsed_patch, 1)
        origin_profile = patch_profile.copy()
        origin_profile.update(dtype="int32", nodata=0)
        with rasterio.open(origin_patch_path, "w", **origin_profile) as destination:
            destination.write(origin_patch, 1)

    collapsed_overlay = write_overlay_vrt(
        base_path=parent_mask_path,
        overlay_path=collapsed_patch_path,
        output_path=collapsed_vrt_path,
    )
    origin_overlay = write_overlay_vrt(
        base_path=parent_mask_path,
        overlay_path=origin_patch_path,
        output_path=origin_vrt_path,
        output_dtype="int32",
    )
    manifest_path = output_dir / "virtual_mask_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "local-patch-virtual-mask-v1",
                "publication_mode": "virtual_patch",
                "changed_pixels": int(np.count_nonzero(changed)),
                "collapsed": collapsed_overlay.to_manifest(),
                "origin": origin_overlay.to_manifest(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return collapsed_vrt_path, origin_vrt_path, int(np.count_nonzero(changed)), manifest_path


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


def _selected_units_from_hierarchy_products(
    *,
    selected_unit_ids: Sequence[int],
    unit_summary: pd.DataFrame,
    unit_link_membership: pd.DataFrame,
    parent_links: gpd.GeoDataFrame,
) -> dict[int, StructuralUnit]:
    """Rebuild the graph fields needed by shared scientific anchoring.

    The base hierarchy already stores each selected unit's link membership and
    summary. Reconstructing only those units avoids rerunning full unit
    detection for every local child while retaining the exact parent edge IDs
    used by the original analysis.
    """
    summary = unit_summary.copy()
    summary["unit_id"] = pd.to_numeric(summary["unit_id"], errors="raise").astype(int)
    membership = unit_link_membership.copy()
    membership["unit_id"] = pd.to_numeric(
        membership["unit_id"], errors="raise"
    ).astype(int)
    membership["id_link"] = pd.to_numeric(
        membership["id_link"], errors="raise"
    ).astype(int)
    links = parent_links.copy()
    links["id_link"] = pd.to_numeric(links["id_link"], errors="raise").astype(int)
    link_index = links.set_index("id_link", drop=False)

    result: dict[int, StructuralUnit] = {}
    for unit_id in tuple(dict.fromkeys(int(value) for value in selected_unit_ids)):
        rows = summary.loc[summary["unit_id"].eq(unit_id)]
        if len(rows) != 1:
            raise ValueError(
                f"Expected one hierarchy summary row for selected unit {unit_id}, "
                f"found {len(rows)}."
            )
        row = rows.iloc[0]
        link_ids = tuple(
            sorted(
                membership.loc[membership["unit_id"].eq(unit_id), "id_link"]
                .drop_duplicates()
                .astype(int)
            )
        )
        missing_link_ids = sorted(set(link_ids).difference(link_index.index.astype(int)))
        if not link_ids or missing_link_ids:
            raise ValueError(
                f"Selected unit {unit_id} has invalid base link membership; "
                f"links={list(link_ids)}, missing={missing_link_ids}."
            )
        edge_set = {
            (
                int(link_index.loc[link_id, "id_us_node"]),
                int(link_index.loc[link_id, "id_ds_node"]),
                int(link_id),
            )
            for link_id in link_ids
        }
        node_set = {
            int(node_id)
            for upstream, downstream, _ in edge_set
            for node_id in (upstream, downstream)
        }
        result[unit_id] = StructuralUnit(
            unit_id=int(unit_id),
            bifurcation=int(row["bifurcation"]),
            confluence=int(row["confluence"]),
            n_paths=int(row.get("n_paths", 0)),
            unit_class=str(row.get("class", "")),
            min_path_length=float(row.get("min_path_length", 0.0)),
            max_path_length=float(row.get("max_path_length", 0.0)),
            node_set=node_set,
            edge_set=edge_set,
        )
    return result


def _materialize_local_patch_network_variant_once(
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
    local_patch_mask_publication: str = "materialized",
    computational_context_unit_id: int | None = None,
    local_patch_attempt_subdirectory: str | None = None,
    exclude_closed_two_link_replacement: bool = False,
    minimum_collar_context_m_override: float | None = None,
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
    if local_patch_mask_publication not in {"materialized", "virtual_patch"}:
        raise ValueError(
            "Local-patch mask publication must be 'materialized' or 'virtual_patch'."
        )

    output_dir = Path(output_dir)
    attempt_root = output_dir / "local_patch_attempt"
    attempt_dir = attempt_root
    if local_patch_attempt_subdirectory:
        attempt_dir = attempt_root / "outer_domain_retries" / str(
            local_patch_attempt_subdirectory
        )
    # A rerun may reuse the same state directory. Invalidate success/fallback
    # markers before doing any work so a failed retry cannot be mistaken for
    # the result of an earlier attempt.
    for stale_path in (
        output_dir / "summary" / "variant_manifest.json",
        attempt_dir / "fallback_to_full_raster.json",
        attempt_dir / "fallback_to_panelized.json",
        attempt_dir / "exit_side_retry_summary.json",
    ):
        stale_path.unlink(missing_ok=True)
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
    unit_summary = pd.read_csv(unit_summary_path)
    unit_link_membership = pd.read_csv(membership_path)
    selected_units_by_id = _selected_units_from_hierarchy_products(
        selected_unit_ids=selected_ids,
        unit_summary=unit_summary,
        unit_link_membership=unit_link_membership,
        parent_links=parent_links,
    )
    planning = plan_local_patch(
        example_id=str(example_id),
        full_links=parent_links,
        unit_summary=unit_summary,
        unit_link_membership=unit_link_membership,
        bubble_edge_resistance=pd.read_csv(bubble_path),
        selected_unit_ids=selected_ids,
        selection_type=str(selection_type or "unit"),
        selected_compound_bubble_id=compound_bubble_id,
        computational_context_unit_id=computational_context_unit_id,
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
    if minimum_collar_context_m_override is not None:
        requested_context = float(minimum_collar_context_m_override)
        if requested_context <= 0:
            raise ValueError("minimum_collar_context_m_override must be positive.")
        contract = replace(
            contract,
            raster_policy=replace(
                contract.raster_policy,
                minimum_collar_context_m=max(
                    float(contract.raster_policy.minimum_collar_context_m),
                    requested_context,
                ),
            ),
        )
        planning = replace(planning, contract=contract)
        planning_path = write_local_patch_planning_result(
            planning, attempt_dir / "planning" / "local_patch_planning_result.json"
        )
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
    exit_side_candidates = derive_local_exit_side_candidates(raster_result)
    exit_side_attempts: list[dict[str, Any]] = []
    raw_variant = None
    realized = None
    anchored = None
    first_retryable_error: Exception | None = None
    for attempt_index, candidate_exit_sides in enumerate(exit_side_candidates):
        candidate_root = (
            attempt_dir
            if attempt_index == 0
            else attempt_dir / "exit_side_retries" / candidate_exit_sides
        )
        # RivGraph derives output names from the selected run name. Reusing a
        # candidate directory across reruns or progressively larger outer
        # domains can otherwise mix skeletons, transforms, and vector files
        # from different masks. These directories contain reproducible attempt
        # products only; planning/raster inputs remain untouched.
        for generated_dir in (
            candidate_root / "variant",
            candidate_root / "realized",
            candidate_root / "anchored",
        ):
            if generated_dir.exists():
                shutil.rmtree(generated_dir)
        try:
            candidate_variant = materialize_local_patch_variant(
                contract=contract,
                raster_result=raster_result,
                reviewed_links_path=reviewed_links_path,
                reviewed_nodes_path=reviewed_nodes_path,
                exit_sides=exit_sides,
                output_dir=candidate_root / "variant",
                preferred_width_field=preferred_width_field,
                footprint_buffer_scale=footprint_buffer_scale,
                all_touched=all_touched,
                allow_noop=allow_noop,
                single_thread=single_thread,
                verbose_rivgraph=verbose_rivgraph,
                max_path_cutoff=max_path_cutoff,
                max_paths=max_paths,
                local_exit_sides_override=(
                    None if attempt_index == 0 else candidate_exit_sides
                ),
            )
            candidate_realized = assemble_local_patch_realized_graph(
                contract=contract,
                raster_result=raster_result,
                variant_result=candidate_variant,
                parent_links_path=reviewed_links_path,
                parent_nodes_path=reviewed_nodes_path,
                output_dir=candidate_root / "realized",
                boundary_match_tolerance_pixels=(
                    12.0
                    if minimum_collar_context_m_override is not None
                    and float(minimum_collar_context_m_override) >= 2000.0
                    else 10.0
                    if minimum_collar_context_m_override is not None
                    else 8.0
                ),
            )
            with rasterio.open(candidate_variant.output_paths["local_collapsed_mask"]) as source:
                anchoring_match_tolerance = float(
                    max(abs(source.transform.a), abs(source.transform.e))
                ) * 1.25
            candidate_anchored = anchor_local_patch_realized_graph(
                realized_result=candidate_realized,
                parent_links_path=reviewed_links_path,
                parent_nodes_path=reviewed_nodes_path,
                selected_unit_ids=selected_ids,
                units_by_id=selected_units_by_id,
                output_dir=candidate_root / "anchored",
                match_tolerance_m=anchoring_match_tolerance,
            )
        except ValueError as error:
            exit_side_attempts.append(
                {
                    "exit_sides": candidate_exit_sides,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            if not _retryable_local_exit_failure(error):
                raise
            if first_retryable_error is None:
                first_retryable_error = error
            continue
        exit_side_attempts.append(
            {
                "exit_sides": candidate_exit_sides,
                "status": "selected",
                "error_type": "",
                "error": "",
            }
        )
        raw_variant = candidate_variant
        realized = candidate_realized
        anchored = candidate_anchored
        break
    retry_summary_path = attempt_dir / "exit_side_retry_summary.json"
    retry_summary_path.write_text(
        json.dumps(
            {
                "method": "bounded_validation_driven_exit_side_retry",
                "candidate_exit_sides": list(exit_side_candidates),
                "attempts": exit_side_attempts,
                "selected_exit_sides": next(
                    (
                        row["exit_sides"]
                        for row in exit_side_attempts
                        if row["status"] == "selected"
                    ),
                    None,
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if raw_variant is None or realized is None or anchored is None:
        if first_retryable_error is not None:
            raise first_retryable_error
        raise RuntimeError("No local exit-side candidate produced a mergeable local graph.")
    final_anchored_links = gpd.read_file(anchored.output_paths["anchored_links"])
    parent_cycle_rank = _undirected_cycle_rank(parent_links)
    anchored_cycle_rank = _undirected_cycle_rank(final_anchored_links)
    observed_cycle_reduction = int(parent_cycle_rank - anchored_cycle_rank)
    expected_cycle_reduction = _expected_cycle_reduction(
        raw_variant,
        parent_links=parent_links,
        selected_link_ids=contract.selected_link_ids,
    )
    final_collapse_topology_validation = {
        "method": "final_shared_anchored_selected_parent_cycle_rank_reduction",
        "parent_cycle_rank": int(parent_cycle_rank),
        "anchored_cycle_rank": int(anchored_cycle_rank),
        "observed_cycle_reduction": int(observed_cycle_reduction),
        "expected_cycle_reduction": (
            int(expected_cycle_reduction)
            if expected_cycle_reduction is not None
            else None
        ),
        "passed": bool(
            expected_cycle_reduction is None
            or observed_cycle_reduction == int(expected_cycle_reduction)
        ),
    }
    if not final_collapse_topology_validation["passed"]:
        raise ValueError(
            "Final shared-anchored local graph is inconsistent with the raster edit: "
            f"selected-parent cycle rank={expected_cycle_reduction}, "
            f"observed cycle-rank reduction={observed_cycle_reduction}, "
            f"parent cycle rank={parent_cycle_rank}, "
            f"anchored cycle rank={anchored_cycle_rank}."
        )
    products = materialize_local_patch_width_length_products(
        merge_result=anchored,
        variant_result=raw_variant,
        parent_width_links_path=reviewed_links_path,
        parent_nodes_path=reviewed_nodes_path,
        parent_width_samples_path=parent_samples_path,
        output_dir=attempt_dir / "products",
        example_id=str(example_id),
        variant_id=str(variant_id),
        collapse_corridor_mode=collapse_corridor_mode,
        exclude_closed_two_link_replacement=exclude_closed_two_link_replacement,
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
    virtual_mask_manifest_path: Path | None = None
    if local_patch_mask_publication == "virtual_patch":
        collapsed_path, origin_path, changed_pixels, virtual_mask_manifest_path = (
            _write_virtual_rasters_with_local_edit(
                parent_mask_path=cleaned_mask_path,
                local_base_mask_path=raw_variant.output_paths["local_base_mask"],
                local_collapsed_mask_path=raw_variant.output_paths["local_collapsed_mask"],
                local_origin_mask_path=raw_variant.output_paths["local_origin_mask"],
                cut_barrier_path=raw_variant.output_paths["main_bank_cut_barrier"],
                output_dir=mask_dir,
                name=name,
            )
        )
    else:
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
    virtual_bank_overlay: dict[str, object] | None = None
    if local_patch_mask_publication == "virtual_patch":
        bank_overlay = write_overlay_vrt(
            base_path=boundary_component_labels_path,
            overlay_path=products.output_paths["local_state_bank_labels"],
            output_path=mask_dir / f"{name}_bank_labels.vrt",
            output_dtype="int32",
            base_value_overrides={int(label): 1 for label in removed_labels},
        )
        bank_labels_path = bank_overlay.vrt_path
        virtual_bank_overlay = bank_overlay.to_manifest()
    else:
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
    directed_nodes = _annotate(
        gpd.read_file(anchored.output_paths["anchored_nodes"]), selected_ids
    )
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

    direction_report = dict(
        anchored.summary.get("direction_summary", {}).get("validation_report", {})
    )
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
            "realized_assembly": realized.to_manifest(),
            "shared_anchoring": anchored.to_manifest(),
            "collapse_topology_validation": final_collapse_topology_validation,
            "products": products.to_manifest(),
            "exit_side_retry": {
                "summary_path": str(retry_summary_path.resolve()),
                "candidate_exit_sides": list(exit_side_candidates),
                "attempts": exit_side_attempts,
            },
            "true_collapsed_pixel_count": changed_pixels,
            "mask_publication": {
                "mode": local_patch_mask_publication,
                "manifest": (
                    str(virtual_mask_manifest_path.resolve())
                    if virtual_mask_manifest_path is not None
                    else None
                ),
                "bank_labels": virtual_bank_overlay,
            },
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
            "edit_geometries": str((mask_dir / "collapse_edit_geometries.gpkg").resolve()),
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


def _parse_parent_ids(value: Any) -> tuple[int, ...]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ()
    values = value if isinstance(value, (list, tuple)) else str(value).strip("[]").split(",")
    resolved: list[int] = []
    for item in values:
        try:
            resolved.append(int(float(str(item).strip())))
        except (TypeError, ValueError):
            continue
    return tuple(dict.fromkeys(resolved))


def _outer_context_candidates(
    unit_summary: pd.DataFrame,
    selected_unit_ids: Sequence[int],
) -> tuple[int, ...]:
    """Return progressively larger primary parents for one selected unit."""
    selected = tuple(dict.fromkeys(int(value) for value in selected_unit_ids))
    if len(selected) != 1:
        return ()
    by_id = unit_summary.set_index(pd.to_numeric(unit_summary["unit_id"], errors="coerce"))
    current = int(selected[0])
    candidates: list[int] = []
    visited = {current}
    while current in by_id.index:
        row = by_id.loc[current]
        primary = pd.to_numeric(pd.Series([row.get("primary_parent_id")]), errors="coerce").iloc[0]
        if pd.notna(primary):
            parent = int(primary)
        else:
            parents = _parse_parent_ids(row.get("parents"))
            parent = int(parents[0]) if parents else -1
        if parent < 0 or parent in visited:
            break
        candidates.append(parent)
        visited.add(parent)
        current = parent
    return tuple(candidates)


def _is_outer_context_retryable(error: Exception) -> bool:
    """Return whether a larger enclosing raster domain can address ``error``.

    Outer-domain retries are for insufficient crop/context only.  They cannot
    repair topology validation, shared-anchoring direction, cycle-rank, or
    hydraulic-product failures and previously obscured those root causes with
    an unrelated error at the enclosing unit's boundary.
    """
    message = str(error)
    retryable_fragments = (
        "No safe local crop",
        "No local link or node lies within",
        "Raw local RivGraph has no unambiguous",
        "Required context cut lies beyond",
        "Collar endpoints share one reference-centerline projection",
        "identical reference-centerline projections",
        "Exact anchored boundary is not reached",
    )
    return any(fragment in message for fragment in retryable_fragments)


def _write_outer_domain_retry_summary(
    *,
    output_dir: str | Path,
    selected_ids: Sequence[int],
    attempts: Sequence[Mapping[str, Any]],
    selected_context_unit_id: int | None,
) -> Path:
    summary_path = Path(output_dir) / "local_patch_attempt" / "outer_domain_retry_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "method": "progressive_enclosing_unit_context_retry",
                "selected_unit_ids": [int(value) for value in selected_ids],
                "attempts": [dict(row) for row in attempts],
                "selected_context_unit_id": (
                    int(selected_context_unit_id)
                    if selected_context_unit_id is not None
                    else None
                ),
                "root_error_preserved": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary_path


def _materialize_with_collar_context_retries(
    **kwargs: Any,
) -> NetworkVariantOutputs:
    """Retry only boundary-proximity failures with bounded longer collars."""
    attempts: list[dict[str, Any]] = []
    base_subdirectory = kwargs.get("local_patch_attempt_subdirectory")
    last_error: Exception | None = None
    for context_distance in (None, *COLLAR_CONTEXT_RETRY_DISTANCES_M):
        attempt_kwargs = dict(kwargs)
        if context_distance is not None:
            suffix = f"collar_context_retries/context_{int(context_distance)}m"
            attempt_kwargs["local_patch_attempt_subdirectory"] = (
                f"{base_subdirectory}/{suffix}" if base_subdirectory else suffix
            )
            attempt_kwargs["minimum_collar_context_m_override"] = float(context_distance)
        try:
            result = _materialize_local_patch_network_variant_once(**attempt_kwargs)
        except (LocalPatchFallbackRequested, ValueError) as error:
            last_error = error
            attempts.append(
                {
                    "minimum_collar_context_m": 1000.0 if context_distance is None else float(context_distance),
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            if not _retryable_collar_context_failure(error):
                raise
            continue
        attempts.append(
            {
                "minimum_collar_context_m": 1000.0 if context_distance is None else float(context_distance),
                "status": "selected",
                "error_type": "",
                "error": "",
            }
        )
        relative_summary = (
            Path(str(base_subdirectory)) / "collar_context_retry_summary.json"
            if base_subdirectory
            else Path("collar_context_retry_summary.json")
        )
        summary_path = Path(kwargs["output_dir"]) / "local_patch_attempt" / relative_summary
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        selected_distance = 1000.0 if context_distance is None else float(context_distance)
        summary_path.write_text(
            json.dumps(
                {
                    "method": "bounded_minimum_collar_context_retry",
                    "attempts": attempts,
                    "selected_minimum_collar_context_m": selected_distance,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        result.manifest.setdefault("local_patch", {})["collar_context_retry"] = {
            "summary_path": str(summary_path.resolve()),
            "selected_minimum_collar_context_m": selected_distance,
            "attempts": attempts,
        }
        manifest_path = Path(result.output_dir) / "summary" / "variant_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                "Successful local-patch result is missing its canonical variant manifest: "
                f"{manifest_path}"
            )
        manifest_path.write_text(
            json.dumps(result.manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result
    if last_error is not None:
        raise last_error
    raise RuntimeError("No collar-context attempt was executed.")


def materialize_local_patch_network_variant(**kwargs: Any) -> NetworkVariantOutputs:
    """Run the selected domain, then progressively larger enclosing domains.

    The retry changes computational raster context only.  The selected unit IDs
    and anchored replacement links remain identical, so an outer-domain retry
    cannot broaden the scientific collapse selection.
    """
    try:
        return _materialize_with_collar_context_retries(**kwargs)
    except (LocalPatchFallbackRequested, ValueError) as first_error:
        if not _is_outer_context_retryable(first_error):
            raise
        selected_ids = tuple(int(value) for value in (kwargs.get("unit_ids") or ()))
        if len(selected_ids) != 1:
            raise
        parent_variant_dir = _variant_dir_from_parent_links(kwargs["reviewed_links_path"])
        unit_summary_path = parent_variant_dir.parent / "hierarchy" / "unit_summary.csv"
        if not unit_summary_path.exists():
            raise
        candidates = _outer_context_candidates(pd.read_csv(unit_summary_path), selected_ids)
        if not candidates:
            raise

        attempts: list[dict[str, Any]] = [
            {
                "context_unit_id": None,
                "status": "failed",
                "error_type": type(first_error).__name__,
                "error": str(first_error),
            }
        ]
        for context_unit_id in candidates:
            try:
                result = _materialize_with_collar_context_retries(
                    **kwargs,
                    computational_context_unit_id=int(context_unit_id),
                    local_patch_attempt_subdirectory=f"context_unit_{int(context_unit_id)}",
                )
            except (LocalPatchFallbackRequested, ValueError) as error:
                attempts.append(
                    {
                        "context_unit_id": int(context_unit_id),
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                continue
            attempts.append(
                {
                    "context_unit_id": int(context_unit_id),
                    "status": "selected",
                    "error_type": "",
                    "error": "",
                }
            )
            output_dir = Path(kwargs["output_dir"])
            summary_path = _write_outer_domain_retry_summary(
                output_dir=output_dir,
                selected_ids=selected_ids,
                attempts=attempts,
                selected_context_unit_id=int(context_unit_id),
            )
            result.manifest.setdefault("local_patch", {})["outer_domain_retry"] = {
                "summary_path": str(summary_path.resolve()),
                "selected_context_unit_id": int(context_unit_id),
                "attempts": attempts,
            }
            manifest_path = output_dir / "summary" / "variant_manifest.json"
            manifest_path.write_text(
                json.dumps(result.manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return result
        _write_outer_domain_retry_summary(
            output_dir=kwargs["output_dir"],
            selected_ids=selected_ids,
            attempts=attempts,
            selected_context_unit_id=None,
        )
        raise first_error


__all__ = [
    "LOCAL_PATCH_EXECUTION_VERSION",
    "LocalPatchFallbackRequested",
    "materialize_local_patch_network_variant",
]
