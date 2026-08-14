"""Helpers for the staged RivGraph workflow."""

from __future__ import annotations

import json
import sqlite3
import shutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPoint, Point, Polygon
from shapely.geometry.base import BaseGeometry

from rivgraph_compat import rivgraph_centerline_compat
from rivgraph_centerline.manual_edits import apply_manual_edits, apply_manual_edits_to_panels
from rivgraph_centerline.raster_mask import crop_binary_mask


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "rivgraph_centerline" / "outputs" / "runs"


@dataclass(frozen=True)
class WorkflowPaths:
    """Conventional paths for a single RivGraph workflow run."""

    name: str
    run_dir: Path

    @classmethod
    def from_name(
        cls,
        name: str,
        run_dir: Path | None = None,
        *,
        output_root: Path | None = None,
    ) -> "WorkflowPaths":
        normalized = name.strip()
        if not normalized:
            raise ValueError("Run name cannot be empty.")
        resolved_output_root = DEFAULT_OUTPUT_ROOT if output_root is None else Path(output_root).expanduser().resolve()
        root = resolved_output_root / normalized if run_dir is None else Path(run_dir).expanduser().resolve()
        return cls(name=normalized, run_dir=root)

    @property
    def prepared_dir(self) -> Path:
        return self.run_dir / "masks_prepared"

    @property
    def prepared_mask(self) -> Path:
        return self.prepared_dir / f"{self.name}_binary_projected.tif"

    @property
    def manual_edits_dir(self) -> Path:
        return self.run_dir / "manual_edits"

    @property
    def manual_edits_path(self) -> Path:
        return self.manual_edits_dir / f"{self.name}_manual_edits.gpkg"

    @property
    def cleaned_dir(self) -> Path:
        return self.run_dir / "masks_cleaned"

    @property
    def cleaned_mask(self) -> Path:
        return self.cleaned_dir / f"{self.name}_cleaned.tif"

    @property
    def rivgraph_dir(self) -> Path:
        return self.run_dir / "rivgraph"

    @property
    def prepare_summary(self) -> Path:
        return self.run_dir / "prepare_summary.json"

    @property
    def manual_edits_summary(self) -> Path:
        return self.run_dir / "manual_edits_summary.json"

    @property
    def rivgraph_summary(self) -> Path:
        return self.run_dir / "rivgraph_summary.json"

    @property
    def workflow_summary(self) -> Path:
        return self.run_dir / "workflow_summary.json"

    def ensure_dirs(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.prepared_dir.mkdir(parents=True, exist_ok=True)
        self.manual_edits_dir.mkdir(parents=True, exist_ok=True)
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)
        self.rivgraph_dir.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def existing_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path)
    if resolved.exists() is False:
        return None
    return str(resolved)


def prepare_mask_for_run(
    *,
    name: str,
    source_mask: Path,
    run_dir: Path | None = None,
    output_root: Path | None = None,
    threshold: float = 0.5,
    dst_crs: str | None = None,
    force: bool = False,
) -> tuple[WorkflowPaths, dict[str, object]]:
    paths = WorkflowPaths.from_name(name, run_dir, output_root=output_root)
    paths.ensure_dirs()
    source_mask = Path(source_mask).expanduser().resolve()

    summary = prepare_binary_projected_mask(
        src_path=source_mask,
        dst_path=paths.prepared_mask,
        threshold=threshold,
        dst_crs=dst_crs,
    )
    ensure_manual_edits_gpkg(
        gpkg_path=paths.manual_edits_path,
        target_crs=summary["target_crs"],
    )
    summary.update(
        {
            "run_name": paths.name,
            "run_dir": str(paths.run_dir),
            "manual_edits_path": str(paths.manual_edits_path),
            "cleaned_mask": str(paths.cleaned_mask),
            "reused_existing": False,
        }
    )
    write_json(paths.prepare_summary, summary)
    return paths, summary


def apply_edits_for_run(
    *,
    name: str,
    run_dir: Path | None = None,
    output_root: Path | None = None,
    edits: Path | None = None,
    base_mask: Path | None = None,
    output_mask: Path | None = None,
    layer: str | None = None,
    action_field: str = "action",
    all_touched: bool = True,
    tiny_land_remove_max_pixels: int = 3,
    tiny_land_export_min_pixels: int = 4,
    tiny_land_export_max_pixels: int = 10,
    tiny_land_export_path: Path | None = None,
    diagonal_passage_export_path: Path | None = None,
    keep_largest_water_component: bool = False,
) -> tuple[WorkflowPaths, dict[str, object]]:
    paths = WorkflowPaths.from_name(name, run_dir, output_root=output_root)
    paths.ensure_dirs()
    edits_path = paths.manual_edits_path if edits is None else Path(edits).expanduser().resolve()
    base_mask_path = paths.prepared_mask if base_mask is None else Path(base_mask).expanduser().resolve()
    output_path = paths.cleaned_mask if output_mask is None else Path(output_mask).expanduser().resolve()
    export_path = (
        paths.manual_edits_dir / f"{paths.name}_main_river_small_land_{tiny_land_export_min_pixels}_to_{tiny_land_export_max_pixels}px.gpkg"
        if tiny_land_export_path is None
        else Path(tiny_land_export_path).expanduser().resolve()
    )
    diagonal_export_path = (
        paths.manual_edits_dir / f"{paths.name}_main_river_diagonal_passages.gpkg"
        if diagonal_passage_export_path is None
        else Path(diagonal_passage_export_path).expanduser().resolve()
    )

    summary = apply_manual_edits(
        base_mask=base_mask_path,
        edits=edits_path,
        output=output_path,
        layer=layer,
        action_field=action_field,
        all_touched=all_touched,
        summary_path=paths.manual_edits_summary,
        tiny_land_remove_max_pixels=tiny_land_remove_max_pixels,
        tiny_land_export_min_pixels=tiny_land_export_min_pixels,
        tiny_land_export_max_pixels=tiny_land_export_max_pixels,
        tiny_land_export_path=export_path,
        diagonal_passage_export_path=diagonal_export_path,
        keep_largest_water_component=keep_largest_water_component,
    )
    summary.update(
        {
            "run_name": paths.name,
            "run_dir": str(paths.run_dir),
            "default_manual_edits_path": str(paths.manual_edits_path),
            "default_cleaned_mask": str(paths.cleaned_mask),
        }
    )
    write_json(paths.manual_edits_summary, summary)
    return paths, summary


def apply_panelized_edits_for_run(
    *,
    name: str,
    run_dir: Path | None = None,
    base_mask: Path | None = None,
    plan_path: Path | None = None,
    edits: Path | None = None,
    layer: str | None = None,
    action_field: str = "action",
    all_touched: bool = True,
    tiny_land_remove_max_pixels: int = 3,
    tiny_land_export_min_pixels: int = 4,
    tiny_land_export_max_pixels: int = 10,
    keep_largest_water_component: bool = False,
    overwrite: bool = False,
) -> tuple[WorkflowPaths, dict[str, object]]:
    """Create prepared panels, then apply manual edits and cleanup locally.

    The source is normally the run's prepared mask. The output masks are the
    standard ``panels/P###/mask.tif`` inputs used by panelized RivGraph.
    """
    from rivgraph_centerline.panelization import prepare_panel_masks, write_panelized_mask_mosaic

    paths = WorkflowPaths.from_name(name, run_dir)
    paths.ensure_dirs()
    source_mask = paths.prepared_mask if base_mask is None else Path(base_mask).expanduser().resolve()
    plan = paths.run_dir / "panels" / "panel_plan.gpkg" if plan_path is None else Path(plan_path).expanduser().resolve()
    edits_path = paths.manual_edits_path if edits is None else Path(edits).expanduser().resolve()
    if not source_mask.exists():
        raise FileNotFoundError(f"Prepared mask not found: {source_mask}")
    if not plan.exists():
        raise FileNotFoundError(f"Panel plan not found: {plan}. Create and review panel breaks, then run plan-panels first.")
    source_panel_root = paths.run_dir / "panels" / "_prepared_for_edits"
    # Do not crop the staged inputs to existing water: add polygons may fall in
    # dry parts of a processing area. The final edited masks are cropped later.
    prepare_panel_masks(mask_path=source_mask, plan_path=plan, output_dir=source_panel_root, overwrite=overwrite, crop=False)
    processing = gpd.read_file(plan, layer="processing_areas")
    panel_footprints = dict(zip(processing["panel_id"].astype(str), processing.geometry))
    small_land_export = paths.manual_edits_dir / f"{paths.name}_main_river_small_land_{tiny_land_export_min_pixels}_to_{tiny_land_export_max_pixels}px.gpkg"
    diagonal_export = paths.manual_edits_dir / f"{paths.name}_main_river_diagonal_passages.gpkg"
    summary = apply_manual_edits_to_panels(
        source_panel_root=source_panel_root,
        output_panel_root=paths.run_dir / "panels",
        edits=edits_path,
        layer=layer,
        action_field=action_field,
        all_touched=all_touched,
        tiny_land_remove_max_pixels=tiny_land_remove_max_pixels,
        tiny_land_export_min_pixels=tiny_land_export_min_pixels,
        tiny_land_export_max_pixels=tiny_land_export_max_pixels,
        tiny_land_export_path=small_land_export,
        diagonal_passage_export_path=diagonal_export,
        keep_largest_water_component=keep_largest_water_component,
        panel_footprints=panel_footprints,
        overwrite=overwrite,
    )
    cleaned_mask = write_panelized_mask_mosaic(
        template_mask_path=source_mask,
        plan_path=plan,
        panel_root=paths.run_dir / "panels",
        output_mask_path=paths.cleaned_mask,
        overwrite=overwrite,
    )
    summary.update({
        "run_name": paths.name,
        "run_dir": str(paths.run_dir),
        "base_mask": str(source_mask),
        "plan": str(plan),
        "cleaned_mask": str(cleaned_mask),
        "requirement": "When local largest-component filtering is enabled, the intended river must be the largest 8-connected water component in every buffered processing area.",
    })
    output = paths.run_dir / "panelized_manual_edits_summary.json"
    write_json(output, summary)
    summary["summary_path"] = str(output)
    return paths, summary


def resolve_input_mask(
    *,
    paths: WorkflowPaths,
    explicit_mask: Path | None = None,
    mask_stage: str = "auto",
) -> tuple[Path, str]:
    if explicit_mask is not None:
        return Path(explicit_mask).expanduser().resolve(), "explicit"

    normalized = mask_stage.lower()
    if normalized == "auto":
        if paths.cleaned_mask.exists():
            return paths.cleaned_mask, "cleaned"
        if paths.prepared_mask.exists():
            return paths.prepared_mask, "prepared"
        raise FileNotFoundError(
            f"No prepared or cleaned mask found in {paths.run_dir}. Run prepare-mask first."
        )
    if normalized == "cleaned":
        if not paths.cleaned_mask.exists():
            raise FileNotFoundError(f"Cleaned mask not found: {paths.cleaned_mask}")
        return paths.cleaned_mask, "cleaned"
    if normalized == "prepared":
        if not paths.prepared_mask.exists():
            raise FileNotFoundError(f"Prepared mask not found: {paths.prepared_mask}")
        return paths.prepared_mask, "prepared"
    raise ValueError("mask_stage must be one of: auto, prepared, cleaned")


def _clear_rivgraph_outputs(paths: WorkflowPaths) -> list[str]:
    """Remove stale RivGraph artifacts for this run before regenerating outputs.

    Only files and directories with the current run-name prefix are removed so
    unrelated helper files in the results folder are preserved.
    """

    if not paths.rivgraph_dir.exists():
        return []

    deleted: list[str] = []
    prefix = f"{paths.name}_"
    for candidate in sorted(paths.rivgraph_dir.iterdir()):
        if not candidate.name.startswith(prefix):
            continue
        if candidate.is_dir():
            shutil.rmtree(candidate)
        else:
            candidate.unlink()
        deleted.append(str(candidate))
    return deleted


def _clear_partial_mesh(network: object) -> None:
    for attr in ("meshlines", "meshpolys", "centerline_smooth"):
        if hasattr(network, attr):
            delattr(network, attr)


def _assign_flow_directions_with_mesh_mode(
    network: object,
    *,
    flow_mesh_mode: str,
) -> dict[str, object]:
    if flow_mesh_mode not in {"auto", "valley", "centerline"}:
        raise ValueError("flow_mesh_mode must be one of: auto, valley, centerline")

    if flow_mesh_mode == "centerline":
        try:
            network.single_thread = True
            network.compute_mesh(single_thread=True)
            with _rivgraph_flow_direction_compat():
                network.assign_flow_directions()
            return {
                "requested": flow_mesh_mode,
                "used": "centerline",
                "fallback_applied": False,
                "primary_error": None,
                "secondary_error": None,
                "succeeded": True,
            }
        except Exception as exc:
            return {
                "requested": flow_mesh_mode,
                "used": None,
                "fallback_applied": False,
                "primary_error": f"{type(exc).__name__}: {exc}",
                "secondary_error": None,
                "succeeded": False,
            }

    try:
        network.single_thread = False
        with _rivgraph_flow_direction_compat():
            network.assign_flow_directions()
        return {
            "requested": flow_mesh_mode,
            "used": "valley",
            "fallback_applied": False,
            "primary_error": None,
            "secondary_error": None,
            "succeeded": True,
        }
    except Exception as primary_exc:
        if flow_mesh_mode != "auto":
            raise

        try:
            _clear_partial_mesh(network)
            network.single_thread = True
            network.compute_mesh(single_thread=True)
            with _rivgraph_flow_direction_compat():
                network.assign_flow_directions()
            return {
                "requested": flow_mesh_mode,
                "used": "centerline",
                "fallback_applied": True,
                "primary_error": f"{type(primary_exc).__name__}: {primary_exc}",
                "secondary_error": None,
                "succeeded": True,
            }
        except Exception as secondary_exc:
            return {
                "requested": flow_mesh_mode,
                "used": None,
                "fallback_applied": True,
                "primary_error": f"{type(primary_exc).__name__}: {primary_exc}",
                "secondary_error": f"{type(secondary_exc).__name__}: {secondary_exc}",
                "succeeded": False,
            }


def run_rivgraph_for_run(
    *,
    name: str,
    exit_sides: str,
    run_dir: Path | None = None,
    output_root: Path | None = None,
    mask: Path | None = None,
    mask_stage: str = "auto",
    vector_format: str = "gpkg",
    assign_flow: bool = False,
    verbose: bool = True,
    clean_target: bool = True,
    flow_mesh_mode: str = "auto",
) -> tuple[WorkflowPaths, dict[str, object]]:
    from rivgraph.classes import river
    import rivgraph.rivers.river_utils as ru

    paths = WorkflowPaths.from_name(name, run_dir, output_root=output_root)
    paths.ensure_dirs()
    input_mask, input_stage = resolve_input_mask(paths=paths, explicit_mask=mask, mask_stage=mask_stage)
    deleted_rivgraph_outputs = _clear_rivgraph_outputs(paths) if clean_target else []

    network = river(
        name=paths.name,
        path_to_mask=str(input_mask),
        results_folder=str(paths.rivgraph_dir),
        exit_sides=exit_sides,
        verbose=verbose,
    )
    network.skeletonize()
    network.to_geotiff("skeleton")
    network.compute_network()
    network.links, network.nodes = ru.prune_river(network.links, network.nodes, network.exit_sides, network.Iskel)
    _sanitize_rivgraph_topology_metadata(network.links, network.nodes)
    network.finalize_ids()
    network.compute_link_width_and_length()
    with rivgraph_centerline_compat():
        network.compute_centerline()
    flow_mesh_summary: dict[str, object] | None = None
    flow_directions_assigned = False
    if assign_flow:
        flow_mesh_summary = _assign_flow_directions_with_mesh_mode(
            network,
            flow_mesh_mode=flow_mesh_mode,
        )
        flow_directions_assigned = bool(flow_mesh_summary.get("succeeded"))
        if flow_directions_assigned:
            network.to_geotiff("directions")
    network.to_geovectors("network", ftype=vector_format)
    network.to_geovectors("centerline", ftype=vector_format)

    n_directional_links_certain: int | None = None
    if flow_directions_assigned and "certain" in network.links:
        n_directional_links_certain = int(np.count_nonzero(np.asarray(network.links["certain"])))

    prepare_summary = load_json(paths.prepare_summary) if paths.prepare_summary.exists() else {}
    summary = {
        **prepare_summary,
        "run_name": paths.name,
        "run_dir": str(paths.run_dir),
        "input_mask": str(input_mask),
        "input_mask_stage": input_stage,
        "exit_sides": exit_sides,
        "vector_format": vector_format,
        "flow_directions_requested": assign_flow,
        "flow_directions_assigned": flow_directions_assigned,
        "flow_direction_backend": "rivgraph" if flow_directions_assigned else None,
        "flow_mesh_mode_requested": flow_mesh_mode if assign_flow else None,
        "flow_mesh_mode_used": None if flow_mesh_summary is None else flow_mesh_summary["used"],
        "flow_mesh_fallback_applied": False if flow_mesh_summary is None else flow_mesh_summary["fallback_applied"],
        "flow_mesh_primary_error": None if flow_mesh_summary is None else flow_mesh_summary["primary_error"],
        "flow_mesh_secondary_error": None if flow_mesh_summary is None else flow_mesh_summary["secondary_error"],
        "rivgraph_results_dir": str(paths.rivgraph_dir),
        "cleaned_rivgraph_target": clean_target,
        "deleted_rivgraph_outputs": deleted_rivgraph_outputs,
        "n_links": len(network.links.get("id", [])),
        "n_nodes": len(network.nodes.get("id", [])),
        "inlets": network.nodes.get("inlets", []),
        "outlets": network.nodes.get("outlets", []),
        "n_directional_links_certain": n_directional_links_certain,
        "outputs": {
            "prepared_mask": str(paths.prepared_mask) if paths.prepared_mask.exists() else None,
            "cleaned_mask": str(paths.cleaned_mask) if paths.cleaned_mask.exists() else None,
            "skeleton": network.paths.get("Iskel"),
            "links": network.paths.get("links"),
            "nodes": network.paths.get("nodes"),
            "centerline": network.paths.get("centerline"),
            "direction_raster": existing_path(network.paths.get("linkdirs")) if flow_directions_assigned else None,
            "manual_fixlinks_csv": existing_path(network.paths.get("fixlinks_csv")) if assign_flow else None,
            "log": network.paths.get("log"),
        },
    }
    write_json(paths.rivgraph_summary, summary)
    write_json(paths.workflow_summary, summary)
    return paths, summary


def run_all_for_run(
    *,
    name: str,
    source_mask: Path,
    exit_sides: str,
    run_dir: Path | None = None,
    output_root: Path | None = None,
    threshold: float = 0.5,
    dst_crs: str | None = None,
    edits: Path | None = None,
    vector_format: str = "gpkg",
    assign_flow: bool = False,
    force_prepare: bool = False,
    verbose: bool = True,
    clean_target: bool = True,
    flow_mesh_mode: str = "auto",
    tiny_land_remove_max_pixels: int = 3,
    tiny_land_export_min_pixels: int = 4,
    tiny_land_export_max_pixels: int = 10,
    tiny_land_export_path: Path | None = None,
    diagonal_passage_export_path: Path | None = None,
    keep_largest_water_component: bool = False,
) -> dict[str, object]:
    paths, prepare_summary = prepare_mask_for_run(
        name=name,
        source_mask=source_mask,
        run_dir=run_dir,
        output_root=output_root,
        threshold=threshold,
        dst_crs=dst_crs,
        force=force_prepare,
    )

    edits_path = paths.manual_edits_path if edits is None else Path(edits).expanduser().resolve()
    edits_summary: dict[str, object] | None = None
    if edits_path.exists():
        _, edits_summary = apply_edits_for_run(
            name=name,
            run_dir=paths.run_dir,
            edits=edits_path,
            output_root=output_root,
            tiny_land_remove_max_pixels=tiny_land_remove_max_pixels,
            tiny_land_export_min_pixels=tiny_land_export_min_pixels,
            tiny_land_export_max_pixels=tiny_land_export_max_pixels,
            tiny_land_export_path=tiny_land_export_path,
            diagonal_passage_export_path=diagonal_passage_export_path,
            keep_largest_water_component=keep_largest_water_component,
        )

    _, rivgraph_summary = run_rivgraph_for_run(
        name=name,
        run_dir=paths.run_dir,
        exit_sides=exit_sides,
        output_root=output_root,
        mask_stage="auto",
        vector_format=vector_format,
        assign_flow=assign_flow,
        verbose=verbose,
        clean_target=clean_target,
        flow_mesh_mode=flow_mesh_mode,
    )
    return {
        "prepare": prepare_summary,
        "manual_edits": edits_summary,
        "rivgraph": rivgraph_summary,
    }


def prepare_binary_projected_mask(
    *,
    src_path: Path,
    dst_path: Path,
    threshold: float,
    dst_crs: str | None,
) -> dict[str, object]:
    with rasterio.open(src_path) as src:
        src_array = src.read(1, masked=True)
        binary = (src_array.filled(0) >= threshold).astype("uint8")
        source_water_pixels = int(binary.sum())
        source_total_pixels = int(binary.size)
        original_height = int(src.height)
        original_width = int(src.width)

        target_crs = dst_crs
        if target_crs is None:
            target_crs = choose_projected_crs(src)

        if target_crs == src.crs:
            profile = src.profile.copy()
            output_array, output_transform, crop_window = crop_binary_mask(binary, src.transform)
            profile.update(
                driver="GTiff",
                dtype="uint8",
                count=1,
                nodata=0,
                compress="lzw",
                transform=output_transform,
                width=int(output_array.shape[1]),
                height=int(output_array.shape[0]),
            )
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(dst_path, "w", **profile) as dst:
                dst.write(output_array, 1)
            dst_water_pixels = source_water_pixels
            dst_total_pixels = int(output_array.size)
        else:
            transform, width, height = calculate_default_transform(
                src.crs,
                target_crs,
                src.width,
                src.height,
                *src.bounds,
            )
            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                crs=target_crs,
                transform=transform,
                width=width,
                height=height,
                dtype="uint8",
                count=1,
                nodata=0,
                compress="lzw",
            )
            projected_array = np.zeros((height, width), dtype="uint8")
            reproject(
                source=binary,
                destination=projected_array,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest,
            )
            output_array, output_transform, crop_window = crop_binary_mask(projected_array, transform)
            profile.update(
                transform=output_transform,
                width=int(output_array.shape[1]),
                height=int(output_array.shape[0]),
            )
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(dst_path, "w", **profile) as dst:
                dst.write(output_array, 1)
            dst_water_pixels = int(output_array.sum())
            dst_total_pixels = int(output_array.size)

        return {
            "source_mask": str(src_path),
            "prepared_mask": str(dst_path),
            "threshold": threshold,
            "source_crs": str(src.crs),
            "target_crs": str(target_crs),
            "source_water_fraction": source_water_pixels / source_total_pixels,
            "prepared_water_fraction": dst_water_pixels / dst_total_pixels,
            "source_shape": [original_height, original_width],
            "prepared_shape": [int(profile["height"]), int(profile["width"])],
            "crop_window": crop_window,
        }


def ensure_manual_edits_gpkg(
    *,
    gpkg_path: Path,
    target_crs: str | rasterio.crs.CRS,
    action_field: str = "action",
) -> None:
    if gpkg_path.exists() and _manual_edits_gpkg_is_valid(gpkg_path, action_field=action_field):
        return

    if gpkg_path.exists():
        gpkg_path.unlink()

    gpkg_path.parent.mkdir(parents=True, exist_ok=True)
    placeholder = Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])
    empty = gpd.GeoDataFrame(
        data={action_field: pd.Series(["remove"], dtype="string")},
        geometry=[placeholder],
        crs=target_crs,
    )
    empty.to_file(gpkg_path, driver="GPKG", geometry_type="Polygon")
    layer_name = gpkg_path.stem
    with sqlite3.connect(gpkg_path) as con:
        con.execute(f'DELETE FROM "{layer_name}"')
        con.commit()


def _manual_edits_gpkg_is_valid(gpkg_path: Path, *, action_field: str) -> bool:
    try:
        layer_name = gpkg_path.stem
        with sqlite3.connect(gpkg_path) as con:
            geometry_row = con.execute(
                "SELECT geometry_type_name FROM gpkg_geometry_columns WHERE table_name = ?",
                (layer_name,),
            ).fetchone()
            field_row = con.execute(f'PRAGMA table_info("{layer_name}")').fetchall()
        if geometry_row is None or geometry_row[0].upper() != "POLYGON":
            return False
        field_types = {row[1]: (row[2] or "") for row in field_row}
        action_type = field_types.get(action_field, "").upper()
        if action_type not in {"TEXT", "VARCHAR(10)", "STRING"}:
            return False
        visible_columns = set(gpd.read_file(gpkg_path).columns)
        return action_field in visible_columns
    except Exception:
        return False


def choose_projected_crs(src: rasterio.io.DatasetReader) -> str:
    if src.crs is None:
        raise ValueError("Source raster has no CRS. Provide --dst-crs explicitly.")
    if not src.crs.is_geographic:
        return src.crs

    center_lon = (src.bounds.left + src.bounds.right) / 2
    center_lat = (src.bounds.bottom + src.bounds.top) / 2
    zone = int((center_lon + 180) // 6) + 1
    epsg = 32600 + zone if center_lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


def _sanitize_rivgraph_topology_metadata(links: dict[str, object], nodes: dict[str, object]) -> None:
    """Drop stale provisional references left behind by RivGraph pruning.

    Some RivGraph prune paths delete links without fully cleaning auxiliary
    fields like `parallels`, which then breaks `finalize_ids()` when it
    remaps the surviving topology.
    """

    valid_link_ids = set(links.get("id", []))
    valid_node_ids = set(nodes.get("id", []))

    if "parallels" in links:
        cleaned_parallels = []
        for pair in links["parallels"]:
            kept = [lid for lid in pair if lid in valid_link_ids]
            if len(kept) >= 2:
                cleaned_parallels.append(kept)
        links["parallels"] = cleaned_parallels

    if "arts" in links:
        cleaned_arts = []
        for triad in links["arts"]:
            kept = [lid for lid in triad if lid in valid_link_ids]
            if len(kept) >= 2:
                cleaned_arts.append(kept)
        links["arts"] = cleaned_arts

    if "link_conn" in links:
        links["link_conn"] = [[lid for lid in conn if lid in valid_link_ids] for conn in links["link_conn"]]

    for key in ("inlets", "outlets", "arts"):
        if key in nodes:
            nodes[key] = [nid for nid in nodes[key] if nid in valid_node_ids]

    if "super_apex" in nodes and nodes["super_apex"] not in valid_node_ids:
        del nodes["super_apex"]


def _representative_project_point(geometry: object) -> object:
    """Reduce non-point Shapely intersections to a single representative point."""

    if geometry is None or not hasattr(geometry, "geom_type") or getattr(geometry, "is_empty", False):
        return geometry
    if isinstance(geometry, Point):
        return geometry
    if isinstance(geometry, MultiPoint):
        return list(geometry.geoms)[0] if len(geometry.geoms) else geometry
    if isinstance(geometry, LineString):
        return geometry.interpolate(0.5, normalized=True)
    if isinstance(geometry, MultiLineString):
        lines = [line for line in geometry.geoms if not line.is_empty]
        if not lines:
            return geometry
        longest = max(lines, key=lambda line: line.length)
        return longest.interpolate(0.5, normalized=True)
    if isinstance(geometry, GeometryCollection):
        points = [geom for geom in geometry.geoms if isinstance(geom, Point) and not geom.is_empty]
        if points:
            return points[0]
        lines = [geom for geom in geometry.geoms if isinstance(geom, (LineString, MultiLineString)) and not geom.is_empty]
        if lines:
            return _representative_project_point(max(lines, key=lambda line: line.length))
    if hasattr(geometry, "representative_point"):
        return geometry.representative_point()
    return geometry


def _patched_rivgraph_dir_centerline(
    original: object,
    links: dict[str, object],
    nodes: dict[str, object],
    meshpolys: list[object],
    meshlines: list[object],
    Imask: np.ndarray,
    gt: tuple[float, ...],
    pixlen: float,
) -> dict[str, object]:
    """Compatibility wrapper for RivGraph dir_centerline with ambiguous joins."""

    gpd_module = original.__globals__["gpd"]
    np_module = original.__globals__["np"]
    Point_cls = original.__globals__["Point"]
    Polygon_cls = original.__globals__["Polygon"]
    gu_module = original.__globals__["gu"]
    dy_module = original.__globals__["dy"]
    math_module = original.__globals__["math"]

    alg = dy_module.algmap("cl_dist_guess")
    mp_gdf = gpd_module.GeoDataFrame(geometry=[Polygon_cls(mp) for mp in meshpolys])
    rc = np_module.unravel_index(nodes["idx"], Imask.shape)
    nodecoords = gu_module.xy_to_coords(rc[1], rc[0], gt)
    node_gdf = gpd_module.GeoDataFrame(
        geometry=[Point_cls(x, y) for x, y in zip(nodecoords[0], nodecoords[1])],
        index=nodes["id"],
    )

    intersect = gpd_module.sjoin(node_gdf, mp_gdf, predicate="intersects", rsuffix="right")
    node_mesh_index = intersect.groupby(level=0)["index_right"].mean() if not intersect.empty else pd.Series(dtype=float)

    cldists = np_module.zeros((len(links["id"]), 1))
    for i, lconn in enumerate(links["conn"]):
        first = node_mesh_index.get(lconn[0])
        second = node_mesh_index.get(lconn[1])
        if pd.isna(first) or pd.isna(second):
            continue
        cldists[i] = float(second) - float(first)

    for i, c in enumerate(cldists):
        if c != 0:
            if c > 0:
                links["guess"][i].append(links["conn"][i][0])
                links["guess_alg"][i].append(alg)
            elif c < 0:
                links["guess"][i].append(links["conn"][i][-1])
                links["guess_alg"][i].append(alg)

    links["cldists"] = np_module.abs(cldists)

    alg = dy_module.algmap("cl_ang_guess")
    clangs = np_module.ones((len(links["id"]), 1)) * np_module.nan
    for i, (lconn, lidx) in enumerate(zip(links["conn"], links["idx"])):
        first = node_mesh_index.get(lconn[0])
        second = node_mesh_index.get(lconn[1])
        if pd.isna(first) or pd.isna(second):
            continue
        first_idx = int(np_module.floor(min(first, second)))
        second_idx = int(np_module.floor(max(first, second)))
        if second_idx + 1 >= len(meshlines):
            continue

        first_coords = np_module.asarray(meshlines[first_idx].coords)
        second_coords = np_module.asarray(meshlines[second_idx + 1].coords)
        first_mp = np_module.mean(first_coords, axis=0)
        second_mp = np_module.mean(second_coords, axis=0)

        cl_vec = second_mp - first_mp
        norm = np_module.sqrt(np_module.sum(cl_vec**2))
        if not np_module.isfinite(norm) or norm == 0:
            continue
        cl_vec = cl_vec / norm

        link_vec = dy_module.get_link_vector(links, nodes, links["id"][i], Imask.shape, pixlen=pixlen)
        link_vec_rev = -link_vec
        lva = math_module.atan2(np_module.linalg.det([cl_vec, link_vec]), np_module.dot(cl_vec, link_vec))
        lvar = math_module.atan2(np_module.linalg.det([cl_vec, link_vec_rev]), np_module.dot(cl_vec, link_vec_rev))

        clangs[i] = np_module.min(np_module.abs([lva, lvar]))
        if np_module.abs(lvar) < np_module.abs(lva):
            links["guess"][i].append(links["conn"][i][1])
            links["guess_alg"][i].append(alg)
        else:
            links["guess"][i].append(links["conn"][i][0])
            links["guess_alg"][i].append(alg)
    links["clangs"] = clangs

    return links


@contextmanager
def _rivgraph_flow_direction_compat():
    """Patch RivGraph direction internals for modern Shapely/GeoPandas behavior."""

    try:
        import rivgraph.rivers.river_directionality as rd
    except ModuleNotFoundError:
        yield
        return

    original_project = BaseGeometry.project
    original_dir_centerline = rd.dir_centerline

    def patched_project(self, other, *args, **kwargs):
        return original_project(self, _representative_project_point(other), *args, **kwargs)

    def patched_dir_centerline(links, nodes, meshpolys, meshlines, Imask, gt, pixlen):
        return _patched_rivgraph_dir_centerline(
            original_dir_centerline,
            links,
            nodes,
            meshpolys,
            meshlines,
            Imask,
            gt,
            pixlen,
        )

    BaseGeometry.project = patched_project
    rd.dir_centerline = patched_dir_centerline
    try:
        yield
    finally:
        BaseGeometry.project = original_project
        rd.dir_centerline = original_dir_centerline
