"""Run the staged RivGraph workflow on any mask."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "rivgraph_centerline" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rivgraph_centerline.workflow import (  # noqa: E402
    WorkflowPaths,
    apply_edits_for_run,
    run_all_for_run,
    run_rivgraph_for_run,
    prepare_mask_for_run,
)
from rivgraph_centerline.panelization import create_panel_breaks, plan_panels, prepare_panel_masks  # noqa: E402


def main() -> None:
    args = parse_args()
    payload = dispatch(args)
    print(json.dumps(payload, indent=2))


def parse_optional_bool(value: str | None) -> bool:
    if value is None:
        return True
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value such as true/false.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare-mask", help="Create or rebuild the binary/projected mask.")
    add_run_identity_args(prepare_parser)
    prepare_parser.add_argument("--source-mask", type=Path, required=True, help="Source river mask GeoTIFF.")
    prepare_parser.add_argument("--threshold", type=float, default=0.5, help="Water threshold for the binary mask.")
    prepare_parser.add_argument("--dst-crs", default=None, help="Projected CRS override.")
    prepare_parser.add_argument("--force", action="store_true", help="Deprecated compatibility flag. The prepared mask is rebuilt on every run.")

    edits_parser = subparsers.add_parser("apply-edits", help="Apply manual polygon edits to the prepared mask.")
    add_run_identity_args(edits_parser)
    edits_parser.add_argument("--edits", type=Path, default=None, help="Vector edits file. Defaults to run-dir/manual_edits/<name>_manual_edits.gpkg.")
    edits_parser.add_argument("--base-mask", type=Path, default=None, help="Prepared mask override.")
    edits_parser.add_argument("--output-mask", type=Path, default=None, help="Cleaned mask output override.")
    edits_parser.add_argument("--layer", default=None, help="Vector layer name. Defaults to the first layer.")
    edits_parser.add_argument("--action-field", default="action", help="Field containing add/remove values.")
    edits_parser.add_argument("--center-only", action="store_true", help="Only edit pixels whose centers fall inside polygons.")
    edits_parser.add_argument("--tiny-land-remove-max-pixels", type=int, default=3, help="Remove main-river land components with this many pixels or fewer after edits. Use a negative value to disable.")
    edits_parser.add_argument("--tiny-land-export-min-pixels", type=int, default=4, help="Minimum component size, in pixels, for the exported post-cleanup main-river land QA layer.")
    edits_parser.add_argument("--tiny-land-export-max-pixels", type=int, default=10, help="Maximum component size, in pixels, for the exported post-cleanup main-river land QA layer. Use a negative value to disable export.")
    edits_parser.add_argument("--tiny-land-export-path", type=Path, default=None, help="Optional GeoPackage path for the post-cleanup main-river small-land export.")
    edits_parser.add_argument("--diagonal-passage-export-path", type=Path, default=None, help="Optional GeoPackage path for the post-cleanup main-river diagonal-passage diagnostic points.")
    edits_parser.add_argument("--keep-largest-water-component", action="store_true", help="Discard all 8-connected water components except the largest after manual edits.")

    breaks_parser = subparsers.add_parser("create-panel-breaks", help="Create an editable provisional panel-break GeoPackage.")
    add_run_identity_args(breaks_parser)
    breaks_parser.add_argument("--mask", type=Path, default=None, help="Cleaned mask override.")
    breaks_parser.add_argument("--output", type=Path, default=None, help="GeoPackage output override.")
    breaks_parser.add_argument("--rows", type=int, default=4)
    breaks_parser.add_argument("--columns", type=int, default=4)
    breaks_parser.add_argument("--overwrite", action="store_true")

    plan_parser = subparsers.add_parser("plan-panels", help="Build irregular panel cores and overlap QA layers from edited breaks.")
    add_run_identity_args(plan_parser)
    plan_parser.add_argument("--mask", type=Path, default=None)
    plan_parser.add_argument("--panel-breaks", type=Path, default=None)
    plan_parser.add_argument("--output", type=Path, default=None)
    plan_parser.add_argument("--overlap-m", type=float, default=5000)
    plan_parser.add_argument("--max-panel-pixels", type=int, default=160_000_000)
    plan_parser.add_argument("--overwrite", action="store_true")

    panel_masks_parser = subparsers.add_parser("prepare-panel-masks", help="Create cropped buffered masks for one or more planned panels.")
    add_run_identity_args(panel_masks_parser)
    panel_masks_parser.add_argument("--mask", type=Path, default=None)
    panel_masks_parser.add_argument("--plan", type=Path, default=None)
    panel_masks_parser.add_argument("--panel-id", action="append", default=None, help="Repeat to prepare selected panels only.")
    panel_masks_parser.add_argument("--overwrite", action="store_true")

    panel_run_parser = subparsers.add_parser("run-panel-rivgraph", help="Run RivGraph for one prepared panel mask.")
    add_run_identity_args(panel_run_parser)
    panel_run_parser.add_argument("--panel-id", required=True)
    panel_run_parser.add_argument("--exit-sides", required=True, help="Local panel exit sides, e.g. SN.")
    panel_run_parser.add_argument("--assign-flow", action="store_true")
    panel_run_parser.add_argument("--flow-mesh-mode", choices=("auto", "valley", "centerline"), default="auto")

    run_parser = subparsers.add_parser("run-rivgraph", help="Run RivGraph on the prepared or cleaned mask.")
    add_run_identity_args(run_parser)
    run_parser.add_argument("--mask", type=Path, default=None, help="Explicit input mask override.")
    run_parser.add_argument(
        "--mask-stage",
        choices=("auto", "prepared", "cleaned"),
        default="auto",
        help="Choose cleaned if available, otherwise prepared.",
    )
    run_parser.add_argument("--exit-sides", required=True, help="Two-letter RivGraph river exit sides, upstream first.")
    run_parser.add_argument("--vector-format", default="gpkg", help="Vector export format for RivGraph.")
    run_parser.add_argument(
        "--assign-flow",
        nargs="?",
        const=True,
        default=False,
        type=parse_optional_bool,
        help="Also run RivGraph's native flow-direction assignment. Accepts optional true/false.",
    )
    run_parser.add_argument(
        "--flow-mesh-mode",
        choices=("auto", "valley", "centerline"),
        default="auto",
        help="Mesh mode for RivGraph flow assignment. 'auto' tries valley mesh first, then falls back to centerline mesh if needed.",
    )
    run_parser.add_argument(
        "--keep-existing-rivgraph-files",
        action="store_true",
        help="Do not clear stale run-specific files from the rivgraph output folder before rerunning.",
    )
    run_parser.add_argument("--quiet", action="store_true", help="Disable verbose RivGraph logging.")

    all_parser = subparsers.add_parser(
        "run-all",
        help="Prepare once, apply edits if present, then run RivGraph on the cleaned or prepared mask.",
    )
    add_run_identity_args(all_parser)
    all_parser.add_argument("--source-mask", type=Path, required=True, help="Source river mask GeoTIFF.")
    all_parser.add_argument("--threshold", type=float, default=0.5, help="Water threshold for the binary mask.")
    all_parser.add_argument("--dst-crs", default=None, help="Projected CRS override.")
    all_parser.add_argument("--edits", type=Path, default=None, help="Optional vector edits file override.")
    all_parser.add_argument("--exit-sides", required=True, help="Two-letter RivGraph river exit sides, upstream first.")
    all_parser.add_argument("--vector-format", default="gpkg", help="Vector export format for RivGraph.")
    all_parser.add_argument(
        "--assign-flow",
        nargs="?",
        const=True,
        default=False,
        type=parse_optional_bool,
        help="Also run RivGraph's native flow-direction assignment. Accepts optional true/false.",
    )
    all_parser.add_argument(
        "--flow-mesh-mode",
        choices=("auto", "valley", "centerline"),
        default="auto",
        help="Mesh mode for RivGraph flow assignment. 'auto' tries valley mesh first, then falls back to centerline mesh if needed.",
    )
    all_parser.add_argument(
        "--keep-existing-rivgraph-files",
        action="store_true",
        help="Do not clear stale run-specific files from the rivgraph output folder before rerunning.",
    )
    all_parser.add_argument("--force-prepare", action="store_true", help="Rebuild the prepared mask even if it exists.")
    all_parser.add_argument("--quiet", action="store_true", help="Disable verbose RivGraph logging.")
    all_parser.add_argument("--tiny-land-remove-max-pixels", type=int, default=3, help="Remove main-river land components with this many pixels or fewer after edits. Use a negative value to disable.")
    all_parser.add_argument("--tiny-land-export-min-pixels", type=int, default=4, help="Minimum component size, in pixels, for the exported post-cleanup main-river land QA layer.")
    all_parser.add_argument("--tiny-land-export-max-pixels", type=int, default=10, help="Maximum component size, in pixels, for the exported post-cleanup main-river land QA layer. Use a negative value to disable export.")
    all_parser.add_argument("--tiny-land-export-path", type=Path, default=None, help="Optional GeoPackage path for the post-cleanup main-river small-land export.")
    all_parser.add_argument("--diagonal-passage-export-path", type=Path, default=None, help="Optional GeoPackage path for the post-cleanup main-river diagonal-passage diagnostic points.")
    all_parser.add_argument("--keep-largest-water-component", action="store_true", help="Discard all 8-connected water components except the largest after manual edits.")

    return parser.parse_args()


def add_run_identity_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--name", required=True, help="Run name. Used in folder and file names.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional run directory override. Defaults to rivgraph_centerline/outputs/runs/<name>.",
    )


def dispatch(args: argparse.Namespace) -> dict[str, object]:
    if args.command == "run-panel-rivgraph":
        parent = WorkflowPaths.from_name(args.name, args.run_dir)
        panel_dir = parent.run_dir / "panels" / args.panel_id
        paths, summary = run_rivgraph_for_run(name=f"{parent.name}_{args.panel_id}", run_dir=panel_dir, mask=panel_dir / "mask.tif", mask_stage="explicit", exit_sides=args.exit_sides, assign_flow=args.assign_flow, flow_mesh_mode=args.flow_mesh_mode)
        return {"command": args.command, "run_paths": paths_to_dict(paths), "summary": summary}
    if args.command in {"create-panel-breaks", "plan-panels", "prepare-panel-masks"}:
        paths = WorkflowPaths.from_name(args.name, args.run_dir)
        mask = paths.cleaned_mask if args.mask is None else args.mask
        breaks = paths.manual_edits_dir / f"{paths.name}_panel_breaks.gpkg" if getattr(args, "panel_breaks", None) is None else args.panel_breaks
        if args.command == "create-panel-breaks":
            summary = create_panel_breaks(mask_path=mask, output_path=(paths.manual_edits_dir / f"{paths.name}_panel_breaks.gpkg") if args.output is None else args.output, rows=args.rows, columns=args.columns, overwrite=args.overwrite)
        elif args.command == "plan-panels":
            summary = plan_panels(mask_path=mask, breaks_path=breaks, output_path=(paths.run_dir / "panels" / "panel_plan.gpkg") if args.output is None else args.output, overlap_m=args.overlap_m, max_panel_pixels=args.max_panel_pixels, overwrite=args.overwrite)
        else:
            summary = prepare_panel_masks(mask_path=mask, plan_path=(paths.run_dir / "panels" / "panel_plan.gpkg") if args.plan is None else args.plan, output_dir=paths.run_dir / "panels", panel_ids=set(args.panel_id) if args.panel_id else None, overwrite=args.overwrite)
        return {"command": args.command, "summary": summary}
    if args.command == "prepare-mask":
        paths, summary = prepare_mask_for_run(
            name=args.name,
            source_mask=args.source_mask,
            run_dir=args.run_dir,
            threshold=args.threshold,
            dst_crs=args.dst_crs,
            force=args.force,
        )
        return {"command": args.command, "run_paths": paths_to_dict(paths), "summary": summary}

    if args.command == "apply-edits":
        paths, summary = apply_edits_for_run(
            name=args.name,
            run_dir=args.run_dir,
            edits=args.edits,
            base_mask=args.base_mask,
            output_mask=args.output_mask,
            layer=args.layer,
            action_field=args.action_field,
            all_touched=not args.center_only,
            tiny_land_remove_max_pixels=args.tiny_land_remove_max_pixels,
            tiny_land_export_min_pixels=args.tiny_land_export_min_pixels,
            tiny_land_export_max_pixels=args.tiny_land_export_max_pixels,
            tiny_land_export_path=args.tiny_land_export_path,
            diagonal_passage_export_path=args.diagonal_passage_export_path,
            keep_largest_water_component=args.keep_largest_water_component,
        )
        return {"command": args.command, "run_paths": paths_to_dict(paths), "summary": summary}

    if args.command == "run-rivgraph":
        paths, summary = run_rivgraph_for_run(
            name=args.name,
            run_dir=args.run_dir,
            mask=args.mask,
            mask_stage=args.mask_stage,
            exit_sides=args.exit_sides,
            vector_format=args.vector_format,
            assign_flow=args.assign_flow,
            verbose=not args.quiet,
            clean_target=not args.keep_existing_rivgraph_files,
            flow_mesh_mode=args.flow_mesh_mode,
        )
        return {"command": args.command, "run_paths": paths_to_dict(paths), "summary": summary}

    if args.command == "run-all":
        paths = WorkflowPaths.from_name(args.name, args.run_dir)
        summary = run_all_for_run(
            name=args.name,
            source_mask=args.source_mask,
            exit_sides=args.exit_sides,
            run_dir=args.run_dir,
            threshold=args.threshold,
            dst_crs=args.dst_crs,
            edits=args.edits,
            vector_format=args.vector_format,
            assign_flow=args.assign_flow,
            force_prepare=args.force_prepare,
            verbose=not args.quiet,
            clean_target=not args.keep_existing_rivgraph_files,
            flow_mesh_mode=args.flow_mesh_mode,
            tiny_land_remove_max_pixels=args.tiny_land_remove_max_pixels,
            tiny_land_export_min_pixels=args.tiny_land_export_min_pixels,
            tiny_land_export_max_pixels=args.tiny_land_export_max_pixels,
            tiny_land_export_path=args.tiny_land_export_path,
            diagonal_passage_export_path=args.diagonal_passage_export_path,
            keep_largest_water_component=args.keep_largest_water_component,
        )
        return {"command": args.command, "run_paths": paths_to_dict(paths), "summary": summary}

    raise ValueError(f"Unsupported command: {args.command}")


def paths_to_dict(paths: WorkflowPaths) -> dict[str, str]:
    return {
        "run_dir": str(paths.run_dir),
        "prepared_mask": str(paths.prepared_mask),
        "manual_edits_path": str(paths.manual_edits_path),
        "cleaned_mask": str(paths.cleaned_mask),
        "rivgraph_dir": str(paths.rivgraph_dir),
        "prepare_summary": str(paths.prepare_summary),
        "manual_edits_summary": str(paths.manual_edits_summary),
        "rivgraph_summary": str(paths.rivgraph_summary),
        "workflow_summary": str(paths.workflow_summary),
    }


if __name__ == "__main__":
    main()
