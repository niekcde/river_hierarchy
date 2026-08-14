"""Run the staged RivGraph workflow on any mask."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "rivgraph_centerline" / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rivgraph_centerline.workflow import (  # noqa: E402
    WorkflowPaths,
    apply_edits_for_run,
    apply_panelized_edits_for_run,
    run_all_for_run,
    run_rivgraph_for_run,
    prepare_mask_for_run,
)
from rivgraph_centerline.panelization import create_panel_breaks, plan_panels, prepare_panel_masks, match_panel_boundary_links, concatenate_matched_panel_links, rebuild_pilot_nodes, validate_pilot_topology, plan_boundary_matches, build_core_boundary_network, rebuild_core_network_nodes, assign_core_network_direction, publish_panelized_rivgraph_handoff  # noqa: E402


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

    panel_edits_parser = subparsers.add_parser("apply-panelized-edits", help="Prepare buffered panels from the prepared mask, then apply manual edits locally to each panel.")
    add_run_identity_args(panel_edits_parser)
    panel_edits_parser.add_argument("--mask", type=Path, default=None, help="Prepared mask override.")
    panel_edits_parser.add_argument("--plan", type=Path, default=None, help="Panel-plan GeoPackage override.")
    panel_edits_parser.add_argument("--edits", type=Path, default=None, help="Vector edits file. Defaults to run-dir/manual_edits/<name>_manual_edits.gpkg.")
    panel_edits_parser.add_argument("--layer", default=None, help="Vector layer name. Defaults to the first layer.")
    panel_edits_parser.add_argument("--action-field", default="action", help="Field containing add/remove values.")
    panel_edits_parser.add_argument("--center-only", action="store_true", help="Only edit pixels whose centers fall inside polygons.")
    panel_edits_parser.add_argument("--tiny-land-remove-max-pixels", type=int, default=3, help="Remove main-river land components with this many pixels or fewer in each panel. Use a negative value to disable.")
    panel_edits_parser.add_argument("--tiny-land-export-min-pixels", type=int, default=4, help="Minimum local main-river land-component size for the merged QA layer.")
    panel_edits_parser.add_argument("--tiny-land-export-max-pixels", type=int, default=10, help="Maximum local main-river land-component size for the merged QA layer.")
    panel_edits_parser.add_argument("--keep-largest-water-component", action="store_true", help="Locally retain only the largest 8-connected water component in every buffered panel.")
    panel_edits_parser.add_argument("--overwrite", action="store_true", help="Replace staged prepared panels and cleaned panel masks.")

    breaks_parser = subparsers.add_parser("create-panel-breaks", help="Create an editable provisional panel-break GeoPackage.")
    add_run_identity_args(breaks_parser)
    breaks_parser.add_argument("--mask", type=Path, default=None, help="Prepared mask override. Defaults to the run's prepared mask.")
    breaks_parser.add_argument("--output", type=Path, default=None, help="GeoPackage output override.")
    breaks_parser.add_argument("--rows", type=int, default=None, help="Optional provisional row count; otherwise computed from the pixel budget.")
    breaks_parser.add_argument("--columns", type=int, default=None, help="Optional provisional column count; otherwise computed from the pixel budget.")
    breaks_parser.add_argument("--max-panel-pixels", type=int, default=160_000_000, help="Maximum target pixels per provisional core panel.")
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

    match_parser = subparsers.add_parser("match-panel-boundary", help="Match duplicate panel links across one named break.")
    add_run_identity_args(match_parser)
    match_parser.add_argument("--break-id", required=True)
    match_parser.add_argument("--left-panel", required=True)
    match_parser.add_argument("--right-panel", required=True)
    match_parser.add_argument("--tolerance-m", type=float, default=10)
    match_parser.add_argument("--overwrite", action="store_true")

    merge_parser = subparsers.add_parser("merge-panel-pilot", help="Concatenate two panel link layers using accepted matches.")
    add_run_identity_args(merge_parser)
    merge_parser.add_argument("--left-panel", required=True)
    merge_parser.add_argument("--right-panel", required=True)
    merge_parser.add_argument("--matches", type=Path, required=True)
    merge_parser.add_argument("--overwrite", action="store_true")

    nodes_parser = subparsers.add_parser("rebuild-pilot-nodes", help="Rebuild global nodes for a concatenated pilot link layer.")
    add_run_identity_args(nodes_parser)
    nodes_parser.add_argument("--merged-links", type=Path, required=True)
    nodes_parser.add_argument("--output-dir", type=Path, default=None)
    nodes_parser.add_argument("--overwrite", action="store_true")
    validator_parser = subparsers.add_parser("validate-pilot-topology", help="Validate reconciled pilot links and nodes.")
    add_run_identity_args(validator_parser)
    validator_parser.add_argument("--links", type=Path, required=True)
    validator_parser.add_argument("--nodes", type=Path, required=True)
    match_plan_parser = subparsers.add_parser("plan-boundary-matches", help="Discover panel pairs sharing each edited break.")
    add_run_identity_args(match_plan_parser)
    batch_run_parser = subparsers.add_parser("run-planned-panels", help="Run RivGraph for selected water-bearing panel masks.")
    add_run_identity_args(batch_run_parser)
    batch_run_parser.add_argument("--assign-flow", action="store_true", help="Also assign native RivGraph flow direction in every planned panel.")
    batch_run_parser.add_argument("--flow-mesh-mode", choices=("auto", "valley", "centerline"), default="auto")
    batch_match_parser = subparsers.add_parser("match-planned-boundaries", help="Run matching for every pair in boundary_match_plan.json.")
    add_run_identity_args(batch_match_parser)
    core_merge_parser = subparsers.add_parser("build-core-boundary-network", help="Clip panel links to cores and merge degree-two boundary continuations.")
    add_run_identity_args(core_merge_parser)
    core_merge_parser.add_argument("--tolerance-m", type=float, default=10)
    core_merge_parser.add_argument("--overwrite", action="store_true")
    core_nodes_parser = subparsers.add_parser("rebuild-core-network-nodes", help="Create global nodes and validate the core-merged panel network.")
    add_run_identity_args(core_nodes_parser)
    core_nodes_parser.add_argument("--merged-links", type=Path, default=None, help="Core-merged links override.")
    core_nodes_parser.add_argument("--endpoint-tolerance-m", type=float, default=0.01, help="Maximum endpoint separation treated as one node.")
    core_nodes_parser.add_argument("--overwrite", action="store_true")
    direction_parser = subparsers.add_parser("assign-core-network-direction", help="Orient global merged links using native panel flow assignments.")
    add_run_identity_args(direction_parser)
    direction_parser.add_argument("--endpoint-tolerance-m", type=float, default=0.01)
    direction_parser.add_argument("--overwrite", action="store_true")
    handoff_parser = subparsers.add_parser("publish-panelized-rivgraph", help="Publish panelized vectors under the standard run/rivgraph file names.")
    add_run_identity_args(handoff_parser)
    handoff_parser.add_argument("--overwrite", action="store_true")
    panelized_parser = subparsers.add_parser("run-panelized-rivgraph", help="Create a plan from reviewed breaks, then run the complete panelized RivGraph workflow.")
    add_run_identity_args(panelized_parser)
    panelized_parser.add_argument("--exit-sides", required=True, help="Global river orientation, upstream first (for example SE).")
    panelized_parser.add_argument("--mask", type=Path, default=None, help="Mask to process. Defaults to this run's cleaned mask.")
    panelized_parser.add_argument("--panel-breaks", type=Path, default=None, help="Reviewed panel-break GeoPackage override.")
    panelized_parser.add_argument("--overlap-m", type=float, default=5000)
    panelized_parser.add_argument("--max-panel-pixels", type=int, default=160_000_000)
    panelized_parser.add_argument("--flow-mesh-mode", choices=("auto", "valley", "centerline"), default="auto")
    panelized_parser.add_argument("--boundary-tolerance-m", type=float, default=10)
    panelized_parser.add_argument("--endpoint-tolerance-m", type=float, default=0.01)
    panelized_parser.add_argument("--overwrite", action="store_true", help="Replace prior panel masks, panel RivGraph outputs, and merged outputs in this run directory.")

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
    if args.command == "apply-panelized-edits":
        _, summary = apply_panelized_edits_for_run(
            name=args.name,
            run_dir=args.run_dir,
            base_mask=args.mask,
            plan_path=args.plan,
            edits=args.edits,
            layer=args.layer,
            action_field=args.action_field,
            all_touched=not args.center_only,
            tiny_land_remove_max_pixels=args.tiny_land_remove_max_pixels,
            tiny_land_export_min_pixels=args.tiny_land_export_min_pixels,
            tiny_land_export_max_pixels=args.tiny_land_export_max_pixels,
            keep_largest_water_component=args.keep_largest_water_component,
            overwrite=args.overwrite,
        )
        return {"command": args.command, "summary": summary}
    if args.command == "run-panelized-rivgraph":
        parent = WorkflowPaths.from_name(args.name, args.run_dir)
        mask = parent.cleaned_mask if args.mask is None else args.mask
        plan = parent.run_dir / "panels" / "panel_plan.gpkg"
        breaks = parent.manual_edits_dir / f"{parent.name}_panel_breaks.gpkg" if args.panel_breaks is None else args.panel_breaks
        if not mask.exists():
            raise FileNotFoundError(f"Mask does not exist: {mask}")
        if not breaks.exists():
            raise FileNotFoundError(f"Panel-break file does not exist: {breaks}. Create and review it before running this command.")
        planned = plan_panels(mask_path=mask, breaks_path=breaks, output_path=plan, overlap_m=args.overlap_m, max_panel_pixels=args.max_panel_pixels, overwrite=args.overwrite)
        prepared = prepare_panel_masks(mask_path=mask, plan_path=plan, output_dir=parent.run_dir / "panels", panel_ids=None, overwrite=args.overwrite)
        match_plan = parent.run_dir / "panel_matching" / "boundary_match_plan.json"
        boundary_plan = plan_boundary_matches(plan_path=plan, breaks_path=breaks, mask_path=mask, output_path=match_plan, global_exit_sides=args.exit_sides)
        manifest = json.loads((parent.run_dir / "panels" / "panel_mask_manifest.json").read_text())
        exits = json.loads((parent.run_dir / "panel_matching" / "panel_exit_sides.json").read_text())
        panel_results = []
        for item in manifest:
            if item.get("core_water_pixels", item["water_pixels"]) <= 0:
                continue
            panel = item["panel_id"]
            sides = exits.get(panel, {}).get("exit_sides", "").upper()
            if len(sides) != 2 or len(set(sides)) != 2 or any(side not in "NSEW" for side in sides):
                raise ValueError(f"Panel {panel} requires exactly two distinct valid exit sides; inspect panel_exit_sides.json.")
            _, summary = run_rivgraph_for_run(name=f"{parent.name}_{panel}", run_dir=parent.run_dir / "panels" / panel, mask=parent.run_dir / "panels" / panel / "mask.tif", mask_stage="explicit", exit_sides=sides, assign_flow=True, flow_mesh_mode=args.flow_mesh_mode)
            panel_results.append({"panel_id": panel, "n_links": summary["n_links"], "n_nodes": summary["n_nodes"]})
        core_dir = parent.run_dir / "merged_network" / "core_boundary"
        core = build_core_boundary_network(plan_path=plan, breaks_path=breaks, panel_root=parent.run_dir / "panels", run_name=parent.name, output_dir=core_dir, tolerance_m=args.boundary_tolerance_m, overwrite=args.overwrite, boundary_match_plan_path=match_plan)
        nodes = rebuild_core_network_nodes(merged_links_path=core_dir / "merged_core_links.gpkg", output_dir=core_dir, endpoint_tolerance_m=args.endpoint_tolerance_m, overwrite=args.overwrite)
        direction = assign_core_network_direction(merged_links_path=core_dir / "merged_links.gpkg", core_segments_path=core_dir / "core_segments.gpkg", panel_root=parent.run_dir / "panels", run_name=parent.name, output_dir=core_dir, endpoint_tolerance_m=args.endpoint_tolerance_m, overwrite=args.overwrite)
        handoff = publish_panelized_rivgraph_handoff(directed_links_path=core_dir / "merged_links_directed.gpkg", nodes_path=core_dir / "merged_nodes.gpkg", core_segments_path=core_dir / "core_segments.gpkg", plan_path=plan, panel_root=parent.run_dir / "panels", run_name=parent.name, rivgraph_dir=parent.rivgraph_dir, overwrite=args.overwrite)
        summary_path = parent.run_dir / "merged_network" / "panelized_workflow_summary.json"
        summary = {"mask": str(mask), "plan": planned, "breaks": str(breaks), "global_exit_sides": args.exit_sides.upper(), "prepared": prepared, "boundary_plan": boundary_plan, "panels": panel_results, "core_network": core, "nodes": nodes, "direction": direction, "handoff": handoff}
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return {"command": args.command, "summary_path": str(summary_path), "panel_count": len(panel_results), "directed_links": direction["output"], "ambiguous_link_count": direction["ambiguous_link_count"]}
    if args.command == "publish-panelized-rivgraph":
        parent = WorkflowPaths.from_name(args.name, args.run_dir)
        core_dir = parent.run_dir / "merged_network" / "core_boundary"
        return {"command": args.command, "summary": publish_panelized_rivgraph_handoff(directed_links_path=core_dir / "merged_links_directed.gpkg", nodes_path=core_dir / "merged_nodes.gpkg", core_segments_path=core_dir / "core_segments.gpkg", plan_path=parent.run_dir / "panels" / "panel_plan.gpkg", panel_root=parent.run_dir / "panels", run_name=parent.name, rivgraph_dir=parent.rivgraph_dir, overwrite=args.overwrite)}
    if args.command == "assign-core-network-direction":
        parent = WorkflowPaths.from_name(args.name, args.run_dir)
        output_dir = parent.run_dir / "merged_network" / "core_boundary"
        return {"command": args.command, "summary": assign_core_network_direction(merged_links_path=output_dir / "merged_links.gpkg", core_segments_path=output_dir / "core_segments.gpkg", panel_root=parent.run_dir / "panels", run_name=parent.name, output_dir=output_dir, endpoint_tolerance_m=args.endpoint_tolerance_m, overwrite=args.overwrite)}
    if args.command == "rebuild-core-network-nodes":
        parent = WorkflowPaths.from_name(args.name, args.run_dir)
        output_dir = parent.run_dir / "merged_network" / "core_boundary"
        merged_links = output_dir / "merged_core_links.gpkg" if args.merged_links is None else args.merged_links
        return {"command": args.command, "summary": rebuild_core_network_nodes(merged_links_path=merged_links, output_dir=output_dir, endpoint_tolerance_m=args.endpoint_tolerance_m, overwrite=args.overwrite)}
    if args.command == "build-core-boundary-network":
        parent = WorkflowPaths.from_name(args.name, args.run_dir)
        return {"command": args.command, "summary": build_core_boundary_network(plan_path=parent.run_dir / "panels" / "panel_plan.gpkg", breaks_path=parent.manual_edits_dir / f"{parent.name}_panel_breaks.gpkg", panel_root=parent.run_dir / "panels", run_name=parent.name, output_dir=parent.run_dir / "merged_network" / "core_boundary", tolerance_m=args.tolerance_m, overwrite=args.overwrite)}
    if args.command == "run-planned-panels":
        parent = WorkflowPaths.from_name(args.name, args.run_dir); manifest=json.loads((parent.run_dir / "panels" / "panel_mask_manifest.json").read_text())
        exits=json.loads((parent.run_dir / "panel_matching" / "panel_exit_sides.json").read_text())
        results=[]
        for item in manifest:
            if item.get("core_water_pixels", item["water_pixels"]) <= 0: continue
            panel=item["panel_id"]; sides=exits.get(panel, {}).get("exit_sides", "").upper()
            if len(sides) != 2 or len(set(sides)) != 2 or any(side not in "NSEW" for side in sides):
                raise ValueError(f"Panel {panel} requires exactly two distinct valid exit sides; edit panel_exit_sides.json.")
            _, summary=run_rivgraph_for_run(name=f"{parent.name}_{panel}", run_dir=parent.run_dir / "panels" / panel, mask=parent.run_dir / "panels" / panel / "mask.tif", mask_stage="explicit", exit_sides=sides, assign_flow=args.assign_flow, flow_mesh_mode=args.flow_mesh_mode)
            results.append({"panel_id":panel,"n_links":summary["n_links"],"n_nodes":summary["n_nodes"]})
        out=parent.run_dir / "panels" / "batch_run_summary.json"; out.write_text(json.dumps(results,indent=2)); return {"command":args.command,"summary_path":str(out),"panel_count":len(results)}
    if args.command == "match-planned-boundaries":
        parent=WorkflowPaths.from_name(args.name,args.run_dir); plan=json.loads((parent.run_dir / "panel_matching" / "boundary_match_plan.json").read_text()); results=[]
        for item in plan["pairs"]:
            left,right,br=item["left_panel"],item["right_panel"],item["break_id"]
            links=lambda panel: parent.run_dir / "panels" / panel / "rivgraph" / f"{parent.name}_{panel}_links.gpkg"
            out=parent.run_dir / "panel_matching" / f"{br}_{left}_{right}_matches.gpkg"
            results.append(match_panel_boundary_links(breaks_path=parent.manual_edits_dir / f"{parent.name}_panel_breaks.gpkg",left_links_path=links(left),right_links_path=links(right),break_id=br,left_panel=left,right_panel=right,output_path=out,overwrite=True))
        out=parent.run_dir / "panel_matching" / "batch_match_summary.json"; out.write_text(json.dumps(results,indent=2)); return {"command":args.command,"summary_path":str(out),"match_count":len(results)}
    if args.command == "plan-boundary-matches":
        parent = WorkflowPaths.from_name(args.name, args.run_dir)
        return {"command": args.command, "summary": plan_boundary_matches(plan_path=parent.run_dir / "panels" / "panel_plan.gpkg", breaks_path=parent.manual_edits_dir / f"{parent.name}_panel_breaks.gpkg", mask_path=parent.cleaned_mask, output_path=parent.run_dir / "panel_matching" / "boundary_match_plan.json", global_exit_sides="SE")}
    if args.command == "validate-pilot-topology":
        parent = WorkflowPaths.from_name(args.name, args.run_dir)
        return {"command": args.command, "summary": validate_pilot_topology(links_path=args.links, nodes_path=args.nodes, output_path=parent.run_dir / "merged_network" / "pilot_reconciled" / "topology_validation.json")}
    if args.command == "rebuild-pilot-nodes":
        parent = WorkflowPaths.from_name(args.name, args.run_dir)
        output = parent.run_dir / "merged_network" / "pilot_reconciled" if args.output_dir is None else args.output_dir
        return {"command": args.command, "summary": rebuild_pilot_nodes(merged_links_path=args.merged_links, output_dir=output, overwrite=args.overwrite)}
    if args.command == "merge-panel-pilot":
        parent = WorkflowPaths.from_name(args.name, args.run_dir)
        links = lambda panel: parent.run_dir / "panels" / panel / "rivgraph" / f"{parent.name}_{panel}_links.gpkg"
        summary = concatenate_matched_panel_links(left_links_path=links(args.left_panel), right_links_path=links(args.right_panel), matches_path=args.matches, plan_path=parent.run_dir / "panels" / "panel_plan.gpkg", output_path=parent.run_dir / "merged_network" / f"{args.left_panel}_{args.right_panel}_merged_links.gpkg", left_panel=args.left_panel, right_panel=args.right_panel, overwrite=args.overwrite)
        return {"command": args.command, "summary": summary}
    if args.command == "match-panel-boundary":
        parent = WorkflowPaths.from_name(args.name, args.run_dir)
        links = lambda panel: parent.run_dir / "panels" / panel / "rivgraph" / f"{parent.name}_{panel}_links.gpkg"
        summary = match_panel_boundary_links(breaks_path=parent.manual_edits_dir / f"{parent.name}_panel_breaks.gpkg", left_links_path=links(args.left_panel), right_links_path=links(args.right_panel), break_id=args.break_id, left_panel=args.left_panel, right_panel=args.right_panel, output_path=parent.run_dir / "panel_matching" / f"{args.break_id}_{args.left_panel}_{args.right_panel}_matches.gpkg", tolerance_m=args.tolerance_m, overwrite=args.overwrite)
        return {"command": args.command, "summary": summary}
    if args.command == "run-panel-rivgraph":
        parent = WorkflowPaths.from_name(args.name, args.run_dir)
        panel_dir = parent.run_dir / "panels" / args.panel_id
        paths, summary = run_rivgraph_for_run(name=f"{parent.name}_{args.panel_id}", run_dir=panel_dir, mask=panel_dir / "mask.tif", mask_stage="explicit", exit_sides=args.exit_sides, assign_flow=args.assign_flow, flow_mesh_mode=args.flow_mesh_mode)
        return {"command": args.command, "run_paths": paths_to_dict(paths), "summary": summary}
    if args.command in {"create-panel-breaks", "plan-panels", "prepare-panel-masks"}:
        paths = WorkflowPaths.from_name(args.name, args.run_dir)
        mask = (paths.prepared_mask if args.command == "create-panel-breaks" else paths.cleaned_mask) if args.mask is None else args.mask
        breaks = paths.manual_edits_dir / f"{paths.name}_panel_breaks.gpkg" if getattr(args, "panel_breaks", None) is None else args.panel_breaks
        if args.command == "create-panel-breaks":
            summary = create_panel_breaks(mask_path=mask, output_path=(paths.manual_edits_dir / f"{paths.name}_panel_breaks.gpkg") if args.output is None else args.output, rows=args.rows, columns=args.columns, max_panel_pixels=args.max_panel_pixels, overwrite=args.overwrite)
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
