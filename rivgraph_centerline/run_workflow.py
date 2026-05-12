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

    prepare_parser = subparsers.add_parser("prepare-mask", help="Create the binary/projected mask once.")
    add_run_identity_args(prepare_parser)
    prepare_parser.add_argument("--source-mask", type=Path, required=True, help="Source river mask GeoTIFF.")
    prepare_parser.add_argument("--threshold", type=float, default=0.5, help="Water threshold for the binary mask.")
    prepare_parser.add_argument("--dst-crs", default=None, help="Projected CRS override.")
    prepare_parser.add_argument("--force", action="store_true", help="Rebuild the prepared mask even if it exists.")

    edits_parser = subparsers.add_parser("apply-edits", help="Apply manual polygon edits to the prepared mask.")
    add_run_identity_args(edits_parser)
    edits_parser.add_argument("--edits", type=Path, default=None, help="Vector edits file. Defaults to run-dir/manual_edits/<name>_manual_edits.gpkg.")
    edits_parser.add_argument("--base-mask", type=Path, default=None, help="Prepared mask override.")
    edits_parser.add_argument("--output-mask", type=Path, default=None, help="Cleaned mask output override.")
    edits_parser.add_argument("--layer", default=None, help="Vector layer name. Defaults to the first layer.")
    edits_parser.add_argument("--action-field", default="action", help="Field containing add/remove values.")
    edits_parser.add_argument("--center-only", action="store_true", help="Only edit pixels whose centers fall inside polygons.")

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
    all_parser.add_argument("--force-prepare", action="store_true", help="Rebuild the prepared mask even if it exists.")
    all_parser.add_argument("--quiet", action="store_true", help="Disable verbose RivGraph logging.")

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
