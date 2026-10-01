from __future__ import annotations

"""Generate immutable raw-pass reference-profile artifacts for one experiment."""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from rapid_tools.reference_profile import (
    ReferenceProfileConfig,
    apply_manual_linear_profile_bridge,
    fit_reference_profile,
    select_reference_profile,
    write_reference_profile_artifacts,
)
from rapid_tools.reference_profile_inputs import build_reference_profile_inputs


def _lambda_values(value: str | None) -> list[float] | None:
    if value is None or not value.strip():
        return None
    return [float(token.strip()) for token in value.split(",") if token.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the reviewed endpoint-bounded SWORD corridor, fit the accepted raw-pass "
            "stage-adjusted profile, and write immutable artifacts. Existing hydraulic contexts are untouched."
        )
    )
    parser.add_argument("experiment_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--sword-node-source", required=True)
    parser.add_argument(
        "--raw-observations-path",
        default=str(ROOT.parent / "data_validation" / "swot_sword_17c" / "target_observations.parquet"),
    )
    parser.add_argument("--max-centerline-offset-m", type=float, default=350.0)
    parser.add_argument("--endpoint-buffer-m", type=float, default=1_000.0)
    parser.add_argument("--lambda-values", help="Optional comma-separated lambda grid; default is the complete accepted grid.")
    parser.add_argument(
        "--fixed-lambda",
        type=float,
        help=(
            "Materialise one already reviewed lambda without rerunning CV selection. "
            "Use only a lambda recorded by slope_investigation.ipynb."
        ),
    )
    parser.add_argument(
        "--fixed-lambda-decision",
        default="accepted_notebook_fixed_lambda",
        help="Durable provenance label written when --fixed-lambda is used.",
    )
    parser.add_argument("--weight-mode", choices=("equal", "capped_inverse_variance"), default="equal")
    parser.add_argument("--uncertainty-floor-m", type=float, default=0.15)
    parser.add_argument("--minimum-slope-fraction", type=float, default=0.01)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument(
        "--manual-linear-bridge",
        action="append",
        nargs=3,
        metavar=("ADJUSTMENT_ID", "START_DISTANCE_M", "END_DISTANCE_M"),
        help=(
            "Documented profile-level linear bridge. May be repeated. The bridge is continuous "
            "because it uses fitted heads at the resolved fixed-profile endpoint stations."
        ),
    )
    args = parser.parse_args(argv)

    inputs = build_reference_profile_inputs(
        args.experiment_dir,
        sword_node_source=args.sword_node_source,
        raw_observations_path=args.raw_observations_path,
        max_centerline_offset_m=args.max_centerline_offset_m,
        endpoint_buffer_m=args.endpoint_buffer_m,
    )
    config = ReferenceProfileConfig(
        weight_mode=args.weight_mode,
        uncertainty_floor_m=args.uncertainty_floor_m,
        minimum_slope_fraction=args.minimum_slope_fraction,
        cv_folds=args.cv_folds,
    )
    if args.fixed_lambda is not None:
        if args.fixed_lambda <= 0.0:
            parser.error("--fixed-lambda must be positive.")
        fit = fit_reference_profile(
            inputs.static_profile,
            inputs.observations,
            endpoint_distances_m=inputs.endpoint_distances_m,
            lambda_value=args.fixed_lambda,
            config=config,
        )
        cv = pd.DataFrame(
            [{
                "lambda_value": args.fixed_lambda,
                "cv_mean_mse_m2": pd.NA,
                "cv_sd_mse_m2": pd.NA,
                "cv_se_mse_m2": pd.NA,
                "n_acquisitions": int(fit.summary["n_acquisitions"]),
            }]
        )
        selection = {
            "selection_rule": str(args.fixed_lambda_decision),
            "selected_lambda": float(args.fixed_lambda),
            "best_lambda": float(args.fixed_lambda),
            "one_standard_error_threshold_m2": None,
            "selected_at_grid_upper_edge": False,
            "lambda_grid_count": 1,
            "cv_reused_from_notebook": True,
        }
    else:
        fit, cv, selection = select_reference_profile(
            inputs.static_profile,
            inputs.observations,
            endpoint_distances_m=inputs.endpoint_distances_m,
            lambda_values=_lambda_values(args.lambda_values),
            config=config,
        )
    manual_adjustments = []
    for adjustment_id, start_distance_m, end_distance_m in args.manual_linear_bridge or []:
        fit, record = apply_manual_linear_profile_bridge(
            fit,
            adjustment_id=adjustment_id,
            start_distance_m=float(start_distance_m),
            end_distance_m=float(end_distance_m),
        )
        manual_adjustments.append(record)
    outputs = write_reference_profile_artifacts(
        fit, cv, selection, args.output_dir, config=config, input_metadata=inputs.metadata,
        manual_adjustments=manual_adjustments,
    )
    print(f"Corridor SWORD nodes: {inputs.metadata['n_static_profile_nodes']}")
    print(f"Raw observations/acquisitions: {inputs.metadata['n_raw_observations']}/{inputs.metadata['n_raw_acquisitions']}")
    print(f"Selected lambda: {selection['selected_lambda']:.12g}")
    for record in manual_adjustments:
        print(
            "Manual bridge: "
            f"{record['adjustment_id']} [{record['resolved_start_distance_m']:.3f}, "
            f"{record['resolved_end_distance_m']:.3f}] m at {record['bridge_slope_m_per_m']:.8g} m/m"
        )
    print(f"Artifacts: {outputs['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
