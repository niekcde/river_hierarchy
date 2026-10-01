from __future__ import annotations

"""Fit immutable raw-pass reference-profile artifacts from prepared tabular inputs."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from rapid_tools.reference_profile import (
    ReferenceProfileConfig,
    select_reference_profile,
    write_reference_profile_artifacts,
)


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _lambda_values(value: str | None) -> list[float] | None:
    if value is None or not value.strip():
        return None
    values = [float(token.strip()) for token in value.split(",") if token.strip()]
    if not values:
        raise argparse.ArgumentTypeError("--lambda-values must contain at least one number.")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a raw-pass, acquisition-stage-adjusted monotone reference WSE profile. "
            "Only immutable profile artifacts are written; hydraulic contexts are untouched."
        )
    )
    parser.add_argument("static_profile", help="CSV/Parquet with reference_section_dist_out_m and static_wse_m.")
    parser.add_argument("observations", help="CSV/Parquet with reference_section_dist_out_m, acquisition_id, and wse_m.")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--endpoint-distances-m", nargs=2, type=float, required=True,
        metavar=("OUTLET_X", "INLET_X"),
        help="Exact realised outlet and inlet distance-to-outlet coordinates [m].",
    )
    parser.add_argument("--lambda-values", help="Optional comma-separated positive lambda grid. Default: accepted broad grid.")
    parser.add_argument("--weight-mode", choices=("equal", "capped_inverse_variance"), default="equal")
    parser.add_argument("--uncertainty-floor-m", type=float, default=0.15)
    parser.add_argument("--minimum-slope-fraction", type=float, default=0.01)
    parser.add_argument("--cv-folds", type=int, default=3)
    args = parser.parse_args(argv)

    static_path = Path(args.static_profile).expanduser().resolve()
    observations_path = Path(args.observations).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    config = ReferenceProfileConfig(
        weight_mode=args.weight_mode,
        uncertainty_floor_m=args.uncertainty_floor_m,
        minimum_slope_fraction=args.minimum_slope_fraction,
        cv_folds=args.cv_folds,
    )
    fit, cv, selection = select_reference_profile(
        _read_table(static_path),
        _read_table(observations_path),
        endpoint_distances_m=args.endpoint_distances_m,
        lambda_values=_lambda_values(args.lambda_values),
        config=config,
    )
    outputs = write_reference_profile_artifacts(
        fit, cv, selection, output_dir, config=config,
        input_metadata={
            "static_profile": str(static_path),
            "observations": str(observations_path),
            "endpoint_distances_m": [float(value) for value in args.endpoint_distances_m],
        },
    )
    print(f"Selected lambda: {selection['selected_lambda']:.12g}")
    print(f"Profile nodes: {fit.summary['n_profile_nodes']}")
    print(f"Raw observations: {fit.summary['n_observations']}")
    print(f"Artifacts: {outputs['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
