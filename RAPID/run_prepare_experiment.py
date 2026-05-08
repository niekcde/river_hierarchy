from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rapid_tools.forcing import ForcingConfig
from rapid_tools.prep import RapidPrepConfig, prepare_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare RAPID inputs for every state in a network_variants experiment."
    )
    parser.add_argument("experiment_dir", help="Path to the network_variants experiment directory.")
    parser.add_argument("--forcing-path", help="CSV or parquet file with time/discharge forcing.")
    parser.add_argument("--time-column", default="time", help="Timestamp column in the forcing file.")
    parser.add_argument("--discharge-column", default="discharge", help="Discharge column in the forcing file.")
    parser.add_argument("--width-field", default="wid_adj_wet", help="Preferred link width field for RAPID K preparation.")
    parser.add_argument("--x-value", type=float, default=0.1, help="Muskingum X parameter.")
    parser.add_argument("--kb-value", type=float, default=20.0, help="K-value helper parameter kb.")
    parser.add_argument("--n-manning", type=float, default=0.35, help="Manning roughness used in K preparation.")
    parser.add_argument("--min-width", type=float, default=1.0, help="Minimum positive width used when width is missing or invalid.")
    parser.add_argument("--use-celerity-capping", action="store_true", help="Cap the implied hydraulic celerity before converting it to RAPID K.")
    parser.add_argument("--min-celerity-mps", type=float, default=0.28, help="Minimum celerity bound in m/s when celerity capping is enabled.")
    parser.add_argument("--max-celerity-mps", type=float, default=1.524, help="Maximum celerity bound in m/s when celerity capping is enabled.")
    parser.add_argument("--target-subreach-length-m", type=float, help="Target RAPID-only subreach length in meters. Long links are split using max(1, round(L / target)).")
    parser.add_argument("--min-slope", type=float, default=1e-6, help="Minimum positive slope used when slope is missing or invalid.")
    parser.add_argument("--preferred-length-field", default="len", help="Fallback link-length field if geometry length is unavailable.")
    parser.add_argument("--exclude-base-state", action="store_true", help="Skip the base state and prepare only derived states.")
    parser.add_argument("--allow-missing-sword", action="store_true", help="Allow prep to continue when SWORD WSE columns are missing on directed nodes.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    prep_registry = prepare_experiment(
        args.experiment_dir,
        forcing_path=args.forcing_path,
        forcing_config=ForcingConfig(
            time_column=args.time_column,
            discharge_column=args.discharge_column,
        ),
        prep_config=RapidPrepConfig(
            width_field=args.width_field,
            x_value=args.x_value,
            kb_value=args.kb_value,
            n_manning=args.n_manning,
            min_width=args.min_width,
            use_celerity_capping=args.use_celerity_capping,
            min_celerity_mps=args.min_celerity_mps,
            max_celerity_mps=args.max_celerity_mps,
            target_subreach_length_m=args.target_subreach_length_m,
            min_slope=args.min_slope,
            preferred_length_field=args.preferred_length_field,
            include_base_state=not args.exclude_base_state,
            strict_sword=not args.allow_missing_sword,
        ),
    )
    print(f"Wrote RAPID prep outputs to {Path(args.experiment_dir).expanduser().resolve()}")
    print(f"Prepared states: {int(prep_registry['status'].eq('prepared').sum()) if not prep_registry.empty else 0}")
    print(f"Failed states: {int(prep_registry['status'].eq('failed').sum()) if not prep_registry.empty else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
