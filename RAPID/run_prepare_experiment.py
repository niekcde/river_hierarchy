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
    parser.add_argument("--forcing-station-key", help="Explicit station_key to extract from a multi-station forcing source.")
    parser.add_argument("--station-key-column", default="station_key", help="Station-key column in the forcing file.")
    parser.add_argument("--forcing-start-time", help="Optional inclusive UTC truncation start for forcing normalization.")
    parser.add_argument("--forcing-end-time", help="Optional inclusive UTC truncation end for forcing normalization.")
    parser.add_argument("--forcing-resample-minutes", type=int, help="Optional normalized forcing interval in minutes, e.g. 15.")
    parser.add_argument(
        "--forcing-output-cache-dir",
        help="Optional shared cache directory for normalized forcing tables keyed by source, station, time window, and timestep.",
    )
    parser.add_argument("--time-column", default="time", help="Timestamp column in the forcing file.")
    parser.add_argument("--discharge-column", default="discharge", help="Discharge column in the forcing file.")
    parser.add_argument("--width-field", default="wid_adj_wet", help="Preferred link width field for RAPID K preparation.")
    parser.add_argument("--x-value", type=float, default=0.1, help="Muskingum X parameter.")
    parser.add_argument("--kb-value", type=float, default=20.0, help="K-value helper parameter kb.")
    parser.add_argument("--kb-mode", choices=["fixed", "based_reference_section"], default="fixed", help="How to select kb for RAPID K preparation.")
    parser.add_argument("--kb-model-path", help="Optional local path to the BASED model .ubj file when kb-mode=based_reference_section.")
    parser.add_argument("--kb-width-sample-field", default="width_wet", help="Reference-section width sample field used when kb-mode=based_reference_section.")
    parser.add_argument("--kb-width-percentile", type=float, default=90.0, help="Reference-section width percentile used when kb-mode=based_reference_section.")
    parser.add_argument("--n-manning", type=float, default=0.35, help="Manning roughness used in K preparation.")
    parser.add_argument("--min-width", type=float, default=1.0, help="Minimum positive width used when width is missing or invalid.")
    parser.add_argument("--min-effective-length-for-k-m", type=float, help="Optional K-only lower bound for effective reach length in meters. Geometry and exported link lengths are unchanged.")
    parser.add_argument("--use-celerity-capping", action="store_true", help="Cap the implied hydraulic celerity before converting it to RAPID K.")
    parser.add_argument("--min-celerity-mps", type=float, default=0.28, help="Minimum celerity bound in m/s when celerity capping is enabled.")
    parser.add_argument("--max-celerity-mps", type=float, default=1.524, help="Maximum celerity bound in m/s when celerity capping is enabled.")
    parser.add_argument("--target-subreach-length-m", type=float, help="Target RAPID-only subreach length in meters. Long links are split using max(1, round(L / target)).")
    parser.add_argument("--min-slope", type=float, default=1e-6, help="Minimum positive slope used when slope is missing or invalid.")
    parser.add_argument("--max-slope-for-k", type=float, help="Optional upper bound for local slopes used in RAPID K preparation.")
    parser.add_argument("--section-slope-ratio-min", type=float, help="Optional lower bound on local_slope / section_slope before a local slope is flagged as an outlier.")
    parser.add_argument("--section-slope-ratio-max", type=float, help="Optional upper bound on local_slope / section_slope before a local slope is flagged as an outlier.")
    parser.add_argument("--disable-section-slope-fallback", action="store_true", help="If a local slope is flagged and no valid neighboring slope is found, fall back to the minimum slope instead of the section reference slope.")
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
            station_key_column=args.station_key_column,
            station_key=args.forcing_station_key,
            start_time=args.forcing_start_time,
            end_time=args.forcing_end_time,
            resample_minutes=args.forcing_resample_minutes,
            output_cache_dir=args.forcing_output_cache_dir,
        ),
        prep_config=RapidPrepConfig(
            width_field=args.width_field,
            x_value=args.x_value,
            kb_value=args.kb_value,
            kb_mode=args.kb_mode,
            kb_model_path=args.kb_model_path,
            kb_width_sample_field=args.kb_width_sample_field,
            kb_width_percentile=args.kb_width_percentile,
            n_manning=args.n_manning,
            min_width=args.min_width,
            min_effective_length_for_k_m=args.min_effective_length_for_k_m,
            use_celerity_capping=args.use_celerity_capping,
            min_celerity_mps=args.min_celerity_mps,
            max_celerity_mps=args.max_celerity_mps,
            target_subreach_length_m=args.target_subreach_length_m,
            min_slope=args.min_slope,
            max_slope_for_k=args.max_slope_for_k,
            section_slope_ratio_min=args.section_slope_ratio_min,
            section_slope_ratio_max=args.section_slope_ratio_max,
            use_section_slope_fallback=not args.disable_section_slope_fallback,
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
