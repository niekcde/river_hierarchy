from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
REPO_ROOT = ROOT.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    parser.add_argument("--forcing-resample-minutes", type=float, help="Optional normalized forcing interval in minutes. Fractional values are allowed if they resolve to whole seconds.")
    parser.add_argument("--forcing-resample-seconds", type=int, help="Optional normalized forcing interval in seconds. Prefer this when the admissible RAPID timestep is not a whole number of minutes.")
    parser.add_argument("--spinup-hours", type=float, help="Optional constant-flow spin-up duration in hours using the first normalized gauge discharge value.")
    parser.add_argument("--auto-spinup-from-k", action="store_true", help="Compute spin-up automatically from the longest downstream summed-K path in the prepared RAPID network.")
    parser.add_argument("--spinup-k-multiplier", type=float, default=1.1, help="Multiplier applied to the longest downstream summed-K path when auto-spinup-from-k is enabled.")
    parser.add_argument("--disable-end-padding-with-spinup", action="store_true", help="Do not append a constant tail equal to the spin-up duration using the final normalized gauge discharge value.")
    parser.add_argument(
        "--forcing-output-cache-dir",
        help="Optional shared cache directory for normalized forcing tables keyed by source, station, time window, and timestep.",
    )
    parser.add_argument("--time-column", default="time", help="Timestamp column in the forcing file.")
    parser.add_argument("--discharge-column", default="discharge", help="Discharge column in the forcing file.")
    parser.add_argument("--width-field", default="wid_adj_routing", help="Preferred effective link width field for RAPID K preparation.")
    parser.add_argument("--vary-width-by-subreach", action="store_true", help="Derive a width for each RAPID subreach from variant link_width_samples.csv instead of copying one parent-link width across all segments.")
    parser.add_argument("--subreach-width-sample-field", default="width_wet", help="Width sample field from link_width_samples.csv used when --vary-width-by-subreach is enabled.")
    parser.add_argument(
        "--slope-profile-mode",
        choices=["network_dist", "reference_section"],
        default="network_dist",
        help="Legacy metadata only. Slope/profile construction must already be written to state hydraulic_context outputs during variant creation.",
    )
    parser.add_argument(
        "--reference-section-summary-path",
        help="Legacy compatibility option. Reference-section slope inputs must already be resolved upstream in network_variants.",
    )
    parser.add_argument("--x-value", type=float, default=0.1, help="Muskingum X parameter.")
    parser.add_argument("--kb-value", type=float, default=20.0, help="K-value helper parameter kb.")
    parser.add_argument("--kb-mode", choices=["fixed", "based_reference_section"], default="fixed", help="How to select kb for RAPID K preparation.")
    parser.add_argument("--kb-model-path", help="Optional local path to the BASED model .ubj file when kb-mode=based_reference_section.")
    parser.add_argument("--kb-width-sample-field", default="wid_adj_wet", help="Reference-section width field used when kb-mode=based_reference_section. Summary width fields on the reference link are preferred over raw sample percentiles when available.")
    parser.add_argument("--kb-width-percentile", type=float, default=90.0, help="Reference-section width percentile used only when the selected kb width field is read from raw sample widths.")
    parser.add_argument("--n-manning", type=float, default=0.35, help="Manning roughness used in K preparation.")
    parser.add_argument("--min-width", type=float, default=1.0, help="Minimum positive width used when width is missing or invalid.")
    parser.add_argument("--min-effective-length-for-k-m", type=float, help="Optional K-only lower bound for effective reach length in meters. Geometry and exported link lengths are unchanged.")
    parser.add_argument("--use-celerity-capping", action="store_true", help="Cap the implied hydraulic celerity before converting it to RAPID K.")
    parser.add_argument("--min-celerity-mps", type=float, default=0.28, help="Minimum celerity bound in m/s when celerity capping is enabled.")
    parser.add_argument("--max-celerity-mps", type=float, default=1.524, help="Maximum celerity bound in m/s when celerity capping is enabled.")
    parser.add_argument("--use-stability-effective-length-cap", action="store_true", help="In unstable or near-unstable states, shorten K-only effective lengths on minimum-celerity reaches to the maximum value allowed by the current dt_max and the requested safety margin.")
    parser.add_argument("--stability-effective-length-cap-margin-seconds", type=float, default=10.0, help="Requested safety margin below dt_max when deriving the statewise K-only effective-length cap.")
    parser.add_argument("--stability-effective-length-cap-celerity-rel-buffer", type=float, default=0.03, help="Relative buffer above min-celerity that is still treated as near-floor for the stability effective-length cap.")
    parser.add_argument("--preferred-routing-dt-seconds", type=int, help="Preferred universal routing timestep. When multiple admissible divisors exist, choose the one closest to this value instead of the midpoint of the stability interval.")
    parser.add_argument("--forcing-dt-search-max-offset-seconds", type=int, default=120, help="If the current forcing timestep has no admissible routing divisor, search nearby forcing timesteps within this offset to find a compatible routing timestep.")
    parser.add_argument("--target-subreach-length-m", type=float, help="Target RAPID-only subreach length in meters. Long links are split using max(1, round(L / target)).")
    parser.add_argument("--min-slope", type=float, default=1e-6, help="Legacy metadata only. Effective slope limits should be applied during variant hydraulic-context generation, not RAPID prep.")
    parser.add_argument("--max-slope-for-k", type=float, help="Legacy metadata only. Effective slope limits should be applied during variant hydraulic-context generation, not RAPID prep.")
    parser.add_argument("--section-slope-ratio-min", type=float, help="Legacy metadata only. Section-slope outlier rules should be applied during variant hydraulic-context generation, not RAPID prep.")
    parser.add_argument("--section-slope-ratio-max", type=float, help="Legacy metadata only. Section-slope outlier rules should be applied during variant hydraulic-context generation, not RAPID prep.")
    parser.add_argument("--disable-section-slope-fallback", action="store_true", help="Legacy metadata only. Section-slope fallback is expected to be resolved upstream in network_variants.")
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
            resample_seconds=args.forcing_resample_seconds,
            spinup_hours=args.spinup_hours,
            auto_spinup_from_k=args.auto_spinup_from_k,
            spinup_k_multiplier=args.spinup_k_multiplier,
            pad_end_with_spinup=not args.disable_end_padding_with_spinup,
            output_cache_dir=args.forcing_output_cache_dir,
        ),
        prep_config=RapidPrepConfig(
            width_field=args.width_field,
            vary_width_by_subreach=args.vary_width_by_subreach,
            subreach_width_sample_field=args.subreach_width_sample_field,
            slope_profile_mode=args.slope_profile_mode,
            reference_section_summary_path=args.reference_section_summary_path,
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
            use_stability_effective_length_cap=args.use_stability_effective_length_cap,
            stability_effective_length_cap_margin_seconds=args.stability_effective_length_cap_margin_seconds,
            stability_effective_length_cap_celerity_rel_buffer=args.stability_effective_length_cap_celerity_rel_buffer,
            preferred_routing_dt_seconds=args.preferred_routing_dt_seconds,
            forcing_dt_search_max_offset_seconds=args.forcing_dt_search_max_offset_seconds,
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
