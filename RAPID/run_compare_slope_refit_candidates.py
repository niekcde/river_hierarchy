from __future__ import annotations

"""Read-only slope and RAPID-parameter impact screen for candidate contexts.

This does not prepare or execute RAPID.  It translates changed base-link slopes
to the same Manning/width/celerity expression used in preparation, retaining
the stored experiment cap settings.  Equivalent resistance is explicitly
reported as unchanged because its present solver has no ``slope_used`` input.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

import geopandas as gpd
import numpy as np
import pandas as pd


COMPARISON_SCHEMA_VERSION = "slope-refit-candidate-impact-v1"


def _canonical_id(value: object) -> str:
    return f"{int(float(value)):02d}"


def _cap_status(values: pd.Series, minimum: float, maximum: float, enabled: bool) -> pd.Series:
    if not enabled:
        return pd.Series("cap_disabled", index=values.index)
    return pd.Series(
        np.select(
            [values.lt(minimum), values.gt(maximum)],
            ["below_min_cap", "above_max_cap"],
            default="within_cap_range",
        ),
        index=values.index,
    )


def _one_width_path(experiment_dir: Path) -> Path:
    width_dir = experiment_dir / "states" / "S000_base" / "variant" / "widths"
    preferred = width_dir / "links_with_width_families.gpkg"
    if preferred.exists():
        return preferred
    matches = sorted(width_dir.glob("*width*.gpkg"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one base width GPKG in {width_dir}; found {matches}")
    return matches[0]


def _impact_for_example(item: object) -> pd.DataFrame:
    example_id = _canonical_id(item.example_id)
    experiment = Path(item.experiment_dir)
    candidate = pd.read_parquet(Path(item.candidate_context_dir) / "candidate_link_hydraulic_context.parquet")
    frame = candidate.copy()
    current_slope = pd.to_numeric(frame["slope_used"], errors="coerce")
    candidate_slope = pd.to_numeric(frame["candidate_slope_m_per_m"], errors="coerce")
    frame["example_id"] = example_id
    frame["current_slope_m_per_m"] = current_slope
    frame["candidate_slope_m_per_m"] = candidate_slope
    frame["slope_ratio_candidate_to_current"] = candidate_slope / current_slope
    frame["raw_celerity_ratio_candidate_to_current"] = np.sqrt(frame["slope_ratio_candidate_to_current"])
    frame["raw_k_ratio_candidate_to_current"] = 1.0 / frame["raw_celerity_ratio_candidate_to_current"]
    frame["equivalent_resistance_ratio_candidate_to_current"] = 1.0
    frame["equivalent_resistance_reason"] = "current_equivalent_resistance_uses_geometry_width_roughness_kb_and_normalized_head_only"
    prep_path = experiment / "rapid_prep_manifest.json"
    if not prep_path.exists():
        frame["rapid_parameter_translation_status"] = "rapid_prep_manifest_unavailable"
        for column in (
            "current_celerity_mps_raw", "candidate_celerity_mps_raw",
            "current_celerity_mps_capped", "candidate_celerity_mps_capped",
            "capped_k_ratio_candidate_to_current", "kb_value_used", "n_manning_used",
        ):
            frame[column] = np.nan
        frame["current_celerity_cap_status"] = "rapid_config_unavailable"
        frame["candidate_celerity_cap_status"] = "rapid_config_unavailable"
        frame["celerity_cap_transition"] = "rapid_config_unavailable"
        frame["rapid_parameter_screen_method"] = "not_available_without_stored_rapid_prep_manifest"
        return frame
    manifest = json.loads(prep_path.read_text(encoding="utf-8"))
    config = manifest.get("prep_config", {}) or {}
    resolved_kb = manifest.get("resolved_kb_summary", {}) or {}
    width_field = str(config.get("width_field", "wid_adj_routing"))
    widths = gpd.read_file(_one_width_path(experiment))
    if width_field not in widths.columns:
        raise ValueError(f"Configured width field {width_field!r} is absent from base width output.")
    width_frame = widths[["id_link", width_field]].copy()
    width_frame["id_link"] = pd.to_numeric(width_frame["id_link"], errors="coerce").astype("Int64")
    width_frame[width_field] = pd.to_numeric(width_frame[width_field], errors="coerce")
    frame = frame.merge(width_frame, on="id_link", how="left", validate="1:1")
    width = pd.to_numeric(frame[width_field], errors="coerce").clip(lower=float(config.get("min_width", 1.0)))
    n_manning = float(config.get("n_manning", 0.35))
    kb_value = float(resolved_kb.get("kb_value", config.get("kb_value", 20.0)))
    celerity_factor = (5.0 / (3.0 * n_manning)) * np.power(width, 2.0 / 3.0) / (kb_value ** (2.0 / 3.0))
    frame["current_celerity_mps_raw"] = celerity_factor * np.sqrt(current_slope)
    frame["candidate_celerity_mps_raw"] = celerity_factor * np.sqrt(candidate_slope)
    use_cap = bool(config.get("use_celerity_capping", False))
    min_cap = float(config.get("min_celerity_mps", 0.0))
    max_cap = float(config.get("max_celerity_mps", np.inf))
    frame["current_celerity_cap_status"] = _cap_status(frame["current_celerity_mps_raw"], min_cap, max_cap, use_cap)
    frame["candidate_celerity_cap_status"] = _cap_status(frame["candidate_celerity_mps_raw"], min_cap, max_cap, use_cap)
    frame["current_celerity_mps_capped"] = frame["current_celerity_mps_raw"].clip(lower=min_cap, upper=max_cap) if use_cap else frame["current_celerity_mps_raw"]
    frame["candidate_celerity_mps_capped"] = frame["candidate_celerity_mps_raw"].clip(lower=min_cap, upper=max_cap) if use_cap else frame["candidate_celerity_mps_raw"]
    frame["capped_k_ratio_candidate_to_current"] = frame["current_celerity_mps_capped"] / frame["candidate_celerity_mps_capped"]
    frame["celerity_cap_transition"] = frame["current_celerity_cap_status"] + " -> " + frame["candidate_celerity_cap_status"]
    frame["rapid_parameter_translation_status"] = "computed_from_stored_rapid_prep_manifest"
    frame["rapid_parameter_screen_method"] = "stored_base_width_and_prep_config_link_level_translation"
    frame["kb_value_used"] = kb_value
    frame["n_manning_used"] = n_manning
    return frame


def compare_candidate_batch(candidate_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(candidate_root).expanduser().resolve()
    manifest = pd.read_csv(root / "slope_refit_candidate_batch_manifest.csv")
    frames = []
    for item in manifest.itertuples(index=False):
        if str(item.status) not in {"complete", "already_complete"}:
            raise ValueError(f"Candidate batch is not complete: Example {item.example_id} status={item.status}")
        frames.append(_impact_for_example(item))
    links = pd.concat(frames, ignore_index=True)
    finite = links.replace([np.inf, -np.inf], np.nan).dropna(subset=["slope_ratio_candidate_to_current"]).copy()
    finite["slope_change_over_25pct"] = finite["slope_ratio_candidate_to_current"].lt(0.8) | finite["slope_ratio_candidate_to_current"].gt(1.25)
    finite["capped_k_change_over_25pct"] = finite["capped_k_ratio_candidate_to_current"].notna() & (finite["capped_k_ratio_candidate_to_current"].lt(0.8) | finite["capped_k_ratio_candidate_to_current"].gt(1.25))
    summary = finite.groupby("example_id", as_index=False).agg(
        n_links=("id_link", "size"),
        median_slope_ratio=("slope_ratio_candidate_to_current", "median"),
        p05_slope_ratio=("slope_ratio_candidate_to_current", lambda value: value.quantile(0.05)),
        p95_slope_ratio=("slope_ratio_candidate_to_current", lambda value: value.quantile(0.95)),
        n_slope_change_over_25pct=("slope_change_over_25pct", "sum"),
        median_uncapped_k_ratio=("raw_k_ratio_candidate_to_current", "median"),
        median_capped_k_ratio=("capped_k_ratio_candidate_to_current", "median"),
        n_capped_k_change_over_25pct=("capped_k_change_over_25pct", "sum"),
        n_current_below_min_cap=("current_celerity_cap_status", lambda value: int(value.eq("below_min_cap").sum())),
        n_candidate_below_min_cap=("candidate_celerity_cap_status", lambda value: int(value.eq("below_min_cap").sum())),
        n_current_above_max_cap=("current_celerity_cap_status", lambda value: int(value.eq("above_max_cap").sum())),
        n_candidate_above_max_cap=("candidate_celerity_cap_status", lambda value: int(value.eq("above_max_cap").sum())),
        n_links_with_rapid_translation=("rapid_parameter_translation_status", lambda value: int(value.eq("computed_from_stored_rapid_prep_manifest").sum())),
    )
    return links, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare validated slope-refit candidates to current base contexts without RAPID reruns.")
    parser.add_argument("candidate_root")
    parser.add_argument("output_dir", help="New empty directory for read-only impact artifacts.")
    args = parser.parse_args(argv)
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Impact output must be empty: {output}")
    links, summary = compare_candidate_batch(args.candidate_root)
    output.mkdir(parents=True, exist_ok=False)
    links_path = output / "slope_refit_candidate_link_impact.parquet"
    summary_path = output / "slope_refit_candidate_impact_summary.csv"
    overall_path = output / "slope_refit_candidate_impact_overall.json"
    links.to_parquet(links_path, index=False)
    summary.to_csv(summary_path, index=False)
    finite = links.replace([np.inf, -np.inf], np.nan).dropna(subset=["slope_ratio_candidate_to_current"])
    capped = finite.dropna(subset=["capped_k_ratio_candidate_to_current"])
    overall = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_root": str(Path(args.candidate_root).expanduser().resolve()),
        "n_examples": int(summary["example_id"].nunique()),
        "n_links": int(len(finite)),
        "median_slope_ratio": float(finite["slope_ratio_candidate_to_current"].median()),
        "p05_slope_ratio": float(finite["slope_ratio_candidate_to_current"].quantile(0.05)),
        "p95_slope_ratio": float(finite["slope_ratio_candidate_to_current"].quantile(0.95)),
        "n_slope_change_over_25pct": int((finite["slope_ratio_candidate_to_current"].lt(0.8) | finite["slope_ratio_candidate_to_current"].gt(1.25)).sum()),
        "n_links_with_rapid_translation": int(len(capped)),
        "n_capped_k_change_over_25pct": int((capped["capped_k_ratio_candidate_to_current"].lt(0.8) | capped["capped_k_ratio_candidate_to_current"].gt(1.25)).sum()),
        "n_current_below_min_cap": int(capped["current_celerity_cap_status"].eq("below_min_cap").sum()),
        "n_candidate_below_min_cap": int(capped["candidate_celerity_cap_status"].eq("below_min_cap").sum()),
        "n_current_above_max_cap": int(capped["current_celerity_cap_status"].eq("above_max_cap").sum()),
        "n_candidate_above_max_cap": int(capped["candidate_celerity_cap_status"].eq("above_max_cap").sum()),
        "equivalent_resistance_expected_ratio": 1.0,
        "equivalent_resistance_note": "No current equivalent-resistance input depends on slope_used; this is an exact expected invariance, not a rerun result.",
        "link_impact_parquet": str(links_path),
        "summary_csv": str(summary_path),
    }
    overall_path.write_text(json.dumps(overall, indent=2, sort_keys=True), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(overall, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
