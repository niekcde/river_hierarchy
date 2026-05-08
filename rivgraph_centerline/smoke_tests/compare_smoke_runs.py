"""Compare two RivGraph smoke-test runs on the same mask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd


def main() -> None:
    args = parse_args()
    baseline_summary_path = resolve_summary_path(args.baseline)
    candidate_summary_path = resolve_summary_path(args.candidate)

    baseline_summary = load_summary(baseline_summary_path)
    candidate_summary = load_summary(candidate_summary_path)

    baseline = enrich_run_summary(baseline_summary_path, baseline_summary)
    candidate = enrich_run_summary(candidate_summary_path, candidate_summary)
    report = build_report(baseline, candidate)

    report_json = json.dumps(report, indent=2)
    print(report_json)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_json, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Baseline smoke-test directory or its smoke_summary.json file.",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="Candidate smoke-test directory or its smoke_summary.json file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file path for the comparison report.",
    )
    return parser.parse_args()


def resolve_summary_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / "smoke_summary.json"
    if resolved.name != "smoke_summary.json":
        raise ValueError(f"Expected a run directory or smoke_summary.json, got: {resolved}")
    if not resolved.exists():
        raise FileNotFoundError(f"Smoke summary not found: {resolved}")
    return resolved


def load_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def enrich_run_summary(summary_path: Path, summary: dict[str, object]) -> dict[str, object]:
    outputs = dict(summary.get("outputs", {}))
    return {
        "run_dir": str(summary_path.parent),
        "summary_path": str(summary_path),
        "source_mask": summary.get("source_mask"),
        "prepared_mask": summary.get("prepared_mask"),
        "threshold": summary.get("threshold"),
        "target_crs": summary.get("target_crs"),
        "exit_sides": summary.get("exit_sides"),
        "source_water_fraction": summary.get("source_water_fraction"),
        "prepared_water_fraction": summary.get("prepared_water_fraction"),
        "summary_counts": {
            "n_links": summary.get("n_links"),
            "n_nodes": summary.get("n_nodes"),
            "n_inlets": len(summary.get("inlets", [])),
            "n_outlets": len(summary.get("outlets", [])),
        },
        "vectors": {
            "links": summarize_vector(outputs.get("links")),
            "nodes": summarize_vector(outputs.get("nodes")),
            "centerline": summarize_vector(outputs.get("centerline")),
        },
    }


def summarize_vector(path_str: str | None) -> dict[str, object]:
    if not path_str:
        return {"path": None, "exists": False}

    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        return {"path": str(path), "exists": False}

    frame = gpd.read_file(path)
    geometry_lengths = None
    if not frame.empty and frame.geometry is not None:
        if frame.geom_type.isin(["LineString", "MultiLineString"]).all():
            geometry_lengths = float(frame.geometry.length.sum())

    return {
        "path": str(path),
        "exists": True,
        "feature_count": int(len(frame)),
        "crs": str(frame.crs) if frame.crs is not None else None,
        "geom_types": sorted(frame.geom_type.unique().tolist()) if not frame.empty else [],
        "total_length": geometry_lengths,
    }


def build_report(baseline: dict[str, object], candidate: dict[str, object]) -> dict[str, object]:
    summary_diffs = diff_mapping(
        baseline["summary_counts"],
        candidate["summary_counts"],
        numeric_delta=True,
    )
    link_vector_diffs = diff_mapping(
        baseline["vectors"]["links"],
        candidate["vectors"]["links"],
        keys=("feature_count", "total_length", "crs", "geom_types"),
        numeric_delta=True,
    )
    node_vector_diffs = diff_mapping(
        baseline["vectors"]["nodes"],
        candidate["vectors"]["nodes"],
        keys=("feature_count", "crs", "geom_types"),
        numeric_delta=True,
    )
    centerline_diffs = diff_mapping(
        baseline["vectors"]["centerline"],
        candidate["vectors"]["centerline"],
        keys=("feature_count", "total_length", "crs", "geom_types"),
        numeric_delta=True,
    )

    return {
        "baseline": baseline,
        "candidate": candidate,
        "checks": {
            "same_source_mask": baseline.get("source_mask") == candidate.get("source_mask"),
            "same_threshold": baseline.get("threshold") == candidate.get("threshold"),
            "same_target_crs": baseline.get("target_crs") == candidate.get("target_crs"),
            "same_exit_sides": baseline.get("exit_sides") == candidate.get("exit_sides"),
            "same_source_water_fraction": baseline.get("source_water_fraction") == candidate.get("source_water_fraction"),
            "same_prepared_water_fraction": baseline.get("prepared_water_fraction") == candidate.get("prepared_water_fraction"),
        },
        "differences": {
            "summary_counts": summary_diffs,
            "links": link_vector_diffs,
            "nodes": node_vector_diffs,
            "centerline": centerline_diffs,
        },
    }


def diff_mapping(
    left: dict[str, object],
    right: dict[str, object],
    *,
    keys: tuple[str, ...] | None = None,
    numeric_delta: bool = False,
) -> dict[str, dict[str, object]]:
    keys_to_check = keys or tuple(sorted(set(left) | set(right)))
    diffs: dict[str, dict[str, object]] = {}
    for key in keys_to_check:
        left_value = left.get(key)
        right_value = right.get(key)
        entry: dict[str, object] = {
            "baseline": left_value,
            "candidate": right_value,
            "same": left_value == right_value,
        }
        if numeric_delta and isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
            entry["delta"] = right_value - left_value
        diffs[key] = entry
    return diffs


if __name__ == "__main__":
    main()
