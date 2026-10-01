from __future__ import annotations

"""Validate immutable slope-refit candidate artifacts before comparison/promotion."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

import numpy as np
import pandas as pd


VALIDATION_SCHEMA_VERSION = "slope-refit-candidate-validation-v1"


def _canonical_id(value: object) -> str:
    return f"{int(float(value)):02d}"


def _check_candidate(example_id: str, profile_dir: Path, candidate_dir: Path, reviewed_lambda: float) -> dict[str, object]:
    profile = pd.read_parquet(profile_dir / "reference_profile.parquet").sort_values("reference_section_dist_out_m")
    links = pd.read_parquet(candidate_dir / "candidate_link_hydraulic_context.parquet")
    nodes = pd.read_parquet(candidate_dir / "candidate_node_hydraulic_context.parquet")
    domains = pd.read_csv(candidate_dir / "candidate_domain_head_loss_audit.csv")
    ownership = pd.read_csv(candidate_dir / "candidate_link_ownership_audit.csv")
    selection = json.loads((profile_dir / "reference_profile_selection.json").read_text(encoding="utf-8"))
    x = pd.to_numeric(profile["reference_section_dist_out_m"], errors="coerce").to_numpy(float)
    head = pd.to_numeric(profile["fitted_wse_m"], errors="coerce").to_numpy(float)
    profile_slopes = np.diff(head) / np.diff(x)
    length = pd.to_numeric(links["actual_length_m"], errors="coerce")
    slope = pd.to_numeric(links["candidate_slope_m_per_m"], errors="coerce")
    head_us = pd.to_numeric(links["candidate_head_us_m"], errors="coerce")
    head_ds = pd.to_numeric(links["candidate_head_ds_m"], errors="coerce")
    reconstructed = (head_us - head_ds) / length
    node_heads = nodes.set_index("id_node")["candidate_head_m"]
    node_us = pd.to_numeric(links["id_us_node"], errors="coerce").astype("Int64").map(node_heads)
    node_ds = pd.to_numeric(links["id_ds_node"], errors="coerce").astype("Int64").map(node_heads)
    path_residual = pd.to_numeric(domains.get("max_abs_path_residual_m"), errors="coerce")
    source_counts = links["candidate_slope_source_method"].astype(str).value_counts()
    row = {
        "example_id": example_id,
        "reviewed_lambda": float(reviewed_lambda),
        "artifact_lambda": float(selection["selection"]["selected_lambda"]),
        "lambda_matches_review": bool(np.isclose(float(selection["selection"]["selected_lambda"]), float(reviewed_lambda), rtol=1e-12, atol=0.0)),
        "profile_n_nodes": int(len(profile)),
        "profile_min_interval_slope_m_per_m": float(np.nanmin(profile_slopes)),
        "profile_positive_intervals": bool(np.isfinite(profile_slopes).all() and (profile_slopes > 0.0).all()),
        "n_links": int(len(links)),
        "positive_actual_lengths": bool(np.isfinite(length).all() and length.gt(0.0).all()),
        "n_len_primary": int(links["actual_length_source"].astype(str).eq("len").sum()),
        "n_non_len_primary": int((~links["actual_length_source"].astype(str).eq("len")).sum()),
        "positive_candidate_slopes": bool(np.isfinite(slope).all() and slope.gt(0.0).all()),
        "max_head_slope_reconstruction_abs_error": float(np.nanmax(np.abs(reconstructed - slope))),
        "head_slope_reconstruction_valid": bool(np.allclose(reconstructed, slope, rtol=1e-10, atol=1e-13, equal_nan=False)),
        "max_link_to_node_head_abs_error_m": float(np.nanmax(np.r_[np.abs(head_us - node_us), np.abs(head_ds - node_ds)])),
        "link_to_node_heads_valid": bool(np.allclose(head_us, node_us, rtol=0.0, atol=1e-9, equal_nan=False) and np.allclose(head_ds, node_ds, rtol=0.0, atol=1e-9, equal_nan=False)),
        "n_domains": int(len(domains)),
        "n_domains_solved": int(domains.get("status", pd.Series(dtype=str)).eq("solved").sum()),
        "n_domains_unsolved": int((~domains.get("status", pd.Series(dtype=str)).eq("solved")).sum()),
        "max_domain_path_residual_m": float(path_residual.max()) if len(path_residual) else np.nan,
        "domain_path_losses_valid": bool(domains.get("status", pd.Series(dtype=str)).eq("solved").all() and (path_residual <= 1e-8).all()),
        "ownership_rows_match_links": bool(len(ownership) == len(links) and ownership["id_link"].nunique() == len(links)),
        "n_domain_solved_links": int(source_counts.get("profile_boundary_conditioned_domain", 0)),
        "n_direct_profile_links": int(source_counts.get("frozen_profile_direct_connector_or_boundary", 0)),
    }
    required = [
        "lambda_matches_review", "profile_positive_intervals", "positive_actual_lengths",
        "positive_candidate_slopes", "head_slope_reconstruction_valid", "link_to_node_heads_valid",
        "domain_path_losses_valid", "ownership_rows_match_links",
    ]
    row["validation_passed"] = bool(all(bool(row[name]) for name in required))
    return row


def validate_candidate_batch(candidate_root: str | Path) -> pd.DataFrame:
    root = Path(candidate_root).expanduser().resolve()
    manifest = pd.read_csv(root / "slope_refit_candidate_batch_manifest.csv")
    required = {"example_id", "reviewed_lambda", "profile_dir", "candidate_context_dir", "status"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Batch manifest is missing fields: {missing}")
    rows = []
    for item in manifest.itertuples(index=False):
        example_id = _canonical_id(item.example_id)
        if str(item.status) not in {"complete", "already_complete"}:
            raise ValueError(f"Example {example_id} is not complete in the candidate manifest: {item.status}")
        rows.append(
            _check_candidate(
                example_id,
                Path(item.profile_dir),
                Path(item.candidate_context_dir),
                float(item.reviewed_lambda),
            )
        )
    return pd.DataFrame(rows).sort_values("example_id").reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate all slope-refit candidate artifacts before any promotion.")
    parser.add_argument("candidate_root")
    parser.add_argument("output_dir", help="New empty directory for validation artifacts.")
    args = parser.parse_args(argv)
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Validation output must be empty: {output}")
    result = validate_candidate_batch(args.candidate_root)
    output.mkdir(parents=True, exist_ok=False)
    table_path = output / "slope_refit_candidate_validation.csv"
    summary_path = output / "slope_refit_candidate_validation_summary.json"
    result.to_csv(table_path, index=False)
    summary = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_root": str(Path(args.candidate_root).expanduser().resolve()),
        "n_examples": int(len(result)),
        "n_passed": int(result["validation_passed"].sum()),
        "n_failed": int((~result["validation_passed"]).sum()),
        "max_domain_path_residual_m": float(result["max_domain_path_residual_m"].max()),
        "max_head_slope_reconstruction_abs_error": float(result["max_head_slope_reconstruction_abs_error"].max()),
        "max_link_to_node_head_abs_error_m": float(result["max_link_to_node_head_abs_error_m"].max()),
        "validation_csv": str(table_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(result.to_string(index=False))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
