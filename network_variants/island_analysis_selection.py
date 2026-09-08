"""Freeze reviewed inclusion decisions for island-only analyses.

The raw island analysis manifest is an inventory of every approved island
observation and its realised network product.  It must remain immutable after
execution.  This module derives a separate strict manifest for analyses that
require a one-island/one-cycle simplification, with every exclusion tied to a
small, reviewable decision registry.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd


REVIEW_COLUMNS = {
    "example_id",
    "island_id",
    "strict_analysis_status",
    "review_code",
    "review_comment",
}


def _read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _example_id(experiment_dir: Path) -> str:
    manifest_path = experiment_dir / "experiment_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Island experiment is missing {manifest_path.name}: {experiment_dir}")
    value = _read_json(manifest_path).get("example_id")
    if value is None or not str(value).strip():
        raise ValueError(f"Island experiment has no example_id: {manifest_path}")
    return str(value).zfill(2)


def build_strict_island_analysis_manifest(
    experiment_dir: str | Path,
    *,
    review_registry_path: str | Path,
    output_name: str = "island_analysis_manifest_strict.csv",
) -> pd.DataFrame:
    """Write a non-destructive strict-analysis view for one island experiment.

    ``strict_analysis_status`` is either ``include_primary_one_cycle_analysis``
    or ``exclude_pruning_or_mask_topology_exception``. The unmodified
    ``island_analysis_manifest.csv`` remains the complete provenance inventory.
    """
    root = Path(experiment_dir).expanduser().resolve()
    analysis_path = root / "island_analysis_manifest.csv"
    review_path = Path(review_registry_path).expanduser().resolve()
    if not analysis_path.exists():
        raise FileNotFoundError(f"Island analysis manifest is missing: {analysis_path}")
    if not review_path.exists():
        raise FileNotFoundError(f"Island topology review registry is missing: {review_path}")

    analysis = pd.read_csv(analysis_path, low_memory=False)
    review = pd.read_csv(review_path, low_memory=False)
    missing = sorted(REVIEW_COLUMNS - set(review.columns))
    if missing:
        raise ValueError(f"Island topology review registry is missing columns: {missing}")
    if analysis["island_id"].astype(str).duplicated().any():
        raise ValueError(f"Island analysis manifest has duplicate island IDs: {analysis_path}")

    example_id = _example_id(root)
    decisions = review.loc[
        review["example_id"].astype(str).str.zfill(2).eq(example_id)
    ].copy()
    if decisions["island_id"].astype(str).duplicated().any():
        raise ValueError(f"Topology review registry has duplicate decisions for example {example_id}.")
    known = set(analysis["island_id"].astype(str))
    unknown = set(decisions["island_id"].astype(str)) - known
    if unknown:
        raise ValueError(
            f"Topology review registry references islands absent from {analysis_path}: {sorted(unknown)}"
        )
    if not decisions["strict_analysis_status"].eq(
        "exclude_pruning_or_mask_topology_exception"
    ).all():
        raise ValueError("This strict registry currently accepts exclusion decisions only.")

    result = analysis.copy()
    result["strict_analysis_status"] = "include_primary_one_cycle_analysis"
    result["topology_review_code"] = ""
    result["topology_review_comment"] = ""
    decision_by_id = decisions.set_index(decisions["island_id"].astype(str))
    for index, island_id in result["island_id"].astype(str).items():
        if island_id not in decision_by_id.index:
            continue
        decision = decision_by_id.loc[island_id]
        result.loc[index, "strict_analysis_status"] = decision["strict_analysis_status"]
        result.loc[index, "topology_review_code"] = decision["review_code"]
        result.loc[index, "topology_review_comment"] = decision["review_comment"]

    output_path = root / output_name
    result.to_csv(output_path, index=False)
    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "island-strict-analysis-manifest-v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source_analysis_manifest": str(analysis_path),
                "review_registry": str(review_path),
                "example_id": example_id,
                "n_observations": int(len(result)),
                "n_included": int(result["strict_analysis_status"].eq("include_primary_one_cycle_analysis").sum()),
                "n_excluded": int(result["strict_analysis_status"].eq("exclude_pruning_or_mask_topology_exception").sum()),
                "exclusions": decisions[["island_id", "review_code", "review_comment"]].to_dict("records"),
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return result
