from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from network_variants.width_qc_disagreement_gui import (
    DEFAULT_OUTPUT_DIR as DEFAULT_REVIEW_DIR,
    MANIFEST_FILENAME as REVIEW_MANIFEST_FILENAME,
    REVIEW_FILENAME,
    build_disagreement_frame,
    resolve_review_inputs,
)
from network_variants.width_qc_model import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_OUTPUT_DIR,
    PREDICTIONS_FILENAME,
    SUMMARY_FILENAME,
)
from network_variants.width_review_archive import _atomic_write_json, _sha256


REBASE_SCHEMA_VERSION = "width-qc-frozen-review-rebase-v1"
REBASE_FILENAME = "width_qc_disagreement_review_rebase_v1.json"


def _keys(frame: pd.DataFrame) -> set[tuple[str, int]]:
    normalized = frame[["example_id", "id_link"]].copy()
    normalized["example_id"] = normalized["example_id"].astype("string").str.zfill(2)
    normalized["id_link"] = pd.to_numeric(normalized["id_link"], errors="raise").astype(int)
    return set(normalized.itertuples(index=False, name=None))


def _changed_dataset_keys(old: pd.DataFrame, new: pd.DataFrame) -> list[tuple[str, int]]:
    keys = ["example_id", "id_link"]
    for frame in (old, new):
        frame["example_id"] = frame["example_id"].astype("string").str.zfill(2)
        frame["id_link"] = pd.to_numeric(frame["id_link"], errors="raise").astype(int)
    if _keys(old) != _keys(new):
        raise ValueError("Frozen-review rebase does not allow training-row membership changes.")
    common_columns = [column for column in old.columns if column in new.columns and column not in keys]
    joined = old.merge(new, on=keys, suffixes=("_old", "_new"), validate="1:1")
    changed: set[tuple[str, int]] = set()
    for column in common_columns:
        before = joined[f"{column}_old"]
        after = joined[f"{column}_new"]
        different = ~(before.eq(after) | (before.isna() & after.isna()))
        changed.update(joined.loc[different, keys].itertuples(index=False, name=None))
    return sorted(changed)


def create_frozen_review_rebase(
    *,
    previous_dataset_path: str | Path,
    previous_predictions_path: str | Path,
    current_dataset_path: str | Path = DEFAULT_DATASET_PATH,
    current_predictions_path: str | Path = DEFAULT_MODEL_OUTPUT_DIR / PREDICTIONS_FILENAME,
    current_summary_path: str | Path = DEFAULT_MODEL_OUTPUT_DIR / SUMMARY_FILENAME,
    review_dir: str | Path = DEFAULT_REVIEW_DIR,
) -> dict[str, Any]:
    review_dir = Path(review_dir).expanduser().resolve()
    review_path = review_dir / REVIEW_FILENAME
    review_manifest_path = review_dir / REVIEW_MANIFEST_FILENAME
    previous_dataset_path = Path(previous_dataset_path).expanduser().resolve()
    previous_predictions_path = Path(previous_predictions_path).expanduser().resolve()
    current_dataset_path = Path(current_dataset_path).expanduser().resolve()
    current_predictions_path = Path(current_predictions_path).expanduser().resolve()
    current_summary_path = Path(current_summary_path).expanduser().resolve()
    for path in (
        review_path,
        review_manifest_path,
        previous_dataset_path,
        previous_predictions_path,
        current_dataset_path,
        current_predictions_path,
        current_summary_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    review_manifest = json.loads(review_manifest_path.read_text(encoding="utf-8"))
    original_inputs = review_manifest.get("inputs", {}) or {}
    if _sha256(previous_dataset_path) != original_inputs.get("dataset_sha256"):
        raise ValueError("Previous dataset does not match the completed review manifest.")
    if _sha256(previous_predictions_path) != original_inputs.get("predictions_sha256"):
        raise ValueError("Previous predictions do not match the completed review manifest.")
    if _sha256(review_path) != (review_manifest.get("file_sha256", {}) or {}).get("review"):
        raise ValueError("Completed human-review CSV no longer matches its manifest.")
    if int((review_manifest.get("counts", {}) or {}).get("unreviewed", -1)) != 0:
        raise ValueError("Only a complete adjudication review can be frozen and rebased.")

    old_dataset = pd.read_csv(previous_dataset_path, dtype={"example_id": "string"})
    new_dataset = pd.read_csv(current_dataset_path, dtype={"example_id": "string"})
    review = pd.read_csv(review_path, dtype={"example_id": "string"})
    changed_keys = _changed_dataset_keys(old_dataset, new_dataset)
    changed_key_set = set(changed_keys)
    overlap = sorted(changed_key_set.intersection(_keys(review)))
    if overlap:
        raise ValueError(
            "Dataset rows changed for completed human-review cases; a provenance-only rebase is unsafe: "
            f"{overlap}"
        )

    old_predictions = pd.read_csv(previous_predictions_path, dtype={"example_id": "string"})
    new_predictions = pd.read_csv(current_predictions_path, dtype={"example_id": "string"})
    model_name = str(review_manifest.get("model_name"))
    old_selected = old_predictions.loc[old_predictions["width_model_name"].eq(model_name)].copy()
    new_selected = new_predictions.loc[new_predictions["width_model_name"].eq(model_name)].copy()
    for frame in (old_selected, new_selected):
        frame["example_id"] = frame["example_id"].astype("string").str.zfill(2)
        frame["id_link"] = pd.to_numeric(frame["id_link"], errors="raise").astype(int)
    prediction_comparison = old_selected.merge(
        new_selected,
        on=["example_id", "id_link"],
        suffixes=("_old", "_new"),
        validate="1:1",
    )
    reviewed_keys = _keys(review)
    reviewed_mask = prediction_comparison.set_index(["example_id", "id_link"]).index.isin(reviewed_keys)
    changed_prediction_mask = reviewed_mask & prediction_comparison[
        "width_model_prediction_old"
    ].ne(prediction_comparison["width_model_prediction_new"])
    changed_review_predictions = prediction_comparison.loc[
        changed_prediction_mask,
        [
            "example_id",
            "id_link",
            "width_model_prediction_old",
            "width_model_prediction_new",
        ],
    ].to_dict("records")

    current_inputs = resolve_review_inputs(
        dataset_path=current_dataset_path,
        predictions_path=current_predictions_path,
        summary_path=current_summary_path,
    )
    current_queue = build_disagreement_frame(current_inputs)
    old_queue_keys = _keys(review)
    new_queue_keys = _keys(current_queue)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    version_dir = review_dir / "versions" / f"rebase_{timestamp}"
    version_dir.mkdir(parents=True, exist_ok=False)
    archived_dataset = version_dir / previous_dataset_path.name
    archived_predictions = version_dir / previous_predictions_path.name
    shutil.copy2(previous_dataset_path, archived_dataset)
    shutil.copy2(previous_predictions_path, archived_predictions)

    payload = {
        "schema_version": REBASE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "validated_frozen_adjudication_reuse",
        "policy": (
            "The completed 63-case human adjudication cohort is frozen. No second disagreement loop "
            "feeds back into training. Newly flagged non-cohort links retain their permanent manual labels."
        ),
        "human_review": {
            "review": str(review_path),
            "review_sha256": _sha256(review_path),
            "manifest": str(review_manifest_path),
            "manifest_sha256": _sha256(review_manifest_path),
            "review_case_count": int(len(review)),
        },
        "original_inputs": {
            "dataset_sha256": original_inputs.get("dataset_sha256"),
            "predictions_sha256": original_inputs.get("predictions_sha256"),
            "archived_dataset": str(archived_dataset),
            "archived_dataset_sha256": _sha256(archived_dataset),
            "archived_predictions": str(archived_predictions),
            "archived_predictions_sha256": _sha256(archived_predictions),
        },
        "current_inputs": {
            "dataset": str(current_dataset_path),
            "dataset_sha256": _sha256(current_dataset_path),
            "predictions": str(current_predictions_path),
            "predictions_sha256": _sha256(current_predictions_path),
            "validation_summary": str(current_summary_path),
            "validation_summary_sha256": _sha256(current_summary_path),
            "junction_audit": str(current_inputs.junction_audit_path),
            "junction_audit_sha256": current_inputs.junction_audit_sha256,
            "junction_consensus": str(current_inputs.junction_consensus_path),
            "junction_consensus_sha256": current_inputs.junction_consensus_sha256,
        },
        "validation": {
            "changed_dataset_keys": [
                {"example_id": example_id, "id_link": int(link_id)}
                for example_id, link_id in changed_keys
            ],
            "changed_dataset_keys_intersect_human_review": [],
            "reviewed_cases_with_changed_binary_prediction": changed_review_predictions,
            "removed_from_current_queue": [
                {"example_id": example_id, "id_link": int(link_id)}
                for example_id, link_id in sorted(old_queue_keys - new_queue_keys)
            ],
            "added_to_current_queue": [
                {"example_id": example_id, "id_link": int(link_id)}
                for example_id, link_id in sorted(new_queue_keys - old_queue_keys)
            ],
        },
    }
    output_path = review_dir / REBASE_FILENAME
    _atomic_write_json(payload, output_path)
    payload["rebase_manifest"] = str(output_path)
    payload["rebase_manifest_sha256"] = _sha256(output_path)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate reuse of a frozen completed disagreement review after a narrow dataset rebuild."
    )
    parser.add_argument("--previous-dataset", required=True)
    parser.add_argument("--previous-predictions", required=True)
    parser.add_argument("--current-dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument(
        "--current-predictions", default=str(DEFAULT_MODEL_OUTPUT_DIR / PREDICTIONS_FILENAME)
    )
    parser.add_argument("--current-summary", default=str(DEFAULT_MODEL_OUTPUT_DIR / SUMMARY_FILENAME))
    parser.add_argument("--review-dir", default=str(DEFAULT_REVIEW_DIR))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = create_frozen_review_rebase(
        previous_dataset_path=args.previous_dataset,
        previous_predictions_path=args.previous_predictions,
        current_dataset_path=args.current_dataset,
        current_predictions_path=args.current_predictions,
        current_summary_path=args.current_summary,
        review_dir=args.review_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
