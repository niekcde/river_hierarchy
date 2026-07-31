from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from network_variants.width_review_archive import (
    ARCHIVE_MANIFEST_NAME,
    ARCHIVE_TARGETS_NAME,
    _atomic_write_csv,
    _atomic_write_json,
    _sha256,
    normalize_example_id,
)
from network_variants.width_review_fingerprint import (
    CURRENT_STATUS,
    validate_review_fingerprints,
)


DATASET_SCHEMA_VERSION = "width-qc-training-v1"
DEFAULT_REVIEW_ROOT = Path(__file__).resolve().parents[1] / "rivgraph_centerline" / "outputs" / "runs"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "width_qc_dataset"
DATASET_FILENAME = "width_qc_training_data_v1.csv"
REPORT_FILENAME = "width_qc_training_report_v1.json"
MANIFEST_FILENAME = "width_qc_training_manifest_v1.json"

PRIMARY_PREDICTORS = (
    "length_over_raw_wet_width",
    "split_sensitive",
    "opposition_cos_trimmed_mean",
    "opposition_cos_trimmed_median",
    "invalid_sample_fraction",
    "upstream_continuity_ratio",
    "upstream_continuity_signed_deviation",
    "upstream_continuity_abs_deviation",
)
SECONDARY_PREDICTORS = (
    "downstream_continuity_ratio",
    "downstream_continuity_signed_deviation",
    "downstream_continuity_abs_deviation",
    "mean_bank_chord_difference",
    "mean_relative_dt_difference",
    "max_relative_dt_difference",
    "n_wet_threads_mean",
    "n_wet_threads_max",
    "wet_fraction",
    "lineage_type",
    "raw_wet_width",
    "link_length",
)

DATASET_COLUMNS = (
    "dataset_schema_version",
    "example_id",
    "variant_id",
    "id_link",
    "id_us_node",
    "id_ds_node",
    "manual_decision",
    "requires_correction",
    *PRIMARY_PREDICTORS,
    *SECONDARY_PREDICTORS,
    "width_sample_count",
    "n_invalid_width_samples",
    "n_trimmed_qc_samples",
    "raw_total_width",
    "raw_dry_width",
    "dry_fraction",
    "width_method",
    "legacy_width_update_applied",
    "legacy_width_update_method",
    "legacy_width_update_scale_factor",
    "legacy_width_update_target_context_width",
    "legacy_wid_source",
    "legacy_wid_adj_source",
    "legacy_wid_med_source",
    "review_archive_dir",
    "review_targets_path",
    "review_manifest_path",
)


@dataclass(frozen=True)
class WidthReviewTrainingDataset:
    frame: pd.DataFrame
    archive_dirs: tuple[Path, ...]
    report: dict[str, Any]
    input_files: tuple[Path, ...]


def discover_review_archives(review_root: str | Path = DEFAULT_REVIEW_ROOT) -> list[Path]:
    root = Path(review_root).expanduser().resolve()
    if (root / ARCHIVE_MANIFEST_NAME).exists():
        return [root]
    return sorted(
        (path.parent for path in root.glob(f"*/width_review/{ARCHIVE_MANIFEST_NAME}")),
        key=lambda path: path.parent.name,
    )


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _strings(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(pd.NA, index=frame.index, dtype="string")
    return frame[column].astype("string")


def _serialized_value_count(value: object) -> int:
    if value is None or pd.isna(value):
        return 0
    text = str(value).strip().strip("[]")
    if not text:
        return 0
    return sum(bool(part.strip()) for part in text.split(","))


def _canonicalize_archive_frame(
    source: pd.DataFrame,
    *,
    example_id: str,
    archive_dir: Path,
    targets_path: Path,
    manifest_path: Path,
    provenance_root: Path,
) -> pd.DataFrame:
    required = {"id_link", "review_decision"}
    missing_required = sorted(required.difference(source.columns))
    if missing_required:
        raise ValueError(f"Archive {archive_dir} is missing required fields: {missing_required}")

    decision = _strings(source, "review_decision").str.strip().str.lower()
    invalid_decisions = sorted(decision.loc[~decision.isin(["good", "bad"])].dropna().unique())
    if decision.isna().any() or invalid_decisions:
        raise ValueError(
            f"Archive {archive_dir} is not a complete good/bad review; invalid={invalid_decisions}, "
            f"undecided={int(decision.isna().sum())}"
        )

    link_length = _numeric(source, "len")
    raw_wet_width = _numeric(source, "wid_adj_wet")
    length_over_width = link_length / raw_wet_width.where(raw_wet_width.gt(0))
    upstream_ratio = _numeric(source, "us_node_out_over_in")
    downstream_ratio = _numeric(source, "ds_node_out_over_in")
    upstream_signed = upstream_ratio - 1.0
    downstream_signed = downstream_ratio - 1.0
    sample_count = _strings(source, "wid_pix_total").map(_serialized_value_count).astype(int)
    invalid_count = _numeric(source, "n_invalid_width_samples")
    invalid_fraction = invalid_count / sample_count.replace(0, np.nan)

    split_sensitive = _numeric(source, "split_sensitive")
    upstream_out_count = _numeric(source, "us_out_count")
    derived_split = upstream_out_count.gt(1).astype(float).where(upstream_out_count.notna())
    split_sensitive = split_sensitive.where(split_sensitive.notna(), derived_split)

    lineage_source = next(
        (column for column in ("lineage_type", "link_lineage_type", "match_type") if column in source.columns),
        None,
    )
    lineage = _strings(source, lineage_source) if lineage_source is not None else _strings(source, "__missing__")

    result = pd.DataFrame(index=source.index)
    result["dataset_schema_version"] = DATASET_SCHEMA_VERSION
    result["example_id"] = normalize_example_id(example_id)
    result["variant_id"] = _strings(source, "variant_id").fillna("S000_base")
    result["id_link"] = _numeric(source, "id_link").astype("Int64")
    result["id_us_node"] = _numeric(source, "id_us_node").astype("Int64")
    result["id_ds_node"] = _numeric(source, "id_ds_node").astype("Int64")
    result["manual_decision"] = decision
    result["requires_correction"] = decision.eq("bad").astype(int)
    result["length_over_raw_wet_width"] = length_over_width
    result["split_sensitive"] = split_sensitive.astype("Int64")
    result["opposition_cos_trimmed_mean"] = _numeric(source, "qc_mean_opposition_cos_trimmed")
    result["opposition_cos_trimmed_median"] = _numeric(
        source, "qc_median_opposition_cos_trimmed_from_samples"
    )
    result["invalid_sample_fraction"] = invalid_fraction
    result["upstream_continuity_ratio"] = upstream_ratio
    result["upstream_continuity_signed_deviation"] = upstream_signed
    result["upstream_continuity_abs_deviation"] = upstream_signed.abs()
    result["downstream_continuity_ratio"] = downstream_ratio
    result["downstream_continuity_signed_deviation"] = downstream_signed
    result["downstream_continuity_abs_deviation"] = downstream_signed.abs()
    result["mean_bank_chord_difference"] = _numeric(source, "qc_mean_width_chord_diff")
    result["mean_relative_dt_difference"] = _numeric(source, "qc_mean_relative_dt_diff")
    result["max_relative_dt_difference"] = _numeric(source, "qc_max_relative_dt_diff")
    result["n_wet_threads_mean"] = _numeric(source, "n_wet_threads_mean")
    result["n_wet_threads_max"] = _numeric(source, "n_wet_threads_max")
    result["wet_fraction"] = _numeric(source, "wet_fraction_adj")
    result["lineage_type"] = lineage
    result["raw_wet_width"] = raw_wet_width
    result["link_length"] = link_length
    result["width_sample_count"] = sample_count
    result["n_invalid_width_samples"] = invalid_count.astype("Int64")
    result["n_trimmed_qc_samples"] = _numeric(source, "n_trimmed_qc_samples").astype("Int64")
    result["raw_total_width"] = _numeric(source, "wid_adj_total")
    result["raw_dry_width"] = _numeric(source, "wid_adj_dry")
    result["dry_fraction"] = _numeric(source, "dry_fraction_adj")
    result["width_method"] = _strings(source, "width_method")
    result["legacy_width_update_applied"] = _numeric(source, "width_update_applied").astype("Int64")
    result["legacy_width_update_method"] = _strings(source, "width_update_method")
    result["legacy_width_update_scale_factor"] = _numeric(source, "width_update_scale_factor")
    result["legacy_width_update_target_context_width"] = _numeric(
        source, "width_update_target_context_width"
    )
    result["legacy_wid_source"] = _numeric(source, "wid_source")
    result["legacy_wid_adj_source"] = _numeric(source, "wid_adj_source")
    result["legacy_wid_med_source"] = _numeric(source, "wid_med_source")
    result["review_archive_dir"] = archive_dir.relative_to(provenance_root).as_posix()
    result["review_targets_path"] = targets_path.relative_to(provenance_root).as_posix()
    result["review_manifest_path"] = manifest_path.relative_to(provenance_root).as_posix()
    return result.loc[:, DATASET_COLUMNS]


def _missing_report(frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    report: dict[str, dict[str, float | int]] = {}
    for column in (*PRIMARY_PREDICTORS, *SECONDARY_PREDICTORS):
        missing = int(frame[column].isna().sum())
        report[column] = {
            "missing_rows": missing,
            "missing_fraction": float(missing / len(frame)) if len(frame) else 0.0,
        }
    return report


def build_width_review_training_dataset(
    review_root: str | Path = DEFAULT_REVIEW_ROOT,
    *,
    validate_freshness: bool = True,
) -> WidthReviewTrainingDataset:
    review_root_path = Path(review_root).expanduser().resolve()
    archives = discover_review_archives(review_root_path)
    if not archives:
        raise FileNotFoundError(f"No permanent width-review archives found below {Path(review_root).resolve()}")

    frames: list[pd.DataFrame] = []
    input_files: list[Path] = []
    archive_reports: list[dict[str, Any]] = []
    for archive in archives:
        manifest_path = archive / ARCHIVE_MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        freshness = (
            validate_review_fingerprints(archive)
            if validate_freshness
            else {"status": str(manifest.get("review_status", "not_validated")), "issues": []}
        )
        if validate_freshness and freshness["status"] != CURRENT_STATUS:
            raise ValueError(
                f"Width-review archive is not current: {archive} status={freshness['status']} "
                f"issues={freshness.get('issues', [])}"
            )
        targets_name = str((manifest.get("files", {}) or {}).get("targets", ARCHIVE_TARGETS_NAME))
        targets_path = archive / targets_name
        if not targets_path.exists():
            raise FileNotFoundError(f"Reviewed target table does not exist: {targets_path}")
        source = pd.read_csv(targets_path)
        example_id = normalize_example_id(manifest.get("example_id", archive.parent.name))
        frame = _canonicalize_archive_frame(
            source,
            example_id=example_id,
            archive_dir=archive,
            targets_path=targets_path,
            manifest_path=manifest_path,
            provenance_root=review_root_path if review_root_path != archive else archive.parent,
        )
        frames.append(frame)
        input_files.extend([manifest_path, targets_path])
        archive_reports.append(
            {
                "example_id": example_id,
                "archive_dir": str(archive),
                "freshness_status": freshness["status"],
                "rows": int(len(frame)),
                "good": int(frame["manual_decision"].eq("good").sum()),
                "bad": int(frame["manual_decision"].eq("bad").sum()),
            }
        )

    combined = pd.concat(frames, ignore_index=True)
    if combined[["example_id", "id_link"]].isna().any().any():
        raise ValueError("Every training row must have a non-null example_id and id_link.")
    duplicates = combined.duplicated(["example_id", "id_link"], keep=False)
    if duplicates.any():
        duplicate_keys = combined.loc[duplicates, ["example_id", "id_link"]].to_dict("records")
        raise ValueError(f"Duplicate example/link training keys found: {duplicate_keys}")
    combined = combined.sort_values(["example_id", "id_link"], kind="mergesort").reset_index(drop=True)

    provenance_root = review_root_path if review_root_path not in archives else review_root_path.parent
    input_hashes = {
        path.relative_to(provenance_root).as_posix(): _sha256(path)
        for path in sorted(input_files)
    }
    dataset_identity_payload = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "inputs": input_hashes,
    }
    dataset_id = hashlib.sha256(
        json.dumps(dataset_identity_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    report = {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "review_root": str(review_root_path),
        "archive_count": int(len(archives)),
        "row_count": int(len(combined)),
        "example_count": int(combined["example_id"].nunique()),
        "label_counts": {
            key: int(value) for key, value in combined["manual_decision"].value_counts().sort_index().items()
        },
        "duplicate_example_link_rows": 0,
        "excluded_rows": 0,
        "exclusions": [],
        "primary_predictors": list(PRIMARY_PREDICTORS),
        "secondary_predictors": list(SECONDARY_PREDICTORS),
        "derived_fields": {
            "length_over_raw_wet_width": "len / wid_adj_wet",
            "invalid_sample_fraction": "n_invalid_width_samples / count(wid_pix_total)",
            "upstream_continuity_signed_deviation": "us_node_out_over_in - 1",
            "downstream_continuity_signed_deviation": "ds_node_out_over_in - 1",
            "split_sensitive_fallback": "us_out_count > 1 when the archived split_sensitive field is missing",
        },
        "missing_values": _missing_report(combined),
        "archives": archive_reports,
        "input_file_sha256": input_hashes,
    }
    return WidthReviewTrainingDataset(
        frame=combined,
        archive_dirs=tuple(archives),
        report=report,
        input_files=tuple(sorted(input_files)),
    )


def write_width_review_training_dataset(
    dataset: WidthReviewTrainingDataset,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset_path = output / DATASET_FILENAME
    report_path = output / REPORT_FILENAME
    manifest_path = output / MANIFEST_FILENAME
    _atomic_write_csv(dataset.frame, dataset_path)
    _atomic_write_json(dataset.report, report_path)
    manifest = {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "dataset_id": dataset.report["dataset_id"],
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "report_path": str(report_path),
        "report_sha256": _sha256(report_path),
        "row_count": int(len(dataset.frame)),
        "columns": list(dataset.frame.columns),
        "input_files": [str(path) for path in dataset.input_files],
    }
    _atomic_write_json(manifest, manifest_path)
    return {
        "dataset": str(dataset_path),
        "report": str(report_path),
        "manifest": str(manifest_path),
        "dataset_id": dataset.report["dataset_id"],
        "row_count": int(len(dataset.frame)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the versioned supervised width-QC training dataset.")
    parser.add_argument("--review-root", default=str(DEFAULT_REVIEW_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--skip-freshness-validation",
        action="store_true",
        help="Debugging only: build without recalculating archive mask/network freshness.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = build_width_review_training_dataset(
        args.review_root,
        validate_freshness=not args.skip_freshness_validation,
    )
    result = write_width_review_training_dataset(dataset, args.output_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
