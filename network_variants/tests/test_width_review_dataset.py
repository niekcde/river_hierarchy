from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from network_variants.width_review_archive import ARCHIVE_MANIFEST_NAME, ARCHIVE_TARGETS_NAME
from network_variants.width_review_dataset import (
    DATASET_COLUMNS,
    DATASET_SCHEMA_VERSION,
    build_width_review_training_dataset,
    discover_review_archives,
    write_width_review_training_dataset,
)


def _write_archive(root: Path, example_id: str, rows: list[dict[str, object]]) -> Path:
    archive = root / example_id / "width_review"
    archive.mkdir(parents=True)
    targets = archive / ARCHIVE_TARGETS_NAME
    pd.DataFrame(rows).to_csv(targets, index=False)
    manifest = {
        "example_id": example_id,
        "review_status": "current",
        "files": {"targets": ARCHIVE_TARGETS_NAME},
    }
    (archive / ARCHIVE_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return archive


def _row(link_id: int, decision: str) -> dict[str, object]:
    return {
        "id_link": link_id,
        "id_us_node": 10,
        "id_ds_node": 11,
        "variant_id": "S000_base",
        "review_decision": decision,
        "len": 10.0,
        "wid_adj_wet": 4.0,
        "wid_adj_total": 5.0,
        "wid_adj_dry": 1.0,
        "wid_pix_total": "4.0,5.0,6.0,7.0",
        "n_invalid_width_samples": 1,
        "n_trimmed_qc_samples": 3,
        "us_node_out_over_in": 1.25,
        "ds_node_out_over_in": 0.8,
        "us_out_count": 2,
        "qc_mean_opposition_cos_trimmed": -0.8,
        "qc_median_opposition_cos_trimmed_from_samples": -0.85,
        "qc_mean_width_chord_diff": 0.1,
        "qc_mean_relative_dt_diff": 0.2,
        "qc_max_relative_dt_diff": 0.3,
        "n_wet_threads_mean": 1.5,
        "n_wet_threads_max": 2,
        "wet_fraction_adj": 0.8,
        "dry_fraction_adj": 0.2,
        "width_method": "two_bank_dt",
        "width_update_applied": decision == "bad",
        "width_update_method": "legacy" if decision == "bad" else None,
        "width_update_scale_factor": 0.75 if decision == "bad" else None,
        "width_update_target_context_width": 3.0 if decision == "bad" else None,
        "wid_source": 3.0,
        "wid_adj_source": 3.0,
        "wid_med_source": 3.0,
    }


def test_build_dataset_maps_every_review_and_derives_primary_predictors(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    _write_archive(root, "02", [_row(2, "bad"), _row(1, "good")])
    _write_archive(root, "03", [_row(4, "good")])

    dataset = build_width_review_training_dataset(root, validate_freshness=False)

    assert list(dataset.frame.columns) == list(DATASET_COLUMNS)
    assert dataset.frame[["example_id", "id_link"]].values.tolist() == [
        ["02", 1],
        ["02", 2],
        ["03", 4],
    ]
    bad = dataset.frame.loc[(dataset.frame["example_id"] == "02") & (dataset.frame["id_link"] == 2)].iloc[0]
    assert bad["manual_decision"] == "bad"
    assert bad["requires_correction"] == 1
    assert bad["length_over_raw_wet_width"] == pytest.approx(2.5)
    assert bad["invalid_sample_fraction"] == pytest.approx(0.25)
    assert bad["upstream_continuity_signed_deviation"] == pytest.approx(0.25)
    assert bad["upstream_continuity_abs_deviation"] == pytest.approx(0.25)
    assert bad["split_sensitive"] == 1
    assert dataset.report["excluded_rows"] == 0
    assert dataset.report["label_counts"] == {"bad": 1, "good": 2}


def test_discovery_does_not_descend_into_version_directories(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    archive = _write_archive(root, "02", [_row(1, "good")])
    old = archive / "versions" / "old"
    old.mkdir(parents=True)
    (old / ARCHIVE_MANIFEST_NAME).write_text("{}", encoding="utf-8")

    assert discover_review_archives(root) == [archive.resolve()]


def test_dataset_rejects_duplicate_example_link_keys(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    _write_archive(root, "02", [_row(1, "good"), _row(1, "bad")])

    with pytest.raises(ValueError, match="Duplicate example/link"):
        build_width_review_training_dataset(root, validate_freshness=False)


def test_dataset_rejects_incomplete_decisions(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    _write_archive(root, "02", [_row(1, "good"), _row(2, "")])

    with pytest.raises(ValueError, match="not a complete"):
        build_width_review_training_dataset(root, validate_freshness=False)


def test_dataset_output_is_versioned_and_reproducible(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    _write_archive(root, "02", [_row(1, "good"), _row(2, "bad")])
    dataset = build_width_review_training_dataset(root, validate_freshness=False)

    first = write_width_review_training_dataset(dataset, tmp_path / "output")
    first_bytes = Path(first["dataset"]).read_bytes()
    second = write_width_review_training_dataset(dataset, tmp_path / "output")

    assert first["dataset_id"] == second["dataset_id"]
    assert Path(second["dataset"]).read_bytes() == first_bytes
    manifest = json.loads(Path(second["manifest"]).read_text(encoding="utf-8"))
    assert manifest["dataset_schema_version"] == DATASET_SCHEMA_VERSION
    assert manifest["row_count"] == 2
