from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from network_variants.width_qc_disagreement_gui import (
    ADJUDICATIONS,
    JUNCTION_EVIDENCE_COLUMNS,
    MANIFEST_FILENAME,
    REVIEW_FILENAME,
    ReviewInputs,
    adjudication_result,
    build_disagreement_frame,
    load_or_create_review,
    merge_existing_adjudications,
    write_disagreement_review,
)
from network_variants.width_qc_model import FEATURE_COLUMNS
from network_variants.width_review_archive import _sha256


def _inputs(tmp_path: Path) -> ReviewInputs:
    dataset_path = tmp_path / "dataset.csv"
    predictions_path = tmp_path / "predictions.csv"
    summary_path = tmp_path / "summary.json"
    rows = []
    for link_id, manual in ((10, 0), (11, 1), (12, 1)):
        row: dict[str, object] = {
            "example_id": "02",
            "id_link": link_id,
            "variant_id": "S000_base",
            "id_us_node": 1,
            "id_ds_node": 2,
            "requires_correction": manual,
            "upstream_continuity_signed_deviation": 0.2,
            "downstream_continuity_ratio": 1.1,
            "downstream_continuity_signed_deviation": 0.1,
            "downstream_continuity_abs_deviation": 0.1,
            "raw_wet_width": 10.0,
            "raw_total_width": 10.0,
            "raw_dry_width": 0.0,
            "dry_fraction": 0.0,
            "link_length": 5.0,
            "width_sample_count": 10,
            "n_invalid_width_samples": 0,
            "n_trimmed_qc_samples": 8,
            "lineage_type": "unchanged",
            "review_archive_dir": "02/width_review",
            "review_targets_path": "02/width_review/reviewed_width_targets.csv",
            "review_manifest_path": "02/width_review/base_width_review_manifest.json",
        }
        row.update({column: 0.5 for column in FEATURE_COLUMNS})
        row["split_sensitive"] = 1
        rows.append(row)
    pd.DataFrame(rows).to_csv(dataset_path, index=False)
    pd.DataFrame(
        [
            {
                "example_id": "02",
                "id_link": 10,
                "manual_decision": "good",
                "requires_correction": 0,
                "split_sensitive": 1,
                "width_model_probability": 0.8,
                "width_model_prediction": 1,
                "width_model_threshold": 0.5,
                "width_model_name": "logistic_regression",
                "width_model_version": "test-v1",
                "manual_model_disagreement": True,
                "routing_impact_score": 20.0,
            },
            {
                "example_id": "02",
                "id_link": 11,
                "manual_decision": "bad",
                "requires_correction": 1,
                "split_sensitive": 1,
                "width_model_probability": 0.2,
                "width_model_prediction": 0,
                "width_model_threshold": 0.5,
                "width_model_name": "logistic_regression",
                "width_model_version": "test-v1",
                "manual_model_disagreement": True,
                "routing_impact_score": 30.0,
            },
            {
                "example_id": "02",
                "id_link": 12,
                "manual_decision": "bad",
                "requires_correction": 1,
                "split_sensitive": 1,
                "width_model_probability": 0.9,
                "width_model_prediction": 1,
                "width_model_threshold": 0.5,
                "width_model_name": "logistic_regression",
                "width_model_version": "test-v1",
                "manual_model_disagreement": False,
                "routing_impact_score": 40.0,
            },
        ]
    ).to_csv(predictions_path, index=False)
    summary_path.write_text(json.dumps({"provisional_candidate_model": "logistic_regression"}), encoding="utf-8")
    return ReviewInputs(
        dataset_path=dataset_path,
        predictions_path=predictions_path,
        summary_path=summary_path,
        dataset_sha256=_sha256(dataset_path),
        predictions_sha256=_sha256(predictions_path),
        summary_sha256=_sha256(summary_path),
        model_name="logistic_regression",
    )


def test_build_disagreement_frame_keeps_only_held_out_disagreements(tmp_path: Path) -> None:
    frame = build_disagreement_frame(_inputs(tmp_path))

    assert frame[["id_link", "disagreement_type"]].values.tolist() == [
        [11, "missed_correction"],
        [10, "false_correction"],
    ]
    assert "split_sensitive" in frame.columns
    assert "split_sensitive_x" not in frame.columns
    assert "split_sensitive_y" not in frame.columns
    assert frame["split_sensitive"].eq(1).all()
    assert frame["adjudication"].isna().all()
    assert set(ADJUDICATIONS) == {
        "manual_upheld",
        "model_upheld",
        "ambiguous",
        "measurement_valid_but_routing_unsuitable",
        "data_or_geometry_problem",
        "exclude_from_training",
    }


def test_build_disagreement_frame_attaches_selected_junction_window(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    audit_path = tmp_path / "junction.csv"
    rows = []
    for link_id in (10, 11):
        row: dict[str, object] = {
            "example_id": "02",
            "id_link": link_id,
            "window_id": "p10_min250_max1000",
        }
        row.update({column: 1.0 for column in JUNCTION_EVIDENCE_COLUMNS})
        row["incoming_link_ids"] = "8,9"
        row["outgoing_link_ids"] = "10,11"
        row["incoming_endpoint_windows_nonoverlapping"] = True
        row["outgoing_endpoint_windows_nonoverlapping"] = False
        rows.append(row)
    pd.DataFrame(rows).to_csv(audit_path, index=False)
    inputs = replace(
        inputs,
        junction_audit_path=audit_path,
        junction_audit_sha256=_sha256(audit_path),
        junction_window_id="p10_min250_max1000",
    )

    frame = build_disagreement_frame(inputs)

    assert frame["window_id"].eq("p10_min250_max1000").all()
    assert frame["incoming_link_ids"].eq("8,9").all()
    assert frame["incoming_endpoint_windows_nonoverlapping"].eq(True).all()
    assert frame["outgoing_endpoint_windows_nonoverlapping"].eq(False).all()


def test_review_queue_unions_model_disagreements_and_junction_candidates(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    audit_path = tmp_path / "junction.csv"
    audit_rows = []
    for link_id in (10, 11, 12):
        row: dict[str, object] = {
            "example_id": "02",
            "id_link": link_id,
            "window_id": "p10_min250_max1000",
        }
        row.update({column: 1.0 for column in JUNCTION_EVIDENCE_COLUMNS})
        row["incoming_link_ids"] = "8,9"
        row["outgoing_link_ids"] = "10,11"
        audit_rows.append(row)
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
    consensus_path = tmp_path / "consensus.csv"
    pd.DataFrame(
        {
            "example_id": ["02"],
            "id_link": [12],
            "targeted_review_priority": [1],
            "targeted_review_reason": ["manual_good_but_new_near_junction_discontinuity"],
            "junction_result_stable_within_0_1": [True],
        }
    ).to_csv(consensus_path, index=False)
    inputs = replace(
        inputs,
        junction_audit_path=audit_path,
        junction_audit_sha256=_sha256(audit_path),
        junction_window_id="p10_min250_max1000",
        junction_consensus_path=consensus_path,
        junction_consensus_sha256=_sha256(consensus_path),
        max_junction_review_priority=2,
    )

    frame = build_disagreement_frame(inputs)

    assert set(frame["id_link"]) == {10, 11, 12}
    added = frame.loc[frame["id_link"].eq(12)].iloc[0]
    assert bool(added["manual_model_disagreement"]) is False
    assert added["review_trigger"] == "junction_audit"
    assert added["disagreement_type"] == "junction_audit_candidate"


def test_adjudication_result_preserves_semantic_distinctions() -> None:
    assert adjudication_result(
        "manual_upheld", manual_requires_correction=0, model_requires_correction=1
    ) == (0, True)
    assert adjudication_result(
        "model_upheld", manual_requires_correction=0, model_requires_correction=1
    ) == (1, True)
    assert adjudication_result(
        "measurement_valid_but_routing_unsuitable",
        manual_requires_correction=0,
        model_requires_correction=1,
    ) == (1, True)
    assert adjudication_result(
        "exclude_from_training", manual_requires_correction=0, model_requires_correction=1
    ) == (None, False)
    assert adjudication_result(
        "ambiguous", manual_requires_correction=0, model_requires_correction=1
    ) == (None, None)


def test_existing_adjudications_resume_by_example_and_link(tmp_path: Path) -> None:
    frame = build_disagreement_frame(_inputs(tmp_path))
    existing = frame.copy()
    existing.loc[existing["id_link"].eq(10), "adjudication"] = "model_upheld"
    existing.loc[existing["id_link"].eq(10), "adjudicated_requires_correction"] = 1
    existing.loc[existing["id_link"].eq(10), "include_in_training"] = True
    existing.loc[existing["id_link"].eq(10), "review_comment"] = "manual slip"

    resumed = merge_existing_adjudications(frame, existing)

    row = resumed.loc[resumed["id_link"].eq(10)].iloc[0]
    assert row["adjudication"] == "model_upheld"
    assert row["adjudicated_requires_correction"] == 1
    assert row["review_comment"] == "manual slip"


def test_write_review_records_hashes_counts_and_versions_prior_save(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    frame = build_disagreement_frame(inputs)
    frame.loc[0, "adjudication"] = "manual_upheld"
    frame.loc[0, "adjudicated_requires_correction"] = 1
    frame.loc[0, "include_in_training"] = True

    first = write_disagreement_review(frame, inputs=inputs, output_dir=tmp_path / "review")
    frame.loc[1, "adjudication"] = "exclude_from_training"
    frame.loc[1, "include_in_training"] = False
    second = write_disagreement_review(frame, inputs=inputs, output_dir=tmp_path / "review")

    manifest = json.loads(Path(second["manifest"]).read_text(encoding="utf-8"))
    assert Path(first["review"]).name == REVIEW_FILENAME
    assert Path(second["manifest"]).name == MANIFEST_FILENAME
    assert manifest["counts"]["reviewed"] == 2
    assert manifest["counts"]["unreviewed"] == 0
    assert manifest["inputs"]["dataset_sha256"] == inputs.dataset_sha256
    assert manifest["file_sha256"]["review"] == _sha256(Path(second["review"]))
    versions = list((tmp_path / "review" / "versions").glob("*/" + REVIEW_FILENAME))
    assert len(versions) == 1


def test_resume_rejects_adjudications_from_different_model_inputs(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    frame = build_disagreement_frame(inputs)
    output_dir = tmp_path / "review"
    write_disagreement_review(frame, inputs=inputs, output_dir=output_dir)
    changed = ReviewInputs(
        dataset_path=inputs.dataset_path,
        predictions_path=inputs.predictions_path,
        summary_path=inputs.summary_path,
        dataset_sha256=inputs.dataset_sha256,
        predictions_sha256="different-predictions",
        summary_sha256=inputs.summary_sha256,
        model_name=inputs.model_name,
    )

    with pytest.raises(ValueError, match="different inputs"):
        load_or_create_review(changed, output_dir=output_dir, start_over=False)

    restarted = load_or_create_review(changed, output_dir=output_dir, start_over=True)
    assert len(restarted) == 2
    assert restarted["adjudication"].isna().all()
