from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from network_variants.width_review_archive import _sha256
from network_variants.width_review_dataset import DATASET_SCHEMA_VERSION
from network_variants.width_qc_model import (
    FEATURE_COLUMNS,
    MODEL_VERSION,
    load_model_dataset,
    run_leave_one_example_out,
    select_conservative_threshold,
    write_model_development_outputs,
)


def _model_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for example_index, example_id in enumerate(("02", "03", "04", "05")):
        for row_index in range(10):
            bad = row_index >= 5
            noise = 0.02 * example_index + 0.01 * row_index
            rows.append(
                {
                    "dataset_schema_version": DATASET_SCHEMA_VERSION,
                    "example_id": example_id,
                    "id_link": example_index * 100 + row_index,
                    "manual_decision": "bad" if bad else "good",
                    "requires_correction": int(bad),
                    "length_over_raw_wet_width": (0.5 if bad else 2.0) + noise,
                    "split_sensitive": int(row_index % 2 == 0),
                    "opposition_cos_trimmed_mean": (-0.35 if bad else -0.9) + noise,
                    "opposition_cos_trimmed_median": (-0.4 if bad else -0.92) + noise,
                    "invalid_sample_fraction": 0.1 if bad and row_index == 9 else 0.0,
                    "upstream_continuity_ratio": (1.8 if bad else 1.05) + noise,
                    "upstream_continuity_abs_deviation": (0.8 if bad else 0.05) + noise,
                    "raw_wet_width": 10.0 + row_index,
                }
            )
    frame = pd.DataFrame(rows)
    frame.loc[frame.index[0], "opposition_cos_trimmed_median"] = np.nan
    return frame


def test_leave_one_example_out_produces_one_probability_per_model_and_row(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    _model_frame().to_csv(dataset_path, index=False)
    frame = load_model_dataset(dataset_path)

    result = run_leave_one_example_out(frame, max_split_false_correction_rate=0.05)

    assert set(result.predictions["width_model_name"]) == {
        "logistic_regression",
        "linear_svm",
        "rbf_svm",
    }
    assert len(result.predictions) == len(frame) * 3
    assert result.predictions.groupby("width_model_name").size().eq(len(frame)).all()
    assert result.predictions["width_model_probability"].between(0, 1).all()
    assert result.summary["validation_design"] == "leave_one_complete_example_out"
    assert result.summary["production_authorized"] is False
    assert len(result.per_example_metrics) == 4 * 3
    assert set(result.fitted_models) == {"logistic_regression", "linear_svm", "rbf_svm"}


def test_threshold_selection_respects_split_sensitive_false_correction_constraint() -> None:
    table = pd.DataFrame(
        {
            "threshold": [0.4, 0.6, 0.8],
            "recall": [0.9, 0.7, 0.3],
            "precision": [0.6, 0.8, 1.0],
            "split_sensitive_false_correction_rate": [0.2, 0.05, 0.0],
        }
    )

    threshold, details = select_conservative_threshold(
        table, max_split_false_correction_rate=0.05
    )

    assert threshold == 0.6
    assert details["constraint_met"] is True


def test_written_candidate_bundles_are_traceable_and_not_authorized(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    _model_frame().to_csv(dataset_path, index=False)
    dataset_manifest_path = tmp_path / "dataset_manifest.json"
    dataset_manifest_path.write_text(
        json.dumps(
            {
                "dataset_schema_version": DATASET_SCHEMA_VERSION,
                "dataset_id": "test-dataset-id",
                "dataset_sha256": _sha256(dataset_path),
            }
        ),
        encoding="utf-8",
    )
    frame = load_model_dataset(dataset_path)
    result = run_leave_one_example_out(frame)

    outputs = write_model_development_outputs(
        result,
        dataset_path=dataset_path,
        dataset_manifest_path=dataset_manifest_path,
        output_dir=tmp_path / "model_output",
    )

    manifest = json.loads(Path(outputs["manifest"]).read_text(encoding="utf-8"))
    assert manifest["model_version"] == MODEL_VERSION
    assert manifest["dataset_id"] == "test-dataset-id"
    assert manifest["production_authorized"] is False
    assert set(manifest["candidate_bundles"]) == {
        "logistic_regression",
        "linear_svm",
        "rbf_svm",
    }
    bundle = joblib.load(manifest["candidate_bundles"]["logistic_regression"])
    assert bundle["feature_columns"] == list(FEATURE_COLUMNS)
    assert bundle["dataset_sha256"] == _sha256(dataset_path)
    assert bundle["production_authorized"] is False
