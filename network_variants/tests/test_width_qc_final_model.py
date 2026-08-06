from __future__ import annotations

import pandas as pd

from network_variants.width_qc_final_model import (
    FINAL_FEATURE_COLUMNS,
    HYBRID_FEATURE,
    build_decision_summary,
    build_final_training_dataset,
    build_old_new_prediction_comparison,
    build_validation_comparison,
)


def _source_dataset() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "example_id": ["02", "02", "03"],
            "id_link": [1, 2, 3],
            "manual_decision": ["good", "bad", "good"],
            "requires_correction": [0, 1, 0],
            "split_sensitive": [1, 1, 0],
            "raw_wet_width": [10.0, 20.0, 30.0],
            "upstream_continuity_abs_deviation": [0.1, 0.8, 0.2],
        }
    )
    for feature in FINAL_FEATURE_COLUMNS:
        if feature != HYBRID_FEATURE and feature not in frame.columns:
            frame[feature] = [0.1, 0.2, 0.3]
    return frame


def _review() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "example_id": ["02", "03"],
            "id_link": [2, 3],
            "adjudication": ["model_upheld", "ambiguous"],
            "adjudicated_requires_correction": [0, pd.NA],
            "include_in_training": [True, False],
            "review_comment": ["", "unclear"],
        }
    )


def _hybrid() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "example_id": ["02", "02", "03"],
            "id_link": [1, 2, 3],
            "hybrid_variant_id": ["lw_ge_10_nonoverlap"] * 3,
            "hybrid_continuity_ratio": [1.1, 1.2, 1.3],
            "hybrid_continuity_abs_deviation": [0.1, 0.2, 0.3],
            "incoming_endpoint_link_count": [0, 1, 0],
            "outgoing_endpoint_link_count": [0, 0, 0],
        }
    )


def test_final_dataset_applies_adjudication_and_excludes_ambiguous() -> None:
    included, excluded = build_final_training_dataset(
        _source_dataset(), _review(), _hybrid()
    )

    assert included["id_link"].tolist() == [1, 2]
    assert excluded["id_link"].tolist() == [3]
    row = included.set_index("id_link").loc[2]
    assert row["original_requires_correction"] == 1
    assert row["requires_correction"] == 0
    assert row["manual_decision"] == "good"
    assert bool(row["final_label_changed_by_adjudication"]) is True
    assert row[HYBRID_FEATURE] == 0.2
    assert bool(row["upstream_hybrid_uses_endpoint"]) is True


def test_prediction_comparison_reports_both_decision_change_directions() -> None:
    final_dataset, _ = build_final_training_dataset(
        _source_dataset(), _review(), _hybrid()
    )
    old = pd.DataFrame(
        {
            "example_id": ["02", "02"],
            "id_link": [1, 2],
            "width_model_name": ["logistic_regression"] * 2,
            "width_model_probability": [0.2, 0.8],
            "width_model_prediction": [0, 1],
            "width_model_threshold": [0.5, 0.5],
            "width_model_version": ["old"] * 2,
        }
    )
    new = pd.DataFrame(
        {
            "example_id": ["02", "02"],
            "id_link": [1, 2],
            "requires_correction": [0, 0],
            "width_model_name": ["logistic_regression"] * 2,
            "width_model_probability": [0.7, 0.2],
            "width_model_prediction": [1, 0],
            "width_model_threshold": [0.6, 0.6],
            "width_model_version": ["final"] * 2,
            "routing_impact_score": [1.0, 2.0],
        }
    )

    comparison = build_old_new_prediction_comparison(final_dataset, old, new)
    assert set(comparison["decision_change"]) == {"accept_to_correct", "correct_to_accept"}
    summary = build_decision_summary(comparison, selected_model="logistic_regression")
    total = summary.loc[summary["scope"].eq("all")].iloc[0]
    assert total["decision_changes"] == 2
    assert total["accept_to_correct"] == 1
    assert total["correct_to_accept"] == 1
    assert bool(total["selected_model"]) is True
    validation = build_validation_comparison(comparison)
    assert set(validation["comparison_version"]) == {
        "original_model_on_final_labels",
        "final_hybrid_model",
    }
