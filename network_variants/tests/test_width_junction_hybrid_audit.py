from __future__ import annotations

import pandas as pd
import pytest

from network_variants.width_junction_hybrid_audit import (
    DEFAULT_VARIANTS,
    build_effective_label_frame,
    build_followup_candidates,
    calculate_hybrid_comparisons,
)


def _endpoint_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "example_id": ["02", "02", "02"],
            "id_link": [1, 2, 3],
            "id_us_node": [0, 1, 1],
            "id_ds_node": [1, 2, 3],
            "window_id": ["p10_min250_max1000"] * 3,
            "whole_link_mean_wet_width": [100.0, 120.0, 80.0],
            "width_near_us_node_wet": [110.0, 130.0, 90.0],
            "width_near_ds_node_wet": [200.0, 140.0, 100.0],
            "length_over_wet_width": [10.0, 1.0, 6.0],
            "endpoint_windows_nonoverlapping": [True, True, False],
        }
    )


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "example_id": ["02"],
            "id_link": [2],
            "manual_decision": ["good"],
            "requires_correction": pd.Series([0], dtype="Int64"),
            "split_sensitive": [True],
            "adjudication": ["manual_upheld"],
            "adjudicated_requires_correction": pd.Series([0], dtype="Int64"),
            "include_in_training": [True],
            "review_comment": [""],
            "effective_requires_correction": pd.Series([0], dtype="Int64"),
            "effective_label_source": ["adjudicated_manual_upheld"],
            "in_completed_review": [True],
        }
    )


def test_hybrid_rule_selects_endpoint_width_per_adjacent_link() -> None:
    comparisons = calculate_hybrid_comparisons(_endpoint_rows(), _labels())
    rows = comparisons.set_index("hybrid_variant_id")

    assert rows.loc["whole_mean", "hybrid_continuity_abs_deviation"] == pytest.approx(1.0)
    assert rows.loc["endpoint_all", "hybrid_continuity_abs_deviation"] == pytest.approx(0.1)
    assert rows.loc["lw_ge_5", "hybrid_continuity_abs_deviation"] == pytest.approx(0.05)
    assert rows.loc["lw_ge_10", "hybrid_continuity_abs_deviation"] == pytest.approx(0.0)
    assert rows.loc["lw_ge_5", "incoming_endpoint_link_ids"] == "1"
    assert rows.loc["lw_ge_5", "outgoing_endpoint_link_ids"] == "3"
    assert rows.loc["lw_ge_5_nonoverlap", "outgoing_endpoint_link_ids"] == ""
    assert rows.loc["lw_ge_5_nonoverlap", "hybrid_continuity_abs_deviation"] == pytest.approx(0.0)


def test_effective_labels_apply_resolved_review_and_exclude_ambiguous() -> None:
    dataset = pd.DataFrame(
        {
            "example_id": ["02", "02", "02", "02"],
            "id_link": [1, 2, 3, 4],
            "manual_decision": ["good", "bad", "bad", "good"],
            "requires_correction": [0, 1, 1, 0],
            "split_sensitive": [True] * 4,
        }
    )
    review = pd.DataFrame(
        {
            "example_id": ["02", "02", "02"],
            "id_link": [2, 3, 4],
            "adjudication": ["manual_upheld", "model_upheld", "ambiguous"],
            "adjudicated_requires_correction": [1, 0, pd.NA],
            "include_in_training": [True, True, False],
            "review_comment": ["", "", "inspect"],
        }
    )

    labels = build_effective_label_frame(dataset, review).set_index("id_link")

    assert labels.loc[1, "effective_requires_correction"] == 0
    assert labels.loc[2, "effective_requires_correction"] == 1
    assert labels.loc[3, "effective_requires_correction"] == 0
    assert pd.isna(labels.loc[4, "effective_requires_correction"])
    assert labels.loc[3, "effective_label_source"] == "adjudicated_model_upheld"
    assert labels.loc[4, "effective_label_source"] == "excluded_ambiguous"


def test_followup_contains_ambiguous_and_resolved_rule_sensitive_links() -> None:
    records = []
    for link_id, adjudication, values in (
        (1, "ambiguous", [0.1] * 8),
        (2, "manual_upheld", [0.1, 0.1, 0.1, 0.7, 0.1, 0.1, 0.1, 0.1]),
        (3, "model_upheld", [0.1] * 8),
    ):
        for variant, value in zip(DEFAULT_VARIANTS, values):
            records.append(
                {
                    "example_id": "02",
                    "id_link": link_id,
                    "manual_decision": "good",
                    "requires_correction": 0,
                    "adjudication": adjudication,
                    "adjudicated_requires_correction": pd.NA if adjudication == "ambiguous" else 0,
                    "effective_requires_correction": pd.NA if adjudication == "ambiguous" else 0,
                    "effective_label_source": "test",
                    "review_comment": pd.NA,
                    "hybrid_variant_id": variant.variant_id,
                    "hybrid_continuity_abs_deviation": value,
                }
            )

    followup = build_followup_candidates(pd.DataFrame(records)).set_index("id_link")

    assert set(followup.index) == {1, 2}
    assert followup.loc[1, "followup_reason"] == "ambiguous_adjudication"
    assert followup.loc[2, "followup_reason"] == "resolved_adjudication_sensitive_to_hybrid_rule"

