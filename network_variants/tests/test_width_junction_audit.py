from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from network_variants.width_junction_audit import (
    WindowSpecification,
    build_review_consensus,
    build_reviewed_junction_comparisons,
    calculate_endpoint_widths,
    parse_window,
)


def _single_link_inputs(*, reversed_samples: bool = False):
    families = pd.DataFrame(
        {
            "id_link": [7],
            "id_us_node": [1],
            "id_ds_node": [2],
            "len": [100.0],
            "wid_adj_wet": [60.0],
        }
    )
    links = gpd.GeoDataFrame(
        {
            "id_link": [7],
            "id_us_node": [1],
            "id_ds_node": [2],
            "geometry": [LineString([(0, 0), (100, 0)])],
        },
        crs="EPSG:32611",
    )
    nodes = gpd.GeoDataFrame(
        {"id_node": [1, 2], "geometry": [Point(0, 0), Point(100, 0)]},
        crs=links.crs,
    )
    x = np.arange(0.0, 101.0, 10.0)
    if reversed_samples:
        x = x[::-1]
    samples = pd.DataFrame(
        {
            "id_link": 7,
            "sample_index": np.arange(len(x)),
            "x": x,
            "y": 0.0,
            "width_wet": 10.0 + x,
            "used_in_trimmed_width": True,
            "width_invalid": False,
        }
    )
    return families, samples, links, nodes


@pytest.mark.parametrize(
    ("reversed_samples", "expected_orientation"),
    [(False, "sample_index_us_to_ds"), (True, "sample_index_ds_to_us")],
)
def test_endpoint_widths_follow_directed_nodes_not_sample_index(
    reversed_samples: bool,
    expected_orientation: str,
) -> None:
    families, samples, links, nodes = _single_link_inputs(reversed_samples=reversed_samples)
    result = calculate_endpoint_widths(
        example_id="02",
        families=families,
        samples=samples,
        directed_links=links,
        directed_nodes=nodes,
        windows=[WindowSpecification(0.1, 20.0, 20.0)],
    ).iloc[0]

    assert result["width_near_us_node_wet"] == pytest.approx(20.0)
    assert result["width_near_ds_node_wet"] == pytest.approx(100.0)
    assert result["us_endpoint_sample_count"] == 3
    assert result["ds_endpoint_sample_count"] == 3
    assert result["sample_order_orientation"] == expected_orientation
    assert bool(result["endpoint_windows_nonoverlapping"]) is True


def test_endpoint_widths_use_only_valid_trimmed_samples() -> None:
    families, samples, links, nodes = _single_link_inputs()
    samples.loc[samples["x"].eq(0), "used_in_trimmed_width"] = False
    samples.loc[samples["x"].eq(20), "width_invalid"] = True
    result = calculate_endpoint_widths(
        example_id="02",
        families=families,
        samples=samples,
        directed_links=links,
        directed_nodes=nodes,
        windows=[WindowSpecification(0.1, 20.0, 20.0)],
    ).iloc[0]

    # The window begins at the closest retained sample (x=10), includes x=10
    # and x=30, and excludes invalid x=20.
    assert result["width_near_us_node_wet"] == pytest.approx(30.0)
    assert result["us_trim_gap_length"] == pytest.approx(10.0)
    assert result["us_endpoint_sample_count"] == 2


def test_endpoint_widths_prefer_preserved_raw_width_over_reviewed_width() -> None:
    families, samples, links, nodes = _single_link_inputs()
    families["wid_adj_wet"] = 15.0
    families["wid_adj_wet_raw"] = 60.0

    result = calculate_endpoint_widths(
        example_id="02",
        families=families,
        samples=samples,
        directed_links=links,
        directed_nodes=nodes,
        windows=[WindowSpecification(0.1, 20.0, 20.0)],
    ).iloc[0]

    assert result["whole_link_mean_wet_width"] == pytest.approx(60.0)
    assert result["whole_link_mean_wet_width_source"] == "wid_adj_wet_raw"


def test_reviewed_comparison_exposes_plain_width_sums_and_derived_deviation() -> None:
    endpoints = pd.DataFrame(
        {
            "example_id": ["02", "02", "02"],
            "id_link": [1, 2, 3],
            "id_us_node": [0, 1, 1],
            "id_ds_node": [1, 2, 3],
            "window_id": ["test"] * 3,
            "link_length": [1000.0, 100.0, 100.0],
            "whole_link_mean_wet_width": [50.0, 80.0, 20.0],
            "length_over_wet_width": [20.0, 1.25, 5.0],
            "width_near_us_node_wet": [50.0, 80.0, 20.0],
            "width_near_ds_node_wet": [100.0, 80.0, 20.0],
            "requested_window_length": [100.0] * 3,
            "endpoint_windows_nonoverlapping": [True, False, False],
            "us_endpoint_sample_count": [10, 10, 10],
            "ds_endpoint_sample_count": [10, 10, 10],
            "us_trim_gap_length": [0.0, 0.0, 0.0],
            "ds_trim_gap_length": [0.0, 0.0, 0.0],
            "sample_order_orientation": ["sample_index_us_to_ds"] * 3,
            "sample_orientation_score_margin": [1.0] * 3,
        }
    )
    reviewed = pd.DataFrame(
        {
            "example_id": ["02"],
            "id_link": [2],
            "manual_decision": ["good"],
            "requires_correction": [0],
            "split_sensitive": [1],
            "upstream_continuity_ratio": [2.0],
            "upstream_continuity_abs_deviation": [1.0],
            "review_archive_dir": ["02/width_review"],
            "review_manifest_path": ["02/width_review/manifest.json"],
        }
    )

    row = build_reviewed_junction_comparisons(endpoints, reviewed).iloc[0]

    assert row["incoming_mean_width_sum"] == pytest.approx(50.0)
    assert row["outgoing_mean_width_sum"] == pytest.approx(100.0)
    assert row["incoming_junction_width_sum"] == pytest.approx(100.0)
    assert row["outgoing_junction_width_sum"] == pytest.approx(100.0)
    assert row["incoming_link_ids"] == "1"
    assert row["outgoing_link_ids"] == "2,3"
    assert row["incoming_max_link_length"] == pytest.approx(1000.0)
    assert row["incoming_max_length_over_width"] == pytest.approx(20.0)
    assert bool(row["incoming_endpoint_windows_nonoverlapping"]) is True
    assert bool(row["outgoing_endpoint_windows_nonoverlapping"]) is False
    assert bool(row["all_junction_endpoint_windows_nonoverlapping"]) is False
    assert row["mean_continuity_abs_deviation_recomputed"] == pytest.approx(1.0)
    assert row["junction_continuity_abs_deviation"] == pytest.approx(0.0)
    assert bool(row["mean_high_junction_low"]) is True


def test_parse_window_accepts_percent_or_fraction() -> None:
    assert parse_window("10,250,1000") == WindowSpecification(0.1, 250.0, 1000.0)
    assert parse_window("0.1,250,1000") == WindowSpecification(0.1, 250.0, 1000.0)


def test_consensus_prioritizes_labels_that_may_change() -> None:
    rows = []
    for window_id, local_deviation in (("a", 0.1), ("b", 0.2)):
        rows.append(
            {
                "example_id": "02",
                "id_link": 10,
                "manual_decision": "bad",
                "requires_correction": 1,
                "split_sensitive": True,
                "window_id": window_id,
                "incoming_link_ids": "9",
                "outgoing_link_ids": "10,11",
                "incoming_max_link_length": 1000.0,
                "incoming_max_length_over_width": 10.0,
                "mean_continuity_abs_deviation_recomputed": 1.0,
                "junction_continuity_abs_deviation": local_deviation,
                "absolute_deviation_change": 1.0 - local_deviation,
                "material_continuity_change": True,
                "all_junction_endpoint_windows_nonoverlapping": True,
                "mean_high_junction_low": True,
                "junction_high_mean_low": False,
            }
        )
    row = build_review_consensus(pd.DataFrame(rows)).iloc[0]

    assert row["targeted_review_priority"] == 1
    assert row["targeted_review_reason"] == "manual_bad_but_mean_discontinuity_resolves_near_junction"
    assert bool(row["junction_result_stable_within_0_1"]) is True
