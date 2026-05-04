from __future__ import annotations

import math

import pandas as pd
import pytest

from hierarchy_level_definition.metrics.unit_metrics import (
    compute_unit_metrics_from_units,
    normalized_entropy,
    weighted_harmonic_mean_width,
)
from hierarchy_level_definition.unit_detection.bifurcation_confluence_units import (
    StructuralUnit,
    UnitPath,
    build_unit_context_frame,
    build_summary_frame,
)


def _make_links(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(rows)


def _make_parallel_path(path_id: int, link_id: int, *, total_length: float) -> UnitPath:
    return UnitPath(
        path_id=path_id,
        node_path=[1, 2],
        edge_path=[(1, 2, link_id)],
        id_links=[link_id],
        total_length=total_length,
    )


def _make_unit(
    paths: list[UnitPath],
    *,
    unit_id: int = 1,
    internal_bifurcations: list[int] | None = None,
    internal_confluences: list[int] | None = None,
    children: list[int] | None = None,
) -> StructuralUnit:
    edge_set = {edge for path in paths for edge in path.edge_path}
    path_lengths = [path.total_length for path in paths] or [0.0]
    return StructuralUnit(
        unit_id=unit_id,
        bifurcation=1,
        confluence=2,
        n_paths=len(paths),
        unit_class="simple_bifurcation_confluence_pair",
        min_path_length=min(path_lengths),
        max_path_length=max(path_lengths),
        paths=paths,
        node_set={1, 2},
        edge_set=edge_set,
        internal_bifurcations=internal_bifurcations or [],
        internal_confluences=internal_confluences or [],
        children=children or [],
        parents=[],
    )


def test_weighted_harmonic_mean_width() -> None:
    result = weighted_harmonic_mean_width([100.0, 100.0], [50.0, 100.0])
    assert result == pytest.approx(200.0 / (100.0 / 50.0 + 100.0 / 100.0))


def test_unit_equivalent_width_and_length() -> None:
    links = _make_links(
        [
            {"id_link": 1, "len_adj": 100.0, "wid_adj": 50.0},
            {"id_link": 2, "len_adj": 200.0, "wid_adj": 100.0},
        ]
    )
    unit = _make_unit(
        [
            _make_parallel_path(1, 1, total_length=100.0),
            _make_parallel_path(2, 2, total_length=200.0),
        ]
    )

    unit_metrics, _ = compute_unit_metrics_from_units(links, [unit])
    row = unit_metrics.iloc[0]

    assert row["equivalent_width"] == pytest.approx(150.0)
    assert row["equivalent_length"] == pytest.approx((50.0 * 100.0 + 100.0 * 200.0) / 150.0)


def test_entropy_metrics_for_equal_paths() -> None:
    entropy, evenness, effective_n = normalized_entropy([1.0, 1.0, 1.0, 1.0])
    assert entropy == pytest.approx(math.log(4.0))
    assert evenness == pytest.approx(1.0)
    assert effective_n == pytest.approx(4.0)


def test_dominant_two_path_metrics() -> None:
    links = _make_links(
        [
            {"id_link": 1, "len_adj": 100.0, "wid_adj": 90.0},
            {"id_link": 2, "len_adj": 100.0, "wid_adj": 10.0},
        ]
    )
    unit = _make_unit(
        [
            _make_parallel_path(1, 1, total_length=100.0),
            _make_parallel_path(2, 2, total_length=100.0),
        ]
    )

    unit_metrics, _ = compute_unit_metrics_from_units(links, [unit])
    row = unit_metrics.iloc[0]

    assert row["smaller_width_fraction_2"] == pytest.approx(0.1)
    assert row["dominant_width_fraction_2"] == pytest.approx(0.9)
    assert row["effective_n_paths_width"] < 2.0


def test_pixel_width_samples_drive_extreme_metrics() -> None:
    links = _make_links(
        [
            {
                "id_link": 1,
                "len_adj": 100.0,
                "wid_adj": 80.0,
                "width_samples": "[20, 40, 80, 100, 120]",
            }
        ]
    )
    unit = _make_unit([_make_parallel_path(1, 1, total_length=100.0)])

    _, path_metrics = compute_unit_metrics_from_units(links, [unit], pixel_width_fields=["width_samples"])
    row = path_metrics.iloc[0]

    assert row["path_width_min"] == pytest.approx(20.0)
    assert row["path_width_max"] == pytest.approx(120.0)
    assert row["path_width_p50"] == pytest.approx(80.0)


def test_pixel_width_fallback_uses_representative_widths() -> None:
    links = _make_links([{"id_link": 1, "len_adj": 100.0, "wid_adj": 80.0}])
    unit = _make_unit([_make_parallel_path(1, 1, total_length=100.0)])

    _, path_metrics = compute_unit_metrics_from_units(links, [unit], pixel_width_fields=["width_samples"])
    row = path_metrics.iloc[0]

    assert row["path_width_min"] == pytest.approx(80.0)
    assert row["path_width_max"] == pytest.approx(80.0)
    assert row["path_width_p05"] == pytest.approx(80.0)
    assert row["path_width_p50"] == pytest.approx(80.0)
    assert row["path_width_p95"] == pytest.approx(80.0)


def test_balanced_simple_split_classification() -> None:
    links = _make_links(
        [
            {"id_link": 1, "len_adj": 100.0, "wid_adj": 50.0},
            {"id_link": 2, "len_adj": 100.0, "wid_adj": 50.0},
        ]
    )
    unit = _make_unit(
        [
            _make_parallel_path(1, 1, total_length=100.0),
            _make_parallel_path(2, 2, total_length=100.0),
        ]
    )

    unit_metrics, _ = compute_unit_metrics_from_units(links, [unit])
    assert unit_metrics.iloc[0]["unit_topodynamic_class"] == "balanced_simple_split"


def test_dominant_simple_split_classification() -> None:
    links = _make_links(
        [
            {"id_link": 1, "len_adj": 100.0, "wid_adj": 90.0},
            {"id_link": 2, "len_adj": 100.0, "wid_adj": 10.0},
        ]
    )
    unit = _make_unit(
        [
            _make_parallel_path(1, 1, total_length=100.0),
            _make_parallel_path(2, 2, total_length=100.0),
        ]
    )

    unit_metrics, _ = compute_unit_metrics_from_units(links, [unit])
    assert unit_metrics.iloc[0]["unit_topodynamic_class"] == "dominant_simple_split"


def test_compound_bubble_membership_and_unit_node_ids() -> None:
    links = _make_links(
        [
            {"id_link": 1, "len_adj": 100.0, "wid_adj": 60.0},
            {"id_link": 2, "len_adj": 120.0, "wid_adj": 40.0},
            {"id_link": 3, "len_adj": 70.0, "wid_adj": 30.0},
            {"id_link": 4, "len_adj": 80.0, "wid_adj": 25.0},
            {"id_link": 5, "len_adj": 90.0, "wid_adj": 50.0},
            {"id_link": 6, "len_adj": 110.0, "wid_adj": 45.0},
            {"id_link": 7, "len_adj": 100.0, "wid_adj": 55.0},
            {"id_link": 8, "len_adj": 120.0, "wid_adj": 35.0},
        ]
    )

    root_unit = StructuralUnit(
        unit_id=10,
        bifurcation=1,
        confluence=4,
        n_paths=2,
        unit_class="compound_or_nested_complex",
        min_path_length=100.0,
        max_path_length=120.0,
        paths=[
            UnitPath(1, [1, 4], [(1, 4, 1)], [1], 100.0),
            UnitPath(2, [1, 2, 4], [(1, 2, 2), (2, 4, 3)], [2, 3], 190.0),
        ],
        node_set={1, 2, 3, 4, 5},
        edge_set={(1, 4, 1), (1, 2, 2), (2, 4, 3), (2, 3, 4), (2, 5, 5)},
        internal_bifurcations=[2],
        internal_confluences=[],
        children=[20],
        parents=[],
    )
    child_unit = StructuralUnit(
        unit_id=20,
        bifurcation=2,
        confluence=3,
        n_paths=2,
        unit_class="simple_bifurcation_confluence_pair",
        min_path_length=80.0,
        max_path_length=90.0,
        paths=[
            UnitPath(1, [2, 3], [(2, 3, 4)], [4], 80.0),
            UnitPath(2, [2, 5, 3], [(2, 5, 5), (5, 3, 6)], [5, 6], 200.0),
        ],
        node_set={2, 3, 5},
        edge_set={(2, 3, 4), (2, 5, 5), (5, 3, 6)},
        internal_bifurcations=[],
        internal_confluences=[],
        children=[],
        parents=[10],
    )
    standalone_unit = _make_unit(
        [
            _make_parallel_path(1, 7, total_length=100.0),
            _make_parallel_path(2, 8, total_length=120.0),
        ],
        unit_id=30,
    )

    unit_metrics, _ = compute_unit_metrics_from_units(links, [root_unit, child_unit, standalone_unit])
    summary = build_summary_frame([root_unit, child_unit, standalone_unit])

    metrics_by_id = unit_metrics.set_index("unit_id")
    summary_by_id = summary.set_index("unit_id")

    bubble_id_root = metrics_by_id.loc[10, "compound_bubble_id"]
    bubble_id_child = metrics_by_id.loc[20, "compound_bubble_id"]
    bubble_id_standalone = metrics_by_id.loc[30, "compound_bubble_id"]

    assert pd.notna(bubble_id_root)
    assert bubble_id_root == bubble_id_child
    assert bubble_id_root != bubble_id_standalone

    assert metrics_by_id.loc[10, "compound_bubble_role"] == "bubble_root"
    assert bool(metrics_by_id.loc[10, "in_compound_bubble"]) is True
    assert metrics_by_id.loc[10, "compound_unit_id"] == 10

    assert metrics_by_id.loc[20, "compound_bubble_role"] == "bubble_member"
    assert bool(metrics_by_id.loc[20, "in_compound_bubble"]) is True
    assert metrics_by_id.loc[20, "compound_unit_id"] == 10

    assert pd.notna(metrics_by_id.loc[30, "compound_bubble_id"])
    assert metrics_by_id.loc[30, "compound_bubble_role"] == "standalone"
    assert bool(metrics_by_id.loc[30, "in_compound_bubble"]) is False
    assert pd.isna(metrics_by_id.loc[30, "compound_unit_id"])

    assert metrics_by_id.loc[20, "unit_node_ids"] == "2,3,5"
    assert summary_by_id.loc[20, "unit_node_ids"] == "2,3,5"


def test_build_unit_context_frame_handles_empty_units() -> None:
    context = build_unit_context_frame([])

    assert list(context.columns) == [
        "unit_id",
        "primary_parent_id",
        "root_unit_id",
        "depth_from_root",
        "collapse_level",
        "n_children",
        "n_descendants",
        "is_compound",
        "compound_unit_id",
        "compound_bubble_id",
        "in_compound_bubble",
        "compound_bubble_role",
        "unit_node_ids",
        "unit_node_count",
    ]
    assert context.empty


def test_compute_unit_metrics_from_units_handles_empty_units() -> None:
    links = _make_links([])

    unit_metrics, path_metrics = compute_unit_metrics_from_units(links, [])

    assert unit_metrics.empty
    assert path_metrics.empty
    assert "unit_id" in unit_metrics.columns
    assert "unit_topodynamic_class" in unit_metrics.columns
