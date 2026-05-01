from __future__ import annotations

import math

import pandas as pd
import pytest

from hierarchy_level_definition.metrics.unit_metrics import (
    compute_unit_metrics_from_units,
    normalized_entropy,
    weighted_harmonic_mean_width,
)
from hierarchy_level_definition.unit_detection.bifurcation_confluence_units import StructuralUnit, UnitPath


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
