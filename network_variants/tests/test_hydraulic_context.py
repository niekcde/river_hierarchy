from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from network_variants.hydraulic_context import (
    SlopeConfig,
    compute_hydraulic_context,
    generate_state_hydraulic_context,
)


def _simple_nodes() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        [
            {
                "id_node": 1,
                "is_inlet": True,
                "is_outlet": False,
                "sword_wse": 10.0,
                "sword_wse_field": "wse",
                "sword_wse_fill_method": "requested_field",
                "geometry": Point(0.0, 0.0),
            },
            {
                "id_node": 2,
                "is_inlet": False,
                "is_outlet": True,
                "sword_wse": 9.0,
                "sword_wse_field": "wse",
                "sword_wse_fill_method": "requested_field",
                "geometry": Point(100.0, 0.0),
            },
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )


def _simple_links() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        [
            {
                "id_link": 1,
                "id_us_node": 1,
                "id_ds_node": 2,
                "len": 100.0,
                "geometry": LineString([(0.0, 0.0), (100.0, 0.0)]),
            }
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )


def _write_reference_section_summary(tmp_path: Path) -> Path:
    experiment_dir = tmp_path / "experiment"
    reference_dir = experiment_dir / "reference_section" / "variant"
    directed_dir = reference_dir / "directed"
    rivgraph_dir = reference_dir / "rivgraph"
    directed_dir.mkdir(parents=True)
    rivgraph_dir.mkdir(parents=True)

    centerline_path = rivgraph_dir / "reference_centerline.gpkg"
    directed_nodes_path = directed_dir / "reference_directed_nodes.gpkg"
    sword_nodes_path = rivgraph_dir / "reference_SWORD_nodes.gpkg"
    base_nodes_path = experiment_dir / "states" / "S000_base" / "variant" / "directed" / "base_directed_nodes.gpkg"
    base_nodes_path.parent.mkdir(parents=True)

    centerline = gpd.GeoDataFrame(
        [{"geometry": LineString([(0.0, 0.0), (100.0, 0.0)])}],
        geometry="geometry",
        crs="EPSG:3857",
    )
    centerline.to_file(centerline_path, driver="GPKG")

    reference_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 100, "is_inlet": True, "is_outlet": False, "geometry": Point(0.0, 0.0)},
            {"id_node": 101, "is_inlet": False, "is_outlet": True, "geometry": Point(100.0, 0.0)},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    reference_nodes.to_file(directed_nodes_path, driver="GPKG")

    reference_sword_nodes = gpd.GeoDataFrame(
        [
            {"wse": 12.0, "wse_fill_method": "requested_field", "geometry": Point(0.0, 0.0)},
            {"wse": 10.5, "wse_fill_method": "requested_field", "geometry": Point(50.0, 0.0)},
            {"wse": 9.0, "wse_fill_method": "requested_field", "geometry": Point(100.0, 0.0)},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    reference_sword_nodes.to_file(sword_nodes_path, driver="GPKG")

    base_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 1, "is_inlet": True, "is_outlet": False, "geometry": Point(0.0, 0.0)},
            {"id_node": 2, "is_inlet": False, "is_outlet": True, "geometry": Point(100.0, 0.0)},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    base_nodes.to_file(base_nodes_path, driver="GPKG")

    pd.DataFrame(
        [
            {
                "state_id": "S000_base",
                "state_role": "base",
                "directed_nodes_path": str(base_nodes_path.resolve()),
            }
        ]
    ).to_csv(experiment_dir / "state_registry.csv", index=False)

    summary_path = experiment_dir / "reference_section" / "reference_section_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "paths": {
                    "centerline_path": str(centerline_path.resolve()),
                    "directed_nodes_path": str(directed_nodes_path.resolve()),
                    "reference_section_dir": str((experiment_dir / "reference_section").resolve()),
                }
            }
        ),
        encoding="utf-8",
    )
    return summary_path


def test_compute_hydraulic_context_builds_profile_and_link_slopes() -> None:
    result = compute_hydraulic_context(_simple_links(), _simple_nodes())

    assert list(result.node_context["rapid_wse_for_slope"]) == pytest.approx([10.0, 9.0])
    assert result.node_context["rapid_wse_source_method"].tolist() == [
        "profile_monotone_requested_field_nodes",
        "profile_monotone_requested_field_nodes",
    ]

    row = result.link_context.iloc[0]
    assert row["wse_us"] == pytest.approx(10.0)
    assert row["wse_ds"] == pytest.approx(9.0)
    assert row["slope_used"] == pytest.approx(0.01)
    assert row["slope_source_method"] == "raw_wse"
    assert row["bubble_slope_qc_reason"] == "unavailable_unit_summary"
    assert result.section_slope_summary["slope"] == pytest.approx(0.01)


def test_generate_state_hydraulic_context_writes_artifacts(tmp_path: Path) -> None:
    links_path = tmp_path / "links.gpkg"
    nodes_path = tmp_path / "nodes.gpkg"
    _simple_links().to_file(links_path, driver="GPKG")
    _simple_nodes().to_file(nodes_path, driver="GPKG")

    outputs = generate_state_hydraulic_context(
        directed_links_path=links_path,
        directed_nodes_path=nodes_path,
        hierarchy_output_dir=None,
        output_dir=tmp_path / "hydraulic_context",
        state_id="S000_base",
    )

    assert outputs.link_context_path.exists()
    assert outputs.node_context_path.exists()
    assert outputs.manifest_path.exists()
    assert outputs.manifest["state_id"] == "S000_base"
    assert outputs.manifest["head_concepts"]["normalized_solver_heads"] == "not written here"

    written_links = pd.read_csv(outputs.link_context_path)
    written_nodes = pd.read_csv(outputs.node_context_path)
    assert float(written_links.loc[0, "slope_used"]) == pytest.approx(0.01)
    assert "rapid_wse_for_slope" in written_nodes.columns


def test_compute_hydraulic_context_can_use_reference_section_profile(tmp_path: Path) -> None:
    summary_path = _write_reference_section_summary(tmp_path)
    nodes = _simple_nodes().copy()
    nodes["matched_parent_node_id"] = [1, 2]
    nodes["match_within_tolerance"] = [True, True]

    result = compute_hydraulic_context(
        _simple_links(),
        nodes,
        config=SlopeConfig(
            slope_profile_mode="reference_section",
            reference_section_summary_path=str(summary_path),
        ),
    )

    assert "node_reference_section_dist_out_m" in result.node_context.columns
    assert result.node_context["rapid_wse_source_method"].tolist() == [
        "reference_section_monotone_requested_field_sword_nodes",
        "reference_section_monotone_requested_field_sword_nodes",
    ]
    row = result.link_context.iloc[0]
    assert row["wse_us"] == pytest.approx(12.0)
    assert row["wse_ds"] == pytest.approx(9.0)
    assert row["slope_used"] == pytest.approx(0.03)
    assert result.section_slope_summary["profile_mode"] == "reference_section"


def test_reference_section_profile_avoids_statewise_corridor_anchor_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary_path = _write_reference_section_summary(tmp_path)
    nodes = _simple_nodes().copy()
    nodes["matched_parent_node_id"] = [1, 2]
    nodes["match_within_tolerance"] = [True, True]

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Statewise corridor SWORD anchor loading should not run in reference_section mode.")

    monkeypatch.setattr(
        "network_variants.hydraulic_context._load_reference_section_corridor_sword_anchors",
        fail_if_called,
    )

    result = compute_hydraulic_context(
        _simple_links(),
        nodes,
        config=SlopeConfig(
            slope_profile_mode="reference_section",
            reference_section_summary_path=str(summary_path),
        ),
    )

    row = result.link_context.iloc[0]
    assert row["wse_us"] == pytest.approx(12.0)
    assert row["wse_ds"] == pytest.approx(9.0)
    assert row["slope_used"] == pytest.approx(0.03)
