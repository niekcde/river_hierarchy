from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from network_variants.hydraulic_context import (
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
