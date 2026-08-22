from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point

from hierarchy_level_definition.unit_detection.bifurcation_confluence_units import (
    build_graph,
    build_structural_units,
)


def _directed_frames() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    nodes = gpd.GeoDataFrame(
        {"id_node": [0, 1, 2, 3, 4, 5]},
        geometry=[Point(x, 0) for x in range(6)],
        crs="EPSG:32631",
    )
    endpoints = [(0, 1), (1, 2), (2, 4), (1, 3), (3, 4), (4, 5)]
    links = gpd.GeoDataFrame(
        {
            "id_link": list(range(len(endpoints))),
            "id_nodes": [
                "1, 0",  # Deliberately opposite to the validated direction.
                "1, 2",
                "2, 4",
                "1, 3",
                "3, 4",
                "4, 5",
            ],
            "id_us_node": [item[0] for item in endpoints],
            "id_ds_node": [item[1] for item in endpoints],
            "len": [1.0] * len(endpoints),
        },
        geometry=[LineString([nodes.geometry.iloc[u], nodes.geometry.iloc[v]]) for u, v in endpoints],
        crs=nodes.crs,
    )
    return links, nodes


def test_build_graph_uses_validated_direction_instead_of_id_nodes_order() -> None:
    links, nodes = _directed_frames()

    units = build_structural_units(build_graph(links, nodes))

    assert [(unit.bifurcation, unit.confluence) for unit in units] == [(1, 4)]
    assert units[0].n_paths == 2


def test_build_graph_rejects_direction_fields_with_different_endpoints() -> None:
    links, nodes = _directed_frames()
    links.loc[0, "id_nodes"] = "1, 99"

    with pytest.raises(ValueError, match="inconsistent endpoints"):
        build_graph(links, nodes)
