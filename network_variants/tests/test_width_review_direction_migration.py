from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString

from network_variants.width_review_direction_migration import (
    build_migrated_decision_seed,
    derive_direction_affected_context,
)


def _links(*, reverse_link_2: bool) -> gpd.GeoDataFrame:
    endpoints = [(1, 2), (2, 3), (2, 4), (3, 5)]
    geometries = [
        LineString([(0, 0), (1, 0)]),
        LineString([(1, 0), (2, 0)]),
        LineString([(1, 0), (1, 1)]),
        LineString([(2, 0), (3, 0)]),
    ]
    if reverse_link_2:
        endpoints[1] = (3, 2)
        geometries[1] = LineString(list(geometries[1].coords)[::-1])
    return gpd.GeoDataFrame(
        {
            "id_link": [1, 2, 3, 4],
            "id_us_node": [pair[0] for pair in endpoints],
            "id_ds_node": [pair[1] for pair in endpoints],
        },
        geometry=geometries,
        crs="EPSG:28992",
    )


def test_direction_impact_includes_both_junctions_and_only_candidate_incident_links() -> None:
    impact = derive_direction_affected_context(
        _links(reverse_link_2=False),
        _links(reverse_link_2=True),
        changed_link_ids=[2],
        candidate_link_ids=[1, 2, 4],
    )

    assert impact == {
        "changed_link_ids": [2],
        "affected_node_ids": [2, 3],
        "incident_link_ids": [1, 2, 3, 4],
        "affected_candidate_link_ids": [1, 2, 4],
    }


def test_direction_migration_rejects_unrequested_topology_change() -> None:
    current = _links(reverse_link_2=True)
    current.loc[current["id_link"] == 4, "id_ds_node"] = 6

    with pytest.raises(ValueError, match="do not exactly match"):
        derive_direction_affected_context(
            _links(reverse_link_2=False),
            current,
            changed_link_ids=[2],
            candidate_link_ids=[1, 2, 4],
        )


def test_migrated_seed_clears_affected_and_carries_only_classifications() -> None:
    old = pd.DataFrame(
        {
            "id_link": [1, 2, 4],
            "review_decision": ["good", "bad", "good"],
            "old_numeric_correction": [10.0, 20.0, 30.0],
        }
    )

    seed = build_migrated_decision_seed(
        old,
        candidate_link_ids=[1, 2, 4],
        affected_candidate_link_ids=[2, 4],
    )

    assert list(seed.columns) == ["id_link", "review_decision"]
    assert seed.loc[seed["id_link"] == 1, "review_decision"].iloc[0] == "good"
    assert seed.loc[seed["id_link"].isin([2, 4]), "review_decision"].isna().all()


def test_migrated_seed_rejects_changed_candidate_membership() -> None:
    old = pd.DataFrame({"id_link": [1], "review_decision": ["good"]})
    with pytest.raises(ValueError, match="candidate membership"):
        build_migrated_decision_seed(
            old,
            candidate_link_ids=[1, 2],
            affected_candidate_link_ids=[2],
        )
