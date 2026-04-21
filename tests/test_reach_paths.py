import pandas as pd

from gauge_sword_match.reach_paths import build_downstream_adjacency, build_example_reach_paths, find_reaches_between


def test_find_reaches_between_keeps_all_bifurcation_branches():
    reaches = pd.DataFrame(
        {
            "reach_id": [1, 2, 3, 4, 5, 9],
            "rch_id_dn": ["2,3", "4", "4", "5", "", ""],
            "dist_out": [500, 400, 390, 300, 200, 50],
        }
    )

    selected = find_reaches_between(reaches, [1], [4])

    assert selected == {1, 2, 3, 4}


def test_find_reaches_between_auto_detects_sword_0_0_8_downstream_columns():
    reaches = pd.DataFrame(
        {
            "reach_id": [1, 2, 3, 4],
            "rch_id_dn_main": [2, 4, 4, None],
            "rch_id_dn_1": [3, None, None, None],
        }
    )

    selected = find_reaches_between(reaches, [1], [4])

    assert selected == {1, 2, 3, 4}


def test_build_example_reach_paths_unions_duplicate_example_rows():
    examples = pd.DataFrame(
        {
            "example_id": [10, 10],
            "station_key_up": ["A:up", "A:up"],
            "station_key_dn": ["A:dn1", "A:dn2"],
        }
    )
    station_matches = pd.DataFrame(
        {
            "station_key": ["A:up", "A:dn1", "A:dn2"],
            "reach_id": [100, 4, 5],
            "sword_region": ["na", "na", "na"],
            "confidence_class": ["high", "high", "high"],
        }
    )
    topologies = {
        "na": pd.DataFrame(
            {
                "reach_id": [100, 20, 30, 4, 5],
                "rch_id_dn": ["20,30", "4", "4", "5", ""],
                "dist_out": [500, 400, 390, 300, 200],
            }
        )
    }

    reach_paths, summary = build_example_reach_paths(
        examples,
        station_matches,
        topologies_by_region=topologies,
    )

    assert reach_paths["reach_id"].tolist() == [100, 20, 30, 4, 5]
    assert reach_paths["path_order"].tolist() == [0, 1, 2, 3, 4]
    assert len(summary) == 1
    assert bool(summary.loc[0, "route_found"]) is True
    assert bool(summary.loc[0, "all_downstream_reaches_reached"]) is True


def test_downstream_adjacency_parses_numpy_like_lists():
    reaches = pd.DataFrame(
        {
            "reach_id": [1, 2],
            "rch_id_dn": [[2, 3], None],
        }
    )

    assert build_downstream_adjacency(reaches) == {1: {2, 3}, 2: set(), 3: set()}
