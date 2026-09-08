from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from network_variants.routing_width_application import DEFAULT_ENDPOINT_WINDOW
from network_variants.width_junction_audit import calculate_endpoint_widths
from network_variants.width_routing_solver import prepare_hybrid_solver_links
from network_variants.width_trim_selection import (
    NO_VALID_CORE_METHOD,
    SELECTED_SAMPLE_FIELD,
    apply_authoritative_width_trim_selection,
    build_parent_child_width_diagnostics,
    resolve_closed_replacement_exclusions,
)


def _samples(*, invalid_two_bank: bool = True) -> pd.DataFrame:
    x = np.linspace(0.0, 1000.0, 101)
    total = np.full(101, 80.0)
    wet = np.full(101, 70.0)
    if invalid_two_bank:
        total[[0, -1]] = 2500.0
        wet[[0, -1]] = 2400.0
    return pd.DataFrame(
        {
            "id_link": 7,
            "sample_index": np.arange(101),
            "x": x,
            "y": 0.0,
            "width_total": total,
            "width_wet": wet,
            "width_dry": total - wet,
            "n_channels": 1,
            "n_wet_threads": 1,
            "qc_opposition_cos": -1.0,
            "width_invalid": False,
        }
    )


def test_native_fallback_controls_summary_and_endpoint_support() -> None:
    native = np.full(101, 80.0)
    native[[0, -1]] = 100.0
    links = gpd.GeoDataFrame(
        {
            "id_link": [7],
            "id_us_node": [1],
            "id_ds_node": [2],
            "len": [1000.0],
            "wid_pix": [",".join(str(value) for value in native)],
            "wid_adj_total": [500.0],
            "wid_adj_wet": [450.0],
        },
        geometry=[LineString([(0.0, 0.0), (1000.0, 0.0)])],
    )
    enriched, marked, audit = apply_authoritative_width_trim_selection(links, _samples())

    assert audit.loc[0, "width_trim_method"] == "native_rivgraph_fallback"
    assert bool(audit.loc[0, "width_trim_valid_core"])
    assert 0 < int(marked[SELECTED_SAMPLE_FIELD].sum()) < len(marked)
    assert np.isclose(float(enriched.loc[0, "wid_adj_wet"]), 70.0)
    assert np.isclose(float(enriched.loc[0, "wid_adj_total"]), 80.0)

    nodes = gpd.GeoDataFrame(
        {"id_node": [1, 2]}, geometry=[Point(0.0, 0.0), Point(1000.0, 0.0)]
    )
    endpoints = calculate_endpoint_widths(
        example_id="01",
        families=enriched,
        samples=marked,
        directed_links=enriched,
        directed_nodes=nodes,
        windows=[DEFAULT_ENDPOINT_WINDOW],
        whole_width_field="wid_adj_wet",
    )
    assert endpoints.loc[0, "width_trim_method"] == "native_rivgraph_fallback"
    assert np.isclose(float(endpoints.loc[0, "width_near_us_node_wet"]), 70.0)
    assert np.isclose(float(endpoints.loc[0, "width_near_ds_node_wet"]), 70.0)
    assert float(endpoints.loc[0, "width_near_us_node_wet"]) != 2400.0


def test_no_valid_core_retains_complete_legacy_support() -> None:
    links = gpd.GeoDataFrame(
        {
            "id_link": [7],
            "id_us_node": [1],
            "id_ds_node": [2],
            "len": [1000.0],
            "wid_pix": ["1,2"],
            "wid_adj_total": [500.0],
            "wid_adj_wet": [450.0],
        },
        geometry=[LineString([(0.0, 0.0), (1000.0, 0.0)])],
    )
    enriched, marked, _ = apply_authoritative_width_trim_selection(links, _samples())
    assert enriched.loc[0, "width_trim_method"] == NO_VALID_CORE_METHOD
    assert np.isclose(float(enriched.loc[0, "wid_adj_total"]), 500.0)
    assert np.isclose(float(enriched.loc[0, "wid_adj_wet"]), 450.0)
    assert bool(marked[SELECTED_SAMPLE_FIELD].all())
    nodes = gpd.GeoDataFrame(
        {"id_node": [1, 2]}, geometry=[Point(0.0, 0.0), Point(1000.0, 0.0)]
    )
    endpoints = calculate_endpoint_widths(
        example_id="01",
        families=enriched,
        samples=marked,
        directed_links=enriched,
        directed_nodes=nodes,
        windows=[DEFAULT_ENDPOINT_WINDOW],
        whole_width_field="wid_adj_wet",
    )
    solver = prepare_hybrid_solver_links(endpoints)
    assert np.isfinite(float(solver.loc[0, "routing_context_width_us"]))
    assert np.isfinite(float(solver.loc[0, "routing_context_width_ds"]))


def test_native_fallback_rejects_an_unstable_short_candidate() -> None:
    native = np.full(101, 80.0)
    native[0] = 1000.0 / 1.1
    native[-1] = 980.0 / 1.1
    links = gpd.GeoDataFrame(
        {
            "id_link": [7],
            "id_us_node": [1],
            "id_ds_node": [2],
            "len": [1000.0],
            # These endpoints produce a one-sample native candidate.
            "wid_pix": [",".join(str(value) for value in native)],
            "wid_adj_total": [500.0],
            "wid_adj_wet": [450.0],
        },
        geometry=[LineString([(0.0, 0.0), (1000.0, 0.0)])],
    )
    enriched, marked, audit = apply_authoritative_width_trim_selection(links, _samples())
    assert enriched.loc[0, "width_trim_method"] == NO_VALID_CORE_METHOD
    assert bool(marked[SELECTED_SAMPLE_FIELD].all())
    assert "insufficient_support" in audit.loc[0, "width_trim_fallback_reason"]
    assert int(audit.loc[0, "width_trim_native_candidate_sample_count"]) < 6


def test_parent_width_intervention_reenables_collapsed_replacement_review() -> None:
    children = pd.DataFrame(
        {
            "id_link": [544, 600],
            "lineage_type": ["collapsed_many_to_one", "collapsed_many_to_one"],
            "matched_parent_link_ids": ["297,301", "400,401"],
        }
    )
    parents = pd.DataFrame(
        {
            "id_link": [297, 301, 400, 401],
            "wid_adj_wet": [76.5, 931.8, 100.0, 120.0],
            "wid_adj_wet_two_bank_original": [428.9, 931.8, 100.0, 120.0],
            "width_trim_method": [
                "native_rivgraph_fallback",
                "untrimmed_no_valid_core",
                "two_bank_interior",
                "two_bank_interior",
            ],
            "width_correction_applied": [False, True, False, False],
        }
    )
    excluded, audit = resolve_closed_replacement_exclusions(
        children, parents, [544, 600]
    )
    assert excluded == [600]
    by_id = audit.set_index("id_link")
    assert bool(by_id.loc[544, "parent_width_intervention"])
    assert not bool(by_id.loc[544, "closed_replacement_excluded"])
    assert bool(by_id.loc[600, "closed_replacement_excluded"])


def test_parent_child_width_sum_is_diagnostic_only() -> None:
    children = pd.DataFrame(
        {
            "id_link": [544, 700],
            "lineage_type": ["collapsed_many_to_one", "unchanged_1to1"],
            "matched_parent_link_ids": ["297,301", "400"],
            "wid_adj_routing": [477.8, 120.0],
        }
    )
    parents = pd.DataFrame(
        {
            "id_link": [297, 301, 400],
            "wid_adj_routing": [76.5, 420.9, 120.0],
        }
    )
    audit = build_parent_child_width_diagnostics(children, parents)
    assert audit["id_link"].tolist() == [544]
    assert bool(audit.loc[0, "parent_child_width_diagnostic_only"])
    assert np.isclose(float(audit.loc[0, "parent_routing_width_sum"]), 497.4)
    assert np.isclose(float(audit.loc[0, "parent_child_routing_width_difference"]), -19.6)
