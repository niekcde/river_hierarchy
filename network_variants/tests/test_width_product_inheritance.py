from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from network_variants.width_product_inheritance import (
    inherit_unchanged_link_width_products,
)


def _links(records: list[dict[str, object]]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:28992")


def test_inherits_complete_width_product_and_samples_only_for_equal_geometry() -> None:
    parent_links = _links(
        [
            {
                "id_link": 10,
                "wid_adj": 12.0,
                "wid_adj_wet": 11.0,
                "wid_adj_routing": 9.5,
                "width_model_probability": 0.91,
                "width_correction_applied": True,
                "width_qc_reason": "model_correction",
                "width_trim_method": "native_rivgraph_fallback",
                "width_trim_start_sample_index": 0,
                "width_trim_stop_sample_index_exclusive": 2,
                "geometry": LineString([(0, 0), (1, 0)]),
            },
            {
                "id_link": 11,
                "wid_adj": 22.0,
                "wid_adj_wet": 21.0,
                "wid_adj_routing": 19.0,
                "width_model_probability": 0.82,
                "width_correction_applied": True,
                "width_qc_reason": "model_correction",
                "geometry": LineString([(2, 0), (4, 0)]),
            },
        ]
    )
    child_links = _links(
        [
            {
                "id_link": 100,
                "wid_adj": 999.0,
                "wid_adj_wet": 998.0,
                "geometry": LineString([(1, 0), (0, 0)]),
            },
            {
                "id_link": 101,
                "wid_adj": 32.0,
                "wid_adj_wet": 31.0,
                # It maps to one parent but is only a segment of that parent.
                "geometry": LineString([(2, 0), (3, 0)]),
            },
        ]
    )
    lineage = pd.DataFrame(
        {
            "id_link": [100, 101],
            "lineage_type": ["unchanged_1to1", "unchanged_1to1"],
            "dominant_parent_link_id": [10, 11],
        }
    )
    parent_samples = pd.DataFrame(
        {
            "id_link": [10, 10, 11],
            "sample_index": [0, 1, 0],
            "width_wet": [10.0, 12.0, 21.0],
            "variant_id": ["parent", "parent", "parent"],
            "used_in_selected_width": [True, True, False],
        }
    )
    child_samples = pd.DataFrame(
        {
            "id_link": [100, 101],
            "sample_index": [0, 0],
            "width_wet": [998.0, 31.0],
            "variant_id": ["child", "child"],
        }
    )

    links, samples, summary = inherit_unchanged_link_width_products(
        child_links=child_links,
        child_width_samples=child_samples,
        parent_links=parent_links,
        parent_width_samples=parent_samples,
        link_lineage=lineage,
    )

    inherited = links.set_index("id_link").loc[100]
    assert inherited["wid_adj"] == 12.0
    assert inherited["wid_adj_wet"] == 11.0
    assert inherited["wid_adj_routing"] == 9.5
    assert inherited["width_model_probability"] == 0.91
    assert inherited["width_trim_method"] == "native_rivgraph_fallback"
    assert bool(inherited["width_correction_applied"])
    assert inherited["width_inheritance_status"] == "inherited_unchanged_1to1"
    assert inherited["width_inherited_from_parent_link_id"] == 10

    changed = links.set_index("id_link").loc[101]
    assert changed["wid_adj"] == 32.0
    assert changed["wid_adj_wet"] == 31.0
    assert pd.isna(changed["wid_adj_routing"])
    assert pd.isna(changed["width_model_probability"])
    assert changed["width_inheritance_status"] == "changed_requires_evaluation"

    inherited_samples = samples.loc[samples["id_link"].eq(100)]
    assert inherited_samples["width_wet"].tolist() == [10.0, 12.0]
    assert inherited_samples["variant_id"].tolist() == ["parent", "parent"]
    assert inherited_samples["used_in_selected_width"].tolist() == [True, True]
    assert samples.loc[samples["id_link"].eq(101), "width_wet"].tolist() == [31.0]
    assert summary["inherited_link_ids"] == [100]
    assert summary["rejected_candidate_reasons"] == {"geometry_changed": 1}


def test_rejects_non_bijective_parent_mapping() -> None:
    parent_links = _links(
        [{"id_link": 10, "wid_adj_wet": 11.0, "geometry": LineString([(0, 0), (1, 0)])}]
    )
    child_links = _links(
        [
            {"id_link": 100, "wid_adj_wet": 20.0, "geometry": LineString([(0, 0), (1, 0)])},
            {"id_link": 101, "wid_adj_wet": 21.0, "geometry": LineString([(0, 0), (1, 0)])},
        ]
    )
    lineage = pd.DataFrame(
        {
            "id_link": [100, 101],
            "lineage_type": ["unchanged_1to1", "unchanged_1to1"],
            "dominant_parent_link_id": [10, 10],
        }
    )

    links, _, summary = inherit_unchanged_link_width_products(
        child_links=child_links,
        child_width_samples=pd.DataFrame(columns=["id_link", "sample_index"]),
        parent_links=parent_links,
        parent_width_samples=pd.DataFrame(columns=["id_link", "sample_index"]),
        link_lineage=lineage,
    )

    assert links["wid_adj_wet"].tolist() == [20.0, 21.0]
    assert links["width_inheritance_status"].eq("changed_requires_evaluation").all()
    assert summary["inherited_link_count"] == 0
    assert summary["rejected_candidate_reasons"] == {"parent_maps_to_multiple_children": 2}


def test_changed_link_clears_stale_routing_fields_but_keeps_measurement() -> None:
    parent_links = _links(
        [
            {
                "id_link": 10,
                "wid_adj_wet": 11.0,
                "wid_adj_routing": 9.0,
                "width_qc_reason": "parent",
                "geometry": LineString([(0, 0), (1, 0)]),
            }
        ]
    )
    child_links = _links(
        [
            {
                "id_link": 100,
                "wid_adj_wet": 15.0,
                "wid_adj_routing": 99.0,
                "width_qc_reason": "stale",
                "geometry": LineString([(0, 0), (2, 0)]),
            }
        ]
    )
    lineage = pd.DataFrame(
        {
            "id_link": [100],
            "lineage_type": ["collapsed_many_to_one"],
            "dominant_parent_link_id": [10],
        }
    )

    links, _, _ = inherit_unchanged_link_width_products(
        child_links=child_links,
        child_width_samples=pd.DataFrame(),
        parent_links=parent_links,
        parent_width_samples=pd.DataFrame(),
        link_lineage=lineage,
    )

    assert links.loc[0, "wid_adj_wet"] == 15.0
    assert np.isnan(links.loc[0, "wid_adj_routing"])
    assert links.loc[0, "width_qc_reason"] is None


def test_inherited_samples_are_copied_when_changed_sample_table_has_no_schema() -> None:
    parent_links = _links(
        [
            {
                "id_link": 10,
                "wid_adj_wet": 11.0,
                "geometry": LineString([(0, 0), (1, 0)]),
            }
        ]
    )
    child_links = _links(
        [
            {
                "id_link": 100,
                "geometry": LineString([(1, 0), (0, 0)]),
            }
        ]
    )
    lineage = pd.DataFrame(
        {
            "id_link": [100],
            "lineage_type": ["unchanged_1to1"],
            "dominant_parent_link_id": [10],
        }
    )
    parent_samples = pd.DataFrame(
        {
            "id_link": [10, 10],
            "sample_index": [0, 1],
            "width_wet": [10.0, 12.0],
            "x": [0.25, 0.75],
            "y": [0.0, 0.0],
        }
    )

    _links_out, samples, summary = inherit_unchanged_link_width_products(
        child_links=child_links,
        child_width_samples=pd.DataFrame(),
        parent_links=parent_links,
        parent_width_samples=parent_samples,
        link_lineage=lineage,
    )

    assert samples["id_link"].tolist() == [100, 100]
    assert samples["sample_index"].tolist() == [0, 1]
    assert samples["width_wet"].tolist() == [10.0, 12.0]
    assert summary["inherited_link_ids"] == [100]
