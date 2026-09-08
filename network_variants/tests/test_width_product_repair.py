from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from network_variants.width_product_repair import (
    _difference_audit,
    _validate_repaired_products,
)


def _links(records: list[dict[str, object]]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:28992")


def test_repair_audit_preserves_topology_and_detects_width_change() -> None:
    source = _links(
        [{
            "id_link": 1,
            "wid_adj_wet": 100.0,
            "wid_adj_total": 110.0,
            "wid_adj_routing": 90.0,
            "width_trim_method": "untrimmed_no_valid_core",
            "width_trim_valid_core": False,
            "geometry": LineString([(0, 0), (1, 0)]),
        }]
    )
    repaired = source.copy()
    repaired.loc[0, "wid_adj_wet"] = 75.0
    repaired.loc[0, "wid_adj_total"] = 85.0
    repaired.loc[0, "wid_adj_routing"] = 75.0
    repaired.loc[0, "width_trim_method"] = "native_rivgraph_fallback"
    repaired.loc[0, "width_trim_valid_core"] = True
    samples = pd.DataFrame({"id_link": [1, 1], "sample_index": [0, 1]})

    differences = _difference_audit(source, repaired)
    validation = _validate_repaired_products(
        source_links=source,
        repaired_links=repaired,
        repaired_samples=samples,
    )

    assert bool(differences.loc[0, "material_change"])
    assert differences.loc[0, "wid_adj_wet_relative_change"] == -0.25
    assert validation["is_valid"] is True
    assert all(validation["checks"].values())


def test_repair_validation_rejects_geometry_or_unknown_sample_changes() -> None:
    source = _links(
        [{
            "id_link": 1,
            "wid_adj_wet": 100.0,
            "wid_adj_total": 110.0,
            "wid_adj_routing": 90.0,
            "width_trim_method": "two_bank_valid_core",
            "geometry": LineString([(0, 0), (1, 0)]),
        }]
    )
    repaired = source.copy()
    repaired.loc[0, "geometry"] = LineString([(0, 0), (2, 0)])
    samples = pd.DataFrame({"id_link": [999], "sample_index": [0]})

    validation = _validate_repaired_products(
        source_links=source,
        repaired_links=repaired,
        repaired_samples=samples,
    )

    assert validation["is_valid"] is False
    assert validation["checks"]["link_geometries_preserved"] is False
    assert validation["checks"]["samples_reference_known_links"] is False
