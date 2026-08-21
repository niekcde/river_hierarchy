from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point

from hierarchy_level_definition.unit_detection.bifurcation_confluence_units import (
    StructuralUnit,
)
from network_variants.local_patch import LocalPatchCollar, LocalPatchContract, LocalPatchRasterPolicy
from network_variants.local_patch_anchored import (
    _canonicalize_unchanged_parent_links,
    anchor_local_patch_realized_graph,
)
from network_variants.local_patch_merge import (
    _boundary_bindings,
    _force_boundary_node_matches,
    _orient_inserted_components_from_parent_order,
    _expected_cycle_reduction,
    _prune_raw_local_to_cut_corridor,
    _prune_directed_to_anchored_interval,
    _remove_post_direction_reverse_parent_duplicates,
    _snap_and_orient_geometry,
    _trim_to_anchored_boundaries,
    _undirected_cycle_rank,
    merge_local_patch_variant,
)
from network_variants.local_patch_products import (
    _local_link_lineage,
    _strict_width_measurement_link_ids,
    materialize_local_patch_width_length_products,
)
from network_variants.local_patch_realized import (
    _dissolve_computational_cut_interfaces,
    assemble_local_patch_realized_graph,
)
from network_variants.local_patch_raster import build_local_patch_base_raster
from network_variants.local_patch_variant import LocalPatchVariantResult
from network_variants.variant_generation import derive_anchored_graph_from_realized


def test_same_endpoint_projection_uses_only_bounded_exact_parent_connector() -> None:
    raw = LineString([(0, 0), (10, 0)])
    fallback = LineString([(20, 0), (25, 1), (30, 0)])

    reconstructed = _snap_and_orient_geometry(
        raw,
        Point(0, 0),
        Point(10, 0),
        Point(20, 0),
        Point(30, 0),
        same_endpoint_fallback_geometry=fallback,
        same_endpoint_reconstruction_limit_m=20.0,
    )

    assert list(reconstructed.coords) == list(fallback.coords)
    with pytest.raises(ValueError, match="indistinguishable"):
        _snap_and_orient_geometry(
            raw,
            Point(0, 0),
            Point(10, 0),
            Point(20, 0),
            Point(30, 0),
            same_endpoint_fallback_geometry=fallback,
            same_endpoint_reconstruction_limit_m=5.0,
        )


def test_inserted_orientation_retries_with_fixed_parent_node_order_only_after_cycle() -> None:
    nodes = gpd.GeoDataFrame(
        [
            {"id_node": 0, "geometry": Point(0, 0)},
            {"id_node": 1, "geometry": Point(10, 3)},
            {"id_node": 2, "geometry": Point(10, -3)},
            {"id_node": 3, "geometry": Point(20, 0)},
        ],
        crs="EPSG:3857",
    )
    rows = [
        (10, 1, 2, [(10, 3), (10, -3)]),
        (11, 0, 1, [(0, 0), (-50, 0), (10, 3)]),
        (12, 0, 2, [(0, 0), (10, -3)]),
        (13, 1, 3, [(10, 3), (20, 0)]),
        (14, 2, 3, [(10, -3), (70, 0), (20, 0)]),
        (15, 1, 2, [(10, 3), (10, -3)]),
    ]
    links = gpd.GeoDataFrame(
        [
            {
                "id_link": link_id,
                "id_us_node": upstream,
                "id_ds_node": downstream,
                "geometry": LineString(coordinates),
            }
            for link_id, upstream, downstream, coordinates in rows
        ],
        crs="EPSG:3857",
    )

    oriented, summary = _orient_inserted_components_from_parent_order(
        links,
        nodes,
        inserted_link_ids={11, 12, 13, 14, 15},
        fixed_scaffold_link_ids={10},
        parent_node_order={0: 0, 1: 1, 2: 2, 3: 3},
        protected_source_node_ids={0},
        protected_sink_node_ids={3},
    )

    oriented = oriented.set_index("id_link")
    assert (int(oriented.loc[10, "id_us_node"]), int(oriented.loc[10, "id_ds_node"])) == (1, 2)
    assert (int(oriented.loc[15, "id_us_node"]), int(oriented.loc[15, "id_ds_node"])) == (1, 2)
    assert summary["parent_order_retry_component_count"] == 1


def _raster(path: Path, values: np.ndarray, dtype: str) -> Path:
    with rasterio.open(
        path, "w", driver="GTiff", height=values.shape[0], width=values.shape[1],
        count=1, dtype=dtype, crs="EPSG:3857", transform=from_origin(0, 100, 1, 1), nodata=0,
    ) as destination:
        destination.write(values.astype(dtype), 1)
    return path


def test_parent_canonicalization_keeps_nodes_and_incident_links_geometrically_connected() -> None:
    parent_links = gpd.GeoDataFrame(
        [
            {
                "id_link": 10,
                "id_us_node": 1,
                "id_ds_node": 2,
                "geometry": LineString([(0, 0), (10, 0)]),
            }
        ],
        crs="EPSG:3857",
    )
    parent_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 1, "geometry": Point(0, 0)},
            {"id_node": 2, "geometry": Point(10, 0)},
        ],
        crs="EPSG:3857",
    )
    child_links = gpd.GeoDataFrame(
        [
            {
                "id_link": 20,
                "id_us_node": 101,
                "id_ds_node": 102,
                "geometry": LineString([(0.5, 0), (9.5, 0)]),
            },
            {
                "id_link": 21,
                "id_us_node": 102,
                "id_ds_node": 103,
                "geometry": LineString([(9.5, 0), (20, 0)]),
            },
        ],
        crs="EPSG:3857",
    )
    child_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 101, "geometry": Point(0.5, 0)},
            {"id_node": 102, "geometry": Point(9.5, 0)},
            {"id_node": 103, "geometry": Point(20, 0)},
        ],
        crs="EPSG:3857",
    )
    lineage = pd.DataFrame(
        [
            {
                "id_link": 20,
                "lineage_type": "unchanged_1to1",
                "dominant_parent_link_id": 10,
            },
            {
                "id_link": 21,
                "lineage_type": "collapsed_many_to_one",
                "dominant_parent_link_id": pd.NA,
            },
        ]
    )

    links, nodes, summary = _canonicalize_unchanged_parent_links(
        anchored_links=child_links,
        anchored_nodes=child_nodes,
        anchored_lineage=lineage,
        parent_links=parent_links,
        parent_nodes=parent_nodes,
        source_id=101,
        sink_id=103,
        match_tolerance_m=1.0,
    )

    node_by_id = nodes.set_index("id_node")
    link_by_id = links.set_index("id_link")
    assert node_by_id.loc[101].geometry.equals(Point(0, 0))
    assert node_by_id.loc[102].geometry.equals(Point(10, 0))
    assert link_by_id.loc[20].geometry.equals_exact(parent_links.iloc[0].geometry, 0.0)
    assert Point(link_by_id.loc[21].geometry.coords[0]).equals(Point(10, 0))
    assert summary["canonicalized_node_count"] == 2
    assert summary["maximum_node_move_m"] == pytest.approx(0.5)


def test_parent_canonicalization_preserves_post_anchor_match_without_unchanged_provenance() -> None:
    parent_links = gpd.GeoDataFrame(
        [
            {
                "id_link": 10,
                "id_us_node": 1,
                "id_ds_node": 2,
                "geometry": LineString([(0, 0), (10, 0)]),
            }
        ],
        crs="EPSG:3857",
    )
    parent_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 1, "geometry": Point(0, 0)},
            {"id_node": 2, "geometry": Point(10, 0)},
        ],
        crs="EPSG:3857",
    )
    raw_collapsed_geometry = LineString([(0, 0), (5, 0.5), (10, 0)])
    child_links = gpd.GeoDataFrame(
        [
            {
                "id_link": 20,
                "id_us_node": 101,
                "id_ds_node": 102,
                "geometry": raw_collapsed_geometry,
            }
        ],
        crs="EPSG:3857",
    )
    child_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 101, "geometry": Point(0, 0)},
            {"id_node": 102, "geometry": Point(10, 0)},
        ],
        crs="EPSG:3857",
    )
    # The post-anchor matcher sees a strict geometric match, but the empty
    # eligibility set says that this segment descended from a changed realized
    # link and therefore must retain its raw collapsed geometry.
    lineage = pd.DataFrame(
        [
            {
                "id_link": 20,
                "lineage_type": "unchanged_1to1",
                "dominant_parent_link_id": 10,
            }
        ]
    )

    links, nodes, summary = _canonicalize_unchanged_parent_links(
        anchored_links=child_links,
        anchored_nodes=child_nodes,
        anchored_lineage=lineage,
        parent_links=parent_links,
        parent_nodes=parent_nodes,
        source_id=101,
        sink_id=102,
        match_tolerance_m=1.0,
        canonicalizable_child_link_ids=set(),
    )

    assert links.iloc[0].geometry.equals_exact(raw_collapsed_geometry, 0.0)
    assert nodes.set_index("id_node").loc[101].geometry.equals(Point(0, 0))
    assert nodes.set_index("id_node").loc[102].geometry.equals(Point(10, 0))
    assert summary["canonicalized_link_count"] == 0
    assert summary["post_anchor_matches_excluded_by_provenance_count"] == 1


def test_parent_canonicalization_rejects_changed_parent_identity_after_segmentation() -> None:
    parent_links = gpd.GeoDataFrame(
        [
            {
                "id_link": 10,
                "id_us_node": 1,
                "id_ds_node": 2,
                "geometry": LineString([(0, 0), (10, 0)]),
            },
            {
                "id_link": 11,
                "id_us_node": 2,
                "id_ds_node": 3,
                "geometry": LineString([(10, 0), (20, 0)]),
            },
        ],
        crs="EPSG:3857",
    )
    parent_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 1, "geometry": Point(0, 0)},
            {"id_node": 2, "geometry": Point(10, 0)},
            {"id_node": 3, "geometry": Point(20, 0)},
        ],
        crs="EPSG:3857",
    )
    changed_segment = LineString([(10, 0), (15, 1), (20, 0)])
    child_links = gpd.GeoDataFrame(
        [
            {
                "id_link": 20,
                "id_us_node": 101,
                "id_ds_node": 102,
                "geometry": changed_segment,
            }
        ],
        crs="EPSG:3857",
    )
    child_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 101, "geometry": Point(10, 0)},
            {"id_node": 102, "geometry": Point(20, 0)},
        ],
        crs="EPSG:3857",
    )
    lineage = pd.DataFrame(
        [
            {
                "id_link": 20,
                "lineage_type": "unchanged_1to1",
                "dominant_parent_link_id": 11,
            }
        ]
    )

    links, _nodes, summary = _canonicalize_unchanged_parent_links(
        anchored_links=child_links,
        anchored_nodes=child_nodes,
        anchored_lineage=lineage,
        parent_links=parent_links,
        parent_nodes=parent_nodes,
        source_id=101,
        sink_id=102,
        match_tolerance_m=1.0,
        canonicalizable_child_link_ids={20},
        canonicalizable_parent_by_child_link_id={20: 10},
    )

    assert links.iloc[0].geometry.equals_exact(changed_segment, 0.0)
    assert summary["canonicalized_link_count"] == 0
    assert summary["post_anchor_matches_excluded_by_provenance"] == [
        {
            "child_id_link": 20,
            "parent_id_link": 11,
            "reason": "pre_anchor_parent_identity_changed",
            "pre_anchor_parent_id_link": 10,
        }
    ]


def test_parent_canonicalization_reorients_connector_and_transfers_collapsed_continuation() -> None:
    parent_links = gpd.GeoDataFrame(
        [
            {
                "id_link": 10,
                "id_us_node": 1,
                "id_ds_node": 2,
                "geometry": LineString([(0, 0), (10, 0)]),
            }
        ],
        crs="EPSG:3857",
    )
    parent_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 1, "geometry": Point(0, 0)},
            {"id_node": 2, "geometry": Point(10, 0)},
        ],
        crs="EPSG:3857",
    )
    child_links = gpd.GeoDataFrame(
        [
            {
                "id_link": 20,
                "id_us_node": 102,
                "id_ds_node": 101,
                "id_nodes": "102, 101",
                "geometry": LineString([(10, 0), (0, 0)]),
            },
            {
                "id_link": 21,
                "id_us_node": 101,
                "id_ds_node": 103,
                "id_nodes": "101, 103",
                "geometry": LineString([(0, 0), (10, 0), (20, 0)]),
            },
        ],
        crs="EPSG:3857",
    )
    child_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 101, "geometry": Point(0, 0)},
            {"id_node": 102, "geometry": Point(10, 0)},
            {"id_node": 103, "geometry": Point(20, 0)},
        ],
        crs="EPSG:3857",
    )
    lineage = pd.DataFrame(
        [
            {
                "id_link": 20,
                "lineage_type": "unchanged_1to1",
                "dominant_parent_link_id": 10,
            },
            {
                "id_link": 21,
                "lineage_type": "resolved_1to1",
                "dominant_parent_link_id": 10,
            },
        ]
    )

    links, nodes, summary = _canonicalize_unchanged_parent_links(
        anchored_links=child_links,
        anchored_nodes=child_nodes,
        anchored_lineage=lineage,
        parent_links=parent_links,
        parent_nodes=parent_nodes,
        source_id=101,
        sink_id=103,
        match_tolerance_m=2.0,
        canonicalizable_child_link_ids=set(),
    )

    link_by_id = links.set_index("id_link")
    assert (int(link_by_id.loc[20, "id_us_node"]), int(link_by_id.loc[20, "id_ds_node"])) == (101, 102)
    assert (int(link_by_id.loc[21, "id_us_node"]), int(link_by_id.loc[21, "id_ds_node"])) == (102, 103)
    assert link_by_id.loc[20].geometry.equals_exact(LineString([(0, 0), (10, 0)]), 0.0)
    assert link_by_id.loc[21].geometry.equals_exact(LineString([(10, 0), (20, 0)]), 0.0)
    assert summary["parent_direction_correction_count"] == 1
    correction = summary["compressed_interface_continuation_corrections"][0]
    assert correction["continuation_child_id_link"] == 21
    assert correction["projected_trim_distance_m"] == pytest.approx(10.0)
    assert correction["geometry_length_change_m"] == pytest.approx(-10.0)
    assert correction["parent_connector_length_m"] == pytest.approx(10.0)
    assert correction["distance_limit_m"] == pytest.approx(10.0)


def test_parent_canonicalization_keeps_raw_reversed_segment_without_bounded_companion() -> None:
    parent_links = gpd.GeoDataFrame(
        [
            {
                "id_link": 10,
                "id_us_node": 1,
                "id_ds_node": 2,
                "geometry": LineString([(0, 0), (10, 0)]),
            }
        ],
        crs="EPSG:3857",
    )
    parent_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 1, "geometry": Point(0, 0)},
            {"id_node": 2, "geometry": Point(10, 0)},
        ],
        crs="EPSG:3857",
    )
    raw_geometry = LineString([(10, 0), (6, 1), (0, 0)])
    child_links = gpd.GeoDataFrame(
        [
            {
                "id_link": 20,
                "id_us_node": 102,
                "id_ds_node": 101,
                "geometry": raw_geometry,
            }
        ],
        crs="EPSG:3857",
    )
    child_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 101, "geometry": Point(0, 0)},
            {"id_node": 102, "geometry": Point(10, 0)},
        ],
        crs="EPSG:3857",
    )
    lineage = pd.DataFrame(
        [
            {
                "id_link": 20,
                "lineage_type": "unchanged_1to1",
                "dominant_parent_link_id": 10,
            }
        ]
    )

    links, _nodes, summary = _canonicalize_unchanged_parent_links(
        anchored_links=child_links,
        anchored_nodes=child_nodes,
        anchored_lineage=lineage,
        parent_links=parent_links,
        parent_nodes=parent_nodes,
        source_id=102,
        sink_id=101,
        match_tolerance_m=2.0,
        canonicalizable_child_link_ids=set(),
    )

    assert links.iloc[0].geometry.equals_exact(raw_geometry, 0.0)
    assert summary["canonicalized_link_count"] == 0
    excluded = summary["post_anchor_matches_excluded_by_provenance"]
    assert excluded[0]["reason"] == "bounded_interface_repair_not_applicable"
    assert excluded[0]["repair_rejection_reason"].startswith(
        "requires_exactly_one_non_strict_continuation"
    )


def test_parent_canonicalization_rejects_unbounded_connector_correction() -> None:
    parent_links = gpd.GeoDataFrame(
        [
            {
                "id_link": 10,
                "id_us_node": 1,
                "id_ds_node": 2,
                "geometry": LineString([(0, 0), (20, 0)]),
            }
        ],
        crs="EPSG:3857",
    )
    parent_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 1, "geometry": Point(0, 0)},
            {"id_node": 2, "geometry": Point(20, 0)},
        ],
        crs="EPSG:3857",
    )
    child_links = gpd.GeoDataFrame(
        [
            {
                "id_link": 20,
                "id_us_node": 102,
                "id_ds_node": 101,
                "geometry": LineString([(20, 0), (0, 0)]),
            },
            {
                "id_link": 21,
                "id_us_node": 101,
                "id_ds_node": 103,
                "geometry": LineString([(0, 0), (20, 0), (40, 0)]),
            },
        ],
        crs="EPSG:3857",
    )
    child_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 101, "geometry": Point(0, 0)},
            {"id_node": 102, "geometry": Point(20, 0)},
            {"id_node": 103, "geometry": Point(40, 0)},
        ],
        crs="EPSG:3857",
    )
    lineage = pd.DataFrame(
        [
            {
                "id_link": 20,
                "lineage_type": "unchanged_1to1",
                "dominant_parent_link_id": 10,
            },
            {
                "id_link": 21,
                "lineage_type": "resolved_1to1",
                "dominant_parent_link_id": 10,
            },
        ]
    )

    with pytest.raises(ValueError, match="exceeds the bounded compressed-interface"):
        _canonicalize_unchanged_parent_links(
            anchored_links=child_links,
            anchored_nodes=child_nodes,
            anchored_lineage=lineage,
            parent_links=parent_links,
            parent_nodes=parent_nodes,
            source_id=101,
            sink_id=103,
            match_tolerance_m=2.0,
        )


def test_parent_canonicalization_rejects_unbounded_continuation_trim() -> None:
    parent_links = gpd.GeoDataFrame(
        [
            {
                "id_link": 10,
                "id_us_node": 1,
                "id_ds_node": 2,
                "geometry": LineString([(0, 0), (10, 0)]),
            }
        ],
        crs="EPSG:3857",
    )
    parent_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 1, "geometry": Point(0, 0)},
            {"id_node": 2, "geometry": Point(10, 0)},
        ],
        crs="EPSG:3857",
    )
    child_links = gpd.GeoDataFrame(
        [
            {
                "id_link": 20,
                "id_us_node": 102,
                "id_ds_node": 101,
                "geometry": LineString([(10, 0), (0, 0)]),
            },
            {
                "id_link": 21,
                "id_us_node": 101,
                "id_ds_node": 103,
                "geometry": LineString([(0, 0), (0, 20), (10, 0), (30, 0)]),
            },
        ],
        crs="EPSG:3857",
    )
    child_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 101, "geometry": Point(0, 0)},
            {"id_node": 102, "geometry": Point(10, 0)},
            {"id_node": 103, "geometry": Point(30, 0)},
        ],
        crs="EPSG:3857",
    )
    lineage = pd.DataFrame(
        [
            {
                "id_link": 20,
                "lineage_type": "unchanged_1to1",
                "dominant_parent_link_id": 10,
            },
            {
                "id_link": 21,
                "lineage_type": "resolved_1to1",
                "dominant_parent_link_id": 10,
            },
        ]
    )

    with pytest.raises(ValueError, match="requires an unbounded compressed-interface"):
        _canonicalize_unchanged_parent_links(
            anchored_links=child_links,
            anchored_nodes=child_nodes,
            anchored_lineage=lineage,
            parent_links=parent_links,
            parent_nodes=parent_nodes,
            source_id=101,
            sink_id=103,
            match_tolerance_m=2.0,
        )


def test_computational_cut_interfaces_are_dissolved_before_shared_anchoring() -> None:
    nodes = gpd.GeoDataFrame(
        [
            {"id_node": 1, "geometry": Point(0, 0)},
            {"id_node": 10, "geometry": Point(1, 0)},
            {"id_node": 11, "geometry": Point(3, 0)},
            {"id_node": 4, "geometry": Point(4, 0)},
        ],
        crs="EPSG:3857",
    )
    links = gpd.GeoDataFrame(
        [
            {
                "id_link": 101,
                "id_us_node": 1,
                "id_ds_node": 10,
                "id_nodes": "1, 10",
                "geometry": LineString([(0, 0), (1, 0)]),
            },
            {
                "id_link": 200,
                "id_us_node": 10,
                "id_ds_node": 11,
                "id_nodes": "10, 11",
                "geometry": LineString([(1, 0), (2, 1), (3, 0)]),
            },
            {
                "id_link": 103,
                "id_us_node": 11,
                "id_ds_node": 4,
                "id_nodes": "11, 4",
                "geometry": LineString([(3, 0), (4, 0)]),
            },
        ],
        crs=nodes.crs,
    )
    dissolved_links, dissolved_nodes, audit = _dissolve_computational_cut_interfaces(
        links=links,
        nodes=nodes,
        splice_summary={
            "cut_splits": [
                {
                    "role": "upstream",
                    "split_method": "split_parent_link_at_computational_cut",
                    "computational_boundary_node_id": 10,
                    "original_parent_link_id": 101,
                },
                {
                    "role": "downstream",
                    "split_method": "split_parent_link_at_computational_cut",
                    "computational_boundary_node_id": 11,
                    "original_parent_link_id": 103,
                },
            ]
        },
    )

    assert set(dissolved_links["id_link"].astype(int)) == {200}
    assert set(dissolved_nodes["id_node"].astype(int)) == {1, 4}
    row = dissolved_links.iloc[0]
    assert (int(row.id_us_node), int(row.id_ds_node)) == (1, 4)
    assert list(row.geometry.coords) == [
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 1.0),
        (3.0, 0.0),
        (4.0, 0.0),
    ]
    assert audit["dissolved_interface_count"] == 2


def test_collapse_topology_contract_counts_filled_components_and_cycles(
    tmp_path: Path,
) -> None:
    components = pd.DataFrame(
        {
            "component_ownership_audit": [
                json.dumps(
                    [
                        {
                            "candidate_component_id": 1,
                            "ownership_status": "accepted_selected_unit",
                        },
                        {
                            "candidate_component_id": 2,
                            "ownership_status": "accepted_selected_unit",
                        },
                        {
                            "candidate_component_id": 3,
                            "ownership_status": "rejected_neighbour",
                        },
                    ]
                )
            ]
        }
    )
    components_path = tmp_path / "collapse_components.csv"
    components.to_csv(components_path, index=False)
    variant = SimpleNamespace(
        output_paths={"collapse_components": str(components_path)}
    )
    assert _expected_cycle_reduction(variant) == 2

    links = gpd.GeoDataFrame(
        [
            {"id_link": 1, "id_us_node": 1, "id_ds_node": 2, "geometry": LineString([(0, 0), (1, 0)])},
            {"id_link": 2, "id_us_node": 2, "id_ds_node": 3, "geometry": LineString([(1, 0), (2, 0)])},
            {"id_link": 3, "id_us_node": 3, "id_ds_node": 1, "geometry": LineString([(2, 0), (0, 0)])},
            {"id_link": 4, "id_us_node": 1, "id_ds_node": 2, "geometry": LineString([(0, 0), (1, 0.1)])},
        ],
        crs="EPSG:3857",
    )
    assert _undirected_cycle_rank(links) == 2


def test_post_direction_reverse_parent_duplicate_requires_geometry_match() -> None:
    links = gpd.GeoDataFrame(
        [
            {
                "id_link": 1,
                "id_us_node": 0,
                "id_ds_node": 1,
                "geometry": LineString([(0, 0), (5, 0), (10, 0)]),
            },
            {
                "id_link": 10,
                "id_us_node": 1,
                "id_ds_node": 0,
                "geometry": LineString([(10, 0), (5, 0.1), (0, 0)]),
            },
            {
                "id_link": 11,
                "id_us_node": 1,
                "id_ds_node": 0,
                "geometry": LineString([(10, 0), (5, 5), (0, 0)]),
            },
        ],
        crs="EPSG:3857",
    )

    result, removed, audit = _remove_post_direction_reverse_parent_duplicates(
        links,
        inserted_link_ids={10, 11},
        retained_parent_link_ids={1},
        geometry_tolerance_m=0.5,
        minimum_overlap_fraction=0.95,
    )

    assert removed == {10: 1}
    assert set(result["id_link"].astype(int)) == {1, 11}
    assert audit[0]["inserted_link_id"] == 10
    assert audit[0]["retained_parent_link_id"] == 1


def test_width_measurement_scope_uses_strict_geometry_not_lineage_alone() -> None:
    parent = gpd.GeoDataFrame(
        [
            {"id_link": 1, "geometry": LineString([(0, 0), (10, 0)])},
            {"id_link": 2, "geometry": LineString([(10, 0), (20, 0)])},
        ],
        crs="EPSG:3857",
    )
    child = parent.copy()
    child.loc[child["id_link"].eq(2), "geometry"] = [
        LineString([(10, 0), (15, 0.25), (20, 0)])
    ]
    lineage = pd.DataFrame(
        {
            "id_link": [1, 2],
            "lineage_type": ["unchanged_1to1", "unchanged_1to1"],
            "dominant_parent_link_id": [1, 2],
        }
    )

    changed = _strict_width_measurement_link_ids(
        child,
        parent_links=parent,
        parent_samples=pd.DataFrame(),
        lineage=lineage,
    )

    assert changed == {2}


def test_cut_corridor_pruning_removes_only_nonboundary_dangling_spur() -> None:
    nodes = gpd.GeoDataFrame(
        [
            {"id_node": 0, "geometry": Point(0, 0)},
            {"id_node": 1, "geometry": Point(1, 0)},
            {"id_node": 2, "geometry": Point(2, 0)},
            {"id_node": 3, "geometry": Point(1, 1)},
        ],
        crs="EPSG:3857",
    )
    links = gpd.GeoDataFrame(
        [
            {"id_link": 10, "id_nodes": "0, 1", "geometry": LineString([(0, 0), (1, 0)])},
            {"id_link": 11, "id_nodes": "1, 2", "geometry": LineString([(1, 0), (2, 0)])},
            {"id_link": 12, "id_nodes": "1, 3", "geometry": LineString([(1, 0), (1, 1)])},
        ],
        crs=nodes.crs,
    )

    retained_links, retained_nodes, summary = _prune_raw_local_to_cut_corridor(
        links, nodes, source_node_id=0, sink_node_id=2
    )

    assert set(retained_links["id_link"].astype(int)) == {10, 11}
    assert set(retained_nodes["id_node"].astype(int)) == {0, 1, 2}
    assert summary["removed_node_ids"] == [3]
    assert summary["removed_link_ids"] == [12]


def test_directed_anchored_interval_allows_one_interface_downstream_of_another() -> None:
    nodes = gpd.GeoDataFrame(
        [
            {"id_node": node_id, "geometry": Point(float(node_id), 0.0)}
            for node_id in (0, 4, 5, 7, 9)
        ],
        crs="EPSG:3857",
    )
    links = gpd.GeoDataFrame(
        [
            {"id_link": 1, "id_us_node": 0, "id_ds_node": 5, "geometry": LineString([(0, 0), (5, 0)])},
            {"id_link": 7, "id_us_node": 5, "id_ds_node": 4, "geometry": LineString([(5, 0), (4, 0)])},
            {"id_link": 8, "id_us_node": 4, "id_ds_node": 7, "geometry": LineString([(4, 0), (7, 0)])},
            {"id_link": 9, "id_us_node": 5, "id_ds_node": 7, "geometry": LineString([(5, 0), (7, 0)])},
            {"id_link": 10, "id_us_node": 7, "id_ds_node": 9, "geometry": LineString([(7, 0), (9, 0)])},
        ],
        crs=nodes.crs,
    )

    retained_links, retained_nodes, summary = _prune_directed_to_anchored_interval(
        links,
        nodes,
        source_node_ids={4, 5},
        sink_node_ids={7},
    )

    assert set(retained_links["id_link"].astype(int)) == {7, 8, 9}
    assert set(retained_nodes["id_node"].astype(int)) == {4, 5, 7}
    assert summary["retained_link_count"] == 3


def test_forced_anchored_match_demotes_nearby_internal_junction() -> None:
    node_match = pd.DataFrame(
        [
            {
                "child_id_node": 10,
                "matched_parent_node_id": 51,
                "match_within_tolerance": True,
                "parent_node_order": 5,
            },
            {
                "child_id_node": 11,
                "matched_parent_node_id": 51,
                "match_within_tolerance": True,
                "parent_node_order": 5,
            },
            {
                "child_id_node": 12,
                "matched_parent_node_id": np.nan,
                "match_within_tolerance": False,
                "parent_node_order": np.nan,
            },
            {
                "child_id_node": 13,
                "matched_parent_node_id": np.nan,
                "match_within_tolerance": False,
                "parent_node_order": np.nan,
            },
        ]
    )
    bindings = pd.DataFrame(
        [
            {
                "child_id_node": 10,
                "forced_parent_node_id": 51,
                "cut_role": "upstream",
                "cut_distance_m": 0.0,
                "raw_boundary_role_matches": True,
                "binding_method": "existing_local_node_at_anchored_boundary",
                "match_method": "forced_anchored_boundary",
            }
        ]
    )
    child_links = gpd.GeoDataFrame(
        [
            {"id_link": 1, "id_nodes": "10, 11", "geometry": LineString([(0, 0), (1, 0)])},
            {"id_link": 2, "id_nodes": "11, 12", "geometry": LineString([(1, 0), (2, 1)])},
            {"id_link": 3, "id_nodes": "11, 13", "geometry": LineString([(1, 0), (2, -1)])},
        ],
        crs="EPSG:3857",
    )

    result = _force_boundary_node_matches(
        node_match, bindings, {51: 5}, child_links
    ).set_index("child_id_node")

    assert result.loc[10, "matched_parent_node_id"] == 51
    assert result.loc[10, "match_method"] == "forced_anchored_boundary"
    assert not bool(result.loc[11, "match_within_tolerance"])
    assert pd.isna(result.loc[11, "matched_parent_node_id"])
    assert result.loc[11, "match_method"] == (
        "new_internal_node_near_anchored_boundary"
    )


def test_duplicate_parent_match_keeps_external_node_and_demotes_internal_node() -> None:
    node_match = pd.DataFrame(
        [
            {"child_id_node": 10, "matched_parent_node_id": 51, "match_distance": 0.2,
             "match_within_tolerance": True, "parent_node_order": 5},
            {"child_id_node": 11, "matched_parent_node_id": 51, "match_distance": 0.1,
             "match_within_tolerance": True, "parent_node_order": 5},
            {"child_id_node": 12, "matched_parent_node_id": pd.NA, "match_distance": 5.0,
             "match_within_tolerance": False, "parent_node_order": pd.NA},
            {"child_id_node": 13, "matched_parent_node_id": pd.NA, "match_distance": 5.0,
             "match_within_tolerance": False, "parent_node_order": pd.NA},
        ]
    )
    child_links = gpd.GeoDataFrame(
        [
            {"id_link": 1, "id_nodes": "10, 11", "geometry": LineString([(0, 0), (1, 0)])},
            {"id_link": 2, "id_nodes": "11, 12", "geometry": LineString([(1, 0), (2, 1)])},
            {"id_link": 3, "id_nodes": "11, 13", "geometry": LineString([(1, 0), (2, -1)])},
        ],
        crs="EPSG:3857",
    )

    result = _force_boundary_node_matches(
        node_match,
        pd.DataFrame(),
        {51: 5},
        child_links,
    ).set_index("child_id_node")

    assert int(result.loc[10, "matched_parent_node_id"]) == 51
    assert pd.isna(result.loc[11, "matched_parent_node_id"])
    assert result.loc[11, "match_method"] == "new_internal_node_near_anchored_boundary"
    assert result.loc[11, "duplicate_resolution_method"] == "keep_unique_external_terminal"


def test_local_lineage_allows_removed_internal_parent_nodes() -> None:
    parent_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 1, "is_inlet": True, "is_outlet": False, "geometry": Point(0, 0)},
            {"id_node": 2, "is_inlet": False, "is_outlet": False, "geometry": Point(1, 0)},
            {"id_node": 3, "is_inlet": False, "is_outlet": True, "geometry": Point(2, 0)},
        ],
        crs="EPSG:3857",
    )
    parent_links = gpd.GeoDataFrame(
        [
            {"id_link": 1, "id_us_node": 1, "id_ds_node": 2, "id_nodes": "1, 2", "geometry": LineString([(0, 0), (1, 0)])},
            {"id_link": 2, "id_us_node": 2, "id_ds_node": 3, "id_nodes": "2, 3", "geometry": LineString([(1, 0), (2, 0)])},
        ],
        crs=parent_nodes.crs,
    )
    # Node 2 is legitimately absent after both parent links collapse into one.
    child_nodes = parent_nodes.loc[parent_nodes["id_node"].isin([1, 3])].copy()
    child_links = gpd.GeoDataFrame(
        [{"id_link": 3, "id_us_node": 1, "id_ds_node": 3, "id_nodes": "1, 3", "geometry": LineString([(0, 0), (2, 0)])}],
        crs=parent_nodes.crs,
    )

    lineage = _local_link_lineage(
        child_links,
        child_nodes,
        parent_links,
        parent_nodes,
        match_tolerance=0.1,
    )

    assert lineage["id_link"].astype(int).tolist() == [3]
    assert lineage.loc[0, "lineage_type"] == "collapsed_many_to_one"


def test_conservative_merge_binds_cuts_and_retains_outside_parent_links(tmp_path) -> None:
    water = np.zeros((100, 100), dtype=np.uint8)
    water[:, 48:52] = 1
    labels = np.zeros_like(water, dtype=np.int32)
    labels[:, :48], labels[:, 52:] = 1, 2
    mask_path = _raster(tmp_path / "mask.tif", water, "uint8")
    labels_path = _raster(tmp_path / "labels.tif", labels, "int32")

    node_values = [(9, 90), (10, 80), (11, 65), (12, 35), (13, 20), (14, 10)]
    nodes = gpd.GeoDataFrame(
        [{
            "id_node": node_id, "is_inlet": node_id == 9, "is_outlet": node_id == 14,
            "id_links": "", "n_links": 0, "type_io": "internal", "geometry": Point(50, y),
        } for node_id, y in node_values], crs="EPSG:3857",
    )
    link_values = [(4, 9, 10), (1, 10, 11), (2, 11, 12), (3, 12, 13), (5, 13, 14)]
    node_y = dict(node_values)
    links = gpd.GeoDataFrame(
        [{
            "id_link": link_id, "id_us_node": us, "id_ds_node": ds, "id_nodes": f"{us}, {ds}",
            "is_inlet": us == 9, "is_outlet": ds == 14, "type_io": "internal",
            "link_conn": "", "idx_link": "1, 2", "scientific_value": link_id * 100.0,
            "len": float(abs(node_y[us] - node_y[ds])),
            "len_adj": float(abs(node_y[us] - node_y[ds])),
            "wid": 4.0,
            "wid_adj": 4.0,
            "wid_med": 4.0,
            "wid_adj_wet": link_id * 10.0,
            "wid_adj_routing": link_id * 11.0,
            "audit_flag": link_id % 2 == 0,
            "geometry": LineString([(50, node_y[us]), (50, node_y[ds])]),
        } for link_id, us, ds in link_values], crs="EPSG:3857",
    )
    parent_links_path, parent_nodes_path = tmp_path / "parent_links.gpkg", tmp_path / "parent_nodes.gpkg"
    links.to_file(parent_links_path, driver="GPKG")
    nodes.to_file(parent_nodes_path, driver="GPKG")

    contract = LocalPatchContract(
        example_id="synthetic", selection_type="unit", selected_unit_ids=(1,),
        domain_type="closed_unit", core_link_ids=(2,),
        upstream_collar=LocalPatchCollar(
            role="upstream", base_link_id=1, base_upstream_node_id=10,
            base_downstream_node_id=11, domain_boundary_node_id=11,
            outer_splice_anchor_node_id=10,
        ),
        downstream_collar=LocalPatchCollar(
            role="downstream", base_link_id=3, base_upstream_node_id=12,
            base_downstream_node_id=13, domain_boundary_node_id=12,
            outer_splice_anchor_node_id=13,
        ),
        raster_policy=LocalPatchRasterPolicy(
            minimum_halo_m=10, halo_expansion_step_m=10, maximum_halo_m=10,
            minimum_collar_context_m=10,
        ),
    )
    crop = build_local_patch_base_raster(
        contract=contract, full_links=links, full_nodes=nodes,
        cleaned_mask_path=mask_path, boundary_component_labels_path=labels_path,
        output_dir=tmp_path / "crop",
    )

    raw_nodes = gpd.GeoDataFrame([
        {"id_node": 0, "is_inlet": True, "is_outlet": False, "geometry": Point(50, 80)},
        {"id_node": 1, "is_inlet": False, "is_outlet": True, "geometry": Point(50, 20)},
    ], crs="EPSG:3857")
    raw_links = gpd.GeoDataFrame([{
        "id_link": 0, "id_nodes": "0, 1", "is_inlet": True, "is_outlet": True,
        "idx_link": "10, 11", "geometry": LineString([(50, 80), (60, 50), (50, 20)]),
    }], crs="EPSG:3857")
    raw_links_path, raw_nodes_path = tmp_path / "raw_links.gpkg", tmp_path / "raw_nodes.gpkg"
    raw_links.to_file(raw_links_path, driver="GPKG")
    raw_nodes.to_file(raw_nodes_path, driver="GPKG")
    variant = LocalPatchVariantResult(
        status="complete_raw_local_variant", selected_unit_ids=(1,),
        output_paths={
            "raw_rivgraph_links": str(raw_links_path), "raw_rivgraph_nodes": str(raw_nodes_path),
            "local_collapsed_mask": crop.output_paths["local_base_mask"],
            "local_base_mask": crop.output_paths["local_base_mask"],
            "local_origin_mask": crop.output_paths["local_base_mask"],
            "boundary_component_labels": crop.output_paths["boundary_component_labels"],
        }, collapse_summary={},
    )

    result = merge_local_patch_variant(
        contract=contract, raster_result=crop, variant_result=variant,
        parent_links_path=parent_links_path, parent_nodes_path=parent_nodes_path,
        output_dir=tmp_path / "merge",
    )
    merged_links = gpd.read_file(result.output_paths["merged_links"])
    merged_nodes = gpd.read_file(result.output_paths["merged_nodes"])
    assert set(merged_links["id_link"].astype(int)) == {1, 3, 4, 5, 6}
    assert merged_links.set_index("id_link").loc[4, "scientific_value"] == 400.0
    assert merged_links.set_index("id_link").loc[5, "scientific_value"] == 500.0
    assert merged_links.set_index("id_link").loc[4, "audit_flag"] == np.bool_(True)
    assert merged_links.set_index("id_link").loc[5, "audit_flag"] == np.bool_(False)
    assert merged_links["audit_flag"].dtype == bool
    inserted = merged_links.loc[merged_links["id_link"].astype(int) == 6].iloc[0]
    assert (int(inserted.id_us_node), int(inserted.id_ds_node)) == (11, 12)
    assert inserted.idx_link is None
    assert set(merged_nodes["id_node"].astype(int)) == {9, 10, 11, 12, 13, 14}
    assert result.summary["validation_report"]["is_valid"] is True
    node_audit = pd.read_csv(result.output_paths["node_merge_audit"])
    assert node_audit.loc[
        node_audit["match_method"].eq("forced_anchored_boundary"),
        "raw_boundary_role_matches",
    ].all()
    assert set(
        node_audit.loc[
            node_audit["match_method"].eq("forced_anchored_boundary"), "binding_method"
        ]
    ) == {"split_local_link_at_anchored_boundary"}


def test_local_realized_assembly_trims_raster_context_at_closed_domain(
    tmp_path: Path,
) -> None:
    """Raster-cut context is removed while parent links outside the domain remain."""
    nodes = gpd.GeoDataFrame(
        [
            {
                "id_node": node_id,
                "is_inlet": node_id == 0,
                "is_outlet": node_id == 5,
                "geometry": Point(0.0, float(50 - node_id * 10)),
            }
            for node_id in range(6)
        ],
        crs="EPSG:3857",
    )
    links = gpd.GeoDataFrame(
        [
            {
                "id_link": 100 + node_id,
                "id_us_node": node_id,
                "id_ds_node": node_id + 1,
                "id_nodes": f"{node_id}, {node_id + 1}",
                "len": 10.0,
                "geometry": LineString(
                    [
                        (0.0, float(50 - node_id * 10)),
                        (0.0, float(40 - node_id * 10)),
                    ]
                ),
            }
            for node_id in range(5)
        ],
        crs=nodes.crs,
    )
    parent_links_path = tmp_path / "parent_links.gpkg"
    parent_nodes_path = tmp_path / "parent_nodes.gpkg"
    links.to_file(parent_links_path, driver="GPKG")
    nodes.to_file(parent_nodes_path, driver="GPKG")

    contract = LocalPatchContract(
        example_id="synthetic",
        selection_type="unit",
        selected_unit_ids=(1,),
        domain_type="closed_unit",
        core_link_ids=(102,),
        upstream_collar=LocalPatchCollar(
            role="upstream",
            base_link_id=101,
            base_upstream_node_id=1,
            base_downstream_node_id=2,
            domain_boundary_node_id=2,
            outer_splice_anchor_node_id=1,
        ),
        downstream_collar=LocalPatchCollar(
            role="downstream",
            base_link_id=103,
            base_upstream_node_id=3,
            base_downstream_node_id=4,
            domain_boundary_node_id=3,
            outer_splice_anchor_node_id=4,
        ),
    )

    raw_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 10, "is_inlet": True, "is_outlet": False, "geometry": Point(0, 40)},
            {"id_node": 11, "is_inlet": False, "is_outlet": False, "geometry": Point(3, 25)},
            {"id_node": 12, "is_inlet": False, "is_outlet": True, "geometry": Point(0, 10)},
        ],
        crs=nodes.crs,
    )
    raw_links = gpd.GeoDataFrame(
        [
            {
                "id_link": 0,
                "id_nodes": "10, 11",
                "is_inlet": True,
                "is_outlet": False,
                "geometry": LineString([(0, 40), (3, 25)]),
            },
            {
                "id_link": 1,
                "id_nodes": "11, 12",
                "is_inlet": False,
                "is_outlet": True,
                "geometry": LineString([(3, 25), (0, 10)]),
            },
        ],
        crs=nodes.crs,
    )
    raw_links_path = tmp_path / "raw_links.gpkg"
    raw_nodes_path = tmp_path / "raw_nodes.gpkg"
    raw_links.to_file(raw_links_path, driver="GPKG")
    raw_nodes.to_file(raw_nodes_path, driver="GPKG")
    cuts_path = tmp_path / "cuts.gpkg"
    gpd.GeoDataFrame(
        [
            {"role": "upstream", "geometry": LineString([(-5, 40), (5, 40)])},
            {"role": "downstream", "geometry": LineString([(-5, 10), (5, 10)])},
        ],
        crs=nodes.crs,
    ).to_file(cuts_path, driver="GPKG")
    local_mask = _raster(tmp_path / "local_mask.tif", np.ones((50, 50)), "uint8")
    raster_result = SimpleNamespace(output_paths={"main_bank_cuts": str(cuts_path)})
    variant_result = LocalPatchVariantResult(
        status="complete_raw_local_variant",
        selected_unit_ids=(1,),
        output_paths={
            "raw_rivgraph_links": str(raw_links_path),
            "raw_rivgraph_nodes": str(raw_nodes_path),
            "local_collapsed_mask": str(local_mask),
        },
        collapse_summary={},
    )

    result = assemble_local_patch_realized_graph(
        contract=contract,
        raster_result=raster_result,
        variant_result=variant_result,
        parent_links_path=parent_links_path,
        parent_nodes_path=parent_nodes_path,
        output_dir=tmp_path / "realized",
    )

    realized = gpd.read_file(result.output_paths["realized_links"])
    assert set(realized["id_link"].astype(int)).issuperset({100, 101, 103, 104})
    parent_by_id = links.set_index("id_link")
    child_by_id = realized.set_index("id_link")
    for link_id in (100, 101, 103, 104):
        assert child_by_id.loc[link_id].geometry.equals_exact(
            parent_by_id.loc[link_id].geometry, 0.0
        )
        assert (
            int(child_by_id.loc[link_id, "id_us_node"]),
            int(child_by_id.loc[link_id, "id_ds_node"]),
        ) == (
            int(parent_by_id.loc[link_id, "id_us_node"]),
            int(parent_by_id.loc[link_id, "id_ds_node"]),
        )
    assert result.summary["scientific_anchoring_applied"] is False
    assert result.summary["outer_upstream_boundary_node_ids"] == [2]
    assert result.summary["outer_downstream_boundary_node_ids"] == [3]
    assert result.summary["outside_parent_validation"]["passed"] is True
    audit = pd.read_csv(result.output_paths["link_lineage_audit"])
    assert set(audit["realized_lineage_class"]).issubset(
        {
            "unchanged_parent",
            "partial_parent_overlap_interface",
            "changed_local",
        }
    )
    assert not audit.empty

    unit = StructuralUnit(
        unit_id=1,
        bifurcation=2,
        confluence=3,
        n_paths=1,
        unit_class="synthetic",
        min_path_length=10.0,
        max_path_length=10.0,
        node_set={2, 3},
        edge_set={(2, 3, 102)},
    )
    anchored = anchor_local_patch_realized_graph(
        realized_result=result,
        parent_links_path=parent_links_path,
        parent_nodes_path=parent_nodes_path,
        selected_unit_ids=(1,),
        units_by_id={1: unit},
        output_dir=tmp_path / "anchored",
        match_tolerance_m=1.25,
    )
    anchored_links = gpd.read_file(anchored.output_paths["anchored_links"])
    anchored_nodes = gpd.read_file(anchored.output_paths["anchored_nodes"])
    assert anchored.status == "complete_shared_anchored_graph"
    assert anchored.summary["scientific_anchoring_applied"] is True
    assert anchored.summary["anchoring_implementation"] == (
        "derive_anchored_graph_from_realized"
    )
    anchored_node_match = pd.read_csv(anchored.output_paths["anchored_node_match"])
    matched_parent_ids = set(
        pd.to_numeric(
            anchored_node_match.loc[
                anchored_node_match["match_within_tolerance"].fillna(False),
                "matched_parent_node_id",
            ],
            errors="coerce",
        ).dropna().astype(int)
    )
    assert {2, 3}.issubset(matched_parent_ids)
    parent_anchor_geometry = nodes.set_index("id_node").loc[[2, 3]].geometry
    assert all(
        float(anchored_nodes.geometry.distance(geometry).min()) <= 1e-9
        for geometry in parent_anchor_geometry
    )
    assert set(anchored_links["id_us_node"].astype(int)).union(
        anchored_links["id_ds_node"].astype(int)
    ).issuperset(set(anchored_nodes["id_node"].astype(int)))
    assert anchored.summary["direction_summary"]["is_valid"] is True

    # The persisted adapter must be exactly the shared transformation, not a
    # second local anchoring implementation.
    direct = derive_anchored_graph_from_realized(
        parent_links=links,
        parent_nodes=nodes,
        realized_links=realized,
        realized_nodes=gpd.read_file(result.output_paths["realized_nodes"]),
        realized_node_match=pd.read_csv(
            anchored.output_paths["realized_node_match"]
        ),
        realized_link_match=pd.read_csv(
            anchored.output_paths["realized_link_match"]
        ),
        realized_link_lineage=pd.read_csv(
            anchored.output_paths["realized_link_lineage"]
        ),
        selected_unit_ids=(1,),
        units_by_id={1: unit},
        match_tolerance=1.25,
    )
    persisted_links = anchored_links.sort_values("id_link").reset_index(drop=True)
    direct_links = direct.directed_links.sort_values("id_link").reset_index(drop=True)
    assert persisted_links["id_link"].astype(int).tolist() == direct_links[
        "id_link"
    ].astype(int).tolist()
    assert persisted_links[["id_us_node", "id_ds_node"]].astype(int).equals(
        direct_links[["id_us_node", "id_ds_node"]].astype(int)
    )
    assert all(
        first.equals_exact(second, 0.0)
        for first, second in zip(persisted_links.geometry, direct_links.geometry)
    )


def test_anchored_trim_prunes_multi_terminal_context_before_parent_matching() -> None:
    parent_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 11, "geometry": Point(50, 65)},
            {"id_node": 12, "geometry": Point(50, 35)},
        ],
        crs="EPSG:3857",
    )
    local_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 0, "geometry": Point(50, 80)},
            {"id_node": 1, "geometry": Point(50, 65)},
            {"id_node": 2, "geometry": Point(50, 50)},
            {"id_node": 3, "geometry": Point(50, 20)},
            {"id_node": 4, "geometry": Point(45, 10)},
            {"id_node": 5, "geometry": Point(55, 10)},
        ],
        crs="EPSG:3857",
    )
    local_links = gpd.GeoDataFrame(
        [
            {"id_link": 0, "id_us_node": 0, "id_ds_node": 1, "id_nodes": "0, 1", "geometry": LineString([(50, 80), (50, 65)])},
            {"id_link": 1, "id_us_node": 1, "id_ds_node": 2, "id_nodes": "1, 2", "geometry": LineString([(50, 65), (48, 50)])},
            {"id_link": 2, "id_us_node": 1, "id_ds_node": 2, "id_nodes": "1, 2", "geometry": LineString([(50, 65), (52, 50)])},
            {"id_link": 3, "id_us_node": 2, "id_ds_node": 3, "id_nodes": "2, 3", "geometry": LineString([(50, 50), (50, 20)])},
            {"id_link": 4, "id_us_node": 3, "id_ds_node": 4, "id_nodes": "3, 4", "geometry": LineString([(50, 20), (45, 10)])},
            {"id_link": 5, "id_us_node": 3, "id_ds_node": 5, "id_nodes": "3, 5", "geometry": LineString([(50, 20), (55, 10)])},
        ],
        crs="EPSG:3857",
    )
    contract = LocalPatchContract(
        example_id="synthetic",
        selection_type="unit",
        selected_unit_ids=(1,),
        domain_type="closed_unit",
        core_link_ids=(1, 2),
        upstream_collar=LocalPatchCollar(
            role="upstream", base_link_id=10, base_upstream_node_id=10,
            base_downstream_node_id=11, domain_boundary_node_id=11,
            outer_splice_anchor_node_id=10,
        ),
        downstream_collar=LocalPatchCollar(
            role="downstream", base_link_id=11, base_upstream_node_id=12,
            base_downstream_node_id=13, domain_boundary_node_id=12,
            outer_splice_anchor_node_id=13,
        ),
    )
    # Raw RivGraph GPKGs can contain the direction columns before their values
    # have been assigned. Geometry-first trimming must use id_nodes in that
    # case rather than treating column presence as proof of valid endpoints.
    local_links[["id_us_node", "id_ds_node"]] = None
    trimmed_links, _, bindings = _trim_to_anchored_boundaries(
        local_links, local_nodes, parent_nodes, contract, tolerance=1.0
    )
    assert 0 not in set(trimmed_links["id_link"].astype(int))
    assert len(trimmed_links) == 3
    assert set(bindings["binding_method"]) == {
        "existing_local_node_at_anchored_boundary",
        "split_local_link_at_anchored_boundary",
    }
    source = int(bindings.loc[bindings["cut_role"].eq("upstream"), "child_id_node"].iloc[0])
    sink = int(bindings.loc[bindings["cut_role"].eq("downstream"), "child_id_node"].iloc[0])
    assert source == 1
    assert set(trimmed_links.loc[trimmed_links["id_us_node"].astype(int).eq(source), "id_link"]) == {1, 2}
    assert set(trimmed_links.loc[trimmed_links["id_ds_node"].astype(int).eq(sink), "id_link"]) == {3}


def test_anchored_trim_allocates_distinct_children_to_close_parent_boundaries() -> None:
    parent_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 11, "geometry": Point(1.0, 0.0)},
            {"id_node": 12, "geometry": Point(1.4, 0.0)},
        ],
        crs="EPSG:3857",
    )
    local_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 0, "geometry": Point(0.0, 0.0)},
            {"id_node": 1, "geometry": Point(1.1, 0.0)},
            {"id_node": 2, "geometry": Point(3.0, 0.0)},
        ],
        crs=parent_nodes.crs,
    )
    local_links = gpd.GeoDataFrame(
        [
            {"id_link": 0, "id_us_node": 0, "id_ds_node": 1, "id_nodes": "0, 1", "geometry": LineString([(0.0, 0.0), (1.1, 0.0)])},
            {"id_link": 1, "id_us_node": 1, "id_ds_node": 2, "id_nodes": "1, 2", "geometry": LineString([(1.1, 0.0), (3.0, 0.0)])},
        ],
        crs=parent_nodes.crs,
    )
    contract = LocalPatchContract(
        example_id="synthetic",
        selection_type="unit",
        selected_unit_ids=(1,),
        domain_type="closed_unit",
        core_link_ids=(2,),
        upstream_collar=LocalPatchCollar(
            role="upstream", base_link_id=10, base_upstream_node_id=10,
            base_downstream_node_id=11, domain_boundary_node_id=11,
            outer_splice_anchor_node_id=10,
        ),
        downstream_collar=LocalPatchCollar(
            role="downstream", base_link_id=11, base_upstream_node_id=12,
            base_downstream_node_id=13, domain_boundary_node_id=12,
            outer_splice_anchor_node_id=13,
        ),
    )

    trimmed_links, trimmed_nodes, bindings = _trim_to_anchored_boundaries(
        local_links, local_nodes, parent_nodes, contract, tolerance=0.5
    )

    assert bindings["child_id_node"].nunique() == 2
    assert bindings["forced_parent_node_id"].nunique() == 2
    assert set(bindings["binding_method"]) == {
        "existing_local_node_at_anchored_boundary",
        "split_local_link_at_anchored_boundary",
    }
    source = int(bindings.loc[bindings["cut_role"].eq("upstream"), "child_id_node"].iloc[0])
    sink = int(bindings.loc[bindings["cut_role"].eq("downstream"), "child_id_node"].iloc[0])
    assert source != sink
    assert np.isclose(
        trimmed_nodes.set_index("id_node").loc[source].geometry.x, 1.0
    )
    assert np.isclose(
        trimmed_nodes.set_index("id_node").loc[sink].geometry.x, 1.4
    )
    assert len(trimmed_links) == 1
    assert int(trimmed_links.iloc[0]["id_us_node"]) == source
    assert int(trimmed_links.iloc[0]["id_ds_node"]) == sink


def test_anchored_trim_allows_one_parent_node_to_hold_both_boundary_roles() -> None:
    parent_nodes = gpd.GeoDataFrame(
        [{"id_node": 11, "geometry": Point(1.0, 0.0)}],
        crs="EPSG:3857",
    )
    local_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 0, "geometry": Point(0.0, 0.0)},
            {"id_node": 1, "geometry": Point(1.0, 0.0)},
            {"id_node": 2, "geometry": Point(2.0, 0.0)},
        ],
        crs=parent_nodes.crs,
    )
    local_links = gpd.GeoDataFrame(
        [
            {"id_link": 0, "id_nodes": "0, 1", "geometry": LineString([(0, 0), (1, 0)])},
            {"id_link": 1, "id_nodes": "1, 2", "geometry": LineString([(1, 0), (2, 0)])},
        ],
        crs=parent_nodes.crs,
    )
    contract = LocalPatchContract(
        example_id="synthetic",
        selection_type="unit",
        selected_unit_ids=(1,),
        domain_type="closed_unit",
        core_link_ids=(0, 1),
        upstream_collar=LocalPatchCollar(
            role="upstream", base_link_id=10, base_upstream_node_id=10,
            base_downstream_node_id=12, domain_boundary_node_id=12,
            outer_splice_anchor_node_id=10,
        ),
        downstream_collar=LocalPatchCollar(
            role="downstream", base_link_id=11, base_upstream_node_id=13,
            base_downstream_node_id=14, domain_boundary_node_id=13,
            outer_splice_anchor_node_id=14,
        ),
        anchored_upstream_boundary_node_ids=(11,),
        anchored_downstream_boundary_node_ids=(11,),
    )

    _, _, bindings = _trim_to_anchored_boundaries(
        local_links,
        local_nodes,
        parent_nodes,
        contract,
        tolerance=0.1,
        prune_undirected_context=False,
    )

    assert bindings["child_id_node"].nunique() == 1
    assert bindings["forced_parent_node_id"].astype(int).tolist() == [11, 11]
    assert set(bindings["cut_role"]) == {"upstream", "downstream"}


def test_anchored_trim_reexpands_compressed_selected_parent_junction() -> None:
    parent_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 11, "geometry": Point(1.0, 0.0)},
            {"id_node": 12, "geometry": Point(1.4, 0.0)},
            {"id_node": 13, "geometry": Point(3.0, 0.0)},
        ],
        crs="EPSG:3857",
    )
    parent_links = gpd.GeoDataFrame(
        [
            {"id_link": 100, "id_us_node": 12, "id_ds_node": 11, "geometry": LineString([(1.4, 0.0), (1.0, 0.0)])},
            {"id_link": 101, "id_us_node": 11, "id_ds_node": 13, "geometry": LineString([(1.0, 0.0), (3.0, 0.0)])},
        ],
        crs=parent_nodes.crs,
    )
    local_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 0, "geometry": Point(1.4, 0.0)},
            {"id_node": 1, "geometry": Point(3.0, 0.0)},
        ],
        crs=parent_nodes.crs,
    )
    local_links = gpd.GeoDataFrame(
        [
            {"id_link": 0, "id_nodes": "0, 1", "geometry": LineString([(1.4, 0.0), (3.0, 0.0)])},
        ],
        crs=parent_nodes.crs,
    )
    contract = LocalPatchContract(
        example_id="synthetic",
        selection_type="unit",
        selected_unit_ids=(1,),
        domain_type="closed_unit",
        core_link_ids=(100, 101),
        selected_link_ids=(100, 101),
        upstream_collar=LocalPatchCollar(
            role="upstream", base_link_id=10, base_upstream_node_id=9,
            base_downstream_node_id=12, domain_boundary_node_id=12,
            outer_splice_anchor_node_id=9,
        ),
        downstream_collar=LocalPatchCollar(
            role="downstream", base_link_id=11, base_upstream_node_id=13,
            base_downstream_node_id=14, domain_boundary_node_id=13,
            outer_splice_anchor_node_id=14,
        ),
        anchored_upstream_boundary_node_ids=(11, 12),
        anchored_downstream_boundary_node_ids=(13,),
    )

    trimmed_links, _, bindings = _trim_to_anchored_boundaries(
        local_links,
        local_nodes,
        parent_nodes,
        contract,
        parent_links=parent_links,
        tolerance=0.5,
        prune_undirected_context=False,
    )

    by_parent = bindings.set_index("forced_parent_node_id")
    child_11 = int(by_parent.loc[11, "child_id_node"])
    child_12 = int(by_parent.loc[12, "child_id_node"])
    assert child_11 != child_12
    assert by_parent.loc[11, "binding_method"] == "expand_compressed_selected_parent_junction"
    connector_rows = trimmed_links.loc[
        trimmed_links["id_nodes"].isin(
            {f"{child_12}, {child_11}", f"{child_11}, {child_12}"}
        )
    ]
    assert len(connector_rows) == 1


def test_global_boundary_binding_accepts_unique_role_within_one_cut_width() -> None:
    contract = LocalPatchContract(
        example_id="synthetic",
        selection_type="unit",
        selected_unit_ids=(1,),
        domain_type="closed_unit",
        core_link_ids=(2,),
        upstream_collar=LocalPatchCollar(
            role="upstream", base_link_id=1, base_upstream_node_id=10,
            base_downstream_node_id=11, domain_boundary_node_id=11,
            outer_splice_anchor_node_id=10,
        ),
        downstream_collar=LocalPatchCollar(
            role="downstream", base_link_id=3, base_upstream_node_id=12,
            base_downstream_node_id=13, domain_boundary_node_id=12,
            outer_splice_anchor_node_id=13,
        ),
    )
    nodes = gpd.GeoDataFrame(
        [
            {"id_node": 0, "is_inlet": True, "is_outlet": False, "geometry": Point(0, 200)},
            {"id_node": 1, "is_inlet": False, "is_outlet": True, "geometry": Point(0, -195)},
        ],
        crs="EPSG:3857",
    )
    links = gpd.GeoDataFrame(
        [{"id_link": 1, "id_nodes": "0, 1", "geometry": LineString([(0, 200), (0, -195)])}],
        crs=nodes.crs,
    )
    cuts = gpd.GeoDataFrame(
        [
            {"role": "upstream", "geometry": LineString([(-200, 0), (200, 0)])},
            {"role": "downstream", "geometry": LineString([(-20, -200), (20, -200)])},
        ],
        crs=nodes.crs,
    )

    bindings = _boundary_bindings(
        links,
        nodes,
        cuts,
        contract,
        tolerance=50.0,
        global_boundary_roles={"upstream": 10, "downstream": 13},
    ).set_index("cut_role")

    assert bindings.loc["upstream", "cut_distance_m"] == 200.0
    assert bindings.loc["upstream", "binding_method"] == (
        "unique_global_boundary_role_extended_tolerance"
    )
    assert bindings.loc["downstream", "binding_method"] == (
        "cut_proximity_and_raw_boundary_role"
    )


def test_internal_boundary_does_not_use_extended_role_tolerance() -> None:
    contract = LocalPatchContract(
        example_id="synthetic",
        selection_type="unit",
        selected_unit_ids=(1,),
        domain_type="closed_unit",
        core_link_ids=(2,),
        upstream_collar=LocalPatchCollar(
            role="upstream", base_link_id=1, base_upstream_node_id=10,
            base_downstream_node_id=11, domain_boundary_node_id=11,
            outer_splice_anchor_node_id=10,
        ),
        downstream_collar=LocalPatchCollar(
            role="downstream", base_link_id=3, base_upstream_node_id=12,
            base_downstream_node_id=13, domain_boundary_node_id=12,
            outer_splice_anchor_node_id=13,
        ),
    )
    nodes = gpd.GeoDataFrame(
        [
            {"id_node": 0, "is_inlet": True, "is_outlet": False, "geometry": Point(0, 200)},
            {"id_node": 1, "is_inlet": False, "is_outlet": True, "geometry": Point(0, -195)},
        ],
        crs="EPSG:3857",
    )
    links = gpd.GeoDataFrame(
        [{"id_link": 1, "id_nodes": "0, 1", "geometry": LineString([(0, 200), (0, -195)])}],
        crs=nodes.crs,
    )
    cuts = gpd.GeoDataFrame(
        [
            {"role": "upstream", "geometry": LineString([(-200, 0), (200, 0)])},
            {"role": "downstream", "geometry": LineString([(-20, -200), (20, -200)])},
        ],
        crs=nodes.crs,
    )

    with pytest.raises(ValueError, match="Upstream cut has 0 local terminal candidates"):
        _boundary_bindings(
            links,
            nodes,
            cuts,
            contract,
            tolerance=50.0,
            global_boundary_roles={"upstream": 999, "downstream": 13},
        )


def test_phase6_inherits_unchanged_and_measures_only_inserted_link(tmp_path) -> None:
    # Reuse the complete Phase-5 fixture above by running it as a compact local
    # setup with the same deterministic straight-channel inputs.
    water = np.zeros((100, 100), dtype=np.uint8)
    water[:, 35:65] = 1
    labels = np.zeros_like(water, dtype=np.int32)
    labels[:, :35], labels[:, 65:] = 1, 2
    mask_path = _raster(tmp_path / "mask.tif", water, "uint8")
    labels_path = _raster(tmp_path / "labels.tif", labels, "int32")
    origin_path = _raster(tmp_path / "origin.tif", water, "uint16")

    nodes = gpd.GeoDataFrame(
        [
            {"id_node": 1, "is_inlet": True, "is_outlet": False, "geometry": Point(50, 90)},
            {"id_node": 2, "is_inlet": False, "is_outlet": False, "geometry": Point(50, 65)},
            {"id_node": 3, "is_inlet": False, "is_outlet": False, "geometry": Point(50, 35)},
            {"id_node": 4, "is_inlet": False, "is_outlet": True, "geometry": Point(50, 10)},
        ],
        crs="EPSG:3857",
    )
    parent = gpd.GeoDataFrame(
        [
            {"id_link": 1, "id_us_node": 1, "id_ds_node": 2, "id_nodes": "1, 2", "len": 25.0, "len_adj": 24.0, "wid": 10.0, "wid_adj": 10.0, "wid_med": 10.0, "wid_adj_wet": 10.0, "wid_adj_routing": 11.0, "geometry": LineString([(50, 90), (50, 65)])},
            {"id_link": 2, "id_us_node": 2, "id_ds_node": 3, "id_nodes": "2, 3", "len": 30.0, "len_adj": 29.0, "wid": 20.0, "wid_adj": 20.0, "wid_med": 20.0, "wid_adj_wet": 20.0, "wid_adj_routing": 22.0, "geometry": LineString([(50, 65), (50, 35)])},
            {"id_link": 3, "id_us_node": 3, "id_ds_node": 4, "id_nodes": "3, 4", "len": 25.0, "len_adj": 24.0, "wid": 30.0, "wid_adj": 30.0, "wid_med": 30.0, "wid_adj_wet": 30.0, "wid_adj_routing": 33.0, "geometry": LineString([(50, 35), (50, 10)])},
        ],
        crs="EPSG:3857",
    )
    child = parent.loc[parent["id_link"].isin([1, 3])].copy()
    inserted = parent.loc[parent["id_link"].eq(2)].copy()
    inserted["id_link"] = 4
    inserted["geometry"] = [LineString([(50, 65), (60, 50), (50, 35)])]
    inserted["len"] = np.nan
    inserted["len_adj"] = np.nan
    inserted["wid_adj_wet"] = np.nan
    inserted["wid_adj_routing"] = np.nan
    child = gpd.GeoDataFrame(pd.concat([child, inserted], ignore_index=True), crs=parent.crs)

    merged_links_path = tmp_path / "merged_links.gpkg"
    merged_nodes_path = tmp_path / "merged_nodes.gpkg"
    parent_path = tmp_path / "parent_width_links.gpkg"
    child.to_file(merged_links_path, driver="GPKG")
    nodes.to_file(merged_nodes_path, driver="GPKG")
    parent.to_file(parent_path, driver="GPKG")
    parent_samples_path = tmp_path / "parent_samples.csv"
    pd.DataFrame(
        [{"id_link": link_id, "sample_index": 0, "width_total": link_id * 10.0}
         for link_id in (1, 2, 3)]
    ).to_csv(parent_samples_path, index=False)

    merge_result = type("Merge", (), {
        "status": "complete_conservative_merge",
        "output_paths": {"merged_links": str(merged_links_path), "merged_nodes": str(merged_nodes_path)},
        "summary": {"match_tolerance_m": 1.0},
    })()
    variant_result = LocalPatchVariantResult(
        status="complete_raw_local_variant",
        selected_unit_ids=(1,),
        output_paths={
            "local_collapsed_mask": str(mask_path),
            "local_base_mask": str(mask_path),
            "local_origin_mask": str(origin_path),
            "boundary_component_labels": str(labels_path),
        },
        collapse_summary={},
    )
    measured_calls = []

    def fake_width_runner(changed_links, **kwargs):
        measured_calls.append(sorted(changed_links["id_link"].astype(int).tolist()))
        enriched = changed_links.copy()
        enriched["wid_adj_total"] = 44.0
        enriched["wid_adj_wet"] = 40.0
        enriched["wid_adj_dry"] = 4.0
        samples = pd.DataFrame([
            {"id_link": 4, "sample_index": 0, "width_total": 44.0, "width_wet": 40.0},
            {"id_link": 4, "sample_index": 1, "width_total": 44.0, "width_wet": 40.0},
        ])
        return enriched, samples

    def fake_routing_runner(enriched, samples, child_nodes, **kwargs):
        assert kwargs["evaluation_link_ids"] == [4]
        routed = enriched.copy()
        routed.loc[routed["id_link"].astype(int).eq(4), "wid_adj_routing"] = 41.0
        return routed, pd.DataFrame({"id_link": routed["id_link"]}), {"status": "fake_complete"}

    products = materialize_local_patch_width_length_products(
        merge_result=merge_result,
        variant_result=variant_result,
        parent_width_links_path=parent_path,
        parent_nodes_path=merged_nodes_path,
        parent_width_samples_path=parent_samples_path,
        output_dir=tmp_path / "products",
        example_id="synthetic",
        variant_id="unit_1",
        width_runner=fake_width_runner,
        routing_width_runner=fake_routing_runner,
    )
    output = gpd.read_file(products.output_paths["links_with_width_families"]).set_index("id_link")
    assert measured_calls == [[4]]
    assert output.loc[1, "wid_adj_wet"] == 10.0
    assert output.loc[1, "wid_adj_routing"] == 11.0
    assert output.loc[1, "len"] == 25.0
    assert output.loc[1, "len_adj"] == 24.0
    assert output.loc[1, "product_source_parent_link_id"] == 1
    assert output.loc[3, "wid_adj_wet"] == 30.0
    assert output.loc[4, "wid_adj_wet"] == 40.0
    assert output.loc[4, "wid_adj_routing"] == 41.0
    assert output.loc[4, "len"] == output.loc[4].geometry.length
    assert output.loc[1, "length_product_status"] == "inherited_unchanged_1to1"
    assert output.loc[4, "length_product_status"] == "recalculated_local_geometry"
    assert products.summary["inherited_link_count"] == 2
    assert products.summary["recalculated_link_ids"] == [4]
    assert products.summary["length_product_inheritance"]["inherited_fields"] == [
        "len",
        "len_adj",
    ]

    # Production Phase 5 supplies the final shared-anchored graph and its
    # canonical post-anchor lineage. Width scope must remain identical and may
    # not fall back to the pre-anchor splice audit.
    anchored_lineage = _local_link_lineage(
        child,
        nodes,
        parent,
        nodes,
        match_tolerance=1.25,
    )
    anchored_lineage_path = tmp_path / "anchored_link_lineage.csv"
    anchored_lineage.to_csv(anchored_lineage_path, index=False)
    anchored_result = type(
        "Anchored",
        (),
        {
            "status": "complete_shared_anchored_graph",
            "output_paths": {
                "anchored_links": str(merged_links_path),
                "anchored_nodes": str(merged_nodes_path),
                "anchored_link_lineage": str(anchored_lineage_path),
            },
            "summary": {},
        },
    )()
    measured_calls.clear()
    anchored_products = materialize_local_patch_width_length_products(
        merge_result=anchored_result,
        variant_result=variant_result,
        parent_width_links_path=parent_path,
        parent_nodes_path=merged_nodes_path,
        parent_width_samples_path=parent_samples_path,
        output_dir=tmp_path / "anchored_products",
        example_id="synthetic",
        variant_id="unit_1",
        width_runner=fake_width_runner,
        routing_width_runner=fake_routing_runner,
    )
    anchored_output = gpd.read_file(
        anchored_products.output_paths["links_with_width_families"]
    ).set_index("id_link")
    assert measured_calls == [[4]]
    assert anchored_products.summary["graph_product_stage"] == (
        "shared_scientific_anchored"
    )
    assert anchored_products.summary["recalculated_link_ids"] == [4]
    assert anchored_output.loc[1, "wid_adj_wet"] == 10.0
    assert anchored_output.loc[4, "wid_adj_wet"] == 40.0
