from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from hierarchy_level_definition.metrics import BubbleEquivalentResistanceConfig, compute_K_manning
import network_variants.equivalent_resistance_experiment as equivalent_resistance_experiment_module
from network_variants.equivalent_resistance_experiment import (
    load_manifest_equivalent_resistance_config,
    main,
    recompute_state_hierarchy_equivalent_resistance,
    run_equivalent_resistance_experiment,
)


def _write_links(path: Path, rows: list[dict[str, object]]) -> None:
    frame = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:3857")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_file(path, driver="GPKG")


def _write_nodes(path: Path, rows: list[dict[str, object]]) -> None:
    frame = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:3857")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_file(path, driver="GPKG")


def _build_synthetic_experiment(tmp_path: Path) -> Path:
    experiment_dir = tmp_path / "experiment"
    base_state_dir = experiment_dir / "states" / "S000_base"
    child_state_dir = experiment_dir / "states" / "S001_unit_99"

    base_variant_dir = base_state_dir / "variant"
    child_variant_dir = child_state_dir / "variant"
    base_hierarchy_dir = base_state_dir / "hierarchy"
    child_hierarchy_dir = child_state_dir / "hierarchy"

    for directory in (
        base_variant_dir / "summary",
        base_variant_dir / "directed",
        base_variant_dir / "widths",
        child_variant_dir / "matching",
        child_variant_dir / "directed",
        child_variant_dir / "widths",
        base_hierarchy_dir,
        child_hierarchy_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    base_links_path = base_variant_dir / "directed" / "synthetic__S000_base_directed_links.gpkg"
    base_nodes_path = base_variant_dir / "directed" / "synthetic__S000_base_directed_nodes.gpkg"
    child_links_path = child_variant_dir / "directed" / "synthetic__S001_unit_99_directed_links.gpkg"
    child_nodes_path = child_variant_dir / "directed" / "synthetic__S001_unit_99_directed_nodes.gpkg"

    _write_nodes(
        base_nodes_path,
        [
            {"id_node": 1, "geometry": Point(0.0, 0.0)},
            {"id_node": 2, "geometry": Point(1.0, 0.0)},
            {"id_node": 3, "geometry": Point(2.0, 0.0)},
            {"id_node": 4, "geometry": Point(3.0, 0.0)},
        ],
    )
    _write_links(
        base_links_path,
        [
            {"id_link": 10, "id_us_node": 1, "id_ds_node": 2, "len": 1.0, "wid_adj": 10.0, "geometry": LineString([(0.0, 0.0), (1.0, 0.0)])},
            {"id_link": 11, "id_us_node": 2, "id_ds_node": 3, "len": 1.0, "wid_adj": 8.0, "geometry": LineString([(1.0, 0.0), (2.0, 0.05)])},
            {"id_link": 12, "id_us_node": 2, "id_ds_node": 3, "len": 1.0, "wid_adj": 4.0, "geometry": LineString([(1.0, 0.0), (2.0, -0.05)])},
            {"id_link": 13, "id_us_node": 3, "id_ds_node": 4, "len": 1.0, "wid_adj": 10.0, "geometry": LineString([(2.0, 0.0), (3.0, 0.0)])},
        ],
    )

    _write_nodes(
        child_nodes_path,
        [
            {"id_node": 100, "geometry": Point(0.0, 0.0)},
            {"id_node": 101, "geometry": Point(3.0, 0.0)},
        ],
    )
    _write_links(
        child_links_path,
        [
            {"id_link": 20, "id_us_node": 100, "id_ds_node": 101, "len": 3.0, "wid_adj": 6.0, "geometry": LineString([(0.0, 0.0), (3.0, 0.0)])},
        ],
    )

    pd.DataFrame(
        [
            {
                "unit_id": 99,
                "bifurcation": 2,
                "confluence": 3,
                "compound_bubble_id": "",
                "compound_bubble_role": "standalone",
            }
        ]
    ).to_csv(base_hierarchy_dir / "unit_summary.csv", index=False)

    pd.DataFrame(
        [
            {"unit_id": 99, "id_link": 11, "path_ids": "1", "n_paths_through_link": 1},
            {"unit_id": 99, "id_link": 12, "path_ids": "2", "n_paths_through_link": 1},
        ]
    ).to_csv(base_variant_dir / "summary" / "unit_link_membership.csv", index=False)

    pd.DataFrame(
        [
            {
                "id_link": 20,
                "lineage_type": "collapsed_many_to_one",
                "matched_parent_link_ids": "10,11,12,13",
                "matched_parent_node_path": "1,2,3,4",
            }
        ]
    ).to_csv(child_variant_dir / "matching" / "link_lineage.csv", index=False)
    pd.DataFrame(
        [
            {"child_id_node": 100, "matched_parent_node_id": 1, "match_within_tolerance": True, "parent_node_order": 1},
            {"child_id_node": 101, "matched_parent_node_id": 4, "match_within_tolerance": True, "parent_node_order": 4},
        ]
    ).to_csv(child_variant_dir / "matching" / "node_match.csv", index=False)

    _write_links(
        base_variant_dir / "widths" / "links_with_width_families.gpkg",
        [
            {"id_link": 10, "id_us_node": 1, "id_ds_node": 2, "len": 1.0, "wid_adj": 10.0, "wid_adj_wet": 10.0, "geometry": LineString([(0.0, 0.0), (1.0, 0.0)])},
            {"id_link": 11, "id_us_node": 2, "id_ds_node": 3, "len": 1.0, "wid_adj": 8.0, "wid_adj_wet": 8.0, "geometry": LineString([(1.0, 0.0), (2.0, 0.05)])},
            {"id_link": 12, "id_us_node": 2, "id_ds_node": 3, "len": 1.0, "wid_adj": 4.0, "wid_adj_wet": 4.0, "geometry": LineString([(1.0, 0.0), (2.0, -0.05)])},
            {"id_link": 13, "id_us_node": 3, "id_ds_node": 4, "len": 1.0, "wid_adj": 10.0, "wid_adj_wet": 10.0, "geometry": LineString([(2.0, 0.0), (3.0, 0.0)])},
        ],
    )
    _write_links(
        child_variant_dir / "widths" / "links_with_width_families.gpkg",
        [
            {"id_link": 20, "id_us_node": 100, "id_ds_node": 101, "len": 3.0, "wid_adj": 6.0, "wid_adj_wet": 6.0, "geometry": LineString([(0.0, 0.0), (3.0, 0.0)])},
        ],
    )

    pd.DataFrame(
        [
            {
                "state_id": "S000_base",
                "state_role": "base",
                "status": "complete",
                "mode": "independent-units",
                "selection_type": "",
                "selected_unit_ids": "",
                "compound_bubble_id": "",
                "bubble_root_unit_id": "",
                "state_dir": str(base_state_dir.resolve()),
                "variant_output_dir": str(base_variant_dir.resolve()),
                "hierarchy_output_dir": str(base_hierarchy_dir.resolve()),
                "directed_links_path": str(base_links_path.resolve()),
                "directed_nodes_path": str(base_nodes_path.resolve()),
            },
            {
                "state_id": "S001_unit_99",
                "state_role": "derived",
                "status": "complete",
                "mode": "independent-units",
                "selection_type": "unit",
                "selected_unit_ids": "99",
                "compound_bubble_id": "",
                "bubble_root_unit_id": "",
                "state_dir": str(child_state_dir.resolve()),
                "variant_output_dir": str(child_variant_dir.resolve()),
                "hierarchy_output_dir": str(child_hierarchy_dir.resolve()),
                "directed_links_path": str(child_links_path.resolve()),
                "directed_nodes_path": str(child_nodes_path.resolve()),
            },
        ]
    ).to_csv(experiment_dir / "state_registry.csv", index=False)
    return experiment_dir


def _build_synthetic_unstable_node_experiment(tmp_path: Path) -> Path:
    experiment_dir = tmp_path / "experiment_unstable"
    base_state_dir = experiment_dir / "states" / "S000_base"
    child_state_dir = experiment_dir / "states" / "S001_unit_99"

    base_variant_dir = base_state_dir / "variant"
    child_variant_dir = child_state_dir / "variant"
    base_hierarchy_dir = base_state_dir / "hierarchy"
    child_hierarchy_dir = child_state_dir / "hierarchy"

    for directory in (
        base_variant_dir / "summary",
        base_variant_dir / "directed",
        child_variant_dir / "matching",
        child_variant_dir / "directed",
        base_hierarchy_dir,
        child_hierarchy_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    base_links_path = base_variant_dir / "directed" / "synthetic__S000_base_directed_links.gpkg"
    base_nodes_path = base_variant_dir / "directed" / "synthetic__S000_base_directed_nodes.gpkg"
    child_links_path = child_variant_dir / "directed" / "synthetic__S001_unit_99_directed_links.gpkg"
    child_nodes_path = child_variant_dir / "directed" / "synthetic__S001_unit_99_directed_nodes.gpkg"

    _write_nodes(
        base_nodes_path,
        [
            {"id_node": 1, "geometry": Point(0.0, 0.0)},
            {"id_node": 2, "geometry": Point(1.0, 0.0)},
            {"id_node": 3, "geometry": Point(2.0, 0.0)},
            {"id_node": 4, "geometry": Point(3.0, 0.0)},
        ],
    )
    _write_links(
        base_links_path,
        [
            {"id_link": 10, "id_us_node": 1, "id_ds_node": 2, "len": 1.0, "wid_adj": 10.0, "geometry": LineString([(0.0, 0.0), (1.0, 0.0)])},
            {"id_link": 11, "id_us_node": 2, "id_ds_node": 3, "len": 1.0, "wid_adj": 8.0, "geometry": LineString([(1.0, 0.0), (2.0, 0.0)])},
            {"id_link": 12, "id_us_node": 3, "id_ds_node": 4, "len": 1.0, "wid_adj": 10.0, "geometry": LineString([(2.0, 0.0), (3.0, 0.0)])},
        ],
    )

    _write_nodes(
        child_nodes_path,
        [
            {"id_node": 100, "geometry": Point(0.0, 0.0)},
            {"id_node": 101, "geometry": Point(1.5, 0.0)},
            {"id_node": 102, "geometry": Point(3.0, 0.0)},
        ],
    )
    _write_links(
        child_links_path,
        [
            {"id_link": 20, "id_us_node": 100, "id_ds_node": 101, "len": 1.5, "wid_adj": 8.0, "geometry": LineString([(0.0, 0.0), (1.5, 0.0)])},
            {"id_link": 21, "id_us_node": 101, "id_ds_node": 102, "len": 1.5, "wid_adj": 8.0, "geometry": LineString([(1.5, 0.0), (3.0, 0.0)])},
        ],
    )

    pd.DataFrame(
        [
            {
                "unit_id": 99,
                "bifurcation": 2,
                "confluence": 3,
                "compound_bubble_id": "",
                "compound_bubble_role": "standalone",
            }
        ]
    ).to_csv(base_hierarchy_dir / "unit_summary.csv", index=False)

    pd.DataFrame(
        [
            {"unit_id": 99, "id_link": 11, "path_ids": "1", "n_paths_through_link": 1},
        ]
    ).to_csv(base_variant_dir / "summary" / "unit_link_membership.csv", index=False)

    pd.DataFrame(
        [
            {
                "id_link": 20,
                "lineage_type": "unchanged_1to1",
                "matched_parent_link_ids": "10",
                "matched_parent_node_path": "",
            },
            {
                "id_link": 21,
                "lineage_type": "unchanged_1to1",
                "matched_parent_link_ids": "12",
                "matched_parent_node_path": "",
            },
        ]
    ).to_csv(child_variant_dir / "matching" / "link_lineage.csv", index=False)
    pd.DataFrame(
        [
            {"child_id_node": 100, "matched_parent_node_id": 1, "match_within_tolerance": True, "parent_node_order": 1},
            {"child_id_node": 101, "matched_parent_node_id": "", "match_within_tolerance": False, "parent_node_order": ""},
            {"child_id_node": 102, "matched_parent_node_id": 4, "match_within_tolerance": True, "parent_node_order": 4},
        ]
    ).to_csv(child_variant_dir / "matching" / "node_match.csv", index=False)

    pd.DataFrame(
        [
            {
                "state_id": "S000_base",
                "state_role": "base",
                "status": "complete",
                "mode": "independent-units",
                "selection_type": "",
                "selected_unit_ids": "",
                "compound_bubble_id": "",
                "bubble_root_unit_id": "",
                "state_dir": str(base_state_dir.resolve()),
                "variant_output_dir": str(base_variant_dir.resolve()),
                "hierarchy_output_dir": str(base_hierarchy_dir.resolve()),
                "directed_links_path": str(base_links_path.resolve()),
                "directed_nodes_path": str(base_nodes_path.resolve()),
            },
            {
                "state_id": "S001_unit_99",
                "state_role": "derived",
                "status": "complete",
                "mode": "independent-units",
                "selection_type": "unit",
                "selected_unit_ids": "99",
                "compound_bubble_id": "",
                "bubble_root_unit_id": "",
                "state_dir": str(child_state_dir.resolve()),
                "variant_output_dir": str(child_variant_dir.resolve()),
                "hierarchy_output_dir": str(child_hierarchy_dir.resolve()),
                "directed_links_path": str(child_links_path.resolve()),
                "directed_nodes_path": str(child_nodes_path.resolve()),
            },
        ]
    ).to_csv(experiment_dir / "state_registry.csv", index=False)
    return experiment_dir


def test_run_equivalent_resistance_experiment_detects_expanded_boundaries(tmp_path: Path) -> None:
    experiment_dir = _build_synthetic_experiment(tmp_path)

    results = run_equivalent_resistance_experiment(
        experiment_dir,
        equivalent_resistance_config=BubbleEquivalentResistanceConfig(
            resistance_law="manning",
            geometry_method="wide",
            kb_mode="fixed",
            kb_value=20.0,
            default_manning_n=0.035,
        ),
    )

    assert len(results.base_nominal_targets) == 1
    nominal = results.base_nominal_targets.iloc[0]
    assert nominal["nominal_upstream_node"] == 2
    assert nominal["nominal_downstream_node"] == 3
    assert nominal["nominal_base_link_ids"] == "11,12"
    assert bool(nominal["solver_success_nominal"]) is True

    assert len(results.child_realized_objects) == 1
    child = results.child_realized_objects.iloc[0]
    assert child["child_object_type"] == "single_link"
    assert child["matched_parent_node_path"] == "1,2,3,4"
    assert child["boundary_relation"] == "expanded_both"
    assert bool(child["boundaries_match_nominal_target"]) is False

    assert len(results.matched_base_corridors) == 1
    matched = results.matched_base_corridors.iloc[0]
    assert matched["matched_base_upstream_node"] == 1
    assert matched["matched_base_downstream_node"] == 4
    assert matched["matched_base_node_path"] == "1,2,3,4"
    assert bool(matched["solver_success_matched"]) is True
    assert matched["K_eq_matched_base"] > nominal["K_eq_nominal_base"]

    assert len(results.comparison) == 1
    comparison = results.comparison.iloc[0]
    assert bool(comparison["comparison_valid_for_nominal"]) is False
    assert bool(comparison["comparison_valid_for_matched"]) is True
    assert pd.isna(comparison["Lambda_nominal"])
    assert comparison["Lambda_matched"] > 0

    output_dir = experiment_dir / "equivalent_resistance"
    assert output_dir.exists()
    assert (output_dir / "base_nominal_targets.csv").exists()
    assert (output_dir / "child_realized_objects.csv").exists()
    assert (output_dir / "matched_base_corridors.csv").exists()
    assert (output_dir / "equivalent_resistance_comparison.csv").exists()
    assert (output_dir / "equivalent_resistance_manifest.json").exists()


def test_nominal_boundaries_win_when_only_internal_target_nodes_disappear(tmp_path: Path) -> None:
    experiment_dir = _build_synthetic_experiment(tmp_path)
    base_state = experiment_dir / "states" / "S000_base"

    unit_summary_path = base_state / "hierarchy" / "unit_summary.csv"
    unit_summary = pd.read_csv(unit_summary_path)
    unit_summary.loc[unit_summary["unit_id"].eq(99), "bifurcation"] = 1
    unit_summary.loc[unit_summary["unit_id"].eq(99), "confluence"] = 4
    unit_summary.to_csv(unit_summary_path, index=False)

    pd.DataFrame(
        [
            {"unit_id": 99, "id_link": link_id, "path_ids": "1", "n_paths_through_link": 1}
            for link_id in (10, 11, 12, 13)
        ]
    ).to_csv(
        base_state / "variant" / "summary" / "unit_link_membership.csv",
        index=False,
    )

    results = run_equivalent_resistance_experiment(
        experiment_dir,
        equivalent_resistance_config=BubbleEquivalentResistanceConfig(
            resistance_law="manning",
            geometry_method="wide",
            kb_mode="fixed",
            kb_value=20.0,
            default_manning_n=0.035,
        ),
    )

    child = results.child_realized_objects.iloc[0]
    assert int(child["child_boundary_match_upstream_parent_node"]) == 1
    assert int(child["child_boundary_match_downstream_parent_node"]) == 4
    assert int(child["child_upstream_node"]) == 100
    assert int(child["child_downstream_node"]) == 101
    assert child["matched_parent_node_path"] == "1,2,3,4"
    assert bool(child["boundaries_match_nominal_target"]) is True
    assert child["boundary_relation"] == "exact_match"


def test_run_equivalent_resistance_experiment_recovers_component_from_unstable_node_match(tmp_path: Path) -> None:
    experiment_dir = _build_synthetic_unstable_node_experiment(tmp_path)

    results = run_equivalent_resistance_experiment(
        experiment_dir,
        equivalent_resistance_config=BubbleEquivalentResistanceConfig(
            resistance_law="manning",
            geometry_method="wide",
            kb_mode="fixed",
            kb_value=20.0,
            default_manning_n=0.035,
        ),
    )

    assert len(results.base_nominal_targets) == 1
    assert len(results.child_realized_objects) == 1
    child = results.child_realized_objects.iloc[0]
    assert child["child_upstream_node"] == 100
    assert child["child_downstream_node"] == 102
    assert child["child_link_ids"] == "20,21"
    assert child["boundary_relation"] == "expanded_both"
    assert bool(child["solver_success_variant"]) is True

    assert len(results.matched_base_corridors) == 1
    matched = results.matched_base_corridors.iloc[0]
    assert matched["matched_base_upstream_node"] == 1
    assert matched["matched_base_downstream_node"] == 4
    assert matched["matched_base_link_ids"] == "10,11,12"
    assert bool(matched["solver_success_matched"]) is True

    assert len(results.comparison) == 1
    comparison = results.comparison.iloc[0]
    assert bool(comparison["comparison_valid_for_matched"]) is True
    assert comparison["Lambda_matched"] > 0


def test_selected_hydraulic_domain_replaces_isolated_member_aggregate(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "13_indep__anchored"
    state_id = "S037_unit_48"
    domain_dir = experiment_dir / "states" / state_id / "equivalent_resistance"
    domain_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"domain_role": "parent", "id_link": 181, "id_us_node": 95, "id_ds_node": 120},
            {"domain_role": "parent", "id_link": 185, "id_us_node": 128, "id_ds_node": 120},
            {"domain_role": "child", "id_link": 900, "id_us_node": 112, "id_ds_node": 84},
        ]
    ).to_csv(domain_dir / "selected_hydraulic_domain_links.csv", index=False)
    pd.DataFrame(
        [
            {
                "transition_id": "T037",
                "child_state_id": state_id,
                "selected_domain_status": "complete",
                "parent_solver_success": True,
                "child_solver_success": True,
                "selected_unit_ids": "48",
                "hydraulic_domain_type": "containing_compound_context",
                "parent_upstream_boundary_node": 112,
                "parent_downstream_boundary_node": 84,
                "child_upstream_boundary_node": 112,
                "child_downstream_boundary_node": 84,
                "parent_solver_method": "energy_sparse",
                "child_solver_method": "energy_sparse",
                "K_before": 4.934218e-7,
                "K_after": 5.345743e-7,
                "G_before": 1.0 / 4.934218e-7,
                "G_after": 1.0 / 5.345743e-7,
                "Q_unit_before": 1423.6,
                "Q_unit_after": 1367.7,
                "Lambda_K": 1.083402,
                "Lambda_G": 0.923018,
            }
        ]
    ).to_csv(experiment_dir / "equivalent_transition_registry.csv", index=False)

    nominal = pd.DataFrame(
        [{"state_id": state_id, "target_type": "unit", "target_id": 48, "nominal_upstream_node": 120, "nominal_downstream_node": 123, "K_eq_nominal_base": 9.0}]
    )
    child = pd.DataFrame(
        [{"state_id": state_id, "child_object_id": f"{state_id}__obj_001", "child_link_ids": "175,176", "child_node_ids": "120,123", "K_variant_realized": float("nan"), "G_variant_realized": float("nan"), "Q_unit_variant": float("nan")}]
    )
    matched = pd.DataFrame(
        [{"state_id": state_id, "child_object_id": f"{state_id}__obj_001", "matched_base_link_ids": "175,176", "matched_base_node_ids": "120,123", "K_eq_matched_base": float("nan"), "G_eq_matched_base": float("nan"), "Q_unit_matched_base": float("nan")}]
    )
    comparison = pd.DataFrame(
        [{"state_id": state_id, "child_object_id": f"{state_id}__obj_001", "K_eq_matched_base": float("nan"), "K_variant_realized": float("nan"), "Lambda_matched": float("nan"), "Gamma_matched": float("nan"), "boundary_relation": "ambiguous"}]
    )

    child, matched, comparison, count = equivalent_resistance_experiment_module._apply_selected_hydraulic_domain_aggregates(
        experiment_path=experiment_dir,
        nominal_targets=nominal,
        child_realized_objects=child,
        matched_base_corridors=matched,
        comparison=comparison,
    )

    assert count == 1
    assert child.iloc[0]["child_link_ids"] == "900"
    assert matched.iloc[0]["matched_base_link_ids"] == "181,185"
    assert comparison.iloc[0]["K_eq_matched_base"] == pytest.approx(4.934218e-7)
    assert comparison.iloc[0]["K_variant_realized"] == pytest.approx(5.345743e-7)
    assert comparison.iloc[0]["Lambda_matched"] == pytest.approx(1.083402)
    assert comparison.iloc[0]["boundary_relation"] == "selected_hydraulic_domain"
    assert "residual_corridor_Lambda_matched_diagnostic" in comparison.columns


def test_main_passes_equivalent_length_field_into_equivalent_resistance_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_equivalent_resistance_experiment(experiment_dir: str | Path, **kwargs: object):
        calls.append({"experiment_dir": experiment_dir, **dict(kwargs)})
        output_dir = Path(kwargs.get("output_dir") or Path(experiment_dir) / "equivalent_resistance")
        return equivalent_resistance_experiment_module.EquivalentResistanceExperimentOutputs(
            base_nominal_targets=pd.DataFrame(),
            child_realized_objects=pd.DataFrame(),
            matched_base_corridors=pd.DataFrame(),
            comparison=pd.DataFrame(),
            output_dir=output_dir,
            manifest={},
        )

    monkeypatch.setattr(
        equivalent_resistance_experiment_module,
        "run_equivalent_resistance_experiment",
        fake_run_equivalent_resistance_experiment,
    )

    exit_code = main(
        [
            str(tmp_path / "experiment"),
            "--equivalent-length-field",
            "len",
        ]
    )

    assert exit_code == 0
    assert len(calls) == 1
    config = calls[0]["equivalent_resistance_config"]
    assert isinstance(config, BubbleEquivalentResistanceConfig)
    assert config.length_field_candidates == ("len", "len_adj", "link_length_m")


def test_recompute_state_hierarchy_uses_manifest_config_and_updates_only_resistance_artifacts(
    tmp_path: Path,
) -> None:
    experiment_dir = tmp_path / "experiment"
    hierarchy_dir = experiment_dir / "states" / "S000_base" / "hierarchy"
    hierarchy_dir.mkdir(parents=True)
    links_path = experiment_dir / "links.gpkg"
    nodes_path = experiment_dir / "nodes.gpkg"
    links_path.touch()
    nodes_path.touch()
    config_payload = {
        "resistance_law": "chezy",
        "geometry_method": "wide",
        "kb_mode": "based_reference_section",
        "kb_value": 20.0,
        "reference_kb_summary": {
            "kb_value": 123.0,
            "kb_source_method": "based_reference_section",
        },
        "length_field_candidates": ["len", "len_adj", "link_length_m"],
        "default_chezy_cz": 22.0,
    }
    (experiment_dir / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "options": {
                    "hierarchy": {
                        "max_path_cutoff": 77,
                        "max_paths": 888,
                        "equivalent_resistance_config": config_payload,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame.from_records(
        [
            {
                "state_id": "S000_base",
                "status": "complete",
                "hierarchy_status": "complete",
                "analysis_links_path": str(links_path),
                "directed_nodes_path": str(nodes_path),
                "hierarchy_output_dir": str(hierarchy_dir),
                "n_equivalent_bubbles": 1,
                "n_equivalent_bubbles_solver_success": 0,
            }
        ]
    ).to_csv(experiment_dir / "state_registry.csv", index=False)
    pd.DataFrame.from_records(
        [
            {
                "compound_bubble_id": 1,
                "upstream_boundary_node": 10,
                "downstream_boundary_node": 11,
                "solver_success": False,
                "K_eq": float("nan"),
            }
        ]
    ).to_csv(hierarchy_dir / "bubble_equivalent_metrics.csv", index=False)
    (hierarchy_dir / "unit_workflow_manifest.json").write_text(
        json.dumps({"unrelated_metric": "preserved"}),
        encoding="utf-8",
    )

    calls: list[dict[str, object]] = []

    def fake_metrics_runner(*args: object, **kwargs: object):
        calls.append({"args": args, **kwargs})
        metrics = pd.DataFrame.from_records(
            [
                {
                    "compound_bubble_id": 1,
                    "upstream_boundary_node": 10,
                    "downstream_boundary_node": 12,
                    "solver_success": True,
                    "solver_initialization": "neutral_internal_heads",
                    "neutral_fallback_attempted": True,
                    "neutral_fallback_solver_success": True,
                    "K_eq": 2.5,
                    "G_eq": 0.4,
                    "Q_unit": 0.4,
                    "residual_relative": 0.0,
                }
            ]
        )
        metrics.attrs["resolved_kb_context"] = {
            "kb_value": 123.0,
            "kb_mode": "based_reference_section",
        }
        edges = pd.DataFrame.from_records([{"compound_bubble_id": 1, "edge_id": 5}])
        heads = pd.DataFrame.from_records([{"compound_bubble_id": 1, "node_id": 10}])
        return metrics, edges, heads

    results = recompute_state_hierarchy_equivalent_resistance(
        experiment_dir,
        metrics_runner=fake_metrics_runner,
    )

    assert len(calls) == 1
    assert calls[0]["max_path_cutoff"] == 77
    assert calls[0]["max_paths"] == 888
    config = calls[0]["config"]
    assert isinstance(config, BubbleEquivalentResistanceConfig)
    assert config.resistance_law == "chezy"
    assert config.reference_kb_summary["kb_value"] == pytest.approx(123.0)
    assert config.length_field_candidates == ("len", "len_adj", "link_length_m")

    refreshed = pd.read_csv(hierarchy_dir / "bubble_equivalent_metrics.csv")
    assert bool(refreshed.loc[0, "solver_success"]) is True
    assert refreshed.loc[0, "downstream_boundary_node"] == 12
    assert bool(results.audit.loc[0, "boundary_changed"]) is True
    assert bool(results.audit.loc[0, "K_changed"]) is True
    registry = pd.read_csv(experiment_dir / "state_registry.csv")
    assert registry.loc[0, "n_equivalent_bubbles_solver_success"] == 1
    workflow_manifest = json.loads(
        (hierarchy_dir / "unit_workflow_manifest.json").read_text(encoding="utf-8")
    )
    assert workflow_manifest["unrelated_metric"] == "preserved"
    assert workflow_manifest["equivalent_resistance_recompute_scope"] == "state_hierarchy_only"
    assert results.audit_csv_path.exists()
    assert results.summary_json_path.exists()


def test_load_manifest_equivalent_resistance_config_requires_stored_config(
    tmp_path: Path,
) -> None:
    (tmp_path / "experiment_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="equivalent_resistance_config"):
        load_manifest_equivalent_resistance_config(tmp_path)


def test_state_hierarchy_recompute_audit_accepts_states_without_bubbles() -> None:
    audit = equivalent_resistance_experiment_module._audit_state_recompute(
        state_id="S999_empty",
        old_metrics=pd.DataFrame(),
        new_metrics=pd.DataFrame(),
    )
    assert audit.empty
    assert "boundary_changed" in audit.columns
    assert "K_changed" in audit.columns


def test_equivalent_resistance_prefers_width_family_link_source(tmp_path: Path) -> None:
    experiment_dir = _build_synthetic_experiment(tmp_path)

    child_widths_path = (
        experiment_dir
        / "states"
        / "S001_unit_99"
        / "variant"
        / "widths"
        / "links_with_width_families.gpkg"
    )
    _write_links(
        child_widths_path,
        [
            {"id_link": 20, "id_us_node": 100, "id_ds_node": 101, "len": 3.0, "wid_adj": 6.0, "wid_adj_wet": 12.0, "geometry": LineString([(0.0, 0.0), (3.0, 0.0)])},
        ],
    )

    results = run_equivalent_resistance_experiment(
        experiment_dir,
        equivalent_resistance_config=BubbleEquivalentResistanceConfig(
            resistance_law="manning",
            geometry_method="wide",
            kb_mode="fixed",
            kb_value=20.0,
            default_manning_n=0.035,
        ),
    )

    child = results.child_realized_objects.iloc[0]
    expected_k = compute_K_manning(3.0, 12.0, 0.035, 20.0)
    assert child["K_variant_realized"] == pytest.approx(expected_k)
