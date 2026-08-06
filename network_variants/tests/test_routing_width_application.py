from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

from network_variants import routing_width_application as application


class _AlwaysCorrectPipeline:
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.column_stack([np.zeros(len(features)), np.ones(len(features))])


def test_dry_corridor_provenance_preserves_measurements_and_flags_impossible_target() -> None:
    raw = pd.DataFrame(
        {
            "id_link": [1, 2, 3],
            "wid_adj_wet": [100.0, 100.0, 100.0],
            "wid_adj_total": [160.0, 110.0, np.nan],
            "wid_adj_dry": [60.0, 10.0, np.nan],
            "dry_fraction_adj": [0.375, np.nan, np.nan],
        }
    )
    audit = pd.DataFrame(
        {
            "id_link": [1, 2, 3],
            "wid_adj_wet": [100.0, 100.0, 100.0],
            "wid_adj_routing": [120.0, 125.0, 90.0],
        }
    )

    result = application._add_dry_corridor_provenance(audit, raw).set_index("id_link")

    assert result.loc[1, "dry_corridor_meaningful"]
    assert result.loc[1, "dry_corridor_fraction"] == pytest.approx(0.375)
    assert result.loc[1, "routing_to_wet_width_ratio"] == pytest.approx(1.2)
    assert result.loc[1, "routing_to_total_width_ratio"] == pytest.approx(0.75)
    assert not result.loc[1, "routing_corridor_consistency_flag"]
    assert result.loc[1, "wid_adj_wet"] == 100.0
    assert result.loc[1, "wid_adj_total"] == 160.0
    assert result.loc[1, "wid_adj_dry"] == 60.0

    assert result.loc[2, "dry_corridor_meaningful"]
    assert result.loc[2, "routing_exceeds_total_corridor"]
    assert result.loc[2, "routing_corridor_consistency_flag"]
    assert result.loc[2, "routing_corridor_consistency_reason"] == (
        "effective_routing_width_exceeds_total_corridor"
    )

    assert not result.loc[3, "dry_corridor_meaningful"]
    assert not result.loc[3, "routing_corridor_consistency_flag"]
    assert result.loc[3, "routing_corridor_consistency_reason"] == "total_corridor_width_unavailable"


def test_derives_missing_unit_membership_from_current_child_topology() -> None:
    links = gpd.GeoDataFrame(
        {
            "id_link": [1, 2, 3, 4, 5],
            "id_nodes": ["0,1", "1,2", "1,3", "2,4", "3,4"],
            "id_us_node": [0, 1, 1, 2, 3],
            "id_ds_node": [1, 2, 3, 4, 4],
            "len": [1.0] * 5,
            "wid_adj_wet": [1.0] * 5,
        },
        geometry=[
            LineString([(0, 0), (1, 0)]),
            LineString([(1, 0), (2, 1)]),
            LineString([(1, 0), (2, -1)]),
            LineString([(2, 1), (3, 0)]),
            LineString([(2, -1), (3, 0)]),
        ],
        crs="EPSG:32631",
    )
    nodes = gpd.GeoDataFrame(
        {"id_node": [0, 1, 2, 3, 4]},
        geometry=[Point(0, 0), Point(1, 0), Point(2, 1), Point(2, -1), Point(3, 0)],
        crs=links.crs,
    )

    result, source = application._ensure_current_unit_membership(links, nodes)

    assert source == "derived_from_current_child_topology"
    assert result.set_index("id_link").loc[[2, 3, 4, 5], "n_units"].eq(1).all()
    assert result.set_index("id_link").loc[1, "n_units"] == 0


def test_routing_application_preserves_measured_fields_and_adds_separate_routing_width(
    monkeypatch,
    tmp_path: Path,
) -> None:
    links = gpd.GeoDataFrame(
        {
            "id_link": [1, 2, 3],
            "id_us_node": [0, 1, 1],
            "id_ds_node": [1, 2, 3],
            "len": [1000.0, 50.0, 50.0],
            "n_units": [0, 1, 0],
            "wid_adj_wet": [100.0, 140.0, 90.0],
            "wid_adj_total": [100.0, 140.0, 90.0],
            "wid_adj_dry": [0.0, 0.0, 0.0],
            "wid_pix_total": ["100,100"] * 3,
            "n_invalid_width_samples": [0, 0, 0],
            "qc_mean_opposition_cos_trimmed": [-1.0, 0.8, -1.0],
        },
        geometry=[
            LineString([(0, 0), (10, 0)]),
            LineString([(10, 0), (20, 5)]),
            LineString([(10, 0), (20, -5)]),
        ],
        crs="EPSG:32631",
    )
    nodes = gpd.GeoDataFrame(
        {"id_node": [0, 1, 2, 3]},
        geometry=[Point(0, 0), Point(10, 0), Point(20, 5), Point(20, -5)],
        crs=links.crs,
    )
    samples = pd.DataFrame(
        [
            {
                "id_link": link_id,
                "sample_index": sample_index,
                "x": x,
                "y": y,
                "width_total": width,
                "width_wet": width,
                "width_invalid": False,
                "qc_opposition_cos": opposition,
            }
            for link_id, width, opposition, coordinates in (
                (1, 100.0, -1.0, [(0, 0), (5, 0), (10, 0)]),
                (2, 100.0, 0.8, [(10, 0), (15, 2.5), (20, 5)]),
                (3, 90.0, -1.0, [(10, 0), (15, -2.5), (20, -5)]),
            )
            for sample_index, (x, y) in enumerate(coordinates)
        ]
    )
    endpoints = pd.DataFrame(
        {
            "example_id": ["99"] * 3,
            "id_link": [1, 2, 3],
            "id_us_node": [0, 1, 1],
            "id_ds_node": [1, 2, 3],
            "whole_link_mean_wet_width": [100.0, 100.0, 90.0],
            "width_near_us_node_wet": [100.0, 100.0, 90.0],
            "width_near_ds_node_wet": [100.0, 100.0, 90.0],
            "length_over_wet_width": [10.0, 0.5, 0.56],
            "endpoint_windows_nonoverlapping": [False, False, False],
        }
    )
    bundle_path = tmp_path / "model.joblib"
    bundle_path.write_bytes(b"test model placeholder")
    workflow_path = tmp_path / "workflow.json"
    workflow_path.write_text("{}", encoding="utf-8")
    bundle = {
        "pipeline": _AlwaysCorrectPipeline(),
        "feature_columns": list(application.FINAL_FEATURE_COLUMNS),
        "model_version": application.FINAL_MODEL_VERSION,
        "model_name": "logistic_regression",
    }
    monkeypatch.setattr(
        application,
        "_load_model_contract",
        lambda _: (
            {"selected_threshold": 0.4},
            {},
            workflow_path,
            bundle_path,
        ),
    )
    monkeypatch.setattr(application.joblib, "load", lambda _: bundle)
    monkeypatch.setattr(application, "calculate_endpoint_widths", lambda **_: endpoints)

    output, audit, summary = application.apply_routing_width_model(
        links,
        samples,
        nodes,
        example_id="99",
        variant_id="S000_base",
        apply_known_base_exclusions=False,
    )
    result = output.set_index("id_link")

    assert result.loc[2, "wid_adj_wet"] == 140.0
    assert result.loc[2, "wid_adj_routing"] != result.loc[2, "wid_adj_wet"]
    assert result.loc[3, "wid_adj_routing"] == 90.0
    assert result.loc[2, "width_qc_decision"] in {"correct", "correct_flag"}
    assert summary["measurement_source_field"] == "wid_adj_wet"
    assert summary["candidate_count"] == 1
    assert summary["predicted_correction_count"] == 1
    assert not any(column.endswith(("_x", "_y")) for column in output.columns)
    assert "wid_adj_wet_raw" not in output.columns
    assert "wid_adj_routing_raw" not in output.columns
    assert "wid_adj_routing_raw" not in audit.columns
    assert set(audit["width_algorithm_version"]) == {application.ROUTING_WIDTH_APPLICATION_VERSION}

    paths = application.write_routing_width_audit(audit, summary, output_dir=tmp_path / "widths")
    manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
    assert manifest["files"]["link_audit"] == application.ROUTING_AUDIT_FILENAME


def test_scoped_child_application_preserves_inherited_links_as_fixed_context(
    monkeypatch,
    tmp_path: Path,
) -> None:
    links = gpd.GeoDataFrame(
        {
            "id_link": [1, 2, 3],
            "id_us_node": [0, 1, 1],
            "id_ds_node": [1, 2, 3],
            "len": [1000.0, 50.0, 50.0],
            "n_units": [0, 1, 0],
            "wid_adj_wet": [100.0, 140.0, 90.0],
            "wid_adj_routing": [80.0, np.nan, 90.0],
            "wid_pix_total": ["100,100"] * 3,
            "n_invalid_width_samples": [0, 0, 0],
            "qc_mean_opposition_cos_trimmed": [-1.0, 0.8, -1.0],
            "width_model_probability": [0.2, np.nan, 0.1],
            "width_qc_reason": ["inherited_one", None, "inherited_three"],
            "width_inheritance_status": [
                "inherited_unchanged_1to1",
                "changed_requires_evaluation",
                "inherited_unchanged_1to1",
            ],
        },
        geometry=[
            LineString([(0, 0), (10, 0)]),
            LineString([(10, 0), (20, 5)]),
            LineString([(10, 0), (20, -5)]),
        ],
        crs="EPSG:32631",
    )
    nodes = gpd.GeoDataFrame(
        {"id_node": [0, 1, 2, 3]},
        geometry=[Point(0, 0), Point(10, 0), Point(20, 5), Point(20, -5)],
        crs=links.crs,
    )
    samples = pd.DataFrame(
        [
            {
                "id_link": link_id,
                "sample_index": sample_index,
                "x": x,
                "y": y,
                "width_total": width,
                "width_wet": width,
                "width_invalid": False,
                "qc_opposition_cos": opposition,
            }
            for link_id, width, opposition, coordinates in (
                (1, 100.0, -1.0, [(0, 0), (5, 0), (10, 0)]),
                (2, 140.0, 0.8, [(10, 0), (15, 2.5), (20, 5)]),
                (3, 90.0, -1.0, [(10, 0), (15, -2.5), (20, -5)]),
            )
            for sample_index, (x, y) in enumerate(coordinates)
        ]
    )
    endpoints = pd.DataFrame(
        {
            "example_id": ["99"] * 3,
            "id_link": [1, 2, 3],
            "id_us_node": [0, 1, 1],
            "id_ds_node": [1, 2, 3],
            "whole_link_mean_wet_width": [100.0, 140.0, 90.0],
            "width_near_us_node_wet": [100.0, 140.0, 90.0],
            "width_near_ds_node_wet": [100.0, 140.0, 90.0],
            "length_over_wet_width": [10.0, 0.36, 0.56],
            "endpoint_windows_nonoverlapping": [False, False, False],
        }
    )
    bundle_path = tmp_path / "model.joblib"
    bundle_path.write_bytes(b"test model placeholder")
    workflow_path = tmp_path / "workflow.json"
    workflow_path.write_text("{}", encoding="utf-8")
    bundle = {
        "pipeline": _AlwaysCorrectPipeline(),
        "feature_columns": list(application.FINAL_FEATURE_COLUMNS),
        "model_version": application.FINAL_MODEL_VERSION,
        "model_name": "logistic_regression",
    }
    monkeypatch.setattr(
        application,
        "_load_model_contract",
        lambda _: ({"selected_threshold": 0.4}, {}, workflow_path, bundle_path),
    )
    monkeypatch.setattr(application.joblib, "load", lambda _: bundle)
    monkeypatch.setattr(application, "calculate_endpoint_widths", lambda **_: endpoints)

    output, audit, summary = application.apply_routing_width_model(
        links,
        samples,
        nodes,
        example_id="99",
        variant_id="S001_unit_1",
        apply_known_base_exclusions=False,
        evaluation_link_ids=[2],
    )
    result = output.set_index("id_link")
    audit_by_id = audit.set_index("id_link")

    assert result.loc[1, "wid_adj_routing"] == 80.0
    assert result.loc[1, "width_model_probability"] == 0.2
    assert result.loc[1, "width_qc_reason"] == "inherited_one"
    assert result.loc[3, "wid_adj_routing"] == 90.0
    assert result.loc[2, "wid_adj_routing"] > 0
    assert result.loc[2, "width_model_probability"] == 1.0
    assert audit_by_id.loc[1, "width_evaluation_status"] == "preserved_inherited_link"
    assert audit_by_id.loc[2, "width_evaluation_status"] == "evaluated_changed_link"
    assert summary["evaluation_scope"] == "selected_changed_links"
    assert summary["evaluated_link_count"] == 1
    assert summary["fixed_inherited_link_count"] == 2
    assert summary["candidate_count"] == 1
