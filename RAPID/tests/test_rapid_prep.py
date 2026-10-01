from __future__ import annotations

import json
import gzip
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from scipy.io import netcdf_file
from shapely.geometry import LineString, Point

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "RAPID" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rapid_tools.engine import RapidResultStorageConfig, run_prepared_experiment, write_qout_netcdf
from rapid_tools.forcing import ForcingConfig
from rapid_tools.hydrograph import HydrographMetricConfig, _integrate_trapezoid, _load_qout, summarize_outlet_hydrograph
from rapid_tools.analysis import load_reach_wave_stack
from rapid_tools.k_values import (
    KValueConfig,
    apply_stability_effective_length_cap,
    compute_k_values,
    compute_routing_dt_seconds,
    find_nearby_compatible_forcing_and_routing_dt,
    select_coordinated_routing_timesteps,
)
import rapid_tools.prep as rapid_prep_module
from rapid_tools.prep import RapidPrepConfig, build_rapid_graph, compute_reach_ratios, create_conn_file, prepare_experiment
from rapid_tools.reference_section import (
    _build_reference_section_variant,
    _load_width_summary,
    _reference_section_paths,
    _validate_single_source_sink_path,
    compute_reference_section_kb,
    ensure_reference_section_geometry,
)
from rapid_tools.slope import SlopeConfig, compute_link_slopes


def _write_single_path_graph(links_path: Path, nodes_path: Path) -> None:
    crs = "EPSG:32631"
    gpd.GeoDataFrame(
        {"id_link": [10], "id_us_node": [1], "id_ds_node": [2]},
        geometry=[LineString([(0, 0), (10, 0)])],
        crs=crs,
    ).to_file(links_path, driver="GPKG")
    gpd.GeoDataFrame(
        {"id_node": [1, 2]},
        geometry=[Point(0, 0), Point(10, 0)],
        crs=crs,
    ).to_file(nodes_path, driver="GPKG")


def _write_state(
    tmp_path: Path,
    *,
    experiment_name: str = "experiment",
    link_id: int = 11,
    link_length: float = 1000.0,
    width_samples_wet: list[float] | None = None,
    write_hydraulic_context: bool = True,
) -> Path:
    experiment_dir = tmp_path / experiment_name
    state_dir = experiment_dir / "states" / "S001_unit_1"
    variant_dir = state_dir / "variant"
    hierarchy_dir = state_dir / "hierarchy"
    directed_dir = variant_dir / "directed"
    widths_dir = variant_dir / "widths"
    directed_dir.mkdir(parents=True)
    widths_dir.mkdir(parents=True)
    hierarchy_dir.mkdir(parents=True)

    nodes = gpd.GeoDataFrame(
        {
            "id_node": [1, 2],
            "is_inlet": [True, False],
            "is_outlet": [False, True],
            "sword_wse": [10.0, 9.0],
            "sword_wse_field": ["wse_obs_p50", "wse_obs_p50"],
            "sword_wse_fill_method": ["requested_field", "requested_field"],
            "sword_wse_fallback_used": [False, False],
            "sword_node_id": [101, 102],
        },
        geometry=[Point(0.0, 0.0), Point(link_length, 0.0)],
        crs="EPSG:3857",
    )
    links = gpd.GeoDataFrame(
        {
            "id_link": [link_id],
            "id_us_node": [1],
            "id_ds_node": [2],
            "is_inlet": [True],
            "is_outlet": [True],
            "len": [link_length],
            "wid_adj_wet": [50.0],
            "wid_adj_total": [60.0],
        },
        geometry=[LineString([(0.0, 0.0), (link_length, 0.0)])],
        crs="EPSG:3857",
    )

    directed_links_path = directed_dir / "test_directed_links.gpkg"
    directed_nodes_path = directed_dir / "test_directed_nodes.gpkg"
    widths_path = widths_dir / "links_with_width_families.gpkg"
    width_samples_path = widths_dir / "link_width_samples.csv"
    links.to_file(directed_links_path, driver="GPKG")
    nodes.to_file(directed_nodes_path, driver="GPKG")
    links.to_file(widths_path, driver="GPKG")
    wet_widths = width_samples_wet or [50.0] * 11
    sample_count = len(wet_widths)
    sample_x = np.linspace(0.0, link_length, sample_count)
    width_samples = pd.DataFrame(
        {
            "id_link": [link_id] * sample_count,
            "sample_index": list(range(sample_count)),
            "x": sample_x,
            "y": [0.0] * sample_count,
            "width_total": wet_widths,
            "width_wet": wet_widths,
            "width_dry": [0.0] * sample_count,
            "wet_fraction": [1.0] * sample_count,
            "dry_fraction": [0.0] * sample_count,
            "n_wet_threads": [1] * sample_count,
        }
    )
    width_samples.to_csv(width_samples_path, index=False)
    pd.DataFrame(
        [
            {
                "unit_id": 1,
                "bifurcation": 1,
                "confluence": 2,
                "unit_node_ids": "1,2",
                "compound_bubble_id": 1,
                "in_compound_bubble": False,
                "compound_bubble_role": "standalone",
            }
        ]
    ).to_csv(hierarchy_dir / "unit_summary.csv", index=False)

    if write_hydraulic_context:
        hydraulic_context_dir = state_dir / "hydraulic_context"
        hydraulic_context_dir.mkdir(parents=True, exist_ok=True)
        unit_summary = pd.read_csv(hierarchy_dir / "unit_summary.csv")
        link_context = compute_link_slopes(links, nodes, unit_summary=unit_summary, config=SlopeConfig())
        link_context.to_csv(hydraulic_context_dir / "link_hydraulic_context.csv", index=False)
        pd.DataFrame(
            {
                "id_node": nodes["id_node"],
                "rapid_wse_for_slope": nodes["sword_wse"],
            }
        ).to_csv(hydraulic_context_dir / "node_hydraulic_context.csv", index=False)
        (hydraulic_context_dir / "hydraulic_context_manifest.json").write_text(
            json.dumps({"state_id": "S001_unit_1"}),
            encoding="utf-8",
        )

    registry = pd.DataFrame(
        [
            {
                "state_id": "S001_unit_1",
                "parent_state_id": "S000_base",
                "depth": 1,
                "mode": "independent-units",
                "state_role": "derived",
                "state_dir": str(state_dir.resolve()),
                "variant_output_dir": str(variant_dir.resolve()),
                "hierarchy_output_dir": str(hierarchy_dir.resolve()),
                "directed_links_path": str(directed_links_path.resolve()),
                "directed_nodes_path": str(directed_nodes_path.resolve()),
                "status": "complete",
            }
        ]
    )
    registry.to_csv(experiment_dir / "state_registry.csv", index=False)

    forcing = pd.DataFrame(
        {
            "time": [
                "2020-01-01T00:00:00Z",
                "2020-01-01T01:00:00Z",
                "2020-01-01T02:00:00Z",
            ],
            "discharge": [100.0, 110.0, 120.0],
        }
    )
    forcing.to_csv(experiment_dir / "forcing.csv", index=False)
    return experiment_dir


def _write_reference_section_summary(experiment_dir: Path) -> Path:
    reference_dir = experiment_dir / "reference_section" / "variant"
    directed_dir = reference_dir / "directed"
    rivgraph_dir = reference_dir / "rivgraph"
    directed_dir.mkdir(parents=True, exist_ok=True)
    rivgraph_dir.mkdir(parents=True, exist_ok=True)

    centerline_path = rivgraph_dir / "reference_centerline.gpkg"
    directed_nodes_path = directed_dir / "reference_directed_nodes.gpkg"

    centerline = gpd.GeoDataFrame(
        [{"geometry": LineString([(0.0, 0.0), (1000.0, 0.0)])}],
        geometry="geometry",
        crs="EPSG:3857",
    )
    centerline.to_file(centerline_path, driver="GPKG")

    reference_nodes = gpd.GeoDataFrame(
        [
            {"id_node": 100, "is_inlet": True, "is_outlet": False, "geometry": Point(0.0, 0.0)},
            {"id_node": 101, "is_inlet": False, "is_outlet": True, "geometry": Point(1000.0, 0.0)},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    reference_nodes.to_file(directed_nodes_path, driver="GPKG")

    summary_path = experiment_dir / "reference_section" / "reference_section_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "paths": {
                    "centerline_path": str(centerline_path.resolve()),
                    "directed_nodes_path": str(directed_nodes_path.resolve()),
                    "reference_section_dir": str((experiment_dir / "reference_section").resolve()),
                }
            }
        ),
        encoding="utf-8",
    )
    return summary_path


def _write_multistation_forcing_csv(experiment_dir: Path) -> Path:
    forcing = pd.DataFrame(
        {
            "station_key": [
                "BR:EXAMPLE_A",
                "BR:EXAMPLE_A",
                "BR:EXAMPLE_A",
                "BR:EXAMPLE_B",
                "BR:EXAMPLE_B",
                "BR:EXAMPLE_B",
            ],
            "time": [
                "2020-01-01T00:00:00Z",
                "2020-01-01T00:30:00Z",
                "2020-01-01T01:00:00Z",
                "2020-01-01T00:00:00Z",
                "2020-01-01T00:30:00Z",
                "2020-01-01T01:00:00Z",
            ],
            "discharge": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
        }
    )
    path = experiment_dir / "forcing_multistation.csv"
    forcing.to_csv(path, index=False)
    return path


def test_reference_section_width_summary_prefers_link_family_field(tmp_path: Path) -> None:
    samples_path = tmp_path / "link_width_samples.csv"
    pd.DataFrame({"width_wet": [100.0, 200.0, 400.0]}).to_csv(samples_path, index=False)

    families_path = tmp_path / "link_width_families.csv"
    pd.DataFrame({"wid_adj_wet": [275.0]}).to_csv(families_path, index=False)

    summary = _load_width_summary(
        samples_path,
        families_path,
        width_sample_field="wid_adj_wet",
        width_percentile=None,
    )

    assert summary["width_reference_m"] == 275.0
    assert summary["source"] == "link_width_families"
    assert summary["statistic"] == "direct_single_link"
    assert summary["percentile"] is None


def test_reference_section_raw_width_samples_require_explicit_percentile(tmp_path: Path) -> None:
    samples_path = tmp_path / "link_width_samples.csv"
    pd.DataFrame({"width_wet": [100.0, 200.0, 400.0]}).to_csv(samples_path, index=False)

    with pytest.raises(ValueError, match="percentile must be provided"):
        _load_width_summary(
            samples_path,
            tmp_path / "missing_link_width_families.csv",
            width_sample_field="width_wet",
            width_percentile=None,
        )


def test_reference_section_cache_refreshes_width_summary_without_rebuilding_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_dir = tmp_path / "experiment"
    reference_dir = experiment_dir / "reference_section"
    variant_dir = reference_dir / "variant"
    widths_dir = variant_dir / "widths"
    directed_dir = variant_dir / "directed"
    widths_dir.mkdir(parents=True)
    directed_dir.mkdir(parents=True)

    pd.DataFrame([{"state_role": "base"}]).to_csv(experiment_dir / "state_registry.csv", index=False)
    (experiment_dir / "experiment_manifest.json").write_text(json.dumps({"example_id": "02"}))
    samples_path = widths_dir / "link_width_samples.csv"
    pd.DataFrame({"width_wet": [100.0, 200.0, 400.0]}).to_csv(samples_path, index=False)
    pd.DataFrame({"wid_adj_routing": [225.0]}).to_csv(
        widths_dir / "link_width_families.csv", index=False
    )
    directed_links_path = directed_dir / "links.gpkg"
    directed_nodes_path = directed_dir / "nodes.gpkg"
    _write_single_path_graph(directed_links_path, directed_nodes_path)

    summary_path = reference_dir / "reference_section_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "width_summary": {
                    "sample_field": "width_wet",
                    "percentile": 90.0,
                    "width_reference_m": 400.0,
                },
                "slope_summary": {"slope": 0.001},
                "paths": {
                    "variant_output_dir": str(variant_dir.resolve()),
                    "directed_links_path": str(directed_links_path.resolve()),
                    "directed_nodes_path": str(directed_nodes_path.resolve()),
                    "link_width_samples_path": str(samples_path.resolve()),
                },
            }
        )
    )

    monkeypatch.setattr(
        "rapid_tools.reference_section._build_reference_section_variant",
        lambda **_: pytest.fail("cached reference geometry should not be rebuilt"),
    )
    summary = ensure_reference_section_geometry(
        experiment_dir,
        width_sample_field="wid_adj_routing",
        width_percentile=90.0,
    )

    assert summary["width_summary"]["width_reference_m"] == 225.0
    assert summary["width_summary"]["sample_field"] == "wid_adj_routing"
    assert summary["width_summary"]["percentile"] is None
    assert json.loads(summary_path.read_text())["width_summary"] == summary["width_summary"]


def test_reference_section_cache_refreshes_nonfinite_slope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_dir = tmp_path / "experiment"
    reference_dir = experiment_dir / "reference_section"
    variant_dir = reference_dir / "variant"
    widths_dir = variant_dir / "widths"
    directed_dir = variant_dir / "directed"
    widths_dir.mkdir(parents=True)
    directed_dir.mkdir(parents=True)
    pd.DataFrame([{"state_role": "base"}]).to_csv(
        experiment_dir / "state_registry.csv", index=False
    )
    (experiment_dir / "experiment_manifest.json").write_text(
        json.dumps({"example_id": "41"})
    )
    samples_path = widths_dir / "link_width_samples.csv"
    pd.DataFrame({"wid_adj_routing": [225.0]}).to_csv(samples_path, index=False)
    pd.DataFrame({"wid_adj_routing": [225.0]}).to_csv(
        widths_dir / "link_width_families.csv", index=False
    )
    links_path = directed_dir / "links.gpkg"
    nodes_path = directed_dir / "nodes.gpkg"
    _write_single_path_graph(links_path, nodes_path)
    summary_path = reference_dir / "reference_section_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "width_summary": {"sample_field": "wid_adj_routing"},
                "slope_summary": {
                    "slope": float("nan"),
                    "source_method": "missing_dist_out",
                    "anchor_count": 0,
                },
                "paths": {
                    "variant_output_dir": str(variant_dir.resolve()),
                    "directed_links_path": str(links_path.resolve()),
                    "directed_nodes_path": str(nodes_path.resolve()),
                    "link_width_samples_path": str(samples_path.resolve()),
                },
            }
        )
    )
    monkeypatch.setattr(
        "rapid_tools.reference_section.compute_section_slope_reference",
        lambda nodes, config: {
            "slope": 0.0001,
            "source_method": "test_refresh",
            "anchor_count": 2,
        },
    )

    summary = ensure_reference_section_geometry(
        experiment_dir,
        width_sample_field="wid_adj_routing",
        width_percentile=None,
    )

    assert summary["slope_summary"]["slope"] == 0.0001
    assert summary["slope_summary"]["source_method"] == "test_refresh"


def test_reference_section_geometry_uses_injected_panelized_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir()
    pd.DataFrame([{"state_role": "base"}]).to_csv(
        experiment_dir / "state_registry.csv", index=False
    )
    (experiment_dir / "experiment_manifest.json").write_text(
        json.dumps({"example_id": "41", "exit_sides": "EW"}),
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    def builder(**kwargs):
        calls.append(kwargs)
        variant_dir = Path(kwargs["reference_paths"]["variant"])
        widths = variant_dir / "widths"
        directed = variant_dir / "directed"
        widths.mkdir(parents=True)
        directed.mkdir(parents=True)
        pd.DataFrame({"wid_adj_routing": [310.0]}).to_csv(
            widths / "link_width_families.csv", index=False
        )
        pd.DataFrame({"wid_adj_routing": [310.0]}).to_csv(
            widths / "link_width_samples.csv", index=False
        )
        links_path = directed / "links.gpkg"
        nodes_path = directed / "nodes.gpkg"
        _write_single_path_graph(links_path, nodes_path)
        return {
            "construction_backend": "panelized",
            "example_id": "41",
            "exit_sides": "EW",
            "unit_ids": [1, 2],
            "variant_output_dir": variant_dir,
            "directed_links_path": links_path,
            "directed_nodes_path": nodes_path,
            "centerline_path": None,
            "collapsed_mask_path": variant_dir / "mask.vrt",
            "boundary_component_labels_path": None,
            "link_width_samples_path": widths / "link_width_samples.csv",
            "manifest_path": variant_dir / "summary" / "variant_manifest.json",
        }

    monkeypatch.setattr(
        "rapid_tools.reference_section.compute_section_slope_reference",
        lambda nodes, config: {
            "slope": 0.001,
            "source_method": "test",
            "anchor_count": 2,
        },
    )

    summary = ensure_reference_section_geometry(
        experiment_dir,
        width_sample_field="wid_adj_routing",
        width_percentile=None,
        variant_builder=builder,
    )

    assert len(calls) == 1
    assert summary["construction_backend"] == "panelized"
    assert summary["n_reference_units_collapsed"] == 2
    assert summary["width_summary"]["width_reference_m"] == 310.0


def test_reference_section_rejects_parallel_source_to_sink_paths(tmp_path: Path) -> None:
    links_path = tmp_path / "links.gpkg"
    nodes_path = tmp_path / "nodes.gpkg"
    crs = "EPSG:32631"
    gpd.GeoDataFrame(
        {
            "id_link": [10, 11],
            "id_us_node": [1, 1],
            "id_ds_node": [2, 2],
        },
        geometry=[
            LineString([(0, 0), (10, 0)]),
            LineString([(0, 0), (5, 1), (10, 0)]),
        ],
        crs=crs,
    ).to_file(links_path, driver="GPKG")
    gpd.GeoDataFrame(
        {"id_node": [1, 2]},
        geometry=[Point(0, 0), Point(10, 0)],
        crs=crs,
    ).to_file(nodes_path, driver="GPKG")

    with pytest.raises(ValueError, match="exactly one directed source-to-sink path"):
        _validate_single_source_sink_path(links_path, nodes_path)


def test_reference_section_kb_accepts_direct_summary_without_percentile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_dir = tmp_path / "experiment"
    monkeypatch.setattr(
        "rapid_tools.reference_section.ensure_reference_section_geometry",
        lambda *args, **kwargs: {
            "example_id": "02",
            "width_summary": {
                "sample_field": "wid_adj_routing",
                "percentile": None,
                "width_reference_m": 200.0,
            },
            "slope_summary": {
                "slope": 0.001,
                "source_method": "test",
                "anchor_count": 2,
            },
        },
    )
    monkeypatch.setattr("rapid_tools.reference_section.predict_based_depth", lambda **kwargs: 4.0)
    monkeypatch.setattr(
        "rapid_tools.reference_section.resolve_based_model_path",
        lambda model_path=None: Path("model.ubj"),
    )
    forcing = pd.DataFrame({"discharge_cms": [10.0, 20.0]})

    summary = compute_reference_section_kb(
        experiment_dir,
        forcing=forcing,
        width_sample_field="wid_adj_routing",
        width_percentile=None,
    )

    assert summary["kb_value"] == 50.0
    assert summary["width_sample_field"] == "wid_adj_routing"
    assert summary["width_percentile"] is None


def test_reference_section_variant_reuses_base_boundary_component_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_dir = tmp_path / "experiment"
    base_dir = experiment_dir / "states" / "S000_base"
    variant_dir = base_dir / "variant"
    hierarchy_dir = base_dir / "hierarchy"
    summary_dir = variant_dir / "summary"
    summary_dir.mkdir(parents=True)
    hierarchy_dir.mkdir(parents=True)

    analysis_links = variant_dir / "widths" / "links_with_width_families.gpkg"
    directed_nodes = variant_dir / "directed" / "base_nodes.gpkg"
    cleaned_mask = variant_dir / "mask" / "base_collapsed.tif"
    for path in (analysis_links, directed_nodes, cleaned_mask):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    pd.DataFrame({"unit_id": [1, 2]}).to_csv(hierarchy_dir / "unit_summary.csv", index=False)

    labels = experiment_dir / "boundary_components" / "02_full_extent_land_water_labels_with_snapped_buffered_border_cut.tif"
    labels.parent.mkdir(parents=True)
    labels.touch()
    (summary_dir / "variant_manifest.json").write_text(
        json.dumps(
            {
                "exit_sides": "SN",
                "output_paths": {"boundary_component_labels": str(labels)},
            }
        ),
        encoding="utf-8",
    )

    registry = pd.DataFrame(
        [
            {
                "state_role": "base",
                "variant_output_dir": str(variant_dir),
                "hierarchy_output_dir": str(hierarchy_dir),
                "analysis_links_path": str(analysis_links),
                "directed_nodes_path": str(directed_nodes),
                "cleaned_mask_path": str(cleaned_mask),
            }
        ]
    )
    captured: dict[str, object] = {}

    def fake_generate_network_variant(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        output_dir = Path(str(kwargs["output_dir"]))
        return SimpleNamespace(
            output_dir=output_dir,
            directed_links_path=output_dir / "directed" / "links.gpkg",
            directed_nodes_path=output_dir / "directed" / "nodes.gpkg",
            rivgraph_centerline_path=None,
            collapsed_mask_path=output_dir / "mask" / "collapsed.tif",
        )

    monkeypatch.setattr(
        "network_variants.variant_generation.generate_network_variant",
        fake_generate_network_variant,
    )

    result = _build_reference_section_variant(
        experiment_dir=experiment_dir,
        experiment_manifest={"example_id": "02", "options": {"variant": {}}},
        registry=registry,
        reference_paths=_reference_section_paths(experiment_dir),
    )

    assert captured["boundary_component_labels_path"] == labels.resolve()
    assert result["boundary_component_labels_path"] == labels.resolve()


def test_prepare_experiment_writes_state_outputs(tmp_path: Path) -> None:
    experiment_dir = _write_state(tmp_path)
    registry = prepare_experiment(
        experiment_dir,
        forcing_path=experiment_dir / "forcing.csv",
        prep_config=RapidPrepConfig(),
    )
    assert registry.loc[0, "status"] == "prepared"

    prep_dir = experiment_dir / "states" / "S001_unit_1" / "rapid" / "prep"
    assert (prep_dir / "rapid_prep_manifest.json").exists()
    assert (prep_dir / "rapid_link_attributes.csv").exists()
    assert (prep_dir / "rapid_reach_attributes.gpkg").exists()
    assert (prep_dir / "kfc.csv").exists()
    assert (prep_dir / "inflow.nc").exists()


def test_prepare_experiment_prefers_precomputed_hydraulic_context(tmp_path: Path) -> None:
    experiment_dir = _write_state(tmp_path)
    state_dir = experiment_dir / "states" / "S001_unit_1"
    variant_dir = state_dir / "variant"
    hierarchy_dir = state_dir / "hierarchy"
    hydraulic_context_dir = state_dir / "hydraulic_context"

    links = gpd.read_file(variant_dir / "widths" / "links_with_width_families.gpkg")
    nodes = gpd.read_file(variant_dir / "directed" / "test_directed_nodes.gpkg")
    unit_summary = pd.read_csv(hierarchy_dir / "unit_summary.csv")
    link_context = compute_link_slopes(links, nodes, unit_summary=unit_summary, config=SlopeConfig())
    link_context["slope_used"] = 0.123456
    link_context.to_csv(hydraulic_context_dir / "link_hydraulic_context.csv", index=False)
    pd.DataFrame({"id_node": nodes["id_node"]}).to_csv(
        hydraulic_context_dir / "node_hydraulic_context.csv",
        index=False,
    )
    (hydraulic_context_dir / "hydraulic_context_manifest.json").write_text(
        json.dumps({"state_id": "S001_unit_1"}),
        encoding="utf-8",
    )

    registry = prepare_experiment(
        experiment_dir,
        forcing_path=experiment_dir / "forcing.csv",
        prep_config=RapidPrepConfig(),
    )

    assert registry.loc[0, "status"] == "prepared"
    assert registry.loc[0, "hydraulic_context_source"] == "precomputed_state_hydraulic_context"
    assert Path(registry.loc[0, "hydraulic_context_link_csv"]).exists()

    prep_dir = state_dir / "rapid" / "prep"
    rapid_links = pd.read_csv(prep_dir / "rapid_link_attributes.csv")
    assert set(np.round(rapid_links["slope_used"], 6)) == {0.123456}

    manifest = json.loads((prep_dir / "rapid_prep_manifest.json").read_text())
    assert manifest["hydraulic_context"]["source"] == "precomputed_state_hydraulic_context"
    assert Path(manifest["hydraulic_context"]["link_context_csv"]).exists()


def test_prepare_experiment_uses_precomputed_hydraulic_context_for_reference_section_mode(
    tmp_path: Path,
) -> None:
    experiment_dir = _write_state(tmp_path)
    summary_path = _write_reference_section_summary(experiment_dir)
    state_dir = experiment_dir / "states" / "S001_unit_1"
    hydraulic_context_dir = state_dir / "hydraulic_context"
    pd.DataFrame(
        [
            {
                "id_link": 11,
                "id_us_node": 1,
                "id_ds_node": 2,
                "link_length_m": 1000.0,
                "slope_used": 0.123456,
                "slope_adjusted": False,
            }
        ]
    ).to_csv(hydraulic_context_dir / "link_hydraulic_context.csv", index=False)

    registry = prepare_experiment(
        experiment_dir,
        forcing_path=experiment_dir / "forcing.csv",
        prep_config=RapidPrepConfig(
            slope_profile_mode="reference_section",
            reference_section_summary_path=str(summary_path),
        ),
    )

    assert registry.loc[0, "status"] == "prepared"
    assert registry.loc[0, "hydraulic_context_source"] == "precomputed_state_hydraulic_context"
    prep_dir = state_dir / "rapid" / "prep"
    rapid_links = pd.read_csv(prep_dir / "rapid_link_attributes.csv")
    assert set(np.round(rapid_links["slope_used"], 6)) == {0.123456}


def test_prepare_experiment_requires_state_hydraulic_context(tmp_path: Path) -> None:
    experiment_dir = _write_state(tmp_path, write_hydraulic_context=False)

    registry = prepare_experiment(
        experiment_dir,
        forcing_path=experiment_dir / "forcing.csv",
        prep_config=RapidPrepConfig(),
    )

    assert registry.loc[0, "status"] == "failed"
    assert "link_hydraulic_context.csv" in str(registry.loc[0, "error"])


def test_prepare_experiment_normalizes_station_key_forcing(tmp_path: Path) -> None:
    experiment_dir = _write_state(tmp_path)
    forcing_path = _write_multistation_forcing_csv(experiment_dir)
    cache_dir = tmp_path / "forcing_cache"

    registry = prepare_experiment(
        experiment_dir,
        forcing_path=forcing_path,
        forcing_config=ForcingConfig(
            station_key="BR:EXAMPLE_A",
            start_time="2020-01-01T00:00:00Z",
            end_time="2020-01-01T01:00:00Z",
            resample_minutes=15,
            output_cache_dir=str(cache_dir),
        ),
        prep_config=RapidPrepConfig(),
    )

    assert registry.loc[0, "status"] == "prepared"
    assert registry.loc[0, "forcing_station_key"] == "BR:EXAMPLE_A"
    assert not bool(registry.loc[0, "forcing_loaded_from_cache"])

    prep_dir = experiment_dir / "states" / "S001_unit_1" / "rapid" / "prep"
    forcing_normalized = pd.read_csv(prep_dir / "forcing_normalized.csv")
    assert len(forcing_normalized) == 5
    assert forcing_normalized["discharge_cms"].round(6).tolist() == [10.0, 15.0, 20.0, 25.0, 30.0]

    manifest = json.loads((prep_dir / "rapid_prep_manifest.json").read_text())
    forcing_metadata = manifest["forcing_metadata"]
    assert forcing_metadata["selected_station_key"] == "BR:EXAMPLE_A"
    assert forcing_metadata["normalized_dt_seconds"] == 900
    assert forcing_metadata["forcing_loaded_from_cache"] is False
    assert Path(forcing_metadata["forcing_cache_csv"]).exists()


def test_prepare_experiment_normalizes_station_key_forcing_with_second_resolution(tmp_path: Path) -> None:
    experiment_dir = _write_state(tmp_path)
    forcing_path = _write_multistation_forcing_csv(experiment_dir)

    registry = prepare_experiment(
        experiment_dir,
        forcing_path=forcing_path,
        forcing_config=ForcingConfig(
            station_key="BR:EXAMPLE_A",
            start_time="2020-01-01T00:00:00Z",
            end_time="2020-01-01T01:00:00Z",
            resample_seconds=990,
        ),
        prep_config=RapidPrepConfig(),
    )

    assert registry.loc[0, "status"] == "prepared"

    prep_dir = experiment_dir / "states" / "S001_unit_1" / "rapid" / "prep"
    manifest = json.loads((prep_dir / "rapid_prep_manifest.json").read_text())
    forcing_metadata = manifest["forcing_metadata"]
    assert forcing_metadata["forcing_resample_seconds"] == 990
    assert forcing_metadata["normalized_dt_seconds"] == 990


def test_prepare_experiment_accepts_fractional_resample_minutes(tmp_path: Path) -> None:
    experiment_dir = _write_state(tmp_path)
    forcing_path = _write_multistation_forcing_csv(experiment_dir)

    registry = prepare_experiment(
        experiment_dir,
        forcing_path=forcing_path,
        forcing_config=ForcingConfig(
            station_key="BR:EXAMPLE_A",
            start_time="2020-01-01T00:00:00Z",
            end_time="2020-01-01T01:00:00Z",
            resample_minutes=16.5,
        ),
        prep_config=RapidPrepConfig(),
    )

    assert registry.loc[0, "status"] == "prepared"

    prep_dir = experiment_dir / "states" / "S001_unit_1" / "rapid" / "prep"
    manifest = json.loads((prep_dir / "rapid_prep_manifest.json").read_text())
    forcing_metadata = manifest["forcing_metadata"]
    assert forcing_metadata["forcing_resample_minutes"] == pytest.approx(16.5)
    assert forcing_metadata["forcing_resample_seconds"] == 990
    assert forcing_metadata["normalized_dt_seconds"] == 990


def test_prepare_experiment_prepends_constant_spinup(tmp_path: Path) -> None:
    experiment_dir = _write_state(tmp_path)

    registry = prepare_experiment(
        experiment_dir,
        forcing_path=experiment_dir / "forcing.csv",
        forcing_config=ForcingConfig(spinup_hours=2.0),
        prep_config=RapidPrepConfig(),
    )

    assert registry.loc[0, "status"] == "prepared"

    prep_dir = experiment_dir / "states" / "S001_unit_1" / "rapid" / "prep"
    forcing_normalized = pd.read_csv(prep_dir / "forcing_normalized.csv")
    forcing_times = pd.to_datetime(forcing_normalized["time"], utc=True)
    assert len(forcing_normalized) == 7
    assert forcing_normalized["discharge_cms"].round(6).tolist() == [100.0, 100.0, 100.0, 110.0, 120.0, 120.0, 120.0]
    assert forcing_times.iloc[0].isoformat() == "2019-12-31T22:00:00+00:00"
    assert forcing_times.iloc[2].isoformat() == "2020-01-01T00:00:00+00:00"
    assert forcing_times.iloc[-1].isoformat() == "2020-01-01T04:00:00+00:00"

    manifest = json.loads((prep_dir / "rapid_prep_manifest.json").read_text())
    forcing_metadata = manifest["forcing_metadata"]
    assert forcing_metadata["spinup_enabled"] is True
    assert forcing_metadata["spinup_hours"] == 2.0
    assert forcing_metadata["spinup_duration_seconds"] == 7200
    assert forcing_metadata["spinup_row_count"] == 2
    assert forcing_metadata["spinup_discharge_cms"] == 100.0
    assert forcing_metadata["end_padding_enabled"] is True
    assert forcing_metadata["end_padding_row_count"] == 2
    assert forcing_metadata["end_padding_discharge_cms"] == 120.0
    assert manifest["forcing_config"]["spinup_hours"] == 2.0


def test_prepare_experiment_computes_auto_spinup_from_longest_k_path(tmp_path: Path) -> None:
    experiment_dir = _write_state(tmp_path)

    registry = prepare_experiment(
        experiment_dir,
        forcing_path=experiment_dir / "forcing.csv",
        forcing_config=ForcingConfig(auto_spinup_from_k=True, spinup_k_multiplier=1.1),
        prep_config=RapidPrepConfig(),
    )

    assert registry.loc[0, "status"] == "prepared"

    prep_dir = experiment_dir / "states" / "S001_unit_1" / "rapid" / "prep"
    rapid_links = pd.read_csv(prep_dir / "rapid_link_attributes.csv")
    rapid_k = float(rapid_links.loc[0, "rapid_k"])

    manifest = json.loads((prep_dir / "rapid_prep_manifest.json").read_text())
    forcing_metadata = manifest["forcing_metadata"]
    expected_spinup_hours = rapid_k * 1.1 / 3600.0

    assert forcing_metadata["spinup_source"] == "longest_downstream_k_path"
    assert forcing_metadata["longest_downstream_k_seconds"] == pytest.approx(rapid_k)
    assert forcing_metadata["spinup_k_multiplier"] == pytest.approx(1.1)
    assert forcing_metadata["spinup_hours"] == pytest.approx(expected_spinup_hours)
    assert forcing_metadata["end_padding_enabled"] is True
    assert manifest["forcing_config"]["spinup_hours"] == pytest.approx(expected_spinup_hours)
    assert manifest["forcing_config"]["auto_spinup_from_k"] is True


def test_prepare_experiment_requires_station_key_for_multistation_forcing(tmp_path: Path) -> None:
    experiment_dir = _write_state(tmp_path)
    forcing_path = _write_multistation_forcing_csv(experiment_dir)

    registry = prepare_experiment(
        experiment_dir,
        forcing_path=forcing_path,
        forcing_config=ForcingConfig(),
        prep_config=RapidPrepConfig(),
    )

    assert registry.loc[0, "status"] == "failed"
    assert "multiple station keys" in str(registry.loc[0, "error"])


def test_prepare_experiment_reuses_station_key_forcing_cache(tmp_path: Path) -> None:
    experiment_dir = _write_state(tmp_path)
    forcing_path = _write_multistation_forcing_csv(experiment_dir)
    cache_dir = tmp_path / "forcing_cache"
    forcing_config = ForcingConfig(
        station_key="BR:EXAMPLE_A",
        start_time="2020-01-01T00:00:00Z",
        end_time="2020-01-01T01:00:00Z",
        resample_minutes=15,
        output_cache_dir=str(cache_dir),
    )

    prepare_experiment(
        experiment_dir,
        forcing_path=forcing_path,
        forcing_config=forcing_config,
        prep_config=RapidPrepConfig(),
    )
    prep_dir = experiment_dir / "states" / "S001_unit_1" / "rapid" / "prep"
    first = pd.read_csv(prep_dir / "forcing_normalized.csv")

    modified = pd.DataFrame(
        {
            "station_key": [
                "BR:EXAMPLE_A",
                "BR:EXAMPLE_A",
                "BR:EXAMPLE_A",
                "BR:EXAMPLE_B",
                "BR:EXAMPLE_B",
                "BR:EXAMPLE_B",
            ],
            "time": [
                "2020-01-01T00:00:00Z",
                "2020-01-01T00:30:00Z",
                "2020-01-01T01:00:00Z",
                "2020-01-01T00:00:00Z",
                "2020-01-01T00:30:00Z",
                "2020-01-01T01:00:00Z",
            ],
            "discharge": [999.0, 999.0, 999.0, 100.0, 200.0, 300.0],
        }
    )
    modified.to_csv(forcing_path, index=False)

    registry = prepare_experiment(
        experiment_dir,
        forcing_path=forcing_path,
        forcing_config=forcing_config,
        prep_config=RapidPrepConfig(),
    )

    second = pd.read_csv(prep_dir / "forcing_normalized.csv")
    pd.testing.assert_frame_equal(first, second)
    assert registry.loc[0, "status"] == "prepared"
    assert bool(registry.loc[0, "forcing_loaded_from_cache"])


def test_prepare_experiment_reuses_stored_reference_section_kb(tmp_path: Path) -> None:
    experiment_dir = _write_state(tmp_path)
    reference_dir = experiment_dir / "reference_section"
    reference_dir.mkdir(parents=True, exist_ok=True)
    stored_summary = {
        "kb_source_method": "based_reference_section",
        "kb_value": 12.5,
        "reference_discharge_cms": 120.0,
        "reference_width_m": 320.0,
        "reference_slope": 1.2e-4,
        "reference_depth_m": 25.6,
        "reference_section_summary_path": str(reference_dir / "reference_section_summary.json"),
        "based_model_path": "/tmp/based_model_v2.ubj",
    }
    (reference_dir / "reference_kb_summary.json").write_text(
        json.dumps(stored_summary),
        encoding="utf-8",
    )

    registry = prepare_experiment(
        experiment_dir,
        forcing_path=experiment_dir / "forcing.csv",
        prep_config=RapidPrepConfig(
            kb_mode="based_reference_section",
            kb_width_sample_field="width_wet",
            kb_width_percentile=90.0,
        ),
    )

    assert registry.loc[0, "status"] == "prepared"
    assert registry.loc[0, "rapid_kb_value"] == 12.5
    assert registry.loc[0, "rapid_kb_source_method"] == "based_reference_section"

    prep_dir = experiment_dir / "states" / "S001_unit_1" / "rapid" / "prep"
    rapid_links = pd.read_csv(prep_dir / "rapid_link_attributes.csv")
    assert set(rapid_links["rapid_kb_value"].round(6)) == {12.5}
    assert set(rapid_links["rapid_kb_source_method"]) == {"based_reference_section"}
    assert set(rapid_links["rapid_kb_reference_discharge_cms"].round(6)) == {120.0}
    assert set(rapid_links["rapid_kb_reference_width_m"].round(6)) == {320.0}
    assert set(rapid_links["rapid_kb_reference_slope"].round(9)) == {0.00012}
    assert set(rapid_links["rapid_kb_reference_depth_m"].round(6)) == {25.6}

    manifest = json.loads((prep_dir / "rapid_prep_manifest.json").read_text())
    assert manifest["kb_summary"]["kb_value"] == 12.5
    assert manifest["kb_summary"]["kb_source_method"] == "based_reference_section"

    experiment_manifest = json.loads((experiment_dir / "rapid_prep_manifest.json").read_text())
    assert experiment_manifest["reference_kb_summary"]["kb_value"] == 12.5
    assert experiment_manifest["reference_kb_summary"]["kb_source_method"] == "based_reference_section"
    assert experiment_manifest["resolved_kb_summary"]["kb_mode"] == "based_reference_section"
    assert experiment_manifest["resolved_kb_summary"]["kb_value"] == 12.5
    assert experiment_manifest["reference_kb_recomputed"] is False
    assert experiment_manifest["reference_kb_summary_source"].endswith(
        "reference_section/reference_kb_summary.json"
    )


def test_prepare_experiment_does_not_recalculate_missing_reference_kb(tmp_path: Path) -> None:
    experiment_dir = _write_state(tmp_path)

    registry = prepare_experiment(
        experiment_dir,
        forcing_path=experiment_dir / "forcing.csv",
        prep_config=RapidPrepConfig(kb_mode="based_reference_section"),
    )

    assert registry.loc[0, "status"] == "failed"
    assert "RAPID no longer recalculates BASED" in str(registry.loc[0, "error"])


def test_run_prepared_experiment_writes_qout(tmp_path: Path) -> None:
    experiment_dir = _write_state(tmp_path)
    prepare_experiment(
        experiment_dir,
        forcing_path=experiment_dir / "forcing.csv",
        prep_config=RapidPrepConfig(),
    )
    run_registry = run_prepared_experiment(experiment_dir)
    assert run_registry.loc[0, "status"] == "ran"
    assert run_registry.loc[0, "hydrograph_status"] == "computed"

    qout_path = Path(run_registry.loc[0, "qout_nc"])
    assert qout_path.exists()
    with netcdf_file(qout_path, "r", mmap=False) as ds:
        qout = ds.variables["Qout"].data.copy()
        assert qout.shape == (3, 1)
    assert Path(run_registry.loc[0, "outlet_hydrograph_csv"]).exists()
    assert Path(run_registry.loc[0, "hydrograph_metrics_csv"]).exists()
    hydrograph_metrics = pd.read_csv(run_registry.loc[0, "hydrograph_metrics_csv"])
    assert "peak_discharge_cms" in hydrograph_metrics.columns
    assert "time_to_peak_seconds" in hydrograph_metrics.columns
    assert hydrograph_metrics.loc[0, "event_start_source"] == "auto_input_global_min_prepeak_fallback"
    assert hydrograph_metrics.loc[0, "event_end_source"] == "auto_outlet_series_end_fallback"


def test_run_prepared_experiment_can_publish_compact_results_and_drop_prep(tmp_path: Path) -> None:
    experiment_dir = _write_state(tmp_path)
    work_root = tmp_path / "rapid-work"
    prep_registry = prepare_experiment(
        experiment_dir,
        forcing_path=experiment_dir / "forcing.csv",
        prep_config=RapidPrepConfig(),
        work_root=work_root,
    )
    prep_row = prep_registry.iloc[0]
    assert Path(prep_row["rapid_prep_dir"]).is_relative_to(work_root)
    assert Path(prep_row["rapid_run_dir"]).is_relative_to(experiment_dir)
    run_registry = run_prepared_experiment(
        experiment_dir,
        storage_config=RapidResultStorageConfig(
            compact_qout=True,
            retain_prepared_inputs=False,
            write_mapping_parquet=True,
        ),
    )

    row = run_registry.iloc[0]
    assert row["status"] == "ran"
    assert str(row["qout_nc"]).endswith(".nc.gz")
    assert Path(row["qout_nc"]).exists()
    assert Path(row["rapid_run_dir"]).is_relative_to(experiment_dir)
    assert Path(row["rapid_link_mapping_parquet"]).exists()
    assert Path(row["rapid_reach_attributes_gpkg"]).exists()
    assert not Path(row["rapid_prep_dir"]).exists()
    wave_stack = load_reach_wave_stack(
        row["rapid_prep_dir"], row["qout_nc"], run_dir=row["rapid_run_dir"]
    )
    assert len(wave_stack.reaches) == 1


def test_run_prepared_experiment_recovers_compact_result_after_prep_cleanup(tmp_path: Path) -> None:
    """A restart must recognise publication when the disposable prep is gone."""
    experiment_dir = _write_state(tmp_path)
    work_root = tmp_path / "rapid-work"
    prep_registry = prepare_experiment(
        experiment_dir,
        forcing_path=experiment_dir / "forcing.csv",
        prep_config=RapidPrepConfig(),
        work_root=work_root,
    )
    storage = RapidResultStorageConfig(
        compact_qout=True,
        retain_prepared_inputs=False,
        write_mapping_parquet=True,
    )
    first = run_prepared_experiment(experiment_dir, storage_config=storage)
    prep_dir = Path(prep_registry.loc[0, "rapid_prep_dir"])
    assert first.loc[0, "status"] == "ran"
    assert not prep_dir.exists()

    recovered = run_prepared_experiment(experiment_dir, storage_config=storage)
    row = recovered.iloc[0]
    assert row["status"] == "ran"
    assert bool(row["recovered_existing_published_result"])
    assert not bool(row["rapid_prep_retained"])
    assert Path(row["qout_nc"]).exists()


def test_compact_publication_moves_reach_geometry_on_the_same_volume(tmp_path: Path) -> None:
    """Avoid a temporary second large GeoPackage before disposable prep is removed."""
    from rapid_tools.engine import _publish_compact_result

    prep = tmp_path / "prep"
    run = tmp_path / "run"
    prep.mkdir()
    run.mkdir()
    reach_geometry = prep / "rapid_reach_attributes.gpkg"
    reach_geometry.write_bytes(b"complete geometry")
    # Simulate the partial destination produced by a disk-full interrupted copy.
    retained = run / reach_geometry.name
    retained.write_bytes(b"partial")
    qout = run / "Qout.nc"
    qout.write_bytes(b"qout")

    result = _publish_compact_result(
        prep_dir=prep,
        run_dir=run,
        qout_path=qout,
        storage=RapidResultStorageConfig(retain_prepared_inputs=False),
    )

    assert not reach_geometry.exists()
    assert retained.read_bytes() == b"complete geometry"
    assert Path(result["rapid_reach_attributes_gpkg"]) == retained


def test_hydrograph_loader_reads_compressed_qout(tmp_path: Path) -> None:
    raw_qout = tmp_path / "qout.nc"
    river_ids = np.array([1, 2], dtype=np.int64)
    time_seconds = np.array([0.0, 300.0], dtype=np.float64)
    expected_qout = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    write_qout_netcdf(raw_qout, river_ids, time_seconds, expected_qout)
    compressed_qout = raw_qout.with_suffix(".nc.gz")
    with raw_qout.open("rb") as source, gzip.open(compressed_qout, "wb") as target:
        shutil.copyfileobj(source, target)

    loaded_ids, loaded_time, loaded_qout = _load_qout(compressed_qout)

    np.testing.assert_array_equal(loaded_ids, river_ids)
    np.testing.assert_allclose(loaded_time, time_seconds)
    np.testing.assert_allclose(loaded_qout, expected_qout)


def test_compute_k_values_exports_kb_provenance() -> None:
    frame = pd.DataFrame(
        {
            "link_length_m": [1000.0],
            "wid_adj_wet": [50.0],
            "slope_used": [1e-3],
        }
    )
    prepared = compute_k_values(
        frame,
        config=KValueConfig(
            kb_value=18.0,
            kb_source_method="based_reference_section",
            kb_reference_discharge_cms=345.0,
            kb_reference_width_m=275.0,
            kb_reference_slope=2.5e-4,
            kb_reference_depth_m=15.2777778,
        ),
    )

    assert prepared.loc[0, "rapid_kb_value"] == 18.0
    assert prepared.loc[0, "rapid_kb_source_method"] == "based_reference_section"
    assert prepared.loc[0, "rapid_kb_reference_discharge_cms"] == 345.0
    assert prepared.loc[0, "rapid_kb_reference_width_m"] == 275.0
    assert prepared.loc[0, "rapid_kb_reference_slope"] == 2.5e-4
    assert prepared.loc[0, "rapid_kb_reference_depth_m"] == 15.2777778


def test_compute_k_values_prefers_effective_routing_width() -> None:
    frame = pd.DataFrame(
        {
            "link_length_m": [1000.0],
            "wid_adj_routing": [30.0],
            "wid_adj_wet": [50.0],
            "wid_adj": [80.0],
            "slope_used": [1e-3],
        }
    )

    prepared = compute_k_values(frame, config=KValueConfig())

    assert prepared.loc[0, "rapid_width_source_field"] == "wid_adj_routing"
    assert prepared.loc[0, "rapid_width_m"] == 30.0


def test_compute_k_values_does_not_silently_use_total_corridor_width() -> None:
    frame = pd.DataFrame(
        {
            "link_length_m": [1000.0],
            "wid_adj_total": [90.0],
            "wid_adj": [40.0],
            "slope_used": [1e-3],
        }
    )

    prepared = compute_k_values(frame, config=KValueConfig())

    assert prepared.loc[0, "rapid_width_source_field"] == "wid_adj"
    assert prepared.loc[0, "rapid_width_m"] == 40.0


def test_prepare_experiment_accepts_real_link_id_zero(tmp_path: Path) -> None:
    experiment_dir = _write_state(tmp_path, link_id=0)
    registry = prepare_experiment(
        experiment_dir,
        forcing_path=experiment_dir / "forcing.csv",
        prep_config=RapidPrepConfig(),
    )
    assert registry.loc[0, "status"] == "prepared"


def test_compute_reach_ratios_handles_real_reach_id_zero(tmp_path: Path) -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(1, node_type="source")
    graph.add_node(2, node_type="internal")
    graph.add_node(3, node_type="outlet")
    graph.add_edge(1, 2, key="0", reach_id=0, width=10.0, length=100.0, geometry=LineString([(0.0, 0.0), (1.0, 0.0)]).wkt, slope_local=1e-3)
    graph.add_edge(2, 3, key="5", reach_id=5, width=20.0, length=100.0, geometry=LineString([(1.0, 0.0), (2.0, 0.0)]).wkt, slope_local=1e-3)

    create_conn_file(graph, tmp_path)
    rat_srt = compute_reach_ratios(graph, tmp_path, use_widths=True)

    assert Path(rat_srt).exists()
    rat = pd.read_csv(rat_srt, header=None)
    assert set(rat[0].astype(int).tolist()) == {0, 5}


def test_build_rapid_graph_uses_branch_widths_for_split_ratios(tmp_path: Path) -> None:
    nodes = gpd.GeoDataFrame(
        {
            "id_node": [1, 2, 3, 4],
            "is_inlet": [True, False, False, False],
            "is_outlet": [False, False, True, True],
        },
        geometry=[
            Point(0.0, 0.0),
            Point(1.0, 0.0),
            Point(2.0, 1.0),
            Point(2.0, -1.0),
        ],
        crs="EPSG:3857",
    )
    rapid_links = gpd.GeoDataFrame(
        {
            "reach_id": [10, 11, 12],
            "id_us_node": [1, 2, 2],
            "id_ds_node": [2, 3, 4],
            "rapid_width_m": [50.0, 90.0, 10.0],
            "rapid_branch_width_m": [50.0, 20.0, 80.0],
            "link_length_m": [100.0, 100.0, 100.0],
            "slope_used": [1e-3, 1e-3, 1e-3],
        },
        geometry=[
            LineString([(0.0, 0.0), (1.0, 0.0)]),
            LineString([(1.0, 0.0), (2.0, 1.0)]),
            LineString([(1.0, 0.0), (2.0, -1.0)]),
        ],
        crs="EPSG:3857",
    )

    graph = build_rapid_graph(rapid_links, nodes)
    create_conn_file(graph, tmp_path)
    rat_srt = compute_reach_ratios(graph, tmp_path, use_widths=True)
    rat = pd.read_csv(rat_srt, header=None)
    row = rat.loc[rat[0].astype(int) == 10].iloc[0].tolist()

    assert row[1] == "[11, 0.2]"
    assert row[2] == "[12, 0.8]"


def test_prepare_experiment_splits_long_links_into_target_subreaches(tmp_path: Path) -> None:
    experiment_dir = _write_state(tmp_path, link_length=1100.0)
    registry = prepare_experiment(
        experiment_dir,
        forcing_path=experiment_dir / "forcing.csv",
        prep_config=RapidPrepConfig(target_subreach_length_m=500.0),
    )
    assert registry.loc[0, "status"] == "prepared"

    prep_dir = experiment_dir / "states" / "S001_unit_1" / "rapid" / "prep"
    rapid_links = pd.read_csv(prep_dir / "rapid_link_attributes.csv")
    rapid_nodes = pd.read_csv(prep_dir / "rapid_node_attributes.csv")

    assert len(rapid_links) == 2
    assert set(rapid_links["parent_link_id"]) == {11}
    assert set(rapid_links["subreach_count"]) == {2}
    assert set(rapid_links["subreach_index"]) == {1, 2}
    assert set(np.round(rapid_links["link_length_m"], 6)) == {550.0}
    assert int(rapid_nodes["rapid_node_source"].eq("subreach_virtual").sum()) == 1


def test_prepare_experiment_can_vary_width_by_subreach_from_samples(tmp_path: Path) -> None:
    experiment_dir = _write_state(
        tmp_path,
        link_length=1100.0,
        width_samples_wet=[20.0, 20.0, 30.0, 30.0, 100.0, 110.0, 120.0],
    )
    registry = prepare_experiment(
        experiment_dir,
        forcing_path=experiment_dir / "forcing.csv",
        prep_config=RapidPrepConfig(
            target_subreach_length_m=500.0,
            vary_width_by_subreach=True,
            subreach_width_sample_field="width_wet",
        ),
    )
    assert registry.loc[0, "status"] == "prepared"

    prep_dir = experiment_dir / "states" / "S001_unit_1" / "rapid" / "prep"
    rapid_links = pd.read_csv(prep_dir / "rapid_link_attributes.csv").sort_values("subreach_index").reset_index(drop=True)

    assert set(rapid_links["rapid_width_source_field"]) == {"rapid_subreach_width_m"}
    assert rapid_links["rapid_subreach_width_fallback_used"].tolist() == [False, False]
    assert rapid_links["rapid_subreach_width_sample_count"].tolist() == [3, 4]
    assert rapid_links["rapid_branch_width_m"].round(6).tolist() == [50.0, 50.0]
    assert rapid_links.loc[0, "rapid_width_m"] < rapid_links.loc[1, "rapid_width_m"]
    assert rapid_links["rapid_width_m"].round(6).tolist() == [20.0, 105.0]


def test_run_prepared_experiment_handles_split_subreaches(tmp_path: Path) -> None:
    experiment_dir = _write_state(tmp_path, link_length=1100.0)
    prepare_experiment(
        experiment_dir,
        forcing_path=experiment_dir / "forcing.csv",
        prep_config=RapidPrepConfig(target_subreach_length_m=500.0),
    )

    run_registry = run_prepared_experiment(experiment_dir)
    assert run_registry.loc[0, "status"] == "ran"


def test_compute_k_values_reports_raw_and_capped_celerity() -> None:
    frame = pd.DataFrame(
        {
            "id_link": [11],
            "link_length_m": [1000.0],
            "slope_used": [1e-4],
            "wid_adj_wet": [200.0],
        }
    )

    uncapped = compute_k_values(frame, config=KValueConfig())
    capped = compute_k_values(
        frame,
        config=KValueConfig(
            use_celerity_capping=True,
            min_celerity_mps=0.28,
            max_celerity_mps=1.524,
        ),
    )

    raw_celerity = float(uncapped.loc[0, "rapid_celerity_mps_raw"])
    assert raw_celerity < 0.28
    assert float(uncapped.loc[0, "rapid_celerity_mps"]) == raw_celerity
    assert not bool(uncapped.loc[0, "rapid_celerity_capped"])

    assert abs(float(capped.loc[0, "rapid_celerity_mps"]) - 0.28) < 1e-12
    assert bool(capped.loc[0, "rapid_celerity_capped"])
    assert capped.loc[0, "rapid_k_source_method"] == "hydraulic_celerity_capped"
    assert abs(float(capped.loc[0, "rapid_k"]) - (1000.0 / 0.28)) < 1e-9


def test_compute_k_values_applies_effective_length_floor() -> None:
    frame = pd.DataFrame(
        {
            "id_link": [11, 12],
            "link_length_m": [30.0, 130.0],
            "slope_used": [1e-4, 1e-4],
            "wid_adj_wet": [200.0, 200.0],
        }
    )

    floored = compute_k_values(
        frame,
        config=KValueConfig(
            min_effective_length_m=100.0,
        ),
    )

    assert floored["rapid_effective_length_m"].round(6).tolist() == [100.0, 130.0]
    assert floored["rapid_length_floor_applied"].tolist() == [True, False]
    celerity = float(floored.loc[0, "rapid_celerity_mps"])
    assert abs(float(floored.loc[0, "rapid_k"]) - (100.0 / celerity)) < 1e-9
    assert abs(float(floored.loc[1, "rapid_k"]) - (130.0 / celerity)) < 1e-9


def test_apply_stability_effective_length_cap_targets_min_celerity_reaches() -> None:
    frame = pd.DataFrame(
        {
            "rapid_effective_length_m": [200.0, 219.4320578362918],
            "rapid_effective_length_floor_m": [np.nan, np.nan],
            "rapid_celerity_mps": [0.5747646461475523, 0.07],
            "rapid_celerity_capped": [False, True],
            "rapid_k_source_method": ["hydraulic_celerity_uncapped", "hydraulic_celerity_capped"],
            "rapid_k": [347.96833928973554, 3134.743683375597],
        }
    )

    adjusted = apply_stability_effective_length_cap(
        frame,
        x_value=0.1,
        min_celerity_mps=0.07,
        safety_margin_seconds=10.0,
    )

    expected_cap = (626.3430107215239 - 10.0) * 0.07 / 0.2
    assert bool(adjusted.loc[0, "rapid_stability_length_cap_applied"]) is False
    assert bool(adjusted.loc[1, "rapid_stability_length_cap_applied"]) is True
    assert abs(float(adjusted.loc[1, "rapid_effective_length_m"]) - expected_cap) < 1e-9
    assert adjusted.loc[1, "rapid_k_source_method"] == "hydraulic_celerity_capped+stability_length_cap"
    assert float(adjusted.loc[0, "rapid_stability_length_cap_dt_min_after_seconds"]) < float(
        adjusted.loc[0, "rapid_stability_length_cap_dt_max_after_seconds"]
    )


def test_apply_stability_effective_length_cap_respects_effective_length_floor() -> None:
    frame = pd.DataFrame(
        {
            "rapid_effective_length_m": [200.0, 240.0],
            "rapid_effective_length_floor_m": [np.nan, 218.0],
            "rapid_celerity_mps": [0.5747646461475523, 0.07],
            "rapid_celerity_capped": [False, True],
            "rapid_k_source_method": ["hydraulic_celerity_uncapped", "hydraulic_celerity_capped"],
            "rapid_k": [347.96833928973554, 240.0 / 0.07],
        }
    )

    adjusted = apply_stability_effective_length_cap(
        frame,
        x_value=0.1,
        min_celerity_mps=0.07,
        safety_margin_seconds=10.0,
    )

    assert bool(adjusted.loc[0, "rapid_stability_length_cap_applied"]) is False
    assert bool(adjusted.loc[1, "rapid_stability_length_cap_applied"]) is True
    assert abs(float(adjusted.loc[1, "rapid_effective_length_m"]) - 218.0) < 1e-12


def test_apply_stability_effective_length_cap_targets_near_floor_reaches() -> None:
    frame = pd.DataFrame(
        {
            "rapid_effective_length_m": [200.0, 239.610667],
            "rapid_effective_length_floor_m": [np.nan, np.nan],
            "rapid_celerity_mps": [0.5747646461475523, 0.071441],
            "rapid_celerity_capped": [False, False],
            "rapid_k_source_method": ["hydraulic_celerity_uncapped", "hydraulic_celerity_uncapped"],
            "rapid_k": [347.96833928973554, 239.610667 / 0.071441],
        }
    )

    adjusted = apply_stability_effective_length_cap(
        frame,
        x_value=0.1,
        min_celerity_mps=0.07,
        safety_margin_seconds=10.0,
        celerity_rel_buffer=0.03,
    )

    expected_cap = (626.3430107215239 - 10.0) * 0.071441 / 0.2
    assert bool(adjusted.loc[1, "rapid_stability_length_cap_applied"]) is True
    assert abs(float(adjusted.loc[1, "rapid_effective_length_m"]) - expected_cap) < 1e-9


def test_apply_stability_effective_length_cap_adapts_to_high_k_tail() -> None:
    """A slow reach outside the near-floor buffer must not collapse state dt."""
    frame = pd.DataFrame(
        {
            "rapid_effective_length_m": [293.918117, 224.807529],
            "rapid_effective_length_floor_m": [200.0, 200.0],
            "rapid_celerity_mps": [0.054203, 0.389380],
            "rapid_celerity_capped": [False, False],
            "rapid_k_source_method": [
                "hydraulic_celerity_uncapped",
                "hydraulic_celerity_uncapped",
            ],
        }
    )
    frame["rapid_k"] = (
        frame["rapid_effective_length_m"] / frame["rapid_celerity_mps"]
    )

    adjusted = apply_stability_effective_length_cap(
        frame,
        x_value=0.1,
        min_celerity_mps=0.05,
        safety_margin_seconds=10.0,
        celerity_rel_buffer=0.03,
    )

    assert bool(adjusted.loc[0, "rapid_stability_length_cap_near_floor_candidate"]) is False
    assert bool(adjusted.loc[0, "rapid_stability_length_cap_adaptive_applied"]) is True
    assert adjusted.loc[0, "rapid_stability_length_cap_method"] == "adaptive_high_k_tail"
    assert bool(adjusted.loc[1, "rapid_stability_length_cap_applied"]) is False
    assert bool(adjusted["rapid_stability_length_cap_unresolved"].any()) is False
    assert float(adjusted.loc[0, "rapid_stability_length_cap_dt_min_after_seconds"]) < float(
        adjusted.loc[0, "rapid_stability_length_cap_dt_max_after_seconds"]
    )


def test_compute_routing_dt_seconds_prefers_requested_timestep() -> None:
    k_values = pd.Series([1000.0, 277.77777777777777], dtype=float)
    selected = compute_routing_dt_seconds(
        k_values,
        x_value=0.1,
        forcing_dt_seconds=960,
        preferred_routing_dt_seconds=480,
    )
    assert selected == 480


def test_find_nearby_compatible_forcing_and_routing_dt_prefers_multiple_of_five() -> None:
    k_values = pd.Series([3134.745], dtype=float)
    forcing_dt_seconds, routing_dt_seconds = find_nearby_compatible_forcing_and_routing_dt(
        k_values,
        x_value=0.1,
        forcing_dt_seconds=620,
        preferred_routing_dt_seconds=620,
        search_max_offset_seconds=20,
    )
    assert forcing_dt_seconds == 630
    assert routing_dt_seconds == 630


def test_select_coordinated_routing_timesteps_uses_common_midpoint() -> None:
    plan = select_coordinated_routing_timesteps(
        {
            "S000_base": (731.083, 741.083),
            "S001_unit_1": (733.0, 744.0),
        },
        preferred_routing_dt_seconds=980,
        multiple_seconds=5,
    )

    assert plan["selection_mode"] == "common_intersection"
    assert plan["selected_unique_routing_dt_seconds"] == [735]
    assert plan["states"]["S000_base"]["routing_dt_seconds"] == 735
    assert plan["states"]["S001_unit_1"]["forcing_dt_seconds"] == 735


def test_select_coordinated_routing_timesteps_keeps_disjoint_states_close() -> None:
    plan = select_coordinated_routing_timesteps(
        {
            "S000_base": (731.083, 741.083),
            "S021_unit_15": (795.111, 805.111),
            "CB001_bubble_5": (703.770, 713.770),
        },
        preferred_routing_dt_seconds=980,
        multiple_seconds=5,
    )

    assert plan["selection_mode"] == "coordinated_state_specific"
    assert plan["states"]["S000_base"]["routing_dt_seconds"] == 740
    assert plan["states"]["S021_unit_15"]["routing_dt_seconds"] == 800
    assert plan["states"]["CB001_bubble_5"]["routing_dt_seconds"] == 710


def test_timestep_coordination_discovers_independent_compound_sibling(tmp_path: Path) -> None:
    independent = tmp_path / "13_indep__anchored"
    compound = tmp_path / "13_compound_bubbles__anchored"
    independent.mkdir()
    compound.mkdir()
    (independent / "state_registry.csv").write_text("state_id,state_dir\n", encoding="utf-8")
    (compound / "state_registry.csv").write_text("state_id,state_dir\n", encoding="utf-8")

    assert rapid_prep_module._sibling_experiment_for_timestep_coordination(independent) == compound.resolve()
    assert rapid_prep_module._sibling_experiment_for_timestep_coordination(compound) == independent.resolve()


def test_coordinated_sibling_prepass_uses_and_cleans_isolated_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    independent = _write_state(
        tmp_path / "independent",
        experiment_name="13_indep__anchored",
    )
    compound = _write_state(
        tmp_path / "compound",
        experiment_name="13_compound_bubbles__anchored",
    )
    work_root = tmp_path / "rapid-work"
    monkeypatch.setattr(
        rapid_prep_module,
        "_sibling_experiment_for_timestep_coordination",
        lambda experiment_path: independent if experiment_path == compound.resolve() else None,
    )

    registry = prepare_experiment(
        compound,
        forcing_path=compound / "forcing.csv",
        prep_config=RapidPrepConfig(preferred_routing_dt_seconds=60),
        work_root=work_root,
    )

    assert registry.loc[0, "status"] == "prepared"
    assert (work_root / compound.name / "S001_unit_1" / "prep").exists()
    assert not (work_root / independent.name).exists()
    assert not list((work_root / "_coordination_prepass").glob("*"))

    manifest = json.loads((compound / "rapid_prep_manifest.json").read_text())
    assert manifest["sibling_timestep_prepass_status"] == "removed_after_successful_timestep_selection"


def test_failed_coordinated_sibling_prepass_is_retained_for_diagnosis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    independent = _write_state(
        tmp_path / "independent",
        experiment_name="13_indep__anchored",
    )
    compound = _write_state(
        tmp_path / "compound",
        experiment_name="13_compound_bubbles__anchored",
    )
    work_root = tmp_path / "rapid-work"
    monkeypatch.setattr(
        rapid_prep_module,
        "_sibling_experiment_for_timestep_coordination",
        lambda experiment_path: independent if experiment_path == compound.resolve() else None,
    )
    original_prepare_state = rapid_prep_module.prepare_state

    def fail_only_sibling(context, *args, **kwargs):
        if independent.name in str(context.rapid_prep_dir):
            raise ValueError("synthetic sibling prepass failure")
        return original_prepare_state(context, *args, **kwargs)

    monkeypatch.setattr(rapid_prep_module, "prepare_state", fail_only_sibling)

    registry = prepare_experiment(
        compound,
        forcing_path=compound / "forcing.csv",
        prep_config=RapidPrepConfig(preferred_routing_dt_seconds=60),
        work_root=work_root,
    )

    assert registry.loc[0, "status"] == "failed"
    retained = list((work_root / "_coordination_prepass").glob("*/coordination_prepass_manifest.json"))
    assert len(retained) == 1
    failure = json.loads(retained[0].read_text())
    assert failure["status"] == "retained_timestep_selection_failure"
    assert "synthetic sibling prepass failure" in failure["error"]


def test_integrate_trapezoid_falls_back_to_trapz(monkeypatch) -> None:
    monkeypatch.delattr(np, "trapezoid", raising=False)
    area = _integrate_trapezoid(
        np.array([0.0, 1.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 2.0], dtype=float),
    )
    assert abs(area - 1.0) < 1e-12


def test_compute_link_slopes_interpolates_along_corridor() -> None:
    nodes = gpd.GeoDataFrame(
        {
            "id_node": [1, 2, 3],
            "sword_wse": [10.0, 1.0, 8.0],
            "sword_wse_field": ["wse_obs_p50", "wse", "wse_obs_p50"],
            "sword_wse_fill_method": ["requested_field", "same_node_wse", "requested_field"],
            "sword_wse_fallback_used": [False, True, False],
            "sword_node_id": [101, 102, 103],
            "sword_reach_id": [999, 888, 999],
            "sword_dist_out": [300.0, 200.0, 100.0],
        },
        geometry=[Point(0.0, 0.0), Point(1000.0, 0.0), Point(2000.0, 0.0)],
        crs="EPSG:3857",
    )
    links = gpd.GeoDataFrame(
        {
            "id_link": [11, 12],
            "id_us_node": [1, 2],
            "id_ds_node": [2, 3],
            "len": [1000.0, 1000.0],
        },
        geometry=[
            LineString([(0.0, 0.0), (1000.0, 0.0)]),
            LineString([(1000.0, 0.0), (2000.0, 0.0)]),
        ],
        crs="EPSG:3857",
    )

    slopes = compute_link_slopes(links, nodes, config=SlopeConfig())
    by_link = slopes.set_index("id_link")
    assert by_link.loc[11, "wse_ds_source_method"] == "profile_monotone_requested_field_nodes"
    assert by_link.loc[11, "wse_ds_interpolated"]
    assert abs(by_link.loc[11, "wse_ds"] - 9.0) < 1e-9
    assert abs(by_link.loc[11, "raw_slope"] - 0.001) < 1e-9
    assert by_link.loc[12, "wse_us_source_method"] == "profile_monotone_requested_field_nodes"
    assert abs(by_link.loc[12, "wse_us"] - 9.0) < 1e-9


def test_compute_link_slopes_fills_flat_segment_from_neighbor() -> None:
    nodes = gpd.GeoDataFrame(
        {
            "id_node": [1, 2, 3, 4],
            "sword_wse": [10.0, 9.0, 9.0, 8.0],
            "sword_wse_field": ["wse_obs_p50"] * 4,
            "sword_wse_fill_method": ["requested_field"] * 4,
            "sword_wse_fallback_used": [False] * 4,
            "sword_node_id": [101, 102, 103, 104],
            "sword_dist_out": [400.0, 300.0, 200.0, 100.0],
        },
        geometry=[
            Point(0.0, 0.0),
            Point(1000.0, 0.0),
            Point(2000.0, 0.0),
            Point(3000.0, 0.0),
        ],
        crs="EPSG:3857",
    )
    links = gpd.GeoDataFrame(
        {
            "id_link": [11, 12, 13],
            "id_us_node": [1, 2, 3],
            "id_ds_node": [2, 3, 4],
            "len": [1000.0, 1000.0, 1000.0],
        },
        geometry=[
            LineString([(0.0, 0.0), (1000.0, 0.0)]),
            LineString([(1000.0, 0.0), (2000.0, 0.0)]),
            LineString([(2000.0, 0.0), (3000.0, 0.0)]),
        ],
        crs="EPSG:3857",
    )

    slopes = compute_link_slopes(links, nodes, config=SlopeConfig())
    by_link = slopes.set_index("id_link")
    assert abs(by_link.loc[11, "raw_slope"] - 0.001) < 1e-9
    assert abs(by_link.loc[13, "raw_slope"] - 0.001) < 1e-9
    assert by_link.loc[12, "slope_source_method"] == "nearest_valid_link"
    assert by_link.loc[12, "slope_reason"] == "filled_from_neighbor_link"
    assert int(by_link.loc[12, "slope_neighbor_distance"]) == 1
    assert int(by_link.loc[12, "slope_neighbor_source_link_id"]) in {11, 13}
    assert abs(by_link.loc[12, "slope_used"] - 0.001) < 1e-9


def test_compute_link_slopes_flags_section_ratio_outlier_and_fills_from_neighbor() -> None:
    nodes = gpd.GeoDataFrame(
        {
            "id_node": [1, 2, 3, 4, 5],
            "sword_wse": [10.0, 9.0, 3.0, 2.0, 1.0],
            "sword_wse_field": ["wse_obs_p50"] * 5,
            "sword_wse_fill_method": ["requested_field"] * 5,
            "sword_wse_fallback_used": [False] * 5,
            "sword_node_id": [101, 102, 103, 104, 105],
            "sword_dist_out": [4000.0, 3000.0, 2000.0, 1000.0, 0.0],
        },
        geometry=[
            Point(0.0, 0.0),
            Point(1000.0, 0.0),
            Point(2000.0, 0.0),
            Point(3000.0, 0.0),
            Point(4000.0, 0.0),
        ],
        crs="EPSG:3857",
    )
    links = gpd.GeoDataFrame(
        {
            "id_link": [11, 12, 13, 14],
            "id_us_node": [1, 2, 3, 4],
            "id_ds_node": [2, 3, 4, 5],
            "len": [1000.0, 1000.0, 1000.0, 1000.0],
        },
        geometry=[
            LineString([(0.0, 0.0), (1000.0, 0.0)]),
            LineString([(1000.0, 0.0), (2000.0, 0.0)]),
            LineString([(2000.0, 0.0), (3000.0, 0.0)]),
            LineString([(3000.0, 0.0), (4000.0, 0.0)]),
        ],
        crs="EPSG:3857",
    )

    slopes = compute_link_slopes(
        links,
        nodes,
        config=SlopeConfig(
            section_slope_ratio_max=2.0,
        ),
    )

    by_link = slopes.set_index("id_link")
    assert abs(float(by_link.loc[12, "raw_slope"]) - 0.006) < 1e-9
    assert abs(float(by_link.loc[12, "slope_section_ref"]) - 0.00225) < 1e-9
    assert bool(by_link.loc[12, "slope_outlier_flag"])
    assert by_link.loc[12, "slope_outlier_reason"] == "above_section_ratio"
    assert by_link.loc[12, "slope_source_method"] == "nearest_valid_link"
    assert by_link.loc[12, "slope_reason"] == "filled_from_neighbor_link"
    assert int(by_link.loc[12, "slope_neighbor_distance"]) == 1
    assert int(by_link.loc[12, "slope_neighbor_source_link_id"]) in {11, 13}
    assert abs(float(by_link.loc[12, "slope_used"]) - 0.001) < 1e-9


def test_compute_link_slopes_exports_compound_bubble_qc_for_valid_bubble() -> None:
    nodes = gpd.GeoDataFrame(
        {
            "id_node": [1, 2, 3, 4, 5, 6],
            "sword_wse": [12.0, 11.0, 10.9, 10.3, 10.0, 9.5],
            "sword_wse_field": ["wse_obs_p50"] * 6,
            "sword_wse_fill_method": ["requested_field"] * 6,
            "sword_wse_fallback_used": [False] * 6,
            "sword_node_id": [101, 102, 103, 104, 105, 106],
            "sword_dist_out": [600.0, 500.0, 400.0, 350.0, 200.0, 0.0],
        },
        geometry=[
            Point(0.0, 0.0),
            Point(1.0, 0.0),
            Point(2.0, 1.0),
            Point(2.0, -1.0),
            Point(3.0, 0.0),
            Point(4.0, 0.0),
        ],
        crs="EPSG:3857",
    )
    links = gpd.GeoDataFrame(
        {
            "id_link": [10, 11, 12, 13, 14, 15],
            "id_us_node": [1, 2, 3, 2, 4, 5],
            "id_ds_node": [2, 3, 5, 4, 5, 6],
            "len": [100.0] * 6,
        },
        geometry=[
            LineString([(0.0, 0.0), (1.0, 0.0)]),
            LineString([(1.0, 0.0), (2.0, 1.0)]),
            LineString([(2.0, 1.0), (3.0, 0.0)]),
            LineString([(1.0, 0.0), (2.0, -1.0)]),
            LineString([(2.0, -1.0), (3.0, 0.0)]),
            LineString([(3.0, 0.0), (4.0, 0.0)]),
        ],
        crs="EPSG:3857",
    )
    unit_summary = pd.DataFrame(
        [
            {
                "unit_id": 10,
                "unit_node_ids": "2,3,5",
                "compound_bubble_id": 6,
                "in_compound_bubble": True,
                "compound_bubble_role": "bubble_root",
            },
            {
                "unit_id": 11,
                "unit_node_ids": "2,4,5",
                "compound_bubble_id": 6,
                "in_compound_bubble": True,
                "compound_bubble_role": "bubble_member",
            },
        ]
    )

    slopes = compute_link_slopes(links, nodes, unit_summary=unit_summary, config=SlopeConfig())
    by_link = slopes.set_index("id_link")
    for link_id in (11, 12, 13, 14):
        assert int(by_link.loc[link_id, "compound_bubble_id_link"]) == 6
        assert bool(by_link.loc[link_id, "bubble_slope_qc_candidate"]) is True
        assert bool(by_link.loc[link_id, "bubble_slope_qc_valid"]) is True
        assert bool(by_link.loc[link_id, "bubble_path_drop_qc_failed"]) is False
        assert by_link.loc[link_id, "bubble_slope_qc_reason"] == "ok"
        assert int(by_link.loc[link_id, "bubble_entry_node_id"]) == 2
        assert int(by_link.loc[link_id, "bubble_exit_node_id"]) == 5
        assert int(by_link.loc[link_id, "bubble_n_paths_checked"]) == 2
        assert abs(float(by_link.loc[link_id, "bubble_reference_drop_m"]) - 1.0) < 1e-9
        assert abs(float(by_link.loc[link_id, "bubble_reference_slope"]) - 0.3535533905932738) < 1e-9
        assert abs(float(by_link.loc[link_id, "bubble_link_slope_ref"]) - float(by_link.loc[link_id, "slope_local_raw"])) < 1e-9


def test_compute_link_slopes_exports_compound_bubble_qc_failure_for_adjusted_path_drop() -> None:
    nodes = gpd.GeoDataFrame(
        {
            "id_node": [1, 2, 3, 4, 5, 6],
            "sword_wse": [12.0, 11.0, 10.9, 10.3, 10.0, 9.5],
            "sword_wse_field": ["wse_obs_p50"] * 6,
            "sword_wse_fill_method": ["requested_field"] * 6,
            "sword_wse_fallback_used": [False] * 6,
            "sword_node_id": [101, 102, 103, 104, 105, 106],
            "sword_dist_out": [600.0, 500.0, 400.0, 350.0, 200.0, 0.0],
        },
        geometry=[
            Point(0.0, 0.0),
            Point(1.0, 0.0),
            Point(2.0, 1.0),
            Point(2.0, -1.0),
            Point(3.0, 0.0),
            Point(4.0, 0.0),
        ],
        crs="EPSG:3857",
    )
    links = gpd.GeoDataFrame(
        {
            "id_link": [10, 11, 12, 13, 14, 15],
            "id_us_node": [1, 2, 3, 2, 4, 5],
            "id_ds_node": [2, 3, 5, 4, 5, 6],
            "len": [100.0] * 6,
        },
        geometry=[
            LineString([(0.0, 0.0), (1.0, 0.0)]),
            LineString([(1.0, 0.0), (2.0, 1.0)]),
            LineString([(2.0, 1.0), (3.0, 0.0)]),
            LineString([(1.0, 0.0), (2.0, -1.0)]),
            LineString([(2.0, -1.0), (3.0, 0.0)]),
            LineString([(3.0, 0.0), (4.0, 0.0)]),
        ],
        crs="EPSG:3857",
    )
    unit_summary = pd.DataFrame(
        [
            {
                "unit_id": 10,
                "unit_node_ids": "2,3,5",
                "compound_bubble_id": 6,
                "in_compound_bubble": True,
                "compound_bubble_role": "bubble_root",
            },
            {
                "unit_id": 11,
                "unit_node_ids": "2,4,5",
                "compound_bubble_id": 6,
                "in_compound_bubble": True,
                "compound_bubble_role": "bubble_member",
            },
        ]
    )

    slopes = compute_link_slopes(
        links,
        nodes,
        unit_summary=unit_summary,
        config=SlopeConfig(max_slope_for_k=0.005),
    )
    by_link = slopes.set_index("id_link")
    for link_id in (11, 12, 13, 14):
        assert int(by_link.loc[link_id, "compound_bubble_id_link"]) == 6
        assert bool(by_link.loc[link_id, "bubble_slope_qc_candidate"]) is True
        assert bool(by_link.loc[link_id, "bubble_slope_qc_valid"]) is True
        assert bool(by_link.loc[link_id, "bubble_path_drop_qc_failed"]) is True
        assert by_link.loc[link_id, "bubble_slope_qc_reason"] == "path_drop_mismatch"
        assert float(by_link.loc[link_id, "bubble_path_drop_max_abs_error_m"]) > 0.0


def test_compute_link_slopes_repairs_plateaued_bubble_reference_profile() -> None:
    nodes = gpd.GeoDataFrame(
        {
            "id_node": [1, 2, 3, 4, 5, 6],
            "sword_wse": [6.0, 5.2, 2.87703, 4.07494, 3.755675, 3.14151],
            "sword_wse_field": ["wse_obs_p50"] * 6,
            "sword_wse_fill_method": [
                "requested_field",
                "same_node_wse",
                "requested_field",
                "requested_field",
                "requested_field",
                "requested_field",
            ],
            "sword_wse_fallback_used": [False] * 6,
            "sword_node_id": [101, 102, 103, 104, 105, 106],
            "sword_dist_out": [12000.0, 10000.0, 8900.0, 8100.0, 7300.0, 0.0],
        },
        geometry=[
            Point(0.0, 0.0),
            Point(1.0, 0.0),
            Point(2.0, 1.0),
            Point(2.0, -1.0),
            Point(3.0, 0.0),
            Point(4.0, 0.0),
        ],
        crs="EPSG:3857",
    )
    links = gpd.GeoDataFrame(
        {
            "id_link": [10, 11, 12, 13, 14, 15],
            "id_us_node": [1, 2, 3, 2, 4, 5],
            "id_ds_node": [2, 3, 5, 4, 5, 6],
            "len": [100.0] * 6,
        },
        geometry=[
            LineString([(0.0, 0.0), (1.0, 0.0)]),
            LineString([(1.0, 0.0), (2.0, 1.0)]),
            LineString([(2.0, 1.0), (3.0, 0.0)]),
            LineString([(1.0, 0.0), (2.0, -1.0)]),
            LineString([(2.0, -1.0), (3.0, 0.0)]),
            LineString([(3.0, 0.0), (4.0, 0.0)]),
        ],
        crs="EPSG:3857",
    )
    unit_summary = pd.DataFrame(
        [
            {
                "unit_id": 10,
                "unit_node_ids": "2,3,5",
                "compound_bubble_id": 16,
                "in_compound_bubble": True,
                "compound_bubble_role": "bubble_root",
            },
            {
                "unit_id": 11,
                "unit_node_ids": "2,4,5",
                "compound_bubble_id": 16,
                "in_compound_bubble": True,
                "compound_bubble_role": "bubble_member",
            },
        ]
    )

    slopes = compute_link_slopes(links, nodes, unit_summary=unit_summary, config=SlopeConfig())
    by_link = slopes.set_index("id_link")
    for link_id in (11, 12, 13, 14):
        assert int(by_link.loc[link_id, "compound_bubble_id_link"]) == 16
        assert bool(by_link.loc[link_id, "bubble_slope_qc_valid"]) is True
        assert bool(by_link.loc[link_id, "bubble_path_drop_qc_failed"]) is False
        assert bool(by_link.loc[link_id, "bubble_slope_repair_candidate"]) is True
        assert bool(by_link.loc[link_id, "bubble_slope_repair_applied"]) is True
        assert by_link.loc[link_id, "bubble_slope_repair_method"] == "bubble_weighted_path_drop_projection"
        assert by_link.loc[link_id, "bubble_reference_profile_method"] == "bubble_weighted_path_drop_projection"
        assert by_link.loc[link_id, "slope_source_method"] == "compound_bubble_profile"
        assert by_link.loc[link_id, "slope_reason"] == "filled_from_compound_bubble_profile"
        assert float(by_link.loc[link_id, "bubble_link_slope_ref"]) > 0.0
        assert float(by_link.loc[link_id, "slope_used"]) == float(by_link.loc[link_id, "bubble_link_slope_ref"])
        assert float(by_link.loc[link_id, "bubble_slope_repair_change_factor"]) > 0.0


def test_prepare_experiment_writes_link_attributes_before_forcing_failure(tmp_path: Path, monkeypatch) -> None:
    experiment_dir = _write_state(tmp_path)

    def _raise(*args, **kwargs):
        raise ValueError("synthetic forcing failure")

    monkeypatch.setattr("rapid_tools.prep.compute_routing_dt_seconds", _raise)
    registry = prepare_experiment(
        experiment_dir,
        forcing_path=experiment_dir / "forcing.csv",
        prep_config=RapidPrepConfig(),
    )

    assert registry.loc[0, "status"] == "failed"
    prep_dir = experiment_dir / "states" / "S001_unit_1" / "rapid" / "prep"
    assert (prep_dir / "rapid_link_attributes.csv").exists()
    assert (prep_dir / "rapid_node_attributes.csv").exists()


def test_prepare_experiment_exports_celerity_columns(tmp_path: Path) -> None:
    experiment_dir = _write_state(tmp_path)
    registry = prepare_experiment(
        experiment_dir,
        forcing_path=experiment_dir / "forcing.csv",
        prep_config=RapidPrepConfig(
            min_effective_length_for_k_m=1500.0,
            use_celerity_capping=True,
            min_celerity_mps=0.28,
            max_celerity_mps=1.524,
            use_stability_effective_length_cap=True,
            stability_effective_length_cap_margin_seconds=10.0,
        ),
    )

    assert registry.loc[0, "status"] == "prepared"
    prep_dir = experiment_dir / "states" / "S001_unit_1" / "rapid" / "prep"
    rapid_links = pd.read_csv(prep_dir / "rapid_link_attributes.csv")
    for column in (
        "rapid_celerity_mps_raw",
        "rapid_celerity_mps",
        "rapid_celerity_capped",
        "rapid_k_source_method",
        "rapid_celerity_cap_enabled",
        "rapid_celerity_cap_min_mps",
        "rapid_celerity_cap_max_mps",
        "rapid_effective_length_m",
        "rapid_length_floor_applied",
        "rapid_stability_length_cap_enabled",
        "rapid_effective_length_m_pre_stability_cap",
        "rapid_stability_length_cap_candidate",
        "rapid_stability_length_cap_applied",
        "rapid_stability_length_cap_triggered",
        "rapid_stability_length_cap_target_m",
        "rapid_stability_length_cap_active_m",
        "rapid_stability_length_cap_dt_min_before_seconds",
        "rapid_stability_length_cap_dt_max_before_seconds",
        "rapid_stability_length_cap_dt_min_after_seconds",
        "rapid_stability_length_cap_dt_max_after_seconds",
        "slope_outlier_flag",
        "bubble_slope_qc_candidate",
        "bubble_slope_qc_valid",
        "bubble_slope_qc_reason",
        "compound_bubble_id_link",
        "bubble_path_drop_qc_failed",
        "bubble_slope_repair_candidate",
        "bubble_slope_repair_applied",
        "bubble_slope_repair_method",
        "bubble_slope_repair_change_factor",
        "bubble_slope_repair_large_change_flag",
    ):
        assert column in rapid_links.columns
    for column in (
        "n_source_links",
        "n_links",
        "link_multiplier",
        "pct_celerity_capped",
        "n_length_floor_applied",
        "pct_length_floor_applied",
        "n_slope_outlier_flagged",
        "pct_slope_outlier_flagged",
        "n_compound_bubbles_qc",
        "n_valid_compound_bubbles_qc",
        "n_failed_compound_bubbles_qc",
        "n_repaired_compound_bubbles_qc",
        "rapid_k_min",
        "rapid_k_max",
    ):
        assert column in registry.columns
    assert bool(rapid_links.loc[0, "rapid_length_floor_applied"])


def test_run_prepared_experiment_allows_manual_event_start_time(tmp_path: Path) -> None:
    experiment_dir = _write_state(tmp_path)
    prepare_experiment(
        experiment_dir,
        forcing_path=experiment_dir / "forcing.csv",
        prep_config=RapidPrepConfig(),
    )

    run_registry = run_prepared_experiment(
        experiment_dir,
        hydrograph_config=HydrographMetricConfig(
            event_start_time="2020-01-01T01:00:00Z",
        ),
    )

    assert run_registry.loc[0, "hydrograph_status"] == "computed"
    assert run_registry.loc[0, "event_start_source"] == "manual_input_time"


def test_run_prepared_experiment_supports_start_and_end_window_search(tmp_path: Path) -> None:
    experiment_dir = _write_state(tmp_path)
    forcing = pd.DataFrame(
        {
            "time": [
                "2020-01-01T00:00:00Z",
                "2020-01-01T01:00:00Z",
                "2020-01-01T02:00:00Z",
                "2020-01-01T03:00:00Z",
                "2020-01-01T04:00:00Z",
            ],
            "discharge": [7.0, 5.0, 8.0, 4.0, 9.0],
        }
    )
    forcing.to_csv(experiment_dir / "forcing.csv", index=False)
    prepare_experiment(
        experiment_dir,
        forcing_path=experiment_dir / "forcing.csv",
        prep_config=RapidPrepConfig(),
    )

    run_registry = run_prepared_experiment(
        experiment_dir,
        hydrograph_config=HydrographMetricConfig(
            event_start_time="2020-01-01T01:30:00Z",
            event_start_buffer_hours=1.0,
            event_end_time="2020-01-01T03:30:00Z",
            event_end_buffer_hours=1.0,
        ),
    )

    assert run_registry.loc[0, "hydrograph_status"] == "computed"
    assert run_registry.loc[0, "event_start_source"] == "manual_input_min_window"
    assert run_registry.loc[0, "event_end_source"] == "manual_input_min_window"
    assert run_registry.loc[0, "event_start_time_utc"] == "2020-01-01T01:00:00+00:00"
    assert run_registry.loc[0, "event_end_time_utc"] == "2020-01-01T03:00:00+00:00"


def test_run_prepared_experiment_prefers_nearest_prepeak_local_minimum(tmp_path: Path) -> None:
    experiment_dir = _write_state(tmp_path)
    forcing = pd.DataFrame(
        {
            "time": [
                "2020-01-01T00:00:00Z",
                "2020-01-01T01:00:00Z",
                "2020-01-01T02:00:00Z",
                "2020-01-01T03:00:00Z",
                "2020-01-01T04:00:00Z",
            ],
            "discharge": [4.0, 2.0, 5.0, 3.0, 9.0],
        }
    )
    forcing.to_csv(experiment_dir / "forcing.csv", index=False)
    prepare_experiment(
        experiment_dir,
        forcing_path=experiment_dir / "forcing.csv",
        prep_config=RapidPrepConfig(),
    )

    run_registry = run_prepared_experiment(experiment_dir)

    assert run_registry.loc[0, "hydrograph_status"] == "computed"
    assert run_registry.loc[0, "event_start_source"] == "auto_input_local_min_prepeak"
    assert run_registry.loc[0, "event_start_time_utc"] == "2020-01-01T03:00:00+00:00"


def test_summarize_outlet_hydrograph_detects_postpeak_outlet_minimum(tmp_path: Path) -> None:
    experiment_dir = _write_state(tmp_path)
    forcing = pd.DataFrame(
        {
            "time": [
                "2020-01-01T00:00:00Z",
                "2020-01-01T01:00:00Z",
                "2020-01-01T02:00:00Z",
                "2020-01-01T03:00:00Z",
                "2020-01-01T04:00:00Z",
                "2020-01-01T05:00:00Z",
                "2020-01-01T06:00:00Z",
            ],
            "discharge": [4.0, 2.0, 5.0, 3.0, 9.0, 8.0, 7.0],
        }
    )
    forcing.to_csv(experiment_dir / "forcing.csv", index=False)
    prepare_experiment(
        experiment_dir,
        forcing_path=experiment_dir / "forcing.csv",
        prep_config=RapidPrepConfig(),
    )

    prep_dir = experiment_dir / "states" / "S001_unit_1" / "rapid" / "prep"
    forcing_normalized = pd.read_csv(prep_dir / "forcing_normalized.csv")
    forcing_times = pd.to_datetime(forcing_normalized["time"], utc=True)
    time_seconds = (forcing_times - forcing_times.iloc[0]).dt.total_seconds().to_numpy(dtype=np.float64)
    river_ids = pd.read_csv(prep_dir / "riv.csv", header=None).iloc[:, 0].to_numpy(dtype=np.int64)
    qout_path = prep_dir.parent / "run" / "custom_qout.nc"
    qout_values = np.array([[2.0], [3.0], [5.0], [7.0], [6.0], [4.0], [5.0]], dtype=np.float64)
    write_qout_netcdf(qout_path, river_ids, time_seconds, qout_values)

    _, metrics = summarize_outlet_hydrograph(prep_dir, qout_path, config=HydrographMetricConfig())

    assert metrics["event_start_source"] == "auto_input_local_min_prepeak"
    assert metrics["event_start_time_utc"] == "2020-01-01T03:00:00+00:00"
    assert metrics["event_end_source"] == "auto_outlet_local_min_postpeak"
    assert metrics["event_end_time_utc"] == "2020-01-01T05:00:00+00:00"
    assert metrics["event_end_censored"] is False
