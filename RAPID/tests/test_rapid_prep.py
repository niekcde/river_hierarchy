from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.io import netcdf_file
from shapely.geometry import LineString, Point

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "RAPID" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rapid_tools.engine import run_prepared_experiment, write_qout_netcdf
from rapid_tools.forcing import ForcingConfig
from rapid_tools.hydrograph import HydrographMetricConfig, _integrate_trapezoid, summarize_outlet_hydrograph
from rapid_tools.k_values import KValueConfig, compute_k_values
from rapid_tools.prep import RapidPrepConfig, compute_reach_ratios, create_conn_file, prepare_experiment
from rapid_tools.slope import SlopeConfig, compute_link_slopes


def _write_state(tmp_path: Path, *, link_id: int = 11, link_length: float = 1000.0) -> Path:
    experiment_dir = tmp_path / "experiment"
    state_dir = experiment_dir / "states" / "S001_unit_1"
    variant_dir = state_dir / "variant"
    directed_dir = variant_dir / "directed"
    widths_dir = variant_dir / "widths"
    directed_dir.mkdir(parents=True)
    widths_dir.mkdir(parents=True)

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
    links.to_file(directed_links_path, driver="GPKG")
    nodes.to_file(directed_nodes_path, driver="GPKG")
    links.to_file(widths_path, driver="GPKG")

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
    assert (prep_dir / "kfc.csv").exists()
    assert (prep_dir / "inflow.nc").exists()


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
    assert by_link.loc[11, "wse_ds_source_method"] == "interpolated_along_corridor"
    assert by_link.loc[11, "wse_ds_interpolated"]
    assert abs(by_link.loc[11, "wse_ds"] - 9.0) < 1e-9
    assert abs(by_link.loc[11, "raw_slope"] - 0.001) < 1e-9
    assert by_link.loc[12, "wse_us_source_method"] == "interpolated_along_corridor"
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
        "slope_outlier_flag",
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
