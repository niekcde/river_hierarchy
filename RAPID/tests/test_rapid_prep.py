from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.io import netcdf_file
from shapely.geometry import LineString, Point

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "RAPID" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rapid_tools.engine import run_prepared_experiment
from rapid_tools.hydrograph import HydrographMetricConfig
from rapid_tools.k_values import KValueConfig, compute_k_values
from rapid_tools.prep import RapidPrepConfig, prepare_experiment
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
    assert hydrograph_metrics.loc[0, "event_start_source"] == "auto_input_min_prepeak"


def test_prepare_experiment_accepts_real_link_id_zero(tmp_path: Path) -> None:
    experiment_dir = _write_state(tmp_path, link_id=0)
    registry = prepare_experiment(
        experiment_dir,
        forcing_path=experiment_dir / "forcing.csv",
        prep_config=RapidPrepConfig(),
    )
    assert registry.loc[0, "status"] == "prepared"


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
    ):
        assert column in rapid_links.columns
    for column in (
        "n_source_links",
        "n_links",
        "link_multiplier",
        "pct_celerity_capped",
        "rapid_k_min",
        "rapid_k_max",
    ):
        assert column in registry.columns


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
            "discharge": [10.0, 5.0, 8.0, 4.0, 9.0],
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
