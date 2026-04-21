from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from gauge_sword_match.config import EventDetectionConfig, KinematicConfig, KinematicScreeningConfig
from gauge_sword_match.kinematic_screen import build_screening_inputs, run_kinematic_screen, summarize_kinematic_results
from gauge_sword_match.sword_io import scan_sword_parquet_dir


def test_kinematic_screen_builds_results_and_summary(tmp_path: Path):
    reaches = gpd.GeoDataFrame(
        {
            "reach_id": [101],
            "river_name": ["Main Stem"],
            "width_obs_p50": [100.0],
            "slope_obs_p50": [0.002],
            "n_chan_max": [1],
            "n_chan_mod": [1],
            "slope_obs_reliable": [True],
        },
        geometry=[LineString([(0.0, 0.0), (0.0, 0.08)])],
        crs="EPSG:4326",
    )
    reaches.to_parquet(tmp_path / "na_sword_v17c_beta_reaches.parquet")
    catalog = scan_sword_parquet_dir(tmp_path)

    events = pd.DataFrame(
        [
            {
                "event_id": "US:001:2020-01-01T09:00:00",
                "station_key": "US:001",
                "station_id": "001",
                "country": "US",
                "peak_time": pd.Timestamp("2020-01-01T09:00:00"),
                "q0_pre_event_median": 1000.0,
                "q0_event_start_discharge": 900.0,
                "t0_rise_t10_t90_hours": 48.0,
                "t0_rise_start_to_peak_hours": 60.0,
                "selected_event": True,
            }
        ]
    )
    best_matches = pd.DataFrame(
        [
            {
                "station_key": "US:001",
                "station_id": "001",
                "country": "US",
                "reach_id": 101.0,
                "sword_region": "na",
                "sword_node_id": 101001,
                "confidence_class": "high",
                "distance_m": 120.0,
                "total_score": 0.9,
            }
        ]
    )
    config = KinematicConfig(
        width_field="width_obs_p50",
        slope_field="slope_obs_p50",
        kb_values=[20.0],
        q0_methods=["pre_event_median", "event_start_discharge"],
        t0_methods=["rise_t10_t90", "rise_start_to_peak"],
        allowed_confidence_classes=["high", "medium"],
        event_detection=EventDetectionConfig(),
        screening=KinematicScreeningConfig(
            min_valid_slope=1e-6,
            gravity_m_s2=9.80665,
            regime_tplus_min=80.0,
            regime_froude_t0=0.9,
            regime_tplus_end=1000.0,
            regime_froude_end=0.9,
        ),
    )

    inputs = build_screening_inputs(events, best_matches, catalog, config)
    results = run_kinematic_screen(inputs, config)
    summary = summarize_kinematic_results(results)

    assert len(inputs) == 1
    assert len(results) == 4
    assert results["valid_input"].all()
    assert results["is_kinematic_candidate"].fillna(False).all()
    assert len(summary) == 1
    assert bool(summary.iloc[0]["stable_kinematic_candidate"]) is True
    assert bool(summary.iloc[0]["review_flag"]) is False
