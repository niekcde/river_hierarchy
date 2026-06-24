from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "RAPID" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import rapid_tools.analysis as analysis
from rapid_tools.analysis import (
    build_combined_analysis_frame,
    compare_hydrographs,
    compute_wave_intensity_matrix,
    compute_reach_timing_metrics,
    summarize_matched_corridor_peak_timing,
    write_wave_mp4,
)


def _write_hydrograph(path: Path, q_outlet: list[float], q_inflow: list[float]) -> Path:
    time_seconds = np.arange(len(q_outlet), dtype=float) * 3600.0
    time_utc = pd.to_datetime("2020-01-01T00:00:00Z", utc=True) + pd.to_timedelta(time_seconds, unit="s")
    frame = pd.DataFrame(
        {
            "time_seconds": time_seconds,
            "time_utc": time_utc,
            "q_outlet_cms": np.asarray(q_outlet, dtype=float),
            "q_inflow_cms": np.asarray(q_inflow, dtype=float),
        }
    )
    frame.to_csv(path, index=False)
    return path


def _metric_row(
    *,
    state_id: str,
    prep_dir: Path,
    run_dir: Path,
    hydro_path: Path,
    peak_time_seconds: float,
    peak_discharge_cms: float,
    time_to_peak_seconds: float,
    lag_to_inflow_peak_seconds: float,
    fall_time_seconds: float,
    peak_attenuation_ratio: float,
) -> dict[str, object]:
    peak_time_utc = pd.Timestamp("2020-01-01T00:00:00Z") + pd.to_timedelta(peak_time_seconds, unit="s")
    return {
        "state_id": state_id,
        "rapid_prep_dir": str(prep_dir),
        "rapid_run_dir": str(run_dir),
        "qout_nc": str(run_dir / "Qout_rapid_framework.nc"),
        "status": "ran",
        "hydrograph_status": "computed",
        "outlet_hydrograph_csv": str(hydro_path),
        "hydrograph_metrics_csv": str(run_dir / "hydrograph_metrics.csv"),
        "hydrograph_metrics_json": str(run_dir / "hydrograph_metrics.json"),
        "event_start_time_seconds": 0.0,
        "event_start_time_utc": "2020-01-01T00:00:00+00:00",
        "event_end_time_seconds": 21600.0,
        "event_end_time_utc": "2020-01-01T06:00:00+00:00",
        "event_end_censored": False,
        "peak_time_seconds": peak_time_seconds,
        "peak_time_utc": peak_time_utc.isoformat(),
        "peak_discharge_cms": peak_discharge_cms,
        "time_to_peak_seconds": time_to_peak_seconds,
        "inflow_peak_time_seconds": 7200.0,
        "inflow_peak_time_utc": "2020-01-01T02:00:00+00:00",
        "inflow_peak_discharge_cms": 4.0,
        "lag_to_inflow_peak_seconds": lag_to_inflow_peak_seconds,
        "peak_attenuation_ratio": peak_attenuation_ratio,
        "fall_time_seconds": fall_time_seconds,
        "fall_time_50_seconds": fall_time_seconds / 2.0,
        "fall_time_10_seconds": fall_time_seconds * 0.9,
        "e_folding_time_seconds": fall_time_seconds / 3.0,
        "hydrograph_duration_seconds": 21600.0,
    }


def _write_analysis_experiment(tmp_path: Path, name: str, *, compound: bool) -> Path:
    experiment_dir = tmp_path / name
    states_dir = experiment_dir / "states"
    states_dir.mkdir(parents=True)

    base_state_dir = states_dir / "S000_base"
    base_hierarchy_dir = base_state_dir / "hierarchy"
    base_hierarchy_dir.mkdir(parents=True)
    base_run_dir = base_state_dir / "rapid" / "run"
    base_prep_dir = base_state_dir / "rapid" / "prep"
    base_run_dir.mkdir(parents=True)
    base_prep_dir.mkdir(parents=True)

    inflow = [0.0, 1.0, 4.0, 3.0, 1.0, 0.5, 0.0]
    base_outlet = [0.0, 0.1, 0.8, 2.0, 3.0, 1.5, 0.3]
    base_hydro = _write_hydrograph(base_run_dir / "outlet_hydrograph.csv", base_outlet, inflow)

    if compound:
        child_state_id = "CB001_bubble_6"
        child_outlet = [0.0, 0.5, 3.3, 2.2, 0.8, 0.2, 0.0]
        mode = "compound-bubbles"
        selection_type = "compound_bubble"
        selected_unit_ids = "10,11"
        extra_state = {
            "is_compound_state": True,
            "compound_bubble_id": 6,
            "bubble_root_unit_id": 10,
            "total_unit_nodes_selected": 4,
        }
    else:
        child_state_id = "S046_unit_10"
        child_outlet = [0.0, 0.2, 1.2, 3.2, 2.0, 0.6, 0.1]
        mode = "independent-units"
        selection_type = "unit"
        selected_unit_ids = "10"
        extra_state = {}

    child_state_dir = states_dir / child_state_id
    child_run_dir = child_state_dir / "rapid" / "run"
    child_prep_dir = child_state_dir / "rapid" / "prep"
    child_run_dir.mkdir(parents=True)
    child_prep_dir.mkdir(parents=True)
    child_hydro = _write_hydrograph(child_run_dir / "outlet_hydrograph.csv", child_outlet, inflow)

    run_registry = pd.DataFrame(
        [
            _metric_row(
                state_id="S000_base",
                prep_dir=base_prep_dir,
                run_dir=base_run_dir,
                hydro_path=base_hydro,
                peak_time_seconds=14400.0,
                peak_discharge_cms=3.0,
                time_to_peak_seconds=14400.0,
                lag_to_inflow_peak_seconds=7200.0,
                fall_time_seconds=7200.0,
                peak_attenuation_ratio=0.75,
            ),
            _metric_row(
                state_id=child_state_id,
                prep_dir=child_prep_dir,
                run_dir=child_run_dir,
                hydro_path=child_hydro,
                peak_time_seconds=7200.0 if compound else 10800.0,
                peak_discharge_cms=3.3 if compound else 3.2,
                time_to_peak_seconds=7200.0 if compound else 10800.0,
                lag_to_inflow_peak_seconds=0.0 if compound else 3600.0,
                fall_time_seconds=7200.0,
                peak_attenuation_ratio=0.825 if compound else 0.80,
            ),
        ]
    )
    run_registry.to_csv(experiment_dir / "rapid_run_registry.csv", index=False)

    prep_registry = pd.DataFrame(
        [
            {
                "state_id": "S000_base",
                "rapid_prep_dir": str(base_prep_dir),
                "rapid_link_attributes_csv": str(base_prep_dir / "rapid_link_attributes.csv"),
                "pct_celerity_capped": 0.20,
                "rapid_kb_value": 180.0,
                "status": "prepared",
            },
            {
                "state_id": child_state_id,
                "rapid_prep_dir": str(child_prep_dir),
                "rapid_link_attributes_csv": str(child_prep_dir / "rapid_link_attributes.csv"),
                "pct_celerity_capped": 0.25 if compound else 0.22,
                "rapid_kb_value": 180.0,
                "status": "prepared",
            },
        ]
    )
    prep_registry.to_csv(experiment_dir / "rapid_prep_registry.csv", index=False)

    state_rows = [
        {
            "state_id": "S000_base",
            "parent_state_id": pd.NA,
            "depth": 0,
            "mode": mode,
            "state_role": "base",
            "selection_type": pd.NA,
            "collapsed_selection_label": "base_state",
            "selected_unit_ids": pd.NA,
            "status": "complete",
        },
        {
            "state_id": child_state_id,
            "parent_state_id": "S000_base",
            "depth": 1,
            "mode": mode,
            "state_role": "derived",
            "selection_type": selection_type,
            "collapsed_selection_label": "collapsed_unit_10" if not compound else "collapsed_units_10_11",
            "selected_unit_ids": selected_unit_ids,
            "status": "complete",
            **extra_state,
        },
    ]
    pd.DataFrame(state_rows).to_csv(experiment_dir / "state_registry.csv", index=False)

    if not compound:
        transition_registry = pd.DataFrame(
            [
                {
                    "transition_id": "T001",
                    "step_index": 1,
                    "parent_state_id": "S000_base",
                    "child_state_id": child_state_id,
                    "depth": 1,
                    "mode": mode,
                    "selection_type": "unit",
                    "collapsed_selection_label": "collapsed_unit_10",
                    "selected_unit_ids": "10",
                    "status": "complete",
                }
            ]
        )
        transition_registry.to_csv(experiment_dir / "transition_registry.csv", index=False)

    collapse_ranking = pd.DataFrame(
        [
            {
                "unit_id": 10,
                "class": "compound_or_nested_complex",
                "n_paths": 2,
                "equivalent_length": 1000.0,
                "topologic_complexity_score": 2.0,
                "collapse_priority_score": 0.8,
                "unit_topodynamic_class": "downstream_complex",
                "compound_bubble_id": 6,
                "in_compound_bubble": True,
                "compound_bubble_role": "bubble_root",
                "unit_node_count": 4,
            },
            {
                "unit_id": 11,
                "class": "simple_bifurcation_confluence_pair",
                "n_paths": 2,
                "equivalent_length": 500.0,
                "topologic_complexity_score": 1.0,
                "collapse_priority_score": 0.5,
                "unit_topodynamic_class": "bubble_member",
                "compound_bubble_id": 6,
                "in_compound_bubble": True,
                "compound_bubble_role": "bubble_member",
                "unit_node_count": 2,
            },
        ]
    )
    collapse_ranking.to_csv(base_hierarchy_dir / "collapse_ranking.csv", index=False)

    unit_summary = pd.DataFrame(
        [
            {
                "unit_id": 10,
                "compound_bubble_id": 6,
                "compound_bubble_role": "bubble_root",
                "unit_node_count": 4,
            },
            {
                "unit_id": 11,
                "compound_bubble_id": 6,
                "compound_bubble_role": "bubble_member",
                "unit_node_count": 2,
            },
        ]
    )
    unit_summary.to_csv(base_hierarchy_dir / "unit_summary.csv", index=False)
    return experiment_dir


def test_compare_hydrographs_reports_candidate_earlier() -> None:
    reference = pd.DataFrame(
        {
            "time_utc": pd.date_range("2020-01-01", periods=5, freq="1h", tz="UTC"),
            "time_seconds": np.arange(5) * 3600.0,
            "q_outlet_cms": [0.0, 1.0, 3.0, 1.0, 0.0],
        }
    )
    candidate = pd.DataFrame(
        {
            "time_utc": pd.date_range("2020-01-01", periods=5, freq="1h", tz="UTC"),
            "time_seconds": np.arange(5) * 3600.0,
            "q_outlet_cms": [1.0, 3.0, 1.0, 0.0, 0.0],
        }
    )

    metrics = compare_hydrographs(reference, candidate, max_lag_steps=2)
    assert metrics["hydrograph_lag_vs_reference_steps"] == -1.0
    assert metrics["hydrograph_lag_vs_reference_hours"] == -1.0


def test_compute_wave_intensity_matrix_normalizes_per_reach() -> None:
    qout = np.array(
        [
            [1.0, 2.0],
            [2.0, 2.5],
            [3.0, 2.5],
        ]
    )
    intensity = compute_wave_intensity_matrix(qout, start_index=0, end_index=2)
    expected = np.array(
        [
            [0.0, 0.0],
            [0.5, 1.0],
            [1.0, 1.0],
        ]
    )
    np.testing.assert_allclose(intensity, expected)


def test_compute_reach_timing_metrics_reports_threshold_crossings() -> None:
    qout = np.array(
        [
            [0.0, 1.0],
            [1.0, 1.5],
            [2.0, 2.0],
            [3.0, 2.5],
        ]
    )
    time_seconds = np.array([0.0, 10.0, 20.0, 30.0])

    metrics = compute_reach_timing_metrics(qout, time_seconds, threshold_fractions=(0.1, 0.5))

    assert float(metrics.loc[0, "t10_seconds"]) == 10.0
    assert float(metrics.loc[0, "t50_seconds"]) == 20.0
    assert float(metrics.loc[0, "t_peak_seconds"]) == 30.0
    assert float(metrics.loc[1, "t10_seconds"]) == 10.0
    assert float(metrics.loc[1, "t50_seconds"]) == 20.0
    assert float(metrics.loc[1, "t_peak_seconds"]) == 30.0


def test_shared_scale_difference_suppresses_pre_activation_noise() -> None:
    reference = np.array([10.0, 10.02, 10.20, 11.00])
    target = np.array([10.0, 10.03, 10.35, 11.40])

    difference = analysis._compute_shared_scale_difference(
        reference,
        target,
        activation_fraction=0.10,
    )

    assert difference[0] == 0.0
    assert difference[1] == 0.0
    assert difference[2] > 0.0
    assert difference[3] > difference[2]


def test_select_matched_parent_links_around_change_walks_outward() -> None:
    base_edges = pd.DataFrame(
        {
            "id_link": [100, 101, 102, 103, 104, 105, 106],
            "id_us_node": [0, 1, 2, 3, 4, 5, 2],
            "id_ds_node": [1, 2, 3, 4, 5, 6, 7],
        }
    )
    upstream, downstream = analysis._select_matched_parent_links_around_change(
        base_edges,
        changed_parent_link_ids={102},
        matched_parent_link_ids={100, 101, 103, 104, 105},
        n_upstream=2,
        n_downstream=2,
    )

    assert [row["parent_link_id"] for row in upstream] == [101, 100]
    assert [row["parent_link_id"] for row in downstream] == [103, 104]
    assert [row["graph_distance"] for row in upstream] == [1, 2]
    assert [row["graph_distance"] for row in downstream] == [1, 2]


def test_aggregate_stable_parent_series_uses_dominant_parent_lineage() -> None:
    reaches = pd.DataFrame(
        {
            "parent_link_id": [18, 18, 31, 31],
            "dominant_parent_link_id": [28, 28, 34, 34],
            "lineage_type": ["unchanged_1to1", "unchanged_1to1", "unchanged_1to1", "collapsed_many_to_one"],
        }
    )
    values = np.array(
        [
            [10.0, 14.0, 20.0, 30.0],
            [12.0, 16.0, 22.0, 32.0],
        ]
    )

    stable_ids, lookup = analysis._aggregate_stable_parent_series(reaches, values)

    np.testing.assert_allclose(stable_ids[:3], np.array([28.0, 28.0, 34.0]))
    assert np.isnan(stable_ids[3])
    np.testing.assert_allclose(lookup[28], np.array([12.0, 14.0]))
    np.testing.assert_allclose(lookup[34], np.array([20.0, 22.0]))


def test_summarize_matched_corridor_peak_timing_reports_aligned_peaks() -> None:
    frame = pd.DataFrame(
        {
            "corridor_side": ["upstream"] * 4,
            "corridor_order": [1] * 4,
            "panel_label": ["U1 | parent 28"] * 4,
            "parent_link_id": [28] * 4,
            "graph_distance": [1] * 4,
            "time_seconds": [0.0, 10.0, 20.0, 30.0],
            "time_hours": [0.0, 10.0 / 3600.0, 20.0 / 3600.0, 30.0 / 3600.0],
            "reference_q_cms": [1.0, 4.0, 3.0, 2.0],
            "variant_q_cms": [1.0, 2.0, 5.0, 4.0],
        }
    )

    summary = summarize_matched_corridor_peak_timing(frame)
    row = summary.iloc[0]
    assert float(row["reference_peak_time_seconds"]) == 10.0
    assert float(row["variant_peak_time_seconds"]) == 20.0
    assert float(row["delta_peak_time_hours"]) == 10.0 / 3600.0
    assert float(row["reference_peak_q_cms"]) == 4.0
    assert float(row["variant_peak_q_cms"]) == 5.0


def test_build_all_matched_link_summary_like_frame_sorting_and_deltas() -> None:
    frame = pd.DataFrame(
        {
            "parent_link_id": [10, 11],
            "reference_peak_time_hours": [5.0, 3.0],
            "variant_peak_time_hours": [6.0, 2.0],
            "reference_peak_q_cms": [100.0, 50.0],
            "variant_peak_q_cms": [120.0, 45.0],
            "variant_local_parent_link_ids": ["7", "8"],
            "matched_parent_node_paths": ["1,2", "3,4"],
            "variant_matched_reach_count": [2, 1],
            "state_id": ["Sx", "Sx"],
            "reference_state_id": ["S000_base", "S000_base"],
        }
    )
    frame["delta_peak_time_hours"] = frame["variant_peak_time_hours"] - frame["reference_peak_time_hours"]
    frame["delta_peak_q_cms"] = frame["variant_peak_q_cms"] - frame["reference_peak_q_cms"]

    sorted_frame = frame.sort_values("reference_peak_time_hours").reset_index(drop=True)
    assert int(sorted_frame.loc[0, "parent_link_id"]) == 11
    assert float(sorted_frame.loc[0, "delta_peak_time_hours"]) == -1.0
    assert float(sorted_frame.loc[1, "delta_peak_q_cms"]) == 20.0


def test_build_combined_analysis_frame_merges_unit_and_bubble(tmp_path: Path) -> None:
    indep_dir = _write_analysis_experiment(tmp_path, "03_indep", compound=False)
    compound_dir = _write_analysis_experiment(tmp_path, "03_compound_bubbles", compound=True)

    frame = build_combined_analysis_frame(indep_dir, compound_dir, max_lag_hours=6.0)

    unit_row = frame.loc[frame["state_id"].eq("S046_unit_10")].iloc[0]
    bubble_row = frame.loc[frame["state_id"].eq("CB001_bubble_6")].iloc[0]

    assert unit_row["scenario_family"] == "unit"
    assert bubble_row["scenario_family"] == "compound_bubble"
    assert int(unit_row["selected_unit_count"]) == 1
    assert int(unit_row["base_unit_compound_bubble_id"]) == 6
    assert unit_row["matching_compound_state_id"] == "CB001_bubble_6"
    assert float(unit_row["delta_peak_time_hours"]) == -1.0
    assert float(bubble_row["delta_peak_time_hours"]) == -2.0
    assert float(unit_row["hydrograph_lag_vs_base_hours"]) == -1.0
    assert float(unit_row["lag_to_inflow_peak_relative_to_base"]) == -0.5
    assert float(unit_row["lag_to_inflow_peak_pct_shift_vs_base"]) == -50.0
    assert pd.isna(unit_row["unit_to_bubble_lag_to_inflow_peak_ratio"])


def test_write_wave_mp4_requires_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    indep_dir = _write_analysis_experiment(tmp_path, "03_indep", compound=False)
    monkeypatch.setattr("rapid_tools.analysis.shutil.which", lambda _name: None)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        write_wave_mp4(indep_dir, "S046_unit_10")
