from pathlib import Path

import pandas as pd

from gauge_sword_match.config import load_config
from gauge_sword_match import pipeline_inputs
from gauge_sword_match.pipeline_inputs import load_combined_best_matches, load_combined_timeseries
from gauge_sword_match.timeseries_io import combine_standardized_timeseries


def test_load_config_defaults_grdc_download_paths_to_project_output(tmp_path: Path):
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                "  output_dir: outputs",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.grdc_station_metadata_path == tmp_path / "outputs" / "grdc_station_metadata.parquet"
    assert config.grdc_timeseries_path == tmp_path / "outputs" / "grdc_timeseries.parquet"
    assert config.kinematic.event_runtime.execution_mode == "sequential"
    assert config.kinematic.event_runtime.batch_station_count == 250
    assert config.kinematic.screen_runtime.execution_mode == "sequential"
    assert config.kinematic.screen_runtime.batch_station_count == 100


def test_combine_standardized_timeseries_prefers_lower_priority_rows():
    main = pd.DataFrame(
        [
            {
                "station_key": "US:001",
                "station_id": "001",
                "country": "US",
                "time": pd.Timestamp("2020-01-01"),
                "discharge": 10.0,
                "variable": "discharge",
                "source_function": "usa",
            }
        ]
    )
    grdc = pd.DataFrame(
        [
            {
                "station_key": "US:001",
                "station_id": "001",
                "country": "US",
                "time": pd.Timestamp("2020-01-01"),
                "discharge": 99.0,
                "variable": "discharge",
                "source_function": "grdc",
            }
        ]
    )

    combined = combine_standardized_timeseries([main, grdc], priorities=[0, 1])

    assert len(combined) == 1
    assert combined.loc[0, "discharge"] == 10.0
    assert combined.loc[0, "source_function"] == "usa"


def test_load_combined_timeseries_adds_scope_filtered_grdc_rows(tmp_path: Path, monkeypatch):
    config = _write_config(tmp_path)

    main_standardized = pd.DataFrame(
        {
            "station_key": ["US:001"] * 3,
            "station_id": ["001"] * 3,
            "country": ["US"] * 3,
            "time": pd.date_range("2020-01-01", periods=3, freq="D"),
            "discharge": [10.0, 11.0, 12.0],
            "variable": ["discharge"] * 3,
            "source_function": ["usa"] * 3,
        }
    )

    grdc_standardized = pd.DataFrame(
        {
            "station_key": ["AF:100", "AF:100", "AF:200", "AF:200"],
            "station_id": ["100", "100", "200", "200"],
            "country": ["AF", "AF", "AF", "AF"],
            "time": pd.to_datetime(["2020-02-01", "2020-02-02", "2020-03-01", "2020-03-02"]),
            "discharge": [50.0, 60.0, 70.0, 80.0],
            "variable": ["discharge"] * 4,
            "source_function": ["grdc"] * 4,
        }
    )

    config.grdc_timeseries_path.touch()

    def fake_load_standardized_timeseries_subset(path, station_keys=None, expected_variable=None):
        if Path(path) == config.timeseries.output:
            frame = main_standardized
        elif Path(path) == config.grdc_timeseries_path:
            frame = grdc_standardized
        else:
            raise AssertionError(f"Unexpected timeseries path: {path}")

        if station_keys is None:
            return frame.copy()
        return frame[frame["station_key"].isin(set(station_keys))].copy()

    def fake_load_grdc_event_station_keys(_config):
        assert _config is config
        return ["AF:100"]

    monkeypatch.setattr(pipeline_inputs, "load_standardized_timeseries_subset", fake_load_standardized_timeseries_subset)
    monkeypatch.setattr(pipeline_inputs, "_load_grdc_event_station_keys", fake_load_grdc_event_station_keys)

    combined = load_combined_timeseries(config)

    assert set(combined["station_key"]) == {"US:001", "AF:100"}
    assert len(combined) == 5
    assert sorted(combined["source_function"].unique().tolist()) == ["grdc", "usa"]


def test_load_event_station_keys_uses_crosswalk_scope_for_main_and_grdc(tmp_path: Path, monkeypatch):
    config = _write_config(tmp_path)
    config.crosswalk_best_path.touch()
    config.grdc_crosswalk_best_path.touch()
    config.grdc_timeseries_path.touch()

    main_best = pd.DataFrame(
        [
            {"station_key": "US:001", "station_id": "001", "country": "US", "reach_id": 1, "confidence_class": "high"},
            {"station_key": "US:002", "station_id": "002", "country": "US", "reach_id": 2, "confidence_class": "low"},
        ]
    )
    grdc_best = pd.DataFrame(
        [
            {"station_key": "AF:100", "station_id": "100", "country": "AF", "reach_id": 3, "confidence_class": "medium"},
            {"station_key": "AF:200", "station_id": "200", "country": "AF", "reach_id": pd.NA, "confidence_class": "unmatched"},
        ]
    )

    def fake_read_table(path, columns=None, filters=None):
        if Path(path) == config.crosswalk_best_path:
            return main_best
        if Path(path) == config.grdc_crosswalk_best_path:
            return grdc_best
        raise AssertionError(f"Unexpected table path: {path}")

    monkeypatch.setattr(pipeline_inputs, "read_table", fake_read_table)

    station_keys = pipeline_inputs.load_event_station_keys(config)

    assert station_keys == ["US:001", "AF:100"]


def test_load_combined_best_matches_merges_grdc_with_main_priority(tmp_path: Path, monkeypatch):
    config = _write_config(tmp_path)

    main_best = pd.DataFrame(
        [
            {"station_key": "US:001", "station_id": "001", "reach_id": 1, "confidence_class": "high"},
            {"station_key": "AA:123", "station_id": "123", "reach_id": 11, "confidence_class": "medium"},
        ]
    )

    grdc_best = pd.DataFrame(
        [
            {"station_key": "AF:100", "station_id": "100", "reach_id": 2, "confidence_class": "high"},
            {"station_key": "AA:123", "station_id": "123", "reach_id": 22, "confidence_class": "high"},
        ]
    )

    config.grdc_crosswalk_best_path.parent.mkdir(parents=True, exist_ok=True)
    config.grdc_crosswalk_best_path.touch()

    def fake_read_table(path):
        if Path(path) == config.crosswalk_best_path:
            return main_best
        if Path(path) == config.grdc_crosswalk_best_path:
            return grdc_best
        raise AssertionError(f"Unexpected table path: {path}")

    monkeypatch.setattr(pipeline_inputs, "read_table", fake_read_table)

    combined = load_combined_best_matches(config)
    by_key = combined.set_index("station_key")

    assert set(by_key.index) == {"US:001", "AF:100", "AA:123"}
    assert by_key.loc["AA:123", "reach_id"] == 11


def _write_config(tmp_path: Path):
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                "  output_dir: outputs",
                "grdc:",
                "  output_dir: outputs/grdc",
                "timeseries:",
                "  output: outputs/gauge_timeseries.parquet",
                "  scope: high_medium_matched_only",
                "  variable: discharge",
            ]
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    config.project.output_dir.mkdir(parents=True, exist_ok=True)
    config.grdc.output_dir.mkdir(parents=True, exist_ok=True)
    return config
