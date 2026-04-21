import pandas as pd

from gauge_sword_match.config import EventDetectionConfig
from gauge_sword_match.event_detection import detect_events
from gauge_sword_match.timeseries_io import standardize_timeseries


def test_standardize_timeseries_renames_rivretrieve_columns():
    raw = pd.DataFrame(
        {
            "Date": ["2020-01-01", "2020-01-02"],
            "Q": [10.0, 11.0],
            "station_id": ["001", "001"],
            "country": ["us", "us"],
            "variable": ["discharge", "discharge"],
            "source_function": ["usa", "usa"],
            "fetch_error": [pd.NA, pd.NA],
        }
    )

    standardized = standardize_timeseries(raw, expected_variable="discharge")

    assert list(standardized.columns) == [
        "station_key",
        "station_id",
        "country",
        "time",
        "discharge",
        "variable",
        "source_function",
    ]
    assert standardized.loc[0, "station_key"] == "US:001"
    assert standardized.loc[0, "discharge"] == 10.0


def test_standardize_timeseries_parses_numeric_r_dates():
    raw = pd.DataFrame(
        {
            "Date": [10957.0, 10958.0],
            "Q": [10.0, 11.0],
            "station_id": ["003103A", "003103A"],
            "country": ["AU", "AU"],
            "variable": ["discharge", "discharge"],
            "source_function": ["australia", "australia"],
            "fetch_error": [pd.NA, pd.NA],
        }
    )

    standardized = standardize_timeseries(raw, expected_variable="discharge")

    assert standardized.loc[0, "time"] == pd.Timestamp("2000-01-01")
    assert standardized.loc[1, "time"] == pd.Timestamp("2000-01-02")


def test_detect_events_selects_clean_single_peak():
    times = pd.date_range("2020-01-01", periods=15, freq="h")
    raw = pd.DataFrame(
        {
            "Date": times,
            "Q": [10, 10, 10, 10, 12, 15, 20, 30, 45, 60, 40, 25, 15, 10, 10],
            "station_id": ["001"] * len(times),
            "country": ["US"] * len(times),
            "variable": ["discharge"] * len(times),
            "source_function": ["usa"] * len(times),
            "fetch_error": [pd.NA] * len(times),
        }
    )
    standardized = standardize_timeseries(raw, expected_variable="discharge")
    config = EventDetectionConfig(
        smoothing_window=1,
        min_rise_points=4,
        min_peak_prominence_ratio=0.25,
        min_event_separation_hours=4,
        pre_event_window_hours=3,
        start_search_hours=12,
        end_search_hours=6,
        min_monotonic_rise_fraction=0.7,
        min_event_duration_hours=4,
    )

    events = detect_events(standardized, config)

    assert len(events) == 1
    event = events.iloc[0]
    assert bool(event["selected_event"]) is True
    assert event["q0_pre_event_median"] == 10.0
    assert event["q0_event_start_discharge"] == 10.0
    assert event["rise_points"] >= 4
    assert event["t0_rise_t10_t90_hours"] > 0
    assert event["t0_rise_start_to_peak_hours"] > event["t0_rise_t10_t90_hours"]


def test_detect_events_uses_lowest_minimum_in_search_window():
    times = pd.date_range("2020-01-01", periods=13, freq="h")
    raw = pd.DataFrame(
        {
            "Date": times,
            "Q": [12, 8, 11, 9, 14, 20, 30, 18, 15, 17, 10, 13, 14],
            "station_id": ["002"] * len(times),
            "country": ["US"] * len(times),
            "variable": ["discharge"] * len(times),
            "source_function": ["usa"] * len(times),
            "fetch_error": [pd.NA] * len(times),
        }
    )
    standardized = standardize_timeseries(raw, expected_variable="discharge")
    config = EventDetectionConfig(
        smoothing_window=1,
        min_rise_points=3,
        min_peak_prominence_ratio=0.1,
        min_event_separation_hours=4,
        pre_event_window_hours=3,
        start_search_hours=8,
        end_search_hours=6,
        min_monotonic_rise_fraction=0.5,
        min_event_duration_hours=4,
    )

    events = detect_events(standardized, config)

    assert len(events) == 1
    event = events.iloc[0]
    assert event["peak_time"] == pd.Timestamp("2020-01-01 06:00:00")
    assert event["start_time"] == pd.Timestamp("2020-01-01 01:00:00")
    assert event["end_time"] == pd.Timestamp("2020-01-01 10:00:00")
