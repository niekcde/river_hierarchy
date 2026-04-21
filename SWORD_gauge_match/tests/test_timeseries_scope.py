from pathlib import Path

import pandas as pd

from gauge_sword_match.config import load_config
from gauge_sword_match.timeseries_io import filter_station_table_for_timeseries


def test_load_config_normalizes_timeseries_scope_alias(tmp_path: Path):
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                "  output_dir: outputs",
                "timeseries:",
                "  output: outputs/gauge_timeseries.parquet",
                "  scope: high_medium_matches_only",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.timeseries.scope == "high_medium_matched_only"


def test_filter_station_table_for_high_medium_scope():
    frame = pd.DataFrame(
        [
            {"station_id": "001", "country": "US", "reach_id": 101, "confidence_class": "high", "source_function": "usa"},
            {"station_id": "002", "country": "US", "reach_id": 102, "confidence_class": "medium", "source_function": "usa"},
            {"station_id": "003", "country": "US", "reach_id": 103, "confidence_class": "low", "source_function": "usa"},
            {"station_id": "004", "country": "US", "reach_id": pd.NA, "confidence_class": "unmatched", "source_function": "usa"},
        ]
    )

    filtered = filter_station_table_for_timeseries(frame, "high_medium_matched_only")

    assert filtered["station_id"].tolist() == ["001", "002"]
    assert filtered["country"].tolist() == ["US", "US"]
    assert filtered["source_function"].tolist() == ["usa", "usa"]
