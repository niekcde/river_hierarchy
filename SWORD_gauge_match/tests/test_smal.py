from pathlib import Path

import pandas as pd

from smal import convert_grdc_download, parse_grdc_station_file


SAMPLE_GRDC_FILE = """# Title:                 GRDC STATION DATA FILE
# file generation date:  2026-03-31
# GRDC-No.:              2240200
# River:                 KUNAR SIND
# Station:               PUL-I-KAMA
# Country:               AF
# Latitude (DD):       34.466667
# Longitude (DD):      70.55
# Catchment area (km2):      26005.0
# Altitude (m ASL):        555.0
# Next downstream station:      2240100
# Owner of original data: United States of America - US Geological Survey (USGS)
# Data Set Content:      MEAN DAILY DISCHARGE (Q)
# Unit of measure:                  m3/s
# Time series:           1966-12 - 1979-09
# No. of years:          14
# Last update:           2018-05-25
# Data lines: 4
# DATA
YYYY-MM-DD;hh:mm; Value
1966-12-28;--:--;    120.000
1966-12-29;--:--;   -999.000
1966-12-30;--:--;    118.500
1966-12-30;--:--;    119.500
"""


def test_parse_grdc_station_file_extracts_pipeline_ready_metadata_and_timeseries(tmp_path: Path):
    station_path = tmp_path / "2240200_Q_Day.Cmd.txt"
    station_path.write_text(SAMPLE_GRDC_FILE, encoding="latin-1")

    metadata, timeseries = parse_grdc_station_file(station_path)

    assert metadata["station_id"] == "2240200"
    assert metadata["station_key"] == "AF:2240200"
    assert metadata["station_name"] == "PUL-I-KAMA"
    assert metadata["country"] == "AF"
    assert metadata["drainage_area"] == 26005.0
    assert metadata["record_start_month"] == "1966-12"
    assert metadata["record_end_month"] == "1979-09"

    assert list(timeseries.columns) == [
        "station_key",
        "station_id",
        "country",
        "time",
        "discharge",
        "variable",
        "source_function",
        "grdc_no",
    ]
    assert len(timeseries) == 2
    assert timeseries["station_key"].tolist() == ["AF:2240200", "AF:2240200"]
    assert timeseries["discharge"].tolist() == [120.0, 119.5]
    assert timeseries["variable"].unique().tolist() == ["discharge"]
    assert timeseries["source_function"].unique().tolist() == ["grdc"]


def test_convert_grdc_download_writes_metadata_and_timeseries_parquet(tmp_path: Path):
    input_dir = tmp_path / "GRDC"
    batch_dir = input_dir / "2026-03-31_11-34"
    batch_dir.mkdir(parents=True)

    (batch_dir / "2240200_Q_Day.Cmd.txt").write_text(SAMPLE_GRDC_FILE, encoding="latin-1")
    (batch_dir / "._2240200_Q_Day.Cmd.txt").write_text("ignored", encoding="latin-1")

    metadata_out = tmp_path / "station_metadata.parquet"
    timeseries_out = tmp_path / "timeseries.parquet"

    convert_grdc_download(
        input_dir=input_dir,
        metadata_out=metadata_out,
        timeseries_out=timeseries_out,
        overwrite=True,
        batch_rows=2,
    )

    metadata = pd.read_parquet(metadata_out)
    timeseries = pd.read_parquet(timeseries_out)

    assert len(metadata) == 1
    assert metadata.loc[0, "station_key"] == "AF:2240200"
    assert metadata.loc[0, "source_function"] == "grdc"

    assert len(timeseries) == 2
    assert timeseries["station_id"].tolist() == ["2240200", "2240200"]
    assert timeseries["country"].tolist() == ["AF", "AF"]
    assert timeseries["discharge"].tolist() == [120.0, 119.5]
