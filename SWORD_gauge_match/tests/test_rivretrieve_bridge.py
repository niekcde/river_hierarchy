from datetime import date
from pathlib import Path
from gauge_sword_match.config import load_config
from gauge_sword_match.rivretrieve_bridge import RScriptRivRetrieveBackend


def test_load_config_normalizes_yaml_dates_to_strings(tmp_path: Path):
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                "  output_dir: outputs",
                "timeseries:",
                "  output: outputs/gauge_timeseries.parquet",
                "  start_date: 2000-01-01",
                "  end_date: 2024-12-31",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.timeseries.start_date == "2000-01-01"
    assert config.timeseries.end_date == "2024-12-31"
    assert config.timeseries.max_retries == 3
    assert config.timeseries.retry_backoff_seconds == 2.0
    assert config.timeseries.station_pause_seconds == 0.1
    assert config.timeseries.country_pause_seconds == 2.0


def test_load_config_reads_timeseries_retry_settings(tmp_path: Path):
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                "  output_dir: outputs",
                "timeseries:",
                "  output: outputs/gauge_timeseries.parquet",
                "  max_retries: 5",
                "  retry_backoff_seconds: 1.5",
                "  station_pause_seconds: 0.25",
                "  country_pause_seconds: 4.0",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.timeseries.max_retries == 5
    assert config.timeseries.retry_backoff_seconds == 1.5
    assert config.timeseries.station_pause_seconds == 0.25
    assert config.timeseries.country_pause_seconds == 4.0


def test_run_script_stringifies_date_arguments(monkeypatch, tmp_path: Path):
    backend = RScriptRivRetrieveBackend(executable="Rscript", scripts_dir=tmp_path)
    captured: dict[str, object] = {}

    class FakePopen:
        def __init__(self, command, stdout, stderr, text, bufsize):
            captured["command"] = command
            self.stdout = iter(["Progress 5%\n", "Done\n"])
            self.return_code = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def wait(self):
            return self.return_code

    monkeypatch.setattr("gauge_sword_match.rivretrieve_bridge.subprocess.Popen", FakePopen)

    backend._run_script("fetch_rivretrieve_timeseries.R", ["--start-date", date(2000, 1, 1)])

    assert captured["command"] == [
        "Rscript",
        str(tmp_path / "fetch_rivretrieve_timeseries.R"),
        "--start-date",
        "2000-01-01",
    ]


def test_fetch_timeseries_passes_retry_arguments(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                "  output_dir: outputs",
                "timeseries:",
                "  output: outputs/gauge_timeseries.parquet",
                "  variable: discharge",
                "  max_retries: 4",
                "  retry_backoff_seconds: 1.5",
                "  station_pause_seconds: 0.2",
                "  country_pause_seconds: 3.0",
            ]
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    backend = RScriptRivRetrieveBackend(executable="Rscript", scripts_dir=tmp_path)
    captured: dict[str, object] = {}

    def fake_run_script(script_name, args):
        captured["script_name"] = script_name
        captured["args"] = args

    monkeypatch.setattr(backend, "_run_script", fake_run_script)

    output_path = backend.fetch_timeseries(config, station_table=tmp_path / "stations.csv")

    assert output_path == config.timeseries.output
    assert captured["script_name"] == "fetch_rivretrieve_timeseries.R"
    assert captured["args"] == [
        "--input",
        str(tmp_path / "stations.csv"),
        "--output",
        str(config.timeseries.output),
        "--variable",
        "discharge",
        "--function-map",
        "AU=australia,BR=brazil,CA=canada,CL=chile,FR=france,GB=uk,IE=ireland,JP=japan,UK=uk,US=usa,ZA=southAfrica",
        "--max-retries",
        4,
        "--retry-backoff-seconds",
        1.5,
        "--station-pause-seconds",
        0.2,
        "--country-pause-seconds",
        3.0,
    ]
