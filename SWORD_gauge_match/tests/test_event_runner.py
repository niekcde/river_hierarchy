from pathlib import Path

import pandas as pd

from gauge_sword_match import event_runner
from gauge_sword_match.config import load_config
from gauge_sword_match.event_runner import build_event_batch_specs, run_detect_events_batched


def test_build_event_batch_specs_chunks_station_keys(tmp_path: Path):
    batch_dir = tmp_path / "batches"
    specs = build_event_batch_specs(["A:1", "A:2", "A:3"], 2, batch_dir)

    assert len(specs) == 2
    assert specs[0].station_keys == ["A:1", "A:2"]
    assert specs[1].station_keys == ["A:3"]
    assert specs[0].events_all_path == batch_dir / "batch_0001_events_all.parquet"


def test_run_detect_events_batched_sequential_writes_batch_outputs(tmp_path: Path, monkeypatch):
    config = _write_config(tmp_path)
    stored_frames: dict[Path, pd.DataFrame] = {}

    monkeypatch.setattr(event_runner, "load_event_station_keys", lambda _config: ["US:001", "US:002"])

    def fake_process_event_batch(config_path, logging_level, spec):
        events = pd.DataFrame(
            [
                {
                    "event_id": f"{spec.station_keys[0]}:event",
                    "station_key": spec.station_keys[0],
                    "station_id": spec.station_keys[0].split(":", 1)[1],
                    "country": spec.station_keys[0].split(":", 1)[0],
                    "source_function": "test",
                    "peak_time": pd.Timestamp("2020-01-01"),
                    "selected_event": True,
                }
            ]
        )
        selected = events.copy()
        stored_frames[spec.events_all_path] = events
        stored_frames[spec.events_selected_path] = selected
        spec.events_all_path.touch()
        spec.events_selected_path.touch()
        return {
            "batch_index": spec.batch_index,
            "total_batches": spec.total_batches,
            "station_count": len(spec.station_keys),
            "timeseries_rows": 10,
            "event_count": len(events),
            "selected_count": len(selected),
        }

    monkeypatch.setattr(event_runner, "process_event_batch", fake_process_event_batch)
    monkeypatch.setattr(event_runner, "read_table", lambda path: stored_frames[Path(path)].copy())

    events, selected, batch_dir = run_detect_events_batched(
        config,
        execution_mode="sequential",
        batch_station_count=1,
    )

    batch_files = sorted(batch_dir.glob("*_events_all.parquet"))

    assert len(batch_files) == 2
    assert set(events["station_key"]) == {"US:001", "US:002"}
    assert len(selected) == 2
    assert selected["selected_event"].all()


def _write_config(tmp_path: Path):
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                "  output_dir: outputs",
                "timeseries:",
                "  output: outputs/gauge_timeseries.parquet",
                "  scope: high_medium_matched_only",
                "  variable: discharge",
                "kinematic:",
                "  event_runtime:",
                "    batch_station_count: 1",
                "    execution_mode: sequential",
                "    workers: 2",
            ]
        ),
        encoding="utf-8",
    )
    config = load_config(config_path)
    config.project.output_dir.mkdir(parents=True, exist_ok=True)
    return config
