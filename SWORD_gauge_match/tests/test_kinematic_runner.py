from pathlib import Path

import pandas as pd

from gauge_sword_match import kinematic_runner
from gauge_sword_match.config import AppConfig, KinematicConfig, ProjectConfig, ScreenRuntimeConfig
from gauge_sword_match.kinematic_runner import build_kinematic_batch_specs, run_kinematic_screen_batched


def test_build_kinematic_batch_specs_chunks_station_keys(tmp_path: Path):
    batch_dir = tmp_path / "batches"
    specs = build_kinematic_batch_specs(["A:1", "A:2", "A:3"], 2, batch_dir)

    assert len(specs) == 2
    assert specs[0].station_keys == ["A:1", "A:2"]
    assert specs[1].station_keys == ["A:3"]
    assert specs[0].results_path == batch_dir / "batch_0001_kinematic_results.parquet"


def test_run_kinematic_screen_batched_sequential_streams_results_and_returns_summary(tmp_path: Path, monkeypatch):
    config = _write_config(tmp_path)
    stored_frames: dict[Path, pd.DataFrame] = {}

    monkeypatch.setattr(kinematic_runner, "_load_screening_station_keys", lambda _path: ["US:001", "US:002"])

    def fake_process_kinematic_batch(config_path, logging_level, spec):
        results = pd.DataFrame(
            [
                {
                    "station_key": spec.station_keys[0],
                    "event_id": f"{spec.station_keys[0]}:event",
                    "valid_input": True,
                    "is_kinematic_candidate": True,
                }
            ]
        )
        summary = pd.DataFrame(
            [
                {
                    "station_key": spec.station_keys[0],
                    "any_kinematic_candidate": True,
                    "stable_kinematic_candidate": True,
                    "is_multichannel_hint": False,
                    "kinematic_fraction": 1.0,
                }
            ]
        )
        stored_frames[spec.results_path] = results
        stored_frames[spec.summary_path] = summary
        spec.results_path.touch()
        spec.summary_path.touch()
        return {
            "batch_index": spec.batch_index,
            "total_batches": spec.total_batches,
            "station_count": len(spec.station_keys),
            "result_rows": len(results),
            "valid_result_rows": 1,
            "event_count": 1,
            "kinematic_candidate_rows": 1,
        }

    def fake_read_table(path, columns=None, filters=None):
        return stored_frames[Path(path)].copy()

    def fake_write_combined_result_parquet(paths, output_path, empty_frame):
        combined = pd.concat([stored_frames[Path(path)] for path in paths], ignore_index=True)
        stored_frames[Path(output_path)] = combined
        Path(output_path).touch()
        return Path(output_path)

    monkeypatch.setattr(kinematic_runner, "process_kinematic_batch", fake_process_kinematic_batch)
    monkeypatch.setattr(kinematic_runner, "read_table", fake_read_table)
    monkeypatch.setattr(kinematic_runner, "_write_combined_result_parquet", fake_write_combined_result_parquet)

    summary, metrics, batch_dir = run_kinematic_screen_batched(
        config,
        execution_mode="sequential",
        batch_station_count=1,
    )

    batch_files = sorted(batch_dir.glob("*_kinematic_results.parquet"))

    assert len(batch_files) == 2
    assert set(summary["station_key"]) == {"US:001", "US:002"}
    assert metrics["result_rows"] == 2
    assert metrics["stable_kinematic_station_count"] == 2
    assert config.kinematic_results_path.exists()


def _write_config(tmp_path: Path):
    config = AppConfig(
        config_path=tmp_path / "config.yml",
        project=ProjectConfig(output_dir=tmp_path / "outputs"),
        kinematic=KinematicConfig(
            screen_runtime=ScreenRuntimeConfig(
                batch_station_count=1,
                execution_mode="sequential",
                workers=2,
            )
        ),
    )
    config.project.output_dir.mkdir(parents=True, exist_ok=True)
    config.events_selected_path.touch()
    config.crosswalk_best_path.touch()
    return config
