from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import click
import pandas as pd

from .config import AppConfig, load_config
from .event_detection import detect_events
from .pipeline_inputs import load_combined_timeseries_for_station_keys, load_event_station_keys
from .utils import configure_logging, ensure_directory, get_logger, read_table, write_table

LOGGER = get_logger("event_runner")


@dataclass(slots=True)
class EventBatchSpec:
    batch_index: int
    total_batches: int
    station_keys: list[str]
    events_all_path: Path
    events_selected_path: Path


def run_detect_events_batched(
    config: AppConfig,
    *,
    execution_mode: str | None = None,
    workers: int | None = None,
    batch_station_count: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    runtime = config.kinematic.event_runtime
    resolved_mode = str(execution_mode or runtime.execution_mode).strip().lower()
    resolved_workers = max(1, int(workers or runtime.workers))
    resolved_batch_station_count = max(1, int(batch_station_count or runtime.batch_station_count))

    station_keys = load_event_station_keys(config)
    batch_dir = _prepare_event_batch_dir(config)
    batch_specs = build_event_batch_specs(station_keys, resolved_batch_station_count, batch_dir)
    empty_events = detect_events(pd.DataFrame(), config.kinematic.event_detection)

    LOGGER.info(
        "Running detect-events in %s mode across %s batches (%s stations total, up to %s stations per batch). Batch outputs: %s",
        resolved_mode,
        len(batch_specs),
        len(station_keys),
        resolved_batch_station_count,
        batch_dir,
    )

    if not batch_specs:
        return empty_events, empty_events[empty_events["selected_event"]].copy(), batch_dir

    if resolved_mode == "parallel":
        _run_event_batches_parallel(
            config_path=config.config_path,
            logging_level=config.logging.level,
            batch_specs=batch_specs,
            workers=resolved_workers,
        )
    elif resolved_mode == "sequential":
        _run_event_batches_sequential(
            config_path=config.config_path,
            logging_level=config.logging.level,
            batch_specs=batch_specs,
        )
    else:
        raise ValueError(f"Unsupported event execution mode '{resolved_mode}'")

    events = _combine_event_batch_tables(
        [spec.events_all_path for spec in batch_specs],
        empty_frame=empty_events,
    )
    selected = _combine_event_batch_tables(
        [spec.events_selected_path for spec in batch_specs],
        empty_frame=empty_events[empty_events["selected_event"]].copy(),
    )
    return events, selected, batch_dir


def build_event_batch_specs(
    station_keys: list[str],
    batch_station_count: int,
    batch_dir: str | Path,
) -> list[EventBatchSpec]:
    normalized_station_keys = [str(value) for value in station_keys]
    if not normalized_station_keys:
        return []

    batch_dir = Path(batch_dir)
    chunks = [
        normalized_station_keys[index : index + batch_station_count]
        for index in range(0, len(normalized_station_keys), batch_station_count)
    ]
    total_batches = len(chunks)
    specs: list[EventBatchSpec] = []
    for offset, chunk in enumerate(chunks, start=1):
        batch_label = f"batch_{offset:04d}"
        specs.append(
            EventBatchSpec(
                batch_index=offset,
                total_batches=total_batches,
                station_keys=chunk,
                events_all_path=batch_dir / f"{batch_label}_events_all.parquet",
                events_selected_path=batch_dir / f"{batch_label}_events_selected.parquet",
            )
        )
    return specs


def process_event_batch(
    config_path: str | Path,
    logging_level: str,
    spec: EventBatchSpec,
) -> dict[str, int]:
    config = load_config(config_path)
    configure_logging(logging_level)

    standardized = load_combined_timeseries_for_station_keys(config, spec.station_keys)
    events = detect_events(standardized, config.kinematic.event_detection)
    selected = events[events["selected_event"]].copy()

    write_table(events, spec.events_all_path)
    write_table(selected, spec.events_selected_path)
    return {
        "batch_index": spec.batch_index,
        "total_batches": spec.total_batches,
        "station_count": len(spec.station_keys),
        "timeseries_rows": len(standardized),
        "event_count": len(events),
        "selected_count": len(selected),
    }


def _run_event_batches_sequential(
    *,
    config_path: str | Path,
    logging_level: str,
    batch_specs: list[EventBatchSpec],
) -> None:
    with click.progressbar(length=len(batch_specs), label="Detecting events") as progress:
        for spec in batch_specs:
            LOGGER.info(
                "Starting event batch %s/%s with %s stations",
                spec.batch_index,
                spec.total_batches,
                len(spec.station_keys),
            )
            result = process_event_batch(config_path, logging_level, spec)
            LOGGER.info(
                "Completed event batch %s/%s: %s rows, %s events, %s selected",
                result["batch_index"],
                result["total_batches"],
                result["timeseries_rows"],
                result["event_count"],
                result["selected_count"],
            )
            progress.update(1)


def _run_event_batches_parallel(
    *,
    config_path: str | Path,
    logging_level: str,
    batch_specs: list[EventBatchSpec],
    workers: int,
) -> None:
    with click.progressbar(length=len(batch_specs), label="Detecting events") as progress:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_event_batch, str(config_path), logging_level, spec): spec for spec in batch_specs
            }
            for future in as_completed(futures):
                result = future.result()
                LOGGER.info(
                    "Completed event batch %s/%s: %s stations, %s rows, %s events, %s selected",
                    result["batch_index"],
                    result["total_batches"],
                    result["station_count"],
                    result["timeseries_rows"],
                    result["event_count"],
                    result["selected_count"],
                )
                progress.update(1)


def _combine_event_batch_tables(
    paths: list[Path],
    *,
    empty_frame: pd.DataFrame,
) -> pd.DataFrame:
    frames = [read_table(path) for path in paths if path.exists()]
    if not frames:
        return empty_frame.copy()
    combined = pd.concat(frames, ignore_index=True)
    if {"station_key", "peak_time"}.issubset(combined.columns):
        combined = combined.sort_values(["station_key", "peak_time"]).reset_index(drop=True)
    return combined


def _prepare_event_batch_dir(config: AppConfig) -> Path:
    ensure_directory(config.event_batches_dir)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return ensure_directory(config.event_batches_dir / run_id)
