from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import click
import pandas as pd

from .config import AppConfig, load_config
from .kinematic_screen import build_screening_inputs, run_kinematic_screen, summarize_kinematic_results
from .pipeline_inputs import load_combined_best_matches
from .sword_io import SwordFileCatalog, scan_sword_parquet_dir
from .utils import configure_logging, ensure_directory, get_logger, list_table_columns, read_table, write_table

LOGGER = get_logger("kinematic_runner")
_CATALOG_CACHE: dict[str, SwordFileCatalog] = {}

EVENT_READ_CANDIDATE_COLUMNS = [
    "event_id",
    "station_key",
    "station_id",
    "country",
    "source_function",
    "start_time",
    "peak_time",
    "end_time",
    "start_discharge",
    "peak_discharge",
    "peak_discharge_smooth",
    "q0_pre_event_median",
    "q0_event_start_discharge",
    "event_amplitude",
    "prominence_ratio",
    "rise_points",
    "rise_monotonic_fraction",
    "event_duration_hours",
    "t10_time",
    "t90_time",
    "t0_rise_t10_t90_hours",
    "t0_rise_start_to_peak_hours",
    "event_valid",
    "event_quality_score",
    "selected_event",
    "station_event_rank",
    "event_detection_config",
]


@dataclass(slots=True)
class KinematicBatchSpec:
    batch_index: int
    total_batches: int
    station_keys: list[str]
    results_path: Path
    summary_path: Path


def run_kinematic_screen_batched(
    config: AppConfig,
    *,
    execution_mode: str | None = None,
    workers: int | None = None,
    batch_station_count: int | None = None,
) -> tuple[pd.DataFrame, dict[str, int | float], Path]:
    runtime = config.kinematic.screen_runtime
    resolved_mode = str(execution_mode or runtime.execution_mode).strip().lower()
    resolved_workers = max(1, int(workers or runtime.workers))
    resolved_batch_station_count = max(1, int(batch_station_count or runtime.batch_station_count))

    station_keys = _load_screening_station_keys(config.events_selected_path)
    batch_dir = _prepare_kinematic_batch_dir(config)
    batch_specs = build_kinematic_batch_specs(station_keys, resolved_batch_station_count, batch_dir)
    empty_results = run_kinematic_screen(pd.DataFrame(), config.kinematic)
    empty_summary = summarize_kinematic_results(empty_results)

    LOGGER.info(
        "Running screen-kinematic in %s mode across %s batches (%s stations total, up to %s stations per batch). Batch outputs: %s",
        resolved_mode,
        len(batch_specs),
        len(station_keys),
        resolved_batch_station_count,
        batch_dir,
    )

    if not batch_specs:
        write_table(empty_results, config.kinematic_results_path)
        return empty_summary, _combine_kinematic_batch_metrics([], empty_summary), batch_dir

    if resolved_mode == "parallel":
        batch_metrics = _run_kinematic_batches_parallel(
            config_path=config.config_path,
            logging_level=config.logging.level,
            batch_specs=batch_specs,
            workers=resolved_workers,
        )
    elif resolved_mode == "sequential":
        batch_metrics = _run_kinematic_batches_sequential(
            config_path=config.config_path,
            logging_level=config.logging.level,
            batch_specs=batch_specs,
        )
    else:
        raise ValueError(f"Unsupported kinematic execution mode '{resolved_mode}'")

    _write_combined_result_parquet(
        [spec.results_path for spec in batch_specs],
        output_path=config.kinematic_results_path,
        empty_frame=empty_results,
    )
    summary = _combine_kinematic_summary_batches(
        [spec.summary_path for spec in batch_specs],
        empty_frame=empty_summary,
    )
    metrics = _combine_kinematic_batch_metrics(batch_metrics, summary)
    return summary, metrics, batch_dir


def build_kinematic_batch_specs(
    station_keys: list[str],
    batch_station_count: int,
    batch_dir: str | Path,
) -> list[KinematicBatchSpec]:
    normalized_station_keys = [str(value) for value in station_keys]
    if not normalized_station_keys:
        return []

    batch_dir = Path(batch_dir)
    chunks = [
        normalized_station_keys[index : index + batch_station_count]
        for index in range(0, len(normalized_station_keys), batch_station_count)
    ]
    total_batches = len(chunks)
    specs: list[KinematicBatchSpec] = []
    for offset, chunk in enumerate(chunks, start=1):
        batch_label = f"batch_{offset:04d}"
        specs.append(
            KinematicBatchSpec(
                batch_index=offset,
                total_batches=total_batches,
                station_keys=chunk,
                results_path=batch_dir / f"{batch_label}_kinematic_results.parquet",
                summary_path=batch_dir / f"{batch_label}_kinematic_summary.parquet",
            )
        )
    return specs


def process_kinematic_batch(
    config_path: str | Path,
    logging_level: str,
    spec: KinematicBatchSpec,
) -> dict[str, int]:
    config = load_config(config_path)
    configure_logging(logging_level)

    events = _load_selected_events_for_station_keys(config.events_selected_path, spec.station_keys)
    best_matches = load_combined_best_matches(config)
    if not best_matches.empty:
        best_matches = best_matches[best_matches["station_key"].isin(spec.station_keys)].copy()

    catalog = _get_cached_catalog(config)
    inputs = build_screening_inputs(events, best_matches, catalog, config.kinematic)
    results = run_kinematic_screen(inputs, config.kinematic)
    summary = summarize_kinematic_results(results)

    write_table(results, spec.results_path)
    write_table(summary, spec.summary_path)
    return {
        "batch_index": spec.batch_index,
        "total_batches": spec.total_batches,
        "station_count": len(spec.station_keys),
        "result_rows": len(results),
        "valid_result_rows": int(results["valid_input"].sum()) if "valid_input" in results.columns else 0,
        "event_count": int(results["event_id"].nunique()) if "event_id" in results.columns else 0,
        "kinematic_candidate_rows": (
            int(results["is_kinematic_candidate"].fillna(False).astype(bool).sum())
            if "is_kinematic_candidate" in results.columns
            else 0
        ),
    }


def _run_kinematic_batches_sequential(
    *,
    config_path: str | Path,
    logging_level: str,
    batch_specs: list[KinematicBatchSpec],
) -> list[dict[str, int]]:
    metrics: list[dict[str, int]] = []
    with click.progressbar(length=len(batch_specs), label="Screening kinematic") as progress:
        for spec in batch_specs:
            LOGGER.info(
                "Starting kinematic batch %s/%s with %s stations",
                spec.batch_index,
                spec.total_batches,
                len(spec.station_keys),
            )
            result = process_kinematic_batch(config_path, logging_level, spec)
            LOGGER.info(
                "Completed kinematic batch %s/%s: %s result rows, %s valid, %s candidate rows",
                result["batch_index"],
                result["total_batches"],
                result["result_rows"],
                result["valid_result_rows"],
                result["kinematic_candidate_rows"],
            )
            metrics.append(result)
            progress.update(1)
    return metrics


def _run_kinematic_batches_parallel(
    *,
    config_path: str | Path,
    logging_level: str,
    batch_specs: list[KinematicBatchSpec],
    workers: int,
) -> list[dict[str, int]]:
    metrics: list[dict[str, int]] = []
    with click.progressbar(length=len(batch_specs), label="Screening kinematic") as progress:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_kinematic_batch, str(config_path), logging_level, spec): spec
                for spec in batch_specs
            }
            for future in as_completed(futures):
                result = future.result()
                LOGGER.info(
                    "Completed kinematic batch %s/%s: %s result rows, %s valid, %s candidate rows",
                    result["batch_index"],
                    result["total_batches"],
                    result["result_rows"],
                    result["valid_result_rows"],
                    result["kinematic_candidate_rows"],
                )
                metrics.append(result)
                progress.update(1)
    return metrics


def _load_screening_station_keys(path: str | Path) -> list[str]:
    frame = read_table(path, columns=["station_key"])
    station_keys = frame["station_key"].astype("string").dropna().drop_duplicates().tolist()
    return [str(value) for value in station_keys]


def _load_selected_events_for_station_keys(path: str | Path, station_keys: list[str]) -> pd.DataFrame:
    station_key_list = [str(value) for value in station_keys]
    available_columns = list_table_columns(path)
    read_columns = [column for column in EVENT_READ_CANDIDATE_COLUMNS if column in available_columns]
    frame = read_table(path, columns=read_columns or None, filters=[("station_key", "in", station_key_list)])
    if "station_key" in frame.columns:
        frame = frame[frame["station_key"].astype("string").isin(station_key_list)].copy()
    return frame


def _combine_kinematic_summary_batches(
    paths: list[Path],
    *,
    empty_frame: pd.DataFrame,
) -> pd.DataFrame:
    frames = [read_table(path) for path in paths if path.exists()]
    if not frames:
        return empty_frame.copy()
    combined = pd.concat(frames, ignore_index=True)
    if {"stable_kinematic_candidate", "kinematic_fraction"}.issubset(combined.columns):
        combined = combined.sort_values(
            ["stable_kinematic_candidate", "kinematic_fraction"],
            ascending=[False, False],
        ).reset_index(drop=True)
    return combined


def _combine_kinematic_batch_metrics(
    batch_metrics: list[dict[str, int]],
    summary: pd.DataFrame,
) -> dict[str, int | float]:
    return {
        "result_rows": int(sum(item.get("result_rows", 0) for item in batch_metrics)),
        "valid_result_rows": int(sum(item.get("valid_result_rows", 0) for item in batch_metrics)),
        "event_count": int(sum(item.get("event_count", 0) for item in batch_metrics)),
        "station_count": int(summary["station_key"].nunique()) if "station_key" in summary.columns else 0,
        "kinematic_candidate_rows": int(sum(item.get("kinematic_candidate_rows", 0) for item in batch_metrics)),
        "any_kinematic_station_count": int(summary["any_kinematic_candidate"].sum())
        if "any_kinematic_candidate" in summary.columns
        else 0,
        "stable_kinematic_station_count": int(summary["stable_kinematic_candidate"].sum())
        if "stable_kinematic_candidate" in summary.columns
        else 0,
        "multichannel_station_count": int(summary["is_multichannel_hint"].sum())
        if "is_multichannel_hint" in summary.columns
        else 0,
    }


def _write_combined_result_parquet(
    paths: list[Path],
    *,
    output_path: str | Path,
    empty_frame: pd.DataFrame,
) -> Path:
    output_path = Path(output_path)
    ensure_directory(output_path.parent)

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        frames = [read_table(path) for path in paths if path.exists()]
        combined = pd.concat(frames, ignore_index=True) if frames else empty_frame.copy()
        write_table(combined, output_path)
        return output_path

    writer = None
    try:
        for path in paths:
            if not path.exists():
                continue
            frame = read_table(path)
            if frame.empty:
                continue
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        write_table(empty_frame, output_path)
    return output_path


def _get_cached_catalog(config: AppConfig) -> SwordFileCatalog:
    cache_key = str(config.sword.parquet_dir)
    catalog = _CATALOG_CACHE.get(cache_key)
    if catalog is None:
        catalog = scan_sword_parquet_dir(config.sword.parquet_dir)
        _CATALOG_CACHE[cache_key] = catalog
    return catalog


def _prepare_kinematic_batch_dir(config: AppConfig) -> Path:
    ensure_directory(config.kinematic_batches_dir)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return ensure_directory(config.kinematic_batches_dir / run_id)
