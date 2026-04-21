from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import AppConfig
from .timeseries_io import (
    combine_standardized_timeseries,
    load_standardized_timeseries_subset,
    select_station_keys_for_timeseries,
)
from .utils import get_logger, list_table_columns, read_table

LOGGER = get_logger("pipeline_inputs")


def load_combined_timeseries(config: AppConfig) -> pd.DataFrame:
    main_standardized = load_standardized_timeseries_subset(
        config.timeseries.output,
        expected_variable=config.timeseries.variable,
    )
    frames: list[pd.DataFrame] = [main_standardized]
    priorities = [0]

    if config.grdc_timeseries_path.exists():
        grdc_station_keys = _load_grdc_event_station_keys(config) if config.timeseries.scope != "all_cleaned" else None
        grdc_standardized = load_standardized_timeseries_subset(
            config.grdc_timeseries_path,
            station_keys=grdc_station_keys,
            expected_variable=config.timeseries.variable,
        )

        if not grdc_standardized.empty:
            frames.append(grdc_standardized)
            priorities.append(1)
            LOGGER.info(
                "Including %s standardized GRDC rows across %s stations from %s",
                len(grdc_standardized),
                grdc_standardized["station_key"].nunique(),
                config.grdc_timeseries_path,
            )
        else:
            LOGGER.info("GRDC timeseries found at %s but no rows remained after filtering", config.grdc_timeseries_path)

    combined = combine_standardized_timeseries(frames, priorities=priorities)
    LOGGER.info(
        "Using %s standardized timeseries rows across %s stations for event detection",
        len(combined),
        combined["station_key"].nunique() if "station_key" in combined.columns else 0,
    )
    return combined


def load_combined_timeseries_for_station_keys(
    config: AppConfig,
    station_keys: list[str] | pd.Index | set[str],
) -> pd.DataFrame:
    batch_station_keys = [str(value) for value in station_keys]
    frames: list[pd.DataFrame] = []
    priorities: list[int] = []

    main_standardized = load_standardized_timeseries_subset(
        config.timeseries.output,
        station_keys=batch_station_keys,
        expected_variable=config.timeseries.variable,
    )
    if not main_standardized.empty:
        frames.append(main_standardized)
        priorities.append(0)

    if config.grdc_timeseries_path.exists():
        grdc_standardized = load_standardized_timeseries_subset(
            config.grdc_timeseries_path,
            station_keys=batch_station_keys,
            expected_variable=config.timeseries.variable,
        )
        if not grdc_standardized.empty:
            frames.append(grdc_standardized)
            priorities.append(1)

    return combine_standardized_timeseries(frames, priorities=priorities)


def load_event_station_keys(config: AppConfig) -> list[str]:
    main_station_keys = _load_main_event_station_keys(config)
    grdc_station_keys = _load_grdc_event_station_keys(config)
    combined = list(dict.fromkeys(main_station_keys + grdc_station_keys))
    LOGGER.info(
        "Prepared %s station keys for event detection (%s main, %s GRDC)",
        len(combined),
        len(main_station_keys),
        len(grdc_station_keys),
    )
    return combined


def load_combined_best_matches(config: AppConfig) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    main_best_matches = read_table(config.crosswalk_best_path)
    if not main_best_matches.empty:
        working = main_best_matches.copy()
        working["_provider_priority"] = 0
        frames.append(working)

    if config.grdc_crosswalk_best_path.exists():
        grdc_best_matches = read_table(config.grdc_crosswalk_best_path)
        if not grdc_best_matches.empty:
            working = grdc_best_matches.copy()
            working["_provider_priority"] = 1
            frames.append(working)
            LOGGER.info(
                "Including %s GRDC best-match rows from %s",
                len(grdc_best_matches),
                config.grdc_crosswalk_best_path,
            )

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    if "station_key" in combined.columns:
        combined = (
            combined.sort_values(["_provider_priority"], kind="mergesort")
            .drop_duplicates(subset=["station_key"], keep="first")
            .reset_index(drop=True)
        )
    return combined.drop(columns=["_provider_priority"], errors="ignore")


def _load_main_event_station_keys(config: AppConfig) -> list[str]:
    if config.timeseries.scope != "all_cleaned" and config.crosswalk_best_path.exists():
        crosswalk = read_table(
            config.crosswalk_best_path,
            columns=["station_key", "station_id", "country", "reach_id", "confidence_class"],
        )
        return [str(value) for value in select_station_keys_for_timeseries(crosswalk, config.timeseries.scope)]

    metadata_path = (
        config.gauges_cleaned_path
        if config.gauges_cleaned_path.exists()
        else (config.gauges.metadata_path or config.gauges.metadata_output)
    )
    if metadata_path is not None and Path(metadata_path).exists():
        return _load_station_keys_from_table(metadata_path)

    return _load_station_keys_from_table(config.timeseries.output)


def _load_grdc_event_station_keys(config: AppConfig) -> list[str]:
    if not config.grdc_timeseries_path.exists():
        return []

    if config.timeseries.scope != "all_cleaned" and config.grdc_crosswalk_best_path.exists():
        crosswalk = read_table(
            config.grdc_crosswalk_best_path,
            columns=["station_key", "station_id", "country", "reach_id", "confidence_class"],
        )
        return [str(value) for value in select_station_keys_for_timeseries(crosswalk, config.timeseries.scope)]

    if config.grdc_station_metadata_path.exists():
        return _load_station_keys_from_table(config.grdc_station_metadata_path)

    return _load_station_keys_from_table(config.grdc_timeseries_path)


def _load_station_keys_from_table(path: str | Path) -> list[str]:
    available_columns = list_table_columns(path)
    read_columns = [column for column in ["station_key", "station_id", "country"] if column in available_columns]
    frame = read_table(path, columns=read_columns or None)
    if "station_key" in frame.columns:
        station_keys = frame["station_key"].astype("string")
    else:
        station_keys = (
            frame["country"].astype("string").str.upper() + ":" + frame["station_id"].astype("string").str.strip()
        )
    return [str(value) for value in station_keys.dropna().drop_duplicates().tolist()]
