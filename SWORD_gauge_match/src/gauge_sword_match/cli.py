from __future__ import annotations

from pathlib import Path

import click

from .candidate_search import search_reach_candidates
from .config import AppConfig, load_config
from .event_detection import summarize_events
from .event_runner import run_detect_events_batched
from .gauge_io import clean_gauges, gauges_to_geodataframe, load_gauges
from .grdc_io import build_grdc_request_table, load_grdc_catalog, prepare_grdc_catalog, write_grdc_request_station_names
from .kinematic_runner import run_kinematic_screen_batched
from .kinematic_qa import export_kinematic_review_queue
from .qa_exports import export_qgis_package, export_review_queue, export_summary_metrics
from .resolver import refine_best_matches_with_nodes, resolve_best_matches
from .rivretrieve_bridge import build_backend
from .scoring import score_candidates
from .subdaily_locator import locate_subdaily_from_hierarchy_examples
from .sword_io import scan_sword_parquet_dir
from .timeseries_io import filter_station_table_for_timeseries
from .utils import configure_logging, ensure_directory, get_logger, write_json, write_table

LOGGER = get_logger("cli")


def _load_runtime(config_path: str | Path) -> AppConfig:
    config = load_config(config_path)
    configure_logging(config.logging.level)
    ensure_directory(config.project.output_dir)
    return config


def _warn_if_crosswalk_is_stale(config: AppConfig) -> None:
    if not config.crosswalk_best_path.exists():
        return

    source_candidates = [
        config.gauges_cleaned_path,
        config.gauges.metadata_path,
        config.gauges.metadata_output,
    ]
    existing_sources = [path for path in source_candidates if path is not None and path.exists()]
    if not existing_sources:
        return

    newest_source = max(existing_sources, key=lambda path: path.stat().st_mtime)
    if newest_source.stat().st_mtime > config.crosswalk_best_path.stat().st_mtime:
        LOGGER.warning(
            "Crosswalk at %s is older than %s. export-gpkg republishes the existing crosswalk; "
            "run build-crosswalk first to include newly fetched gauges.",
            config.crosswalk_best_path,
            newest_source,
        )


def _prepare_timeseries_station_table(config: AppConfig) -> Path:
    base_station_table = (
        config.gauges_cleaned_path
        if config.gauges_cleaned_path.exists()
        else (config.gauges.metadata_path or config.gauges.metadata_output)
    )
    if base_station_table is None or not Path(base_station_table).exists():
        raise click.ClickException(
            f"Station table not found at {base_station_table}. Run fetch-gauges first or set gauges.metadata_output."
        )

    scope = config.timeseries.scope
    if scope == "all_cleaned":
        stations = load_gauges(str(base_station_table))
        count = len(stations)
        LOGGER.info("Using timeseries scope '%s' with %s stations from %s", scope, count, base_station_table)
        return Path(base_station_table)

    if not config.crosswalk_best_path.exists():
        raise click.ClickException(
            f"Best crosswalk not found at {config.crosswalk_best_path}. Run build-crosswalk first for timeseries.scope={scope}."
        )

    crosswalk = load_gauges(str(config.crosswalk_best_path))
    filtered = filter_station_table_for_timeseries(crosswalk, scope)
    if filtered.empty:
        raise click.ClickException(
            f"Timeseries scope '{scope}' selected zero stations from {config.crosswalk_best_path}."
        )

    scoped_path = config.timeseries_station_scope_path
    write_table(filtered, scoped_path)
    LOGGER.info(
        "Using timeseries scope '%s' with %s stations written to %s",
        scope,
        len(filtered),
        scoped_path,
    )
    return scoped_path


def _run_matching_workflow(
    *,
    raw_gauges,
    config: AppConfig,
    cleaned_path: Path,
    candidates_path: Path,
    best_path: Path,
    review_path: Path,
    summary_path: Path,
    qgis_path: Path,
) -> tuple:
    catalog = scan_sword_parquet_dir(config.sword.parquet_dir)
    cleaned_gauges = clean_gauges(raw_gauges)
    gauges_gdf = gauges_to_geodataframe(cleaned_gauges)

    candidates = search_reach_candidates(
        gauges=gauges_gdf,
        catalog=catalog,
        search_radius_m=config.sword.search_radius_m,
        max_candidates=config.sword.max_candidates,
        continent=config.sword.continent,
        bbox=config.sword.bbox,
    )
    scored_candidates = score_candidates(
        candidates,
        score_weights=config.matching.score_weights,
        search_radius_m=config.sword.search_radius_m,
        ambiguity_penalty_weight=config.matching.ambiguity_penalty_weight,
        ambiguity_window=config.matching.ambiguity_window,
    )
    best_matches = resolve_best_matches(cleaned_gauges, scored_candidates, config.matching)
    if config.sword.use_node_refinement:
        best_matches = refine_best_matches_with_nodes(
            best_matches=best_matches,
            gauges_gdf=gauges_gdf,
            catalog=catalog,
            search_radius_m=config.sword.search_radius_m,
        )

    write_table(cleaned_gauges, cleaned_path)
    write_table(scored_candidates, candidates_path)
    write_table(best_matches, best_path)
    export_review_queue(best_matches, review_path)
    export_summary_metrics(best_matches, summary_path)
    export_qgis_package(best_matches, gauges_gdf, catalog, qgis_path)
    return cleaned_gauges, best_matches


@click.group()
def main() -> None:
    """Gauge to SWORD matching workflow."""


@main.command("fetch-gauges")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, path_type=Path))
def fetch_gauges(config_path: Path) -> None:
    config = _load_runtime(config_path)
    backend = build_backend(config)
    output_path = backend.fetch_metadata(config)
    LOGGER.info("Gauge metadata written to %s", output_path)


@main.command("build-crosswalk")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, path_type=Path))
def build_crosswalk(config_path: Path) -> None:
    config = _load_runtime(config_path)
    metadata_path = config.gauges.metadata_path or config.gauges.metadata_output
    if metadata_path is None or not metadata_path.exists():
        raise click.ClickException(
            f"Gauge metadata file not found at {metadata_path}. Run fetch-gauges first or set gauges.metadata_path."
        )

    raw_gauges = load_gauges(str(metadata_path))
    _run_matching_workflow(
        raw_gauges=raw_gauges,
        config=config,
        cleaned_path=config.gauges_cleaned_path,
        candidates_path=config.crosswalk_candidates_path,
        best_path=config.crosswalk_best_path,
        review_path=config.review_queue_path,
        summary_path=config.summary_metrics_path,
        qgis_path=config.qgis_export_path,
    )
    LOGGER.info("Crosswalk written to %s", config.crosswalk_best_path)


@main.command("match-grdc-catalog")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, path_type=Path))
def match_grdc_catalog(config_path: Path) -> None:
    config = _load_runtime(config_path)
    if config.grdc.catalog_path is None or not config.grdc.catalog_path.exists():
        raise click.ClickException(
            f"GRDC catalogue not found at {config.grdc.catalog_path}. Set grdc.catalog_path to the downloaded Excel workbook."
        )

    raw_catalog = load_grdc_catalog(config.grdc.catalog_path, sheet_name=config.grdc.sheet_name)
    prepared_catalog = prepare_grdc_catalog(
        raw_catalog,
        daily_only=config.grdc.daily_only,
        min_daily_years=config.grdc.min_daily_years,
    )
    _, best_matches = _run_matching_workflow(
        raw_gauges=prepared_catalog,
        config=config,
        cleaned_path=config.grdc_cleaned_path,
        candidates_path=config.grdc_crosswalk_candidates_path,
        best_path=config.grdc_crosswalk_best_path,
        review_path=config.grdc_review_queue_path,
        summary_path=config.grdc_summary_metrics_path,
        qgis_path=config.grdc_qgis_export_path,
    )

    request_table = build_grdc_request_table(best_matches)
    write_table(request_table, config.grdc_request_stations_path)
    write_grdc_request_station_names(request_table, config.grdc_request_station_names_path)
    LOGGER.info("GRDC crosswalk written to %s", config.grdc_crosswalk_best_path)
    LOGGER.info("GRDC request table written to %s", config.grdc_request_stations_path)


@main.command("fetch-timeseries")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, path_type=Path))
def fetch_timeseries(config_path: Path) -> None:
    config = _load_runtime(config_path)
    backend = build_backend(config)
    station_table = _prepare_timeseries_station_table(config)
    output_path = backend.fetch_timeseries(config, station_table=Path(station_table))
    LOGGER.info("Gauge timeseries written to %s", output_path)


@main.command("locate-subdaily")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--country", required=True, type=str, help="Two-letter country code. US, CA, BR, CL, GF, CO, KH, LA, TH, BG, ML, NG, and RU are currently supported.")
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path))
@click.option("--layer", default="hierarchy_examples_filtered", show_default=True, type=str)
@click.option("--inventory", "inventory_path", default=None, type=click.Path(exists=True, path_type=Path))
@click.option("--search-radius-m", default=5000.0, show_default=True, type=click.FloatRange(min=0.0))
@click.option("--nearby-limit", default=25, show_default=True, type=click.IntRange(min=1))
@click.option("--inventory-snap-distance-m", default=5000.0, show_default=True, type=click.FloatRange(min=0.0))
@click.option("--max-resolution-distance-m", default=5000.0, show_default=True, type=click.FloatRange(min=0.0))
def locate_subdaily_command(
    input_path: Path,
    country: str,
    output_path: Path,
    layer: str,
    inventory_path: Path | None,
    search_radius_m: float,
    nearby_limit: int,
    inventory_snap_distance_m: float,
    max_resolution_distance_m: float,
) -> None:
    results = locate_subdaily_from_hierarchy_examples(
        input_path,
        country=country,
        layer=layer,
        search_radius_m=search_radius_m,
        nearby_limit=nearby_limit,
        inventory_path=inventory_path,
        inventory_snap_distance_m=inventory_snap_distance_m,
        max_resolution_distance_m=max_resolution_distance_m,
    )
    write_table(results, output_path)
    LOGGER.info(
        "Subdaily locator wrote %s station rows for %s to %s",
        len(results),
        str(country).strip().upper(),
        output_path,
    )


@main.command("export-gpkg")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, path_type=Path))
def export_gpkg(config_path: Path) -> None:
    config = _load_runtime(config_path)
    if not config.crosswalk_best_path.exists():
        raise click.ClickException(
            f"Best crosswalk not found at {config.crosswalk_best_path}. Run build-crosswalk first."
        )
    if not config.gauges_cleaned_path.exists():
        raise click.ClickException(
            f"Cleaned gauges not found at {config.gauges_cleaned_path}. Run build-crosswalk first."
        )

    _warn_if_crosswalk_is_stale(config)
    catalog = scan_sword_parquet_dir(config.sword.parquet_dir)
    best_matches = load_gauges(str(config.crosswalk_best_path))
    gauges = load_gauges(str(config.gauges_cleaned_path))
    gauges_gdf = gauges_to_geodataframe(gauges)
    export_qgis_package(best_matches, gauges_gdf, catalog, config.qgis_export_path)
    LOGGER.info("QGIS GeoPackage written to %s", config.qgis_export_path)


@main.command("detect-events")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--execution-mode",
    type=click.Choice(["sequential", "parallel"], case_sensitive=False),
    default=None,
    help="Override the configured event batch execution mode for this run.",
)
@click.option(
    "--workers",
    type=click.IntRange(min=1),
    default=None,
    help="Override the configured worker count for parallel event batches.",
)
@click.option(
    "--batch-station-count",
    type=click.IntRange(min=1),
    default=None,
    help="Override the configured number of stations per event batch.",
)
def detect_events_command(
    config_path: Path,
    execution_mode: str | None,
    workers: int | None,
    batch_station_count: int | None,
) -> None:
    config = _load_runtime(config_path)
    if not config.timeseries.output.exists():
        raise click.ClickException(
            f"Timeseries file not found at {config.timeseries.output}. Run fetch-timeseries first or update timeseries.output."
        )

    events, selected, batch_dir = run_detect_events_batched(
        config,
        execution_mode=execution_mode,
        workers=workers,
        batch_station_count=batch_station_count,
    )

    write_table(events, config.events_all_path)
    write_table(selected, config.events_selected_path)
    write_json(summarize_events(events), config.event_summary_path)
    LOGGER.info("Event candidates written to %s", config.events_all_path)
    LOGGER.info("Event batch outputs written to %s", batch_dir)


@main.command("screen-kinematic")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--execution-mode",
    type=click.Choice(["sequential", "parallel"], case_sensitive=False),
    default=None,
    help="Override the configured screening batch execution mode for this run.",
)
@click.option(
    "--workers",
    type=click.IntRange(min=1),
    default=None,
    help="Override the configured worker count for parallel screening batches.",
)
@click.option(
    "--batch-station-count",
    type=click.IntRange(min=1),
    default=None,
    help="Override the configured number of stations per screening batch.",
)
def screen_kinematic(
    config_path: Path,
    execution_mode: str | None,
    workers: int | None,
    batch_station_count: int | None,
) -> None:
    config = _load_runtime(config_path)
    if not config.events_selected_path.exists():
        raise click.ClickException(
            f"Selected events not found at {config.events_selected_path}. Run detect-events first."
        )
    if not config.crosswalk_best_path.exists():
        raise click.ClickException(
            f"Best crosswalk not found at {config.crosswalk_best_path}. Run build-crosswalk first."
        )

    summary, metrics, batch_dir = run_kinematic_screen_batched(
        config,
        execution_mode=execution_mode,
        workers=workers,
        batch_station_count=batch_station_count,
    )
    write_table(summary, config.kinematic_summary_path)
    export_kinematic_review_queue(summary, config.kinematic_review_queue_path)
    write_json(metrics, config.kinematic_metrics_path)
    LOGGER.info("Kinematic screening written to %s", config.kinematic_results_path)
    LOGGER.info("Kinematic batch outputs written to %s", batch_dir)


if __name__ == "__main__":
    main()
