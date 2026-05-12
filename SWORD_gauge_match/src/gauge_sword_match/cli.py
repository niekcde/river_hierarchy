from __future__ import annotations

from datetime import datetime
from pathlib import Path

import click

from .candidate_search import search_reach_candidates
from .config import AppConfig, load_config
from .example_bboxes import DEFAULT_SWORD_PARQUET_DIR, write_example_bbox_layer
from .event_detection import summarize_events
from .event_runner import run_detect_events_batched
from .gauge_io import clean_gauges, gauges_to_geodataframe, load_gauges
from .grdc_io import build_grdc_request_table, load_grdc_catalog, prepare_grdc_catalog, write_grdc_request_station_names
from .kinematic_runner import run_kinematic_screen_batched
from .kinematic_qa import export_kinematic_review_queue
from .qa_exports import (
    export_qgis_package,
    export_review_queue,
    export_subdaily_hierarchy_package,
    export_summary_metrics,
)
from .resolver import refine_best_matches_with_nodes, resolve_best_matches
from .rivretrieve_bridge import build_backend
from .scoring import score_candidates
from .subdaily_download import download_subdaily_from_audit, download_subdaily_to_country_outputs
from .subdaily_locator import locate_subdaily_from_hierarchy_examples
from .sword_io import scan_sword_parquet_dir
from .timeseries_io import filter_station_table_for_timeseries
from .us_manual_import import import_manual_us_subdaily
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


@main.command("export-subdaily-gpkg")
@click.option("--input-gpkg", "input_gpkg_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--audit", "audit_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path))
@click.option("--layer", default="hierarchy_examples_filtered", show_default=True, type=str)
@click.option("--manifests-dir", "manifests_dir", default=None, type=click.Path(exists=True, path_type=Path))
def export_subdaily_gpkg_command(
    input_gpkg_path: Path,
    audit_path: Path,
    output_path: Path,
    layer: str,
    manifests_dir: Path | None,
) -> None:
    export_subdaily_hierarchy_package(
        input_gpkg_path,
        audit_path,
        output_path,
        layer=layer,
        manifests_dir=manifests_dir,
    )
    LOGGER.info("Subdaily hierarchy GeoPackage written to %s", output_path)


@main.command("build-example-bboxes")
@click.option("--input-gpkg", "input_gpkg_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path))
@click.option("--input-layer", default="subdaily_station_summary", show_default=True, type=str)
@click.option("--output-layer", default="example_bboxes", show_default=True, type=str)
@click.option("--reach-lists", "reach_lists_path", default=None, type=click.Path(exists=True, path_type=Path))
@click.option("--reach-summary", "reach_summary_path", default=None, type=click.Path(exists=True, path_type=Path))
@click.option("--sword-parquet-dir", default=None, type=click.Path(exists=True, path_type=Path))
@click.option("--width-field", default="width_obs_p50", show_default=True, type=str)
@click.option("--buffer-multiplier", default=1.0, show_default=True, type=click.FloatRange(min=0.0))
@click.option("--fallback-buffer-m", default=0.0, show_default=True, type=click.FloatRange(min=0.0))
def build_example_bboxes_command(
    input_gpkg_path: Path,
    output_path: Path,
    input_layer: str,
    output_layer: str,
    reach_lists_path: Path | None,
    reach_summary_path: Path | None,
    sword_parquet_dir: Path | None,
    width_field: str,
    buffer_multiplier: float,
    fallback_buffer_m: float,
) -> None:
    layer = write_example_bbox_layer(
        input_gpkg_path,
        output_path,
        input_layer=input_layer,
        output_layer=output_layer,
        reach_lists_path=reach_lists_path,
        reach_summary_path=reach_summary_path,
        sword_parquet_dir=sword_parquet_dir if sword_parquet_dir is not None else DEFAULT_SWORD_PARQUET_DIR,
        width_field=width_field,
        buffer_multiplier=buffer_multiplier,
        fallback_buffer_m=fallback_buffer_m,
    )
    LOGGER.info(
        "Example bbox layer wrote %s polygons to %s (%s)",
        len(layer),
        output_path,
        output_layer,
    )


@main.command("download-subdaily")
@click.option("--audit", "audit_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output", "output_path", default=None, type=click.Path(path_type=Path), help="Combined subdaily timeseries output path. Use together with --manifest.")
@click.option("--manifest", "manifest_path", default=None, type=click.Path(path_type=Path), help="Combined download manifest path. Use together with --output.")
@click.option("--output-dir", "output_dir", default=None, type=click.Path(path_type=Path), help="Optional root directory for per-country outputs. Writes <dir>/<COUNTRY>/subdaily_timeseries.parquet and <dir>/<COUNTRY>/subdaily_download_manifest.csv.")
@click.option("--country", "countries", multiple=True, type=str, help="Optional two-letter country filter. Repeat to restrict the download set.")
@click.option("--start-date", default="2010-01-01", show_default=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--minimum-completeness", default=0.70, show_default=True, type=click.FloatRange(min=0.0, max=1.0))
@click.option("--max-gap-days", default=183.0, show_default=True, type=click.FloatRange(min=0.0))
@click.option("--progress/--no-progress", default=True, show_default=True, help="Show a per-station progress bar during the download run.")
def download_subdaily_command(
    audit_path: Path,
    output_path: Path | None,
    manifest_path: Path | None,
    output_dir: Path | None,
    countries: tuple[str, ...],
    start_date: datetime,
    minimum_completeness: float,
    max_gap_days: float,
    progress: bool,
) -> None:
    if output_dir is not None and (output_path is not None or manifest_path is not None):
        raise click.ClickException("Use either --output-dir for per-country files or the combined --output/--manifest pair, not both.")
    if output_dir is None and (output_path is None or manifest_path is None):
        raise click.ClickException("Provide either --output-dir or both --output and --manifest.")

    if output_dir is not None:
        summary = download_subdaily_to_country_outputs(
            audit_path,
            output_dir=output_dir,
            countries=list(countries),
            target_start_date=start_date.date(),
            minimum_completeness=minimum_completeness,
            max_gap_days=max_gap_days,
            show_progress=progress,
        )
        LOGGER.info(
            "Per-country subdaily downloader wrote %s country summaries to %s",
            len(summary),
            output_dir,
        )
        return

    timeseries, manifest = download_subdaily_from_audit(
        audit_path,
        output_path=output_path,
        manifest_path=manifest_path,
        countries=list(countries),
        target_start_date=start_date.date(),
        minimum_completeness=minimum_completeness,
        max_gap_days=max_gap_days,
        show_progress=progress,
    )
    LOGGER.info("Subdaily downloader wrote %s rows across %s stations", len(timeseries), len(manifest))


@main.command("import-manual-us-subdaily")
@click.option("--manual-dir", "manual_dir", default=Path("outputs/subdaily_values/US/manual_download"), show_default=True, type=click.Path(exists=True, path_type=Path))
@click.option("--audit", "audit_path", default=Path("outputs/subdaily_daily_audit_manual_updates_with_added_examples.csv"), show_default=True, type=click.Path(exists=True, path_type=Path))
@click.option("--examples-gpkg", "examples_gpkg_path", default=Path("outputs/hierarchy_examples_filtered_manual_updates.gpkg"), show_default=True, type=click.Path(exists=True, path_type=Path))
@click.option("--examples-csv", "examples_csv_path", default=Path("outputs/hierarchy_examples_manual_updates.csv"), show_default=True, type=click.Path(path_type=Path))
@click.option("--subdaily-gpkg", "subdaily_gpkg_path", default=Path("outputs/hierarchy_examples_filtered_subdaily_manual_updates.gpkg"), show_default=True, type=click.Path(path_type=Path))
@click.option("--output-dir", "output_dir", default=Path("outputs/subdaily_values"), show_default=True, type=click.Path(path_type=Path))
@click.option("--gauges-cleaned", "gauges_cleaned_path", default=Path("outputs/gauges_cleaned.parquet"), show_default=True, type=click.Path(exists=True, path_type=Path))
@click.option("--start-date", default="2010-01-01", show_default=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--max-gap-days", default=183.0, show_default=True, type=click.FloatRange(min=0.0))
def import_manual_us_subdaily_command(
    manual_dir: Path,
    audit_path: Path,
    examples_gpkg_path: Path,
    examples_csv_path: Path,
    subdaily_gpkg_path: Path,
    output_dir: Path,
    gauges_cleaned_path: Path,
    start_date: datetime,
    max_gap_days: float,
) -> None:
    result = import_manual_us_subdaily(
        manual_download_dir=manual_dir,
        audit_path=audit_path,
        examples_gpkg_path=examples_gpkg_path,
        examples_csv_path=examples_csv_path,
        subdaily_gpkg_path=subdaily_gpkg_path,
        output_dir=output_dir,
        gauges_cleaned_path=gauges_cleaned_path,
        target_start_date=start_date.date(),
        max_gap_days=max_gap_days,
    )
    LOGGER.info(
        "Manual US subdaily import added/updated %s stations and wrote %s US timeseries rows",
        len(result["imported_station_keys"]),
        result["us_timeseries_rows"],
    )


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
