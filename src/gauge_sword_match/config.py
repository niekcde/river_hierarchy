from __future__ import annotations

from datetime import date, datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .utils import Bbox, resolve_path


@dataclass(slots=True)
class ProjectConfig:
    output_dir: Path = Path("outputs")


@dataclass(slots=True)
class LoggingConfig:
    level: str = "INFO"


@dataclass(slots=True)
class RConfig:
    executable: str = "Rscript"


@dataclass(slots=True)
class SwordConfig:
    parquet_dir: Path = Path("/Volumes/PhD/SWORD/v17c/beta/parquet")
    search_radius_m: float = 5_000.0
    max_candidates: int = 8
    use_node_refinement: bool = True
    continent: list[str] | None = None
    bbox: Bbox | None = None


@dataclass(slots=True)
class GaugesConfig:
    countries: list[str] = field(default_factory=list)
    metadata_output: Path = Path("outputs/gauges_raw.parquet")
    metadata_path: Path | None = None
    country_function_map: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class GrdcConfig:
    catalog_path: Path | None = None
    sheet_name: str = "station_catalogue"
    daily_only: bool = True
    min_daily_years: float = 1.0
    output_dir: Path = Path("outputs/grdc")
    station_metadata_path: Path | None = None
    timeseries_path: Path | None = None


@dataclass(slots=True)
class MatchingConfig:
    score_weights: dict[str, float] = field(
        default_factory=lambda: {
            "distance": 0.6,
            "river_name": 0.2,
            "drainage_area": 0.2,
        }
    )
    ambiguity_penalty_weight: float = 0.1
    ambiguity_window: float = 0.05
    high_confidence_score: float = 0.8
    medium_confidence_score: float = 0.6
    min_score_gap: float = 0.1
    review_distance_m: float = 2_500.0


@dataclass(slots=True)
class TimeseriesConfig:
    output: Path = Path("outputs/gauge_timeseries.parquet")
    scope: str = "high_medium_matched_only"
    variable: str = "discharge"
    start_date: str | None = None
    end_date: str | None = None
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    station_pause_seconds: float = 0.1
    country_pause_seconds: float = 2.0


@dataclass(slots=True)
class EventDetectionConfig:
    smoothing_window: int = 3
    min_rise_points: int = 6
    min_peak_prominence_ratio: float = 0.25
    min_event_separation_hours: float = 72.0
    pre_event_window_hours: float = 72.0
    start_search_hours: float = 240.0
    end_search_hours: float = 240.0
    min_monotonic_rise_fraction: float = 0.7
    min_event_duration_hours: float = 12.0


@dataclass(slots=True)
class EventRuntimeConfig:
    batch_station_count: int = 250
    execution_mode: str = "sequential"
    workers: int = 4


@dataclass(slots=True)
class ScreenRuntimeConfig:
    batch_station_count: int = 100
    execution_mode: str = "sequential"
    workers: int = 4


@dataclass(slots=True)
class KinematicScreeningConfig:
    min_valid_slope: float = 1e-6
    gravity_m_s2: float = 9.80665
    regime_tplus_min: float = 80.0
    regime_froude_t0: float = 0.9
    regime_tplus_end: float = 1_000.0
    regime_froude_end: float = 0.9


@dataclass(slots=True)
class KinematicConfig:
    width_field: str = "width_obs_p50"
    slope_field: str = "slope_obs_p50"
    kb_values: list[float] = field(default_factory=lambda: [10.0, 20.0, 30.0, 40.0])
    q0_methods: list[str] = field(default_factory=lambda: ["pre_event_median", "event_start_discharge"])
    t0_methods: list[str] = field(default_factory=lambda: ["rise_t10_t90", "rise_start_to_peak"])
    allowed_confidence_classes: list[str] = field(default_factory=lambda: ["high", "medium"])
    event_detection: EventDetectionConfig = field(default_factory=EventDetectionConfig)
    event_runtime: EventRuntimeConfig = field(default_factory=EventRuntimeConfig)
    screen_runtime: ScreenRuntimeConfig = field(default_factory=ScreenRuntimeConfig)
    screening: KinematicScreeningConfig = field(default_factory=KinematicScreeningConfig)


@dataclass(slots=True)
class AppConfig:
    config_path: Path
    project: ProjectConfig = field(default_factory=ProjectConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    r: RConfig = field(default_factory=RConfig)
    sword: SwordConfig = field(default_factory=SwordConfig)
    gauges: GaugesConfig = field(default_factory=GaugesConfig)
    grdc: GrdcConfig = field(default_factory=GrdcConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    timeseries: TimeseriesConfig = field(default_factory=TimeseriesConfig)
    kinematic: KinematicConfig = field(default_factory=KinematicConfig)

    @property
    def project_root(self) -> Path:
        return self.config_path.parent

    @property
    def gauges_cleaned_path(self) -> Path:
        return self.project.output_dir / "gauges_cleaned.parquet"

    @property
    def crosswalk_best_path(self) -> Path:
        return self.project.output_dir / "crosswalk_best.parquet"

    @property
    def crosswalk_candidates_path(self) -> Path:
        return self.project.output_dir / "crosswalk_candidates.parquet"

    @property
    def review_queue_path(self) -> Path:
        return self.project.output_dir / "review_queue.parquet"

    @property
    def summary_metrics_path(self) -> Path:
        return self.project.output_dir / "summary_metrics.json"

    @property
    def qgis_export_path(self) -> Path:
        return self.project.output_dir / "matched_qgis.gpkg"

    @property
    def events_all_path(self) -> Path:
        return self.project.output_dir / "events_all.parquet"

    @property
    def events_selected_path(self) -> Path:
        return self.project.output_dir / "events_selected.parquet"

    @property
    def event_summary_path(self) -> Path:
        return self.project.output_dir / "event_summary.json"

    @property
    def event_batches_dir(self) -> Path:
        return self.project.output_dir / "_event_batches"

    @property
    def kinematic_batches_dir(self) -> Path:
        return self.project.output_dir / "_kinematic_batches"

    @property
    def kinematic_results_path(self) -> Path:
        return self.project.output_dir / "kinematic_results.parquet"

    @property
    def kinematic_summary_path(self) -> Path:
        return self.project.output_dir / "kinematic_summary.parquet"

    @property
    def kinematic_review_queue_path(self) -> Path:
        return self.project.output_dir / "kinematic_review_queue.parquet"

    @property
    def kinematic_metrics_path(self) -> Path:
        return self.project.output_dir / "kinematic_metrics.json"

    @property
    def timeseries_station_scope_path(self) -> Path:
        return self.project.output_dir / "_timeseries_station_scope.csv"

    @property
    def grdc_cleaned_path(self) -> Path:
        return self.grdc.output_dir / "gauges_cleaned.parquet"

    @property
    def grdc_crosswalk_best_path(self) -> Path:
        return self.grdc.output_dir / "crosswalk_best.parquet"

    @property
    def grdc_station_metadata_path(self) -> Path:
        if self.grdc.station_metadata_path is not None:
            return self.grdc.station_metadata_path
        return self.project.output_dir / "grdc_station_metadata.parquet"

    @property
    def grdc_timeseries_path(self) -> Path:
        if self.grdc.timeseries_path is not None:
            return self.grdc.timeseries_path
        return self.project.output_dir / "grdc_timeseries.parquet"

    @property
    def grdc_crosswalk_candidates_path(self) -> Path:
        return self.grdc.output_dir / "crosswalk_candidates.parquet"

    @property
    def grdc_review_queue_path(self) -> Path:
        return self.grdc.output_dir / "review_queue.parquet"

    @property
    def grdc_summary_metrics_path(self) -> Path:
        return self.grdc.output_dir / "summary_metrics.json"

    @property
    def grdc_qgis_export_path(self) -> Path:
        return self.grdc.output_dir / "matched_qgis.gpkg"

    @property
    def grdc_request_stations_path(self) -> Path:
        return self.grdc.output_dir / "request_stations.csv"

    @property
    def grdc_request_station_names_path(self) -> Path:
        return self.grdc.output_dir / "request_station_names.txt"


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to load the configuration file. Install the project dependencies first."
        ) from exc

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a YAML mapping.")
    return data


def _parse_bbox(value: Any) -> Bbox | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("bbox must be a list of four numbers: [min_lon, min_lat, max_lon, max_lat]")
    return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))


def _parse_optional_date_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _normalize_timeseries_scope(value: Any) -> str:
    normalized = str(value or "high_medium_matched_only").strip().lower()
    aliases = {
        "high_medium_matches_only": "high_medium_matched_only",
        "high_medium_match_only": "high_medium_matched_only",
    }
    normalized = aliases.get(normalized, normalized)
    allowed = {"all_cleaned", "matched_only", "high_medium_matched_only"}
    if normalized not in allowed:
        raise ValueError(f"Unsupported timeseries.scope '{value}'. Expected one of: {', '.join(sorted(allowed))}")
    return normalized


def _normalize_event_execution_mode(value: Any) -> str:
    normalized = str(value or "sequential").strip().lower()
    allowed = {"sequential", "parallel"}
    if normalized not in allowed:
        raise ValueError(
            f"Unsupported kinematic.event_runtime.execution_mode '{value}'. Expected one of: {', '.join(sorted(allowed))}"
        )
    return normalized


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    payload = _load_yaml(config_path)
    base_dir = config_path.parent

    project_cfg = ProjectConfig(
        output_dir=resolve_path(payload.get("project", {}).get("output_dir", "outputs"), base_dir)
    )
    logging_cfg = LoggingConfig(level=str(payload.get("logging", {}).get("level", "INFO")))
    r_cfg = RConfig(executable=str(payload.get("r", {}).get("executable", "Rscript")))

    sword_payload = payload.get("sword", {})
    sword_cfg = SwordConfig(
        parquet_dir=resolve_path(
            sword_payload.get("parquet_dir", "/Volumes/PhD/SWORD/v17c/beta/parquet"), base_dir
        ),
        search_radius_m=float(sword_payload.get("search_radius_m", 5_000)),
        max_candidates=int(sword_payload.get("max_candidates", 8)),
        use_node_refinement=bool(sword_payload.get("use_node_refinement", True)),
        continent=sword_payload.get("continent"),
        bbox=_parse_bbox(sword_payload.get("bbox")),
    )

    gauges_payload = payload.get("gauges", {})
    gauges_cfg = GaugesConfig(
        countries=[str(item).upper() for item in gauges_payload.get("countries", [])],
        metadata_output=resolve_path(
            gauges_payload.get("metadata_output", "outputs/gauges_raw.parquet"), base_dir
        ),
        metadata_path=resolve_path(gauges_payload.get("metadata_path"), base_dir),
        country_function_map={
            str(key).upper(): str(value)
            for key, value in gauges_payload.get("country_function_map", {}).items()
        },
    )

    grdc_payload = payload.get("grdc", {})
    grdc_cfg = GrdcConfig(
        catalog_path=resolve_path(grdc_payload.get("catalog_path"), base_dir),
        sheet_name=str(grdc_payload.get("sheet_name", "station_catalogue")),
        daily_only=bool(grdc_payload.get("daily_only", True)),
        min_daily_years=max(0.0, float(grdc_payload.get("min_daily_years", 1))),
        output_dir=resolve_path(grdc_payload.get("output_dir", "outputs/grdc"), base_dir),
        station_metadata_path=resolve_path(grdc_payload.get("station_metadata_path"), base_dir),
        timeseries_path=resolve_path(grdc_payload.get("timeseries_path"), base_dir),
    )

    matching_payload = payload.get("matching", {})
    weights = matching_payload.get("score_weights", {}) or {}
    matching_cfg = MatchingConfig(
        score_weights={
            "distance": float(weights.get("distance", 0.6)),
            "river_name": float(weights.get("river_name", 0.2)),
            "drainage_area": float(weights.get("drainage_area", 0.2)),
        },
        ambiguity_penalty_weight=float(matching_payload.get("ambiguity_penalty_weight", 0.1)),
        ambiguity_window=float(matching_payload.get("ambiguity_window", 0.05)),
        high_confidence_score=float(matching_payload.get("high_confidence_score", 0.8)),
        medium_confidence_score=float(matching_payload.get("medium_confidence_score", 0.6)),
        min_score_gap=float(matching_payload.get("min_score_gap", 0.1)),
        review_distance_m=float(matching_payload.get("review_distance_m", 2_500)),
    )

    timeseries_payload = payload.get("timeseries", {})
    timeseries_cfg = TimeseriesConfig(
        output=resolve_path(timeseries_payload.get("output", "outputs/gauge_timeseries.parquet"), base_dir),
        scope=_normalize_timeseries_scope(timeseries_payload.get("scope", "high_medium_matched_only")),
        variable=str(timeseries_payload.get("variable", "discharge")),
        start_date=_parse_optional_date_string(timeseries_payload.get("start_date")),
        end_date=_parse_optional_date_string(timeseries_payload.get("end_date")),
        max_retries=max(0, int(timeseries_payload.get("max_retries", 3))),
        retry_backoff_seconds=max(0.0, float(timeseries_payload.get("retry_backoff_seconds", 2.0))),
        station_pause_seconds=max(0.0, float(timeseries_payload.get("station_pause_seconds", 0.1))),
        country_pause_seconds=max(0.0, float(timeseries_payload.get("country_pause_seconds", 2.0))),
    )

    kinematic_payload = payload.get("kinematic", {})
    event_payload = kinematic_payload.get("event_detection", {}) or {}
    event_runtime_payload = kinematic_payload.get("event_runtime", {}) or {}
    screen_runtime_payload = kinematic_payload.get("screen_runtime", {}) or {}
    screening_payload = kinematic_payload.get("screening", {}) or {}
    kinematic_cfg = KinematicConfig(
        width_field=str(kinematic_payload.get("width_field", "width_obs_p50")),
        slope_field=str(kinematic_payload.get("slope_field", "slope_obs_p50")),
        kb_values=[float(value) for value in kinematic_payload.get("kb_values", [10, 20, 30, 40])],
        q0_methods=[str(value) for value in kinematic_payload.get("q0_methods", ["pre_event_median", "event_start_discharge"])],
        t0_methods=[str(value) for value in kinematic_payload.get("t0_methods", ["rise_t10_t90", "rise_start_to_peak"])],
        allowed_confidence_classes=[
            str(value) for value in kinematic_payload.get("allowed_confidence_classes", ["high", "medium"])
        ],
        event_detection=EventDetectionConfig(
            smoothing_window=int(event_payload.get("smoothing_window", 3)),
            min_rise_points=int(event_payload.get("min_rise_points", 6)),
            min_peak_prominence_ratio=float(event_payload.get("min_peak_prominence_ratio", 0.25)),
            min_event_separation_hours=float(event_payload.get("min_event_separation_hours", 72)),
            pre_event_window_hours=float(event_payload.get("pre_event_window_hours", 72)),
            start_search_hours=float(event_payload.get("start_search_hours", 240)),
            end_search_hours=float(event_payload.get("end_search_hours", 240)),
            min_monotonic_rise_fraction=float(event_payload.get("min_monotonic_rise_fraction", 0.7)),
            min_event_duration_hours=float(event_payload.get("min_event_duration_hours", 12)),
        ),
        event_runtime=EventRuntimeConfig(
            batch_station_count=max(1, int(event_runtime_payload.get("batch_station_count", 250))),
            execution_mode=_normalize_event_execution_mode(event_runtime_payload.get("execution_mode", "sequential")),
            workers=max(1, int(event_runtime_payload.get("workers", 4))),
        ),
        screen_runtime=ScreenRuntimeConfig(
            batch_station_count=max(1, int(screen_runtime_payload.get("batch_station_count", 100))),
            execution_mode=_normalize_event_execution_mode(screen_runtime_payload.get("execution_mode", "sequential")),
            workers=max(1, int(screen_runtime_payload.get("workers", 4))),
        ),
        screening=KinematicScreeningConfig(
            min_valid_slope=float(screening_payload.get("min_valid_slope", 1e-6)),
            gravity_m_s2=float(screening_payload.get("gravity_m_s2", 9.80665)),
            regime_tplus_min=float(screening_payload.get("regime_tplus_min", 80)),
            regime_froude_t0=float(screening_payload.get("regime_froude_t0", 0.9)),
            regime_tplus_end=float(screening_payload.get("regime_tplus_end", 1_000)),
            regime_froude_end=float(screening_payload.get("regime_froude_end", 0.9)),
        ),
    )

    if sum(matching_cfg.score_weights.values()) <= 0:
        raise ValueError("At least one matching score weight must be positive.")
    if not kinematic_cfg.kb_values:
        raise ValueError("kinematic.kb_values must contain at least one value.")
    if not kinematic_cfg.q0_methods:
        raise ValueError("kinematic.q0_methods must contain at least one method.")
    if not kinematic_cfg.t0_methods:
        raise ValueError("kinematic.t0_methods must contain at least one method.")

    return AppConfig(
        config_path=config_path,
        project=project_cfg,
        logging=logging_cfg,
        r=r_cfg,
        sword=sword_cfg,
        gauges=gauges_cfg,
        grdc=grdc_cfg,
        matching=matching_cfg,
        timeseries=timeseries_cfg,
        kinematic=kinematic_cfg,
    )
