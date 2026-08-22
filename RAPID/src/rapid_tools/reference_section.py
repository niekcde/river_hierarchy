from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

from .slope import SlopeConfig, compute_section_slope_reference

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REFERENCE_SECTION_VARIANT_ID = "reference_section"
REFERENCE_SECTION_DIRNAME = "reference_section"
REFERENCE_SECTION_GEOMETRY_SUMMARY = "reference_section_summary.json"
REFERENCE_SECTION_KB_SUMMARY = "reference_kb_summary.json"
DEFAULT_BASED_MODEL_PATH = REPO_ROOT / "RAPID" / "assets" / "based_model_v2.ubj"
BASED_MODEL_SOURCE = "https://github.com/jameshgrn/based_api/blob/master/based_model_v2.ubj"
BASED_SOFTWARE_CITATION = (
    "Gearon, J. (2024). Boost-Assisted Stream Estimator for Depth (BASED) "
    "[Computer software]. Version 1.0.0. https://github.com/JakeGearon/based-api"
)
BASED_PAPER_DOI = "https://doi.org/10.1038/s41586-024-07964-2"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _validate_single_source_sink_path(
    directed_links_path: str | Path,
    directed_nodes_path: str | Path,
) -> dict[str, Any]:
    """Require the all-units reference product to be one complete directed path."""
    links = gpd.read_file(directed_links_path)
    nodes = gpd.read_file(directed_nodes_path)
    required = {"id_link", "id_us_node", "id_ds_node"}
    missing = required.difference(links.columns)
    if missing:
        raise ValueError(
            "Reference-section links are missing directed graph fields: "
            f"{sorted(missing)}."
        )
    if links.empty or nodes.empty:
        raise ValueError("Reference-section graph must contain links and nodes.")
    if links[["id_us_node", "id_ds_node"]].isna().any().any():
        raise ValueError("Reference-section graph contains unresolved link directions.")

    graph = nx.MultiDiGraph()
    graph.add_nodes_from(nodes["id_node"].astype(int).tolist())
    for row in links.itertuples(index=False):
        graph.add_edge(
            int(row.id_us_node),
            int(row.id_ds_node),
            key=int(row.id_link),
        )
    if not nx.is_weakly_connected(graph):
        raise ValueError("Reference-section graph is not weakly connected.")
    if not nx.is_directed_acyclic_graph(nx.DiGraph(graph)):
        raise ValueError("Reference-section graph is not a DAG.")
    sources = sorted(node for node in graph if graph.in_degree(node) == 0)
    sinks = sorted(node for node in graph if graph.out_degree(node) == 0)
    if len(sources) != 1 or len(sinks) != 1:
        raise ValueError(
            "Reference-section graph must have exactly one source and one sink; "
            f"found sources={sources}, sinks={sinks}."
        )

    paths = []
    for edge_path in nx.all_simple_edge_paths(graph, sources[0], sinks[0]):
        paths.append(edge_path)
        if len(paths) > 1:
            break
    if len(paths) != 1:
        raise ValueError(
            "Reference-section graph must contain exactly one directed source-to-sink "
            f"path; found {'more than one' if len(paths) > 1 else 'none'}."
        )
    edge_path = paths[0]
    path_nodes = {sources[0], sinks[0]}
    path_link_ids: list[int] = []
    for upstream, downstream, key in edge_path:
        path_nodes.update((int(upstream), int(downstream)))
        path_link_ids.append(int(key))
    if len(edge_path) != graph.number_of_edges() or path_nodes != set(graph.nodes):
        raise ValueError(
            "Reference-section graph contains links or nodes outside its sole "
            "source-to-sink path."
        )
    return {
        "is_valid": True,
        "source_node_id": int(sources[0]),
        "sink_node_id": int(sinks[0]),
        "path_count": 1,
        "path_link_count": int(len(path_link_ids)),
        "path_link_ids": path_link_ids,
        "node_count": int(graph.number_of_nodes()),
        "link_count": int(graph.number_of_edges()),
    }


def _ensure_repo_imports() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _load_experiment_manifest(experiment_dir: Path) -> dict[str, Any]:
    manifest_path = experiment_dir / "experiment_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Experiment manifest was not found: {manifest_path}")
    return _load_json(manifest_path)


def _resolve_base_state_row(registry: pd.DataFrame) -> pd.Series:
    if "state_role" not in registry.columns:
        raise ValueError("state_registry.csv is missing the 'state_role' column required for reference-section prep.")
    base_rows = registry.loc[registry["state_role"].astype(str).eq("base")]
    if base_rows.empty:
        raise ValueError("Could not resolve a base state row from state_registry.csv.")
    return base_rows.iloc[0]


def _resolve_exit_sides(experiment_dir: Path, registry: pd.DataFrame) -> str:
    for _, row in registry.iterrows():
        variant_output_dir = row.get("variant_output_dir")
        if variant_output_dir is None or pd.isna(variant_output_dir):
            continue
        manifest_path = Path(str(variant_output_dir)).expanduser().resolve() / "summary" / "variant_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = _load_json(manifest_path)
        exit_sides = str(manifest.get("exit_sides", "") or "").strip()
        if exit_sides:
            return exit_sides
    experiment_manifest = _load_experiment_manifest(experiment_dir)
    exit_sides = str(experiment_manifest.get("exit_sides", "") or "").strip()
    if exit_sides:
        return exit_sides
    variant_options = (experiment_manifest.get("options", {}) or {}).get("variant", {}) or {}
    exit_sides = str(variant_options.get("exit_sides", "") or "").strip()
    if exit_sides:
        return exit_sides
    raise ValueError(
        "Could not resolve exit_sides from existing variant manifests. "
        f"Inspect {experiment_dir / 'states'} for missing summary outputs."
    )


def _load_reference_unit_ids(base_hierarchy_dir: Path) -> list[int]:
    unit_summary_path = base_hierarchy_dir / "unit_summary.csv"
    if not unit_summary_path.exists():
        raise FileNotFoundError(f"Reference unit summary was not found: {unit_summary_path}")
    frame = pd.read_csv(unit_summary_path)
    if "unit_id" not in frame.columns:
        raise ValueError(f"{unit_summary_path} is missing the 'unit_id' column.")
    unit_ids = pd.to_numeric(frame["unit_id"], errors="coerce").dropna().astype(int).tolist()
    if not unit_ids:
        raise ValueError(f"No unit IDs were found in {unit_summary_path}.")
    return unit_ids


def _resolve_base_boundary_component_labels(
    *,
    base_row: pd.Series,
    experiment_dir: Path,
    example_id: str,
) -> Path | None:
    """Resolve the global labels already published for the experiment base state."""
    candidates: list[Path] = []
    raw_variant_dir = base_row.get("variant_output_dir")
    variant_dir: Path | None = None
    if raw_variant_dir is not None and not pd.isna(raw_variant_dir):
        variant_dir = Path(str(raw_variant_dir)).expanduser().resolve()
        manifest_path = variant_dir / "summary" / "variant_manifest.json"
        if manifest_path.exists():
            manifest = _load_json(manifest_path)
            for section_name in ("output_paths", "source_paths"):
                section = manifest.get(section_name, {}) or {}
                raw_labels = str(section.get("boundary_component_labels", "") or "").strip()
                if raw_labels:
                    candidates.append(Path(raw_labels).expanduser())

    labels_name = f"{example_id}_full_extent_land_water_labels_with_snapped_buffered_border_cut.tif"
    candidates.append(experiment_dir / "boundary_components" / labels_name)
    if variant_dir is not None:
        candidates.append(variant_dir / "boundary_components" / labels_name)

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None


def _reference_section_paths(experiment_dir: Path) -> dict[str, Path]:
    root = experiment_dir / REFERENCE_SECTION_DIRNAME
    return {
        "root": root,
        "variant": root / "variant",
        "geometry_summary": root / REFERENCE_SECTION_GEOMETRY_SUMMARY,
        "kb_summary": root / REFERENCE_SECTION_KB_SUMMARY,
    }


def _load_width_summary(
    link_width_samples_path: Path,
    link_width_families_path: Path,
    *,
    width_sample_field: str,
    width_percentile: float | None,
) -> dict[str, object]:
    if link_width_families_path.exists():
        families = pd.read_csv(link_width_families_path)
        if width_sample_field in families.columns:
            widths = pd.to_numeric(families[width_sample_field], errors="coerce")
            widths = widths[np.isfinite(widths) & widths.gt(0)]
            if not widths.empty:
                width_reference = float(widths.iloc[0]) if len(widths) == 1 else float(widths.median())
                return {
                    "sample_field": width_sample_field,
                    "percentile": None,
                    "sample_count": int(len(widths)),
                    "positive_sample_count": int(len(widths)),
                    "width_reference_m": width_reference,
                    "width_min_m": float(widths.min()),
                    "width_max_m": float(widths.max()),
                    "source": "link_width_families",
                    "statistic": "direct_single_link" if len(widths) == 1 else "median_multi_link",
                }

    if not link_width_samples_path.exists():
        raise FileNotFoundError(f"Reference-section width samples were not found: {link_width_samples_path}")
    samples = pd.read_csv(link_width_samples_path)
    if width_sample_field not in samples.columns:
        raise ValueError(
            f"Reference-section width inputs do not contain '{width_sample_field}'. "
            f"Available sample columns: {samples.columns.tolist()}"
        )
    widths = pd.to_numeric(samples[width_sample_field], errors="coerce")
    widths = widths[np.isfinite(widths) & widths.gt(0)]
    if widths.empty:
        raise ValueError(
            f"Reference-section width samples did not contain any positive finite '{width_sample_field}' values."
        )
    if width_percentile is None:
        raise ValueError(
            "Reference-section width percentile must be provided when the selected width field "
            "is available only in raw link-width samples. Summary width fields such as "
            "'wid_adj_routing' do not require a percentile."
        )
    return {
        "sample_field": width_sample_field,
        "percentile": float(width_percentile),
        "sample_count": int(len(widths)),
        "positive_sample_count": int(len(widths)),
        "width_reference_m": float(np.nanpercentile(widths.to_numpy(dtype=float), float(width_percentile))),
        "width_min_m": float(widths.min()),
        "width_max_m": float(widths.max()),
        "source": "link_width_samples",
        "statistic": "percentile",
    }


def _build_reference_section_variant(
    *,
    experiment_dir: Path,
    experiment_manifest: Mapping[str, Any],
    registry: pd.DataFrame,
    reference_paths: Mapping[str, Path],
) -> dict[str, Any]:
    _ensure_repo_imports()
    from network_variants.variant_generation import generate_network_variant

    base_row = _resolve_base_state_row(registry)
    base_hierarchy_dir = Path(str(base_row["hierarchy_output_dir"])).expanduser().resolve()
    base_analysis_links = Path(str(base_row["analysis_links_path"])).expanduser().resolve()
    base_directed_nodes = Path(str(base_row["directed_nodes_path"])).expanduser().resolve()
    base_cleaned_mask = Path(str(base_row["cleaned_mask_path"])).expanduser().resolve()
    for label, path in (
        ("base analysis links", base_analysis_links),
        ("base directed nodes", base_directed_nodes),
        ("base cleaned mask", base_cleaned_mask),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Registered {label} path does not exist: {path}")
    unit_ids = _load_reference_unit_ids(base_hierarchy_dir)
    variant_options = (experiment_manifest.get("options", {}) or {}).get("variant", {}) or {}
    exit_sides = _resolve_exit_sides(experiment_dir, registry)
    example_id = str(experiment_manifest.get("example_id") or experiment_manifest.get("root_example_id") or "").strip()
    if not example_id:
        raise ValueError("Experiment manifest is missing example_id/root_example_id.")
    boundary_component_labels = _resolve_base_boundary_component_labels(
        base_row=base_row,
        experiment_dir=experiment_dir,
        example_id=example_id,
    )

    outputs = generate_network_variant(
        cleaned_mask_path=base_cleaned_mask,
        reviewed_links_path=base_analysis_links,
        reviewed_nodes_path=base_directed_nodes,
        exit_sides=exit_sides,
        unit_ids=unit_ids,
        variant_id=REFERENCE_SECTION_VARIANT_ID,
        example_id=example_id,
        output_dir=reference_paths["variant"],
        boundary_component_labels_path=boundary_component_labels,
        preferred_width_field=str(
            variant_options.get("preferred_width_field", "wid_adj_routing")
        ),
        footprint_buffer_scale=float(variant_options.get("footprint_buffer_scale", 0.5)),
        all_touched=bool(variant_options.get("all_touched", True)),
        allow_noop=bool(variant_options.get("allow_noop", False)),
        single_thread=True,
        export_sword=bool(variant_options.get("export_sword", True)),
        transect_scale=float(variant_options.get("transect_scale", 1.5)),
        min_transect_pixels=float(variant_options.get("min_transect_pixels", 5.0)),
        match_tolerance=variant_options.get("match_tolerance"),
        sword_node_source_path=variant_options.get("sword_node_source_path"),
        sword_wse_field=variant_options.get("sword_wse_field"),
        sword_match_tolerance=variant_options.get("sword_match_tolerance"),
        sword_example_station_source_path=variant_options.get("sword_example_station_source_path"),
        sword_station_match_source_path=variant_options.get("sword_station_match_source_path"),
        sword_reach_buffer_steps=int(variant_options.get("sword_reach_buffer_steps", 2)),
        verbose_rivgraph=bool(variant_options.get("verbose_rivgraph", False)),
    )

    return {
        "construction_backend": "full_raster",
        "example_id": example_id,
        "exit_sides": exit_sides,
        "unit_ids": unit_ids,
        "variant_output_dir": outputs.output_dir,
        "directed_links_path": outputs.directed_links_path,
        "directed_nodes_path": outputs.directed_nodes_path,
        "centerline_path": outputs.rivgraph_centerline_path,
        "collapsed_mask_path": outputs.collapsed_mask_path,
        "boundary_component_labels_path": boundary_component_labels,
        "link_width_samples_path": outputs.output_dir / "widths" / "link_width_samples.csv",
        "manifest_path": outputs.output_dir / "summary" / "variant_manifest.json",
    }


def ensure_reference_section_geometry(
    experiment_dir: str | Path,
    *,
    width_sample_field: str,
    width_percentile: float | None,
    variant_builder: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    experiment_path = Path(experiment_dir).expanduser().resolve()
    registry_path = experiment_path / "state_registry.csv"
    if not registry_path.exists():
        raise FileNotFoundError(f"State registry was not found: {registry_path}")
    registry = pd.read_csv(registry_path)
    experiment_manifest = _load_experiment_manifest(experiment_path)
    reference_paths = _reference_section_paths(experiment_path)
    reference_paths["root"].mkdir(parents=True, exist_ok=True)

    if reference_paths["geometry_summary"].exists():
        cached = _load_json(reference_paths["geometry_summary"])
        cached_paths = cached.get("paths", {})
        required = []
        for key in ("variant_output_dir", "directed_links_path", "directed_nodes_path", "link_width_samples_path"):
            raw = str(cached_paths.get(key, "") or "").strip()
            if not raw:
                required = []
                break
            required.append(Path(raw))
        if required and all(path.exists() for path in required):
            variant_output_dir = Path(str(cached_paths["variant_output_dir"])).expanduser().resolve()
            path_validation = _validate_single_source_sink_path(
                cached_paths["directed_links_path"],
                cached_paths["directed_nodes_path"],
            )
            refreshed_width_summary = _load_width_summary(
                Path(str(cached_paths["link_width_samples_path"])).expanduser().resolve(),
                variant_output_dir / "widths" / "link_width_families.csv",
                width_sample_field=width_sample_field,
                width_percentile=width_percentile,
            )
            if cached.get("width_summary") != refreshed_width_summary:
                cached["width_summary"] = refreshed_width_summary
            if cached.get("path_validation") != path_validation:
                cached["path_validation"] = path_validation
            reference_paths["geometry_summary"].write_text(json.dumps(cached, indent=2))
            return cached

    builder = variant_builder or _build_reference_section_variant
    variant_summary = dict(builder(
        experiment_dir=experiment_path,
        experiment_manifest=experiment_manifest,
        registry=registry,
        reference_paths=reference_paths,
    ))
    nodes = gpd.read_file(variant_summary["directed_nodes_path"])
    path_validation = _validate_single_source_sink_path(
        variant_summary["directed_links_path"],
        variant_summary["directed_nodes_path"],
    )
    variant_output_dir = Path(variant_summary["variant_output_dir"])
    width_summary = _load_width_summary(
        Path(variant_summary["link_width_samples_path"]),
        variant_output_dir / "widths" / "link_width_families.csv",
        width_sample_field=width_sample_field,
        width_percentile=width_percentile,
    )
    slope_summary = compute_section_slope_reference(nodes, config=SlopeConfig())

    summary = {
        "example_id": variant_summary["example_id"],
        "reference_variant_id": REFERENCE_SECTION_VARIANT_ID,
        "construction_backend": str(
            variant_summary.get("construction_backend", "full_raster")
        ),
        "single_thread": True,
        "path_validation": path_validation,
        "exit_sides": variant_summary["exit_sides"],
        "n_reference_units_collapsed": int(len(variant_summary["unit_ids"])),
        "reference_unit_ids": variant_summary["unit_ids"],
        "width_summary": width_summary,
        "slope_summary": slope_summary,
        "paths": {
            "reference_section_dir": str(reference_paths["root"]),
            "variant_output_dir": str(variant_summary["variant_output_dir"]),
            "directed_links_path": str(variant_summary["directed_links_path"]),
            "directed_nodes_path": str(variant_summary["directed_nodes_path"]),
            "centerline_path": str(variant_summary["centerline_path"]) if variant_summary["centerline_path"] is not None else "",
            "collapsed_mask_path": str(variant_summary["collapsed_mask_path"]),
            "boundary_component_labels_path": (
                str(variant_summary["boundary_component_labels_path"])
                if variant_summary["boundary_component_labels_path"] is not None
                else ""
            ),
            "link_width_samples_path": str(variant_summary["link_width_samples_path"]),
            "variant_manifest_path": str(variant_summary["manifest_path"]),
        },
        "source_experiment_manifest": str(experiment_path / "experiment_manifest.json"),
    }
    reference_paths["geometry_summary"].write_text(json.dumps(summary, indent=2))
    return summary


def resolve_based_model_path(model_path: str | Path | None = None) -> Path:
    resolved = Path(model_path).expanduser().resolve() if model_path is not None else DEFAULT_BASED_MODEL_PATH
    if not resolved.exists():
        raise FileNotFoundError(
            "The BASED model file was not found. "
            f"Expected: {resolved}. Download based_model_v2.ubj from {BASED_MODEL_SOURCE}."
        )
    return resolved


@lru_cache(maxsize=2)
def _load_based_model(model_path: str) -> Any:
    try:
        import xgboost as xgb
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError(
            "BASED reference-section kb prediction requires the optional 'xgboost' dependency."
        ) from exc
    model = xgb.Booster()
    model.load_model(model_path)
    return model


def predict_based_depth(
    *,
    discharge_cms: float,
    width_m: float,
    slope: float,
    model_path: str | Path | None = None,
) -> float:
    if discharge_cms <= 0 or width_m <= 0 or slope <= 0:
        raise ValueError("BASED depth prediction requires positive discharge, width, and slope.")
    resolved_model_path = resolve_based_model_path(model_path)
    try:
        import xgboost as xgb
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError(
            "BASED reference-section kb prediction requires the optional 'xgboost' dependency."
        ) from exc
    features = pd.DataFrame(
        {
            "log_Q": [np.log10(discharge_cms)],
            "log_w": [np.log10(width_m)],
            "log_S": [np.log10(slope)],
        },
        dtype=float,
    )
    model = _load_based_model(str(resolved_model_path))
    log_depth = model.predict(xgb.DMatrix(features))
    depth_m = float(10 ** float(log_depth[0]))
    if not np.isfinite(depth_m) or depth_m <= 0:
        raise ValueError("BASED produced a non-positive depth prediction.")
    return depth_m


def compute_reference_section_kb(
    experiment_dir: str | Path,
    *,
    forcing: pd.DataFrame,
    width_sample_field: str,
    width_percentile: float | None,
    model_path: str | Path | None = None,
    variant_builder: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    experiment_path = Path(experiment_dir).expanduser().resolve()
    reference_paths = _reference_section_paths(experiment_path)
    geometry_summary = ensure_reference_section_geometry(
        experiment_path,
        width_sample_field=width_sample_field,
        width_percentile=width_percentile,
        variant_builder=variant_builder,
    )
    discharge = pd.to_numeric(forcing["discharge_cms"], errors="coerce")
    discharge = discharge[np.isfinite(discharge)]
    if discharge.empty:
        raise ValueError("Normalized forcing does not contain any finite discharge values.")
    q_ref = float(discharge.max())
    width_ref = float(geometry_summary["width_summary"]["width_reference_m"])
    slope_ref = float(geometry_summary["slope_summary"]["slope"])
    depth_ref = float(
        predict_based_depth(
            discharge_cms=q_ref,
            width_m=width_ref,
            slope=slope_ref,
            model_path=model_path,
        )
    )
    kb_value = float(width_ref / depth_ref)
    width_percentile_value = geometry_summary["width_summary"].get("percentile")
    summary = {
        "example_id": geometry_summary["example_id"],
        "kb_source_method": "based_reference_section",
        "reference_section_construction_backend": str(
            geometry_summary.get("construction_backend", "full_raster")
        ),
        "kb_value": kb_value,
        "reference_discharge_cms": q_ref,
        "reference_width_m": width_ref,
        "reference_slope": slope_ref,
        "reference_depth_m": depth_ref,
        "width_sample_field": str(geometry_summary["width_summary"]["sample_field"]),
        "width_percentile": (
            float(width_percentile_value)
            if width_percentile_value is not None and np.isfinite(float(width_percentile_value))
            else None
        ),
        "slope_source_method": str(geometry_summary["slope_summary"]["source_method"]),
        "slope_anchor_count": int(geometry_summary["slope_summary"]["anchor_count"]),
        "based_model_path": str(resolve_based_model_path(model_path)),
        "based_model_source": BASED_MODEL_SOURCE,
        "based_software_citation": BASED_SOFTWARE_CITATION,
        "based_paper_doi": BASED_PAPER_DOI,
        "reference_section_summary_path": str(reference_paths["geometry_summary"]),
    }
    reference_paths["root"].mkdir(parents=True, exist_ok=True)
    reference_paths["kb_summary"].write_text(json.dumps(summary, indent=2))
    return summary
