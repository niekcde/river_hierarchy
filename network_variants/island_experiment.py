"""Selection contracts for paired island-only simplification experiments.

This module intentionally separates the reviewed island candidate set from the
existing unit-collapse workflow.  The eventual geometry-aware variant runner
consumes :class:`IslandVariantSelection`; it does not change unit selections.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds

from network_variants.width_samples import read_width_samples


@dataclass(frozen=True)
class IslandBackendPolicy:
    """Execution policy inherited from the completed independent experiment."""

    base_backend: str
    local_patch_mask_publication: str
    fallback_backend: str
    panelized_run_root: Path | None = None


def resolve_island_backend_policy(
    base_experiment_dir: str | Path,
    *,
    local_patch_reference_centerline_path: str | Path | None = None,
) -> IslandBackendPolicy:
    """Use the same local/fallback policy as the independent experiment.

    A panelized parent is never silently sent to a full-raster fallback: local
    island patches are published as virtual overlays and failures are retried
    through the established panelized child backend.
    """
    base = Path(base_experiment_dir).expanduser().resolve()
    manifest_path = base / "experiment_manifest.json"
    manifest = _load_json_file(manifest_path) if manifest_path.exists() else {}
    backends = dict(manifest.get("execution_backends") or {})
    panelized = str(backends.get("base_backend", "")) == "panelized"
    if not panelized:
        return IslandBackendPolicy("full_raster", "materialized", "full_raster")

    run_root: Path | None = None
    cleaned_mask = dict(manifest.get("root_paths") or {}).get("cleaned_mask")
    if cleaned_mask:
        candidate = Path(str(cleaned_mask)).expanduser().resolve().parent.parent
        if (candidate / "panels" / "panel_plan.gpkg").exists():
            run_root = candidate
    if run_root is None and local_patch_reference_centerline_path is not None:
        candidate = Path(local_patch_reference_centerline_path).expanduser().resolve().parent.parent
        if (candidate / "panels" / "panel_plan.gpkg").exists():
            run_root = candidate
    if run_root is None:
        raise FileNotFoundError(
            "The independent experiment is panelized but its source panel run could not be "
            "resolved from experiment_manifest.json or --local-patch-reference-centerline."
        )
    return IslandBackendPolicy("panelized", "virtual_patch", "panelized", run_root)


@dataclass(frozen=True)
class IslandVariantSelection:
    """One approved island candidate and its explicit graph-anchor context."""

    island_id: str
    geometry: object
    anchor_unit_id: int
    anchor_bifurcation: int
    anchor_confluence: int
    source_unit_ids: tuple[int, ...]


@dataclass(frozen=True)
class IslandMaskProducts:
    collapsed_mask_path: Path
    origin_mask_path: Path
    edit_geometries_path: Path
    pixels_added: int


@dataclass(frozen=True)
class IslandTopologyRun:
    selection: IslandVariantSelection
    mask_products: IslandMaskProducts
    rivgraph_paths: dict[str, Path | None]


@dataclass(frozen=True)
class IslandDirectedProducts:
    directed_links_path: Path
    directed_nodes_path: Path
    node_match_path: Path
    link_match_path: Path
    lineage_path: Path
    anchor_summary_path: Path


@dataclass(frozen=True)
class IslandWidthProducts:
    bank_labels_path: Path
    enriched_links_path: Path
    width_samples_path: Path
    removed_bank_labels: tuple[int, ...]
    segmented_stub_inheritance: dict[str, object] | None = None


@dataclass(frozen=True)
class IslandRoutingWidthProducts:
    enriched_links_path: Path
    audit_path: Path
    audit_manifest_path: Path


@dataclass(frozen=True)
class IslandFullRasterExecution:
    """Published outcome of one manifest-driven island observation."""

    island_id: str
    execution_action: str
    outcome: str
    state_reference_dir: Path
    run_manifest_path: Path | None


def _parse_unit_ids(value: object) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in str(value).split(",") if part.strip())


def _state_collapse_edit_geometries_path(state_dir: str | Path) -> Path:
    """Resolve the canonical state edit geometry across writer layouts."""
    state = Path(state_dir)
    candidates = (
        state / "variant" / "mask" / "collapse_edit_geometries.gpkg",
        state / "variant" / "summary" / "collapse_edit_geometries.gpkg",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "State is missing collapse_edit_geometries.gpkg under mask/ or summary/: "
        f"{state}"
    )


def load_island_variant_selection(
    setup_dir: str | Path,
    island_id: str,
) -> IslandVariantSelection:
    """Load one approved candidate from a frozen ``island_only_setup`` directory.

    Refuses excluded or absent IDs, preventing a raster artefact from being
    accidentally sent to a costly downstream workflow.
    """

    root = Path(setup_dir)
    manifest = json.loads((root / "island_experiment_manifest.json").read_text())
    approved = set(map(str, manifest.get("approved_island_ids", [])))
    if str(island_id) not in approved:
        raise ValueError(f"Island {island_id!r} is not an approved pilot candidate.")

    registry = pd.read_csv(root / "island_candidate_registry.csv")
    geometries = gpd.read_file(root / "island_candidate_geometries.gpkg")
    row = registry.loc[registry["island_id"].astype(str).eq(str(island_id))]
    geometry_row = geometries.loc[geometries["island_id"].astype(str).eq(str(island_id))]
    if len(row) != 1 or len(geometry_row) != 1:
        raise ValueError(f"Island setup is inconsistent for {island_id!r}.")
    item = row.iloc[0]
    return IslandVariantSelection(
        island_id=str(island_id),
        geometry=geometry_row.geometry.iloc[0],
        anchor_unit_id=int(item["proposed_anchor_unit_id"]),
        anchor_bifurcation=int(item["anchor_bifurcation"]),
        anchor_confluence=int(item["anchor_confluence"]),
        source_unit_ids=_parse_unit_ids(item["current_unit_ids"]),
    )


def build_island_equivalence_registry(
    *,
    setup_dir: str | Path,
    experiment_dir: str | Path,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Classify approved islands as unit aliases or required island reruns.

    This is deliberately a cheap preflight: exact rasterized edit equality,
    frozen-anchor equality, and bank-label removal equality. Later stages can
    add directed-lineage equality before reusing a completed hydraulic state.
    """
    setup, experiment = Path(setup_dir), Path(experiment_dir)
    base_variant = experiment / "states" / "S000_base" / "variant"
    base_mask_path = next((base_variant / "mask").glob("*_collapsed.tif"))
    units = pd.read_csv(experiment / "states" / "S000_base" / "hierarchy" / "unit_summary.csv").set_index("unit_id")
    base_labels_path = next((experiment / "boundary_components").glob("*land_water_labels*.tif"))
    manifest = json.loads((setup / "island_experiment_manifest.json").read_text())
    rows = []
    for island_id in manifest["approved_island_ids"]:
        selection = load_island_variant_selection(setup, island_id)
        state = next((experiment / "states").glob(f"S*_unit_{selection.anchor_unit_id}"))
        edits = gpd.read_file(_state_collapse_edit_geometries_path(state))
        adds = edits.loc[edits["action"].eq("add")]
        # Rasterise only the bounded edit envelope. This is exact, but avoids
        # allocating multi-gigabyte arrays for panelized parent masks.
        edit_geometries = [selection.geometry, *adds.geometry.tolist()]
        minx = min(geometry.bounds[0] for geometry in edit_geometries)
        miny = min(geometry.bounds[1] for geometry in edit_geometries)
        maxx = max(geometry.bounds[2] for geometry in edit_geometries)
        maxy = max(geometry.bounds[3] for geometry in edit_geometries)
        with rasterio.open(base_mask_path) as mask_source, rasterio.open(base_labels_path) as labels_source:
            window = from_bounds(minx, miny, maxx, maxy, transform=mask_source.transform)
            window = window.round_offsets().round_lengths().intersection(
                rasterio.windows.Window(0, 0, mask_source.width, mask_source.height)
            )
            base_water = mask_source.read(1, window=window) > 0
            labels = labels_source.read(1, window=window)
            transform = mask_source.window_transform(window)
            shape = (int(window.height), int(window.width))
        island_add = rasterize([(selection.geometry, 1)], out_shape=shape, transform=transform, dtype="uint8", all_touched=True).astype(bool) & ~base_water
        unit_add = rasterize([(geom, 1) for geom in adds.geometry], out_shape=shape, transform=transform, dtype="uint8", all_touched=True).astype(bool) & ~base_water
        mask_equal = bool(np.array_equal(island_add, unit_add))
        unit = units.loc[int(selection.anchor_unit_id)]
        anchor_equal = int(unit.bifurcation) == selection.anchor_bifurcation and int(unit.confluence) == selection.anchor_confluence
        island_labels = sorted(np.unique(labels[island_add & (labels > 1)]).astype(int).tolist())
        unit_labels = sorted(np.unique(labels[unit_add & (labels > 1)]).astype(int).tolist())
        bank_equal = island_labels == unit_labels
        rows.append({"island_id": island_id, "source_unit_id": selection.anchor_unit_id,
                     "mask_equal": mask_equal, "anchor_equal": anchor_equal, "bank_labels_equal": bank_equal,
                     "island_added_pixels": int(island_add.sum()), "unit_added_pixels": int(unit_add.sum()),
                     "island_bank_labels": ",".join(map(str,island_labels)), "unit_bank_labels": ",".join(map(str,unit_labels)),
                     "equivalence_status": "alias_existing_unit_state" if mask_equal and anchor_equal and bank_equal else "requires_full_island_state"})
    frame = pd.DataFrame(
        rows,
        columns=[
            "island_id", "source_unit_id", "mask_equal", "anchor_equal",
            "bank_labels_equal", "island_added_pixels", "unit_added_pixels",
            "island_bank_labels", "unit_bank_labels", "equivalence_status",
        ],
    )
    if output_path is not None: frame.to_csv(output_path, index=False)
    return frame


def build_island_execution_plan(
    *,
    setup_dir: str | Path,
    experiment_dir: str | Path,
    equivalence_registry_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Backward-compatible alias for :func:`build_island_observation_manifest`.

    The returned table is deliberately an observation crosswalk, not a second
    state registry. Alias rows reference their canonical independent state and
    never copy its products.
    """
    return build_island_observation_manifest(
        setup_dir=setup_dir,
        experiment_dir=experiment_dir,
        equivalence_registry_path=equivalence_registry_path,
        output_dir=output_dir,
    )


def _canonical_unit_state_dirs(experiment: Path, source_unit_ids: pd.Series) -> dict[int, Path]:
    states_dir = experiment / "states"
    resolved: dict[int, Path] = {}
    for unit_id in source_unit_ids.astype(int).unique():
        matches = sorted(states_dir.glob(f"S*_unit_{int(unit_id)}"))
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one canonical independent state for unit {unit_id}; "
                f"found {len(matches)} under {states_dir}."
            )
        resolved[int(unit_id)] = matches[0].resolve()
    return resolved


def _approved_island_geometry_hashes(setup: Path) -> dict[str, str]:
    geometries = gpd.read_file(setup / "island_candidate_geometries.gpkg")
    if geometries["island_id"].duplicated().any():
        raise ValueError("Island candidate geometries must have unique island_id values.")
    return {
        str(row.island_id): hashlib.sha256(bytes(row.geometry.wkb)).hexdigest()
        for row in geometries.itertuples(index=False)
    }


def _load_json_file(path: str | Path) -> dict[str, object]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    return json.loads(source.read_text(encoding="utf-8"))


def create_island_experiment_root(
    analysis_manifest_path: str | Path,
    *,
    base_experiment_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Create a compact standalone island experiment root from validated outputs.

    This promotion writes only durable registries and provenance. Alias
    observations reference independent states, while genuine island
    observations reference their validated island-state directories. No large
    products are copied, moved, or deleted.
    """
    analysis_path = Path(analysis_manifest_path).expanduser().resolve()
    base_dir = Path(base_experiment_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if not analysis_path.exists():
        raise FileNotFoundError(f"Island analysis manifest was not found: {analysis_path}")
    if not (base_dir / "experiment_manifest.json").exists():
        raise FileNotFoundError(f"Base experiment manifest was not found under: {base_dir}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            "Island experiment destination already exists and is non-empty; "
            f"refusing to overwrite it: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    observations = pd.read_csv(analysis_path, low_memory=False)
    required = {"island_id", "analysis_backend", "analysis_state_dir"}
    missing = sorted(required - set(observations.columns))
    if missing:
        raise ValueError(f"Island analysis manifest is missing columns: {missing}")
    if observations["island_id"].astype(str).duplicated().any():
        raise ValueError("Island analysis manifest contains duplicate island IDs.")

    registry = observations.copy()
    registry["observation_role"] = np.where(
        registry["analysis_backend"].astype(str).eq("independent_unit_alias"),
        "alias_existing_independent_state",
        "genuine_island_state",
    )
    registry["artifact_policy"] = np.where(
        registry["observation_role"].eq("alias_existing_independent_state"),
        "reference_independent_state_no_copy",
        "reference_validated_island_state_no_copy",
    )
    registry["base_experiment_dir"] = str(base_dir)
    registry_path = destination / "island_observation_registry.csv"
    registry.to_csv(registry_path, index=False)

    source_root = analysis_path.parent
    referenced_artifacts = {
        "source_analysis_manifest": analysis_path,
        "source_analysis_manifest_json": source_root / "island_analysis_manifest.json",
        "hydraulic_context_registry": source_root / "island_hydraulic_context_registry.csv",
        "rapid_prep_registry": source_root / "island_rapid_prep_registry.csv",
        "rapid_run_registry": source_root / "island_rapid_run_registry.csv",
        "island_equivalent_resistance_registry": (
            source_root / "island_equivalent_resistance" /
            "island_equivalent_resistance_observation_registry.csv"
        ),
        "island_vs_independent_hydrograph_comparison": (
            source_root / "island_vs_independent_hydrograph_comparison.csv"
        ),
    }
    artifact_registry_path = destination / "artifact_reference_registry.csv"
    pd.DataFrame([
        {
            "artifact_name": name,
            "source_path": str(path),
            "exists": bool(path.exists()),
            "storage_policy": "reference_only_no_copy",
        }
        for name, path in referenced_artifacts.items()
    ]).to_csv(artifact_registry_path, index=False)
    manifest_path = destination / "experiment_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "island-experiment-root-v1",
                "experiment_kind": "island-only",
                "mode": "island-anchored",
                "base_experiment_dir": str(base_dir),
                "source_setup_dir": str(source_root),
                "source_analysis_manifest": str(analysis_path),
                "observation_registry": str(registry_path),
                "artifact_reference_registry": str(artifact_registry_path),
                "n_observations": int(len(registry)),
                "n_alias_references": int(registry["observation_role"].eq("alias_existing_independent_state").sum()),
                "n_genuine_island_states": int(registry["observation_role"].eq("genuine_island_state").sum()),
                "storage_contract": {
                    "aliases": "reference existing independent states; never copy",
                    "genuine_islands": "reference validated island states until a deliberate relocation is requested",
                    "promotion_action": "metadata only; no raster, network, RAPID, or debug artifact copied/moved/deleted",
                },
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return {
        "experiment_dir": destination,
        "experiment_manifest": manifest_path,
        "observation_registry": registry_path,
        "artifact_reference_registry": artifact_registry_path,
    }


def freeze_reviewed_island_setup(
    reviewed_setup_dir: str | Path,
    *,
    output_dir: str | Path,
) -> Path:
    """Copy the small reviewed-island contract into a new experiment root.

    Candidate geometry and review metadata are intentionally retained by value:
    they are the reproducibility contract for the island treatment.  Network,
    raster, RivGraph, and routing products are never copied here.
    """
    source = Path(reviewed_setup_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    required = (
        "island_candidate_registry.csv",
        "island_candidate_geometries.gpkg",
        "island_experiment_manifest.json",
        "island_equivalence_registry.csv",
    )
    missing = [name for name in required if not (source / name).exists()]
    if missing:
        raise FileNotFoundError(f"Reviewed island setup is incomplete: {missing}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Frozen island setup destination is non-empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    pd.read_csv(source / "island_candidate_registry.csv").to_csv(
        destination / "island_candidate_registry.csv", index=False
    )
    candidates = gpd.read_file(source / "island_candidate_geometries.gpkg")
    candidates.to_file(destination / "island_candidate_geometries.gpkg", driver="GPKG")
    pd.read_csv(source / "island_equivalence_registry.csv").to_csv(
        destination / "island_equivalence_registry.csv", index=False
    )
    reviewed_manifest = _load_json_file(source / "island_experiment_manifest.json")
    reviewed_manifest.update({
        "frozen_from_reviewed_setup": str(source),
        "frozen_setup_schema_version": "island-reviewed-setup-v1",
    })
    (destination / "island_experiment_manifest.json").write_text(
        json.dumps(reviewed_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def run_island_experiment(
    *,
    reviewed_setup_dir: str | Path,
    base_experiment_dir: str | Path,
    output_dir: str | Path,
    base_mask_path: str | Path,
    parent_links_path: str | Path,
    parent_nodes_path: str | Path,
    boundary_component_labels_path: str | Path,
    wet_reference_mask_path: str | Path,
    exit_sides: str,
    example_id: str,
    single_thread: bool = False,
    execution_mode: str = "full_raster",
    compute_equivalent_resistance: bool = False,
    local_patch_reference_centerline_path: str | Path | None = None,
    sword_node_source_path: str | Path | None = None,
    sword_wse_field: str | None = None,
    sword_match_tolerance: float | None = None,
    sword_example_station_source_path: str | Path | None = None,
    sword_station_match_source_path: str | Path | None = None,
    sword_reach_buffer_steps: int = 2,
    adopt_metadata_root: bool = False,
    resume_existing_experiment: bool = False,
    rebuild_observation_manifest: bool = False,
) -> dict[str, Path]:
    """Materialise one standalone island-only experiment after candidate review.

    ``reviewed_setup_dir`` is a required explicit review gate.  The resulting
    output root contains the entire conceptual-island manifest, but executes
    only the islands that fail the exact independent-state equivalence check.
    """
    root = Path(output_dir).expanduser().resolve()
    base_experiment = Path(base_experiment_dir).expanduser().resolve()
    if adopt_metadata_root and resume_existing_experiment:
        raise ValueError(
            "--adopt-metadata-root and --resume-existing-experiment are mutually exclusive."
        )
    if rebuild_observation_manifest and not resume_existing_experiment:
        raise ValueError(
            "--rebuild-observation-manifest is only valid with "
            "--resume-existing-experiment."
        )

    metadata_root = False
    resuming = False
    if root.exists() and any(root.iterdir()):
        if resume_existing_experiment:
            setup_dir = root / "setup"
            observation_path = root / "island_observation_manifest.csv"
            equivalence_path = setup_dir / "island_equivalence_registry.csv"
            required = [
                setup_dir / "island_experiment_manifest.json",
                equivalence_path,
            ]
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "Cannot resume island experiment: the frozen setup or observation "
                    f"manifest is missing: {missing}"
                )
            resuming = True
        else:
            root_manifest_path = root / "experiment_manifest.json"
            if root_manifest_path.exists():
                try:
                    metadata_root = (
                        _load_json_file(root_manifest_path).get("schema_version")
                        == "island-experiment-root-v1"
                    )
                except (OSError, json.JSONDecodeError):
                    metadata_root = False
            if not (adopt_metadata_root and metadata_root):
                raise FileExistsError(
                    "Island experiment output is non-empty; refusing to overwrite it: "
                    f"{root}"
                )
            if (root / "setup").exists() or (root / "island_reruns").exists():
                raise FileExistsError(
                    "The output root contains state/setup products and cannot be adopted as a "
                    f"metadata-only island root: {root}"
                )
            shutil.copy2(root_manifest_path, root / "promotion_manifest.json")
    if execution_mode not in {"full_raster", "local_patch"}:
        raise ValueError("execution_mode must be 'full_raster' or 'local_patch'.")
    root.mkdir(parents=True, exist_ok=True)
    if resuming:
        setup = root / "setup"
        equivalence_path = setup / "island_equivalence_registry.csv"
        observation_path = root / "island_observation_manifest.csv"
        if observation_path.exists() and not rebuild_observation_manifest:
            observations = pd.read_csv(observation_path, low_memory=False)
        else:
            # An interruption can occur before the small observation crosswalk
            # is written. A reviewed manual adjustment can also deliberately
            # change frozen candidate inclusion/geometry. In both cases,
            # rebuild only this deterministic crosswalk from the frozen review
            # records; never rerun detection.
            observations = build_island_observation_manifest(
                setup_dir=setup,
                experiment_dir=base_experiment,
                equivalence_registry_path=equivalence_path,
                output_dir=root,
            )
    else:
        setup = freeze_reviewed_island_setup(reviewed_setup_dir, output_dir=root / "setup")
        equivalence_path = setup / "island_equivalence_registry.csv"
        if not equivalence_path.is_file():
            raise FileNotFoundError(
                "Reviewed island setup has no precomputed equivalence registry: "
                f"{equivalence_path}. Run run_prepare_island_setup.py first."
            )
        equivalence = pd.read_csv(equivalence_path, low_memory=False)
        observations = build_island_observation_manifest(
            setup_dir=setup,
            experiment_dir=base_experiment,
            equivalence_registry_path=equivalence_path,
            output_dir=root,
        )
        observation_path = root / "island_observation_manifest.csv"
    full_report: pd.DataFrame | None = None
    local_report_path: Path | None = None
    panelized_report: pd.DataFrame | None = None
    backend_policy = resolve_island_backend_policy(
        base_experiment,
        local_patch_reference_centerline_path=local_patch_reference_centerline_path,
    )
    if execution_mode == "full_raster":
        full_report = run_island_full_raster_observations(
            observation_path,
            base_mask_path=base_mask_path,
            parent_links_path=parent_links_path,
            parent_nodes_path=parent_nodes_path,
            boundary_component_labels_path=boundary_component_labels_path,
            wet_reference_mask_path=wet_reference_mask_path,
            exit_sides=exit_sides,
            example_id=str(example_id),
            single_thread=single_thread,
            report_dir=root,
            sword_node_source_path=sword_node_source_path,
            sword_wse_field=sword_wse_field,
            sword_match_tolerance=sword_match_tolerance,
            sword_example_station_source_path=sword_example_station_source_path,
            sword_station_match_source_path=sword_station_match_source_path,
            sword_reach_buffer_steps=sword_reach_buffer_steps,
        )
    else:
        local_report = run_island_local_patch_observations(
            observation_path,
            base_mask_path=base_mask_path,
            parent_links_path=parent_links_path,
            parent_nodes_path=parent_nodes_path,
            boundary_component_labels_path=boundary_component_labels_path,
            exit_sides=exit_sides,
            example_id=str(example_id),
            single_thread=single_thread,
            report_dir=root,
            local_patch_reference_centerline_path=local_patch_reference_centerline_path,
            local_patch_mask_publication=backend_policy.local_patch_mask_publication,
            fallback_backend=backend_policy.fallback_backend,
        )
        local_report_path = root / "island_local_patch_execution.csv"
        # The production report always has island_id; retaining this tolerant
        # path keeps lightweight callers/tests that only report outcomes valid.
        fallback_ids = (
            local_report.loc[
                local_report["outcome"].eq(
                    f"fallback_{backend_policy.fallback_backend}_required"
                ),
                "island_id",
            ].astype(str).tolist()
            if "island_id" in local_report.columns else []
        )
        if fallback_ids and backend_policy.fallback_backend == "full_raster":
            full_report = run_island_full_raster_observations(
                observation_path,
                base_mask_path=base_mask_path,
                parent_links_path=parent_links_path,
                parent_nodes_path=parent_nodes_path,
                boundary_component_labels_path=boundary_component_labels_path,
                wet_reference_mask_path=wet_reference_mask_path,
                exit_sides=exit_sides,
                example_id=str(example_id),
                single_thread=single_thread,
                island_ids=fallback_ids,
                report_dir=root,
                sword_node_source_path=sword_node_source_path,
                sword_wse_field=sword_wse_field,
                sword_match_tolerance=sword_match_tolerance,
                sword_example_station_source_path=sword_example_station_source_path,
                sword_station_match_source_path=sword_station_match_source_path,
                sword_reach_buffer_steps=sword_reach_buffer_steps,
            )
        elif fallback_ids:
            panelized_report = run_island_panelized_observations(
                observation_path,
                base_experiment_dir=base_experiment,
                panelized_run_root=backend_policy.panelized_run_root,
                base_mask_path=base_mask_path,
                parent_links_path=parent_links_path,
                parent_nodes_path=parent_nodes_path,
                boundary_component_labels_path=boundary_component_labels_path,
                exit_sides=exit_sides,
                example_id=str(example_id),
                island_ids=fallback_ids,
                report_dir=root,
                sword_node_source_path=sword_node_source_path,
                sword_wse_field=sword_wse_field,
                sword_match_tolerance=sword_match_tolerance,
                sword_example_station_source_path=sword_example_station_source_path,
                sword_station_match_source_path=sword_station_match_source_path,
                sword_reach_buffer_steps=sword_reach_buffer_steps,
            )
    analysis = build_island_analysis_manifest(observation_path, output_dir=root)
    equivalent_resistance_registry: Path | None = None
    if compute_equivalent_resistance:
        from network_variants.island_equivalent_resistance import (
            run_island_equivalent_resistance,
        )

        run_island_equivalent_resistance(
            root / "island_analysis_manifest.csv",
            base_experiment_dir=base_experiment,
            output_dir=root / "equivalent_resistance",
        )
        equivalent_resistance_registry = (
            root / "equivalent_resistance" /
            "island_equivalent_resistance_observation_registry.csv"
        )
    manifest_path = root / "experiment_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "island-experiment-v1",
                "experiment_kind": "island-only",
                "mode": "island-anchored",
                "example_id": str(example_id),
                "base_experiment_dir": str(base_experiment),
                "reviewed_setup_source": str(Path(reviewed_setup_dir).expanduser().resolve()),
                "frozen_setup_dir": str(setup),
                "execution_mode": execution_mode,
                "backend_policy": {
                    "base_backend": backend_policy.base_backend,
                    "local_patch_mask_publication": backend_policy.local_patch_mask_publication,
                    "fallback_backend": backend_policy.fallback_backend,
                    "panelized_run_root": str(backend_policy.panelized_run_root)
                    if backend_policy.panelized_run_root is not None else None,
                },
                "adopted_metadata_root": bool(adopt_metadata_root and metadata_root),
                "resumed_existing_experiment": bool(resuming),
                "observation_manifest": str(observation_path),
                "analysis_manifest": str(root / "island_analysis_manifest.csv"),
                "equivalent_resistance_registry": (
                    str(equivalent_resistance_registry)
                    if equivalent_resistance_registry is not None else None
                ),
                "n_conceptual_islands": int(len(observations)),
                "n_alias_references": int(observations["execution_action"].eq("alias_existing_unit_state").sum()),
                "n_genuine_island_states": int(observations["execution_action"].eq("run_island_state").sum()),
                "n_full_raster_completed": int(full_report["outcome"].eq("completed_full_raster_run").sum()) if full_report is not None else 0,
                "n_full_raster_reused": int(full_report["outcome"].eq("reused_completed_run").sum()) if full_report is not None else 0,
                "n_panelized_completed": int(panelized_report["outcome"].eq("completed_panelized_run").sum()) if panelized_report is not None else 0,
                "n_local_patch_completed": int(
                    pd.read_csv(local_report_path)["outcome"].eq("completed_local_patch_run").sum()
                ) if local_report_path is not None else 0,
                "n_analysis_ready": int(analysis["analysis_execution_status"].astype(str).str.startswith("ready").sum()),
                "storage_contract": {
                    "aliases": "reference existing independent state products without copying",
                    "genuine_islands": "store only explicitly executed island state products under island_reruns",
                    "reviewed_setup": "small review metadata and geometry are frozen under setup",
                },
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    result = {
        "experiment_dir": root,
        "experiment_manifest": manifest_path,
        "setup_dir": setup,
        "equivalence_registry": equivalence_path,
        "observation_manifest": observation_path,
        "analysis_manifest": root / "island_analysis_manifest.csv",
    }
    if full_report is not None:
        result["full_raster_report"] = root / "island_full_raster_execution.csv"
    if local_report_path is not None:
        result["local_patch_report"] = local_report_path
    if equivalent_resistance_registry is not None:
        result["equivalent_resistance_registry"] = equivalent_resistance_registry
    return result


def build_island_observation_manifest(
    *,
    setup_dir: str | Path,
    experiment_dir: str | Path,
    equivalence_registry_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Write one compact observation row per approved conceptual island.

    This is the durable analysis crosswalk for island-only sensitivity work.
    It is *not* an island state registry: an exact alias merely resolves to an
    existing independent unit state, while a rerun receives a future output
    location. No alias network, width, or routing products are copied.
    """
    setup, experiment = Path(setup_dir), Path(experiment_dir)
    equivalence_path = Path(equivalence_registry_path) if equivalence_registry_path else setup / "island_equivalence_registry.csv"
    equivalence = pd.read_csv(equivalence_path)
    all_candidates = pd.read_csv(setup / "island_candidate_registry.csv").rename(
        columns={"proposed_anchor_unit_id": "source_unit_id"}
    )
    # Candidate-only setups created before the review-status column existed
    # treated every registry row as approved. Preserve that read compatibility;
    # modern reviewed setups must explicitly retain approved_pilot only.
    candidates = (
        all_candidates.loc[all_candidates["selection_status"].eq("approved_pilot")].copy()
        if "selection_status" in all_candidates.columns else all_candidates.copy()
    )
    # CSV inference sees a float column whenever excluded candidates have a
    # missing anchor, while the equivalence table may preserve the same ID as
    # text. Normalize both sides before enforcing the one-to-one crosswalk.
    candidates["source_unit_id"] = pd.to_numeric(
        candidates["source_unit_id"], errors="raise"
    ).astype("Int64")
    equivalence["source_unit_id"] = pd.to_numeric(
        equivalence["source_unit_id"], errors="raise"
    ).astype("Int64")
    plan = candidates.merge(equivalence, on=["island_id", "source_unit_id"], how="inner", validate="1:1")
    if len(plan) != len(candidates):
        raise ValueError("Every approved island must have exactly one equivalence record.")
    source_states = _canonical_unit_state_dirs(experiment, plan["source_unit_id"])
    geometry_hashes = {
        island_id: geometry_hash
        for island_id, geometry_hash in _approved_island_geometry_hashes(setup).items()
        if island_id in set(candidates["island_id"].astype(str))
    }
    if set(plan["island_id"].astype(str)) != set(geometry_hashes):
        raise ValueError("Approved island geometry IDs do not match the observation plan.")
    plan["source_state_dir"] = plan["source_unit_id"].astype(int).map(lambda unit_id: str(source_states[unit_id]))
    plan["execution_action"] = np.where(
        plan["equivalence_status"].eq("alias_existing_unit_state"),
        "alias_existing_unit_state", "run_island_state",
    )
    plan["execution_backend"] = np.where(plan["execution_action"].eq("run_island_state"), "pending_island_execution", "reference_existing_state")
    plan["observation_status"] = np.where(plan["execution_action"].eq("alias_existing_unit_state"), "ready_reference", "pending_island_execution")
    plan["geometry_sha256"] = plan["island_id"].astype(str).map(geometry_hashes)
    plan["canonical_state_dir"] = np.where(plan["execution_action"].eq("alias_existing_unit_state"), plan["source_state_dir"], "")
    plan["island_setup_dir"] = str(setup.resolve())
    root = Path(output_dir) if output_dir is not None else setup
    rerun_root = (root / "island_reruns").resolve()
    plan["island_run_dir"] = np.where(
        plan["execution_action"].eq("run_island_state"),
        plan["island_id"].astype(str).map(lambda island_id: str(rerun_root / island_id)),
        "",
    )
    plan["analysis_state_dir"] = np.where(
        plan["execution_action"].eq("alias_existing_unit_state"),
        plan["canonical_state_dir"],
        plan["island_run_dir"],
    )
    plan = plan.sort_values(["execution_action", "island_id"], kind="mergesort").reset_index(drop=True)
    if output_dir is not None:
        root.mkdir(parents=True, exist_ok=True)
        plan.to_csv(root / "island_observation_manifest.csv", index=False)
        manifest = {
            "schema_version": "island-observation-manifest-v1",
            "n_conceptual_islands": int(len(plan)),
            "n_aliases": int(plan["execution_action"].eq("alias_existing_unit_state").sum()),
            "n_required_island_runs": int(plan["execution_action"].eq("run_island_state").sum()),
            "equivalence_registry": str(equivalence_path.resolve()),
            "setup_dir": str(setup.resolve()),
            "rerun_output_root": str(rerun_root),
            "alias_policy": "reference_existing_independent_state_without_copying_products",
        }
        (root / "island_observation_manifest.json").write_text(json.dumps(manifest, indent=2))
    return plan


def build_island_analysis_manifest(
    observation_manifest_path: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Resolve all conceptual islands to their final analysis products.

    Exact aliases remain references to the already-published independent-unit
    state.  A non-equivalent island resolves to its explicit full-raster run,
    with the bounded local-patch result recorded as paired-validation
    provenance.  The result deliberately contains no copied network products.
    """
    manifest_path = Path(observation_manifest_path).resolve()
    observations = pd.read_csv(manifest_path, low_memory=False)
    required = {
        "island_id", "execution_action", "source_state_dir", "island_run_dir",
        "anchor_bifurcation", "anchor_confluence", "source_unit_id",
    }
    missing = sorted(required - set(observations.columns))
    if missing:
        raise ValueError(f"Island observation manifest is missing columns: {missing}")

    records: list[dict[str, object]] = []
    for row in observations.itertuples(index=False):
        record = row._asdict()
        island_id = str(row.island_id)
        action = str(row.execution_action)
        if action == "alias_existing_unit_state":
            state_dir = Path(str(row.source_state_dir)).resolve()
            variant_dir = state_dir / "variant"
            variant_manifest_path = variant_dir / "summary" / "variant_manifest.json"
            variant_manifest = _load_json_file(variant_manifest_path)
            paths = dict(variant_manifest.get("output_paths", {}))
            required_paths = ("directed_links", "directed_nodes", "links_with_width_families")
            missing_paths = [key for key in required_paths if not paths.get(key) or not Path(paths[key]).exists()]
            if missing_paths:
                raise FileNotFoundError(
                    f"Alias {island_id} has incomplete independent-state products: {missing_paths}"
                )
            record.update({
                "analysis_execution_status": "ready_alias_reference",
                "analysis_backend": "independent_unit_alias",
                "analysis_state_dir": str(state_dir),
                "analysis_variant_dir": str(variant_dir),
                "analysis_directed_links_path": str(paths["directed_links"]),
                "analysis_directed_nodes_path": str(paths["directed_nodes"]),
                "analysis_routing_width_links_path": str(paths["links_with_width_families"]),
                "analysis_product_manifest_path": str(variant_manifest_path),
                "local_patch_status": "not_applicable_alias",
                "local_patch_variant_dir": "",
                "local_patch_routing_width_links_path": "",
            })
        elif action == "run_island_state":
            run_dir = Path(str(row.island_run_dir)).resolve()
            required_paths = ("directed_links", "directed_nodes", "routing_width_links")
            full_manifest_path = run_dir / "island_run_manifest.json"
            local_manifest_path = run_dir / "local_patch" / "island_local_patch_run_manifest.json"
            full_manifest = _load_json_file(full_manifest_path) if full_manifest_path.exists() else None
            local_manifest = _load_json_file(local_manifest_path) if local_manifest_path.exists() else None

            def valid_paths(payload: dict[str, object] | None) -> tuple[dict[str, object], list[str]]:
                paths = dict((payload or {}).get("final_paths", {}))
                missing = [key for key in required_paths if not paths.get(key) or not Path(str(paths[key])).exists()]
                return paths, missing

            full_paths, missing_full = valid_paths(full_manifest)
            local_paths, missing_local = valid_paths(local_manifest)
            full_complete = bool(full_manifest and full_manifest.get("status") == "complete" and not missing_full)
            local_complete = bool(local_manifest and local_manifest.get("status") == "complete" and not missing_local)
            if not full_complete and not local_complete:
                raise FileNotFoundError(
                    f"Island {island_id} has no complete final backend. "
                    f"full_missing={missing_full}; local_missing={missing_local}"
                )
            if full_complete:
                final_backend = str(full_manifest.get("execution_backend", "full_raster"))
                record.update({
                    "analysis_execution_status": (
                        f"ready_{final_backend}_with_local_validation"
                        if local_complete else f"ready_{final_backend}"
                    ),
                    "analysis_backend": f"island_{final_backend}",
                    "analysis_state_dir": str(run_dir),
                    "analysis_variant_dir": str(run_dir / "variant"),
                    "analysis_directed_links_path": str(full_paths["directed_links"]),
                    "analysis_directed_nodes_path": str(full_paths["directed_nodes"]),
                    "analysis_routing_width_links_path": str(full_paths["routing_width_links"]),
                    "analysis_product_manifest_path": str(full_manifest_path),
                    "full_width_pipeline": json.dumps(full_manifest.get("final_width_pipeline", {}), sort_keys=True),
                    "local_patch_status": "validated_complete" if local_complete else "not_run",
                    "local_patch_variant_dir": str(run_dir / "local_patch" / "variant") if local_complete else "",
                    "local_patch_directed_links_path": str(local_paths["directed_links"]) if local_complete else "",
                    "local_patch_directed_nodes_path": str(local_paths["directed_nodes"]) if local_complete else "",
                    "local_patch_routing_width_links_path": str(local_paths["routing_width_links"]) if local_complete else "",
                    "local_patch_product_manifest_path": str(local_manifest_path) if local_complete else "",
                })
            else:
                local_variant_dir = run_dir / "local_patch" / "variant"
                record.update({
                    "analysis_execution_status": "ready_direct_local_patch",
                    "analysis_backend": "island_local_patch",
                    "analysis_state_dir": str(run_dir / "local_patch"),
                    "analysis_variant_dir": str(local_variant_dir),
                    "analysis_directed_links_path": str(local_paths["directed_links"]),
                    "analysis_directed_nodes_path": str(local_paths["directed_nodes"]),
                    "analysis_routing_width_links_path": str(local_paths["routing_width_links"]),
                    "analysis_product_manifest_path": str(local_manifest_path),
                    "full_width_pipeline": "",
                    "local_patch_status": "direct_complete",
                    "local_patch_variant_dir": str(local_variant_dir),
                    "local_patch_directed_links_path": str(local_paths["directed_links"]),
                    "local_patch_directed_nodes_path": str(local_paths["directed_nodes"]),
                    "local_patch_routing_width_links_path": str(local_paths["routing_width_links"]),
                    "local_patch_product_manifest_path": str(local_manifest_path),
                })
        else:
            raise ValueError(f"Unsupported island execution action: {action!r}")
        records.append(record)

    analysis = pd.DataFrame.from_records(records).sort_values("island_id", kind="mergesort").reset_index(drop=True)
    if analysis["island_id"].duplicated().any() or len(analysis) != len(observations):
        raise ValueError("Island analysis manifest must retain one row per conceptual island.")
    root = Path(output_dir).resolve() if output_dir is not None else manifest_path.parent
    root.mkdir(parents=True, exist_ok=True)
    output_path = root / "island_analysis_manifest.csv"
    analysis.to_csv(output_path, index=False)
    summary = {
        "schema_version": "island-analysis-manifest-v1",
        "source_observation_manifest": str(manifest_path),
        "n_conceptual_islands": int(len(analysis)),
        "n_alias_references": int(analysis["analysis_backend"].eq("independent_unit_alias").sum()),
        "n_full_raster_island_states": int(analysis["analysis_backend"].eq("island_full_raster").sum()),
        "n_panelized_island_states": int(analysis["analysis_backend"].eq("island_panelized").sum()),
        "n_direct_local_patch_island_states": int(analysis["analysis_backend"].eq("island_local_patch").sum()),
        "n_validated_local_patch_states": int(analysis["local_patch_status"].eq("validated_complete").sum()),
        "analysis_manifest_csv": str(output_path),
        "alias_policy": "reference_existing_independent_state_without_copying_products",
    }
    (root / "island_analysis_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return analysis


def generate_island_hydraulic_contexts(
    analysis_manifest_path: str | Path,
    *,
    base_link_context_path: str | Path,
    base_node_context_path: str | Path,
    slope_config: object | None = None,
    output_dir: str | Path | None = None,
    refresh_existing: bool = False,
    backup_existing_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Generate hydraulic context only for non-aliased island observations.

    Alias rows are recorded as references to their existing independent-state
    hydraulic products. A genuine island—whether realised by full raster,
    direct local patch, or panelized fallback—receives a context beside its
    realised state and inherits slope context from the established base state.
    """
    from network_variants.hydraulic_context import (
        HYDRAULIC_CONTEXT_MANIFEST,
        generate_state_hydraulic_context,
    )

    manifest_path = Path(analysis_manifest_path).resolve()
    analysis = pd.read_csv(manifest_path, low_memory=False)
    required = {
        "island_id", "analysis_backend", "analysis_state_dir",
        "analysis_directed_links_path", "analysis_directed_nodes_path",
    }
    missing = sorted(required - set(analysis.columns))
    if missing:
        raise ValueError(f"Island analysis manifest is missing columns: {missing}")
    base_link = Path(base_link_context_path).resolve()
    base_node = Path(base_node_context_path).resolve()
    if not base_link.exists() or not base_node.exists():
        raise FileNotFoundError("Island hydraulic context requires complete base link/node context products.")

    backup_root = Path(backup_existing_dir).resolve() if backup_existing_dir is not None else None
    records: list[dict[str, object]] = []
    for row in analysis.itertuples(index=False):
        island_id = str(row.island_id)
        backend = str(row.analysis_backend)
        state_dir = Path(str(row.analysis_state_dir)).resolve()
        strict_status = str(getattr(row, "strict_analysis_status", "")).strip()
        excluded_duplicate = str(getattr(row, "separate_observation_excluded", False)).lower() in {"true", "1", "1.0"}
        if strict_status == "exclude_pruning_or_mask_topology_exception" or excluded_duplicate:
            records.append({
                "island_id": island_id,
                "hydraulic_context_action": "excluded_duplicate_use_independent" if excluded_duplicate else "excluded_topology_review",
                "hydraulic_context_status": "excluded_from_primary_analysis",
                "hydraulic_context_dir": "",
                "link_hydraulic_context_path": "",
                "node_hydraulic_context_path": "",
                "hydraulic_context_manifest_path": "",
            })
            continue
        if backend == "independent_unit_alias":
            context_dir = state_dir / "hydraulic_context"
            required_paths = (
                context_dir / "link_hydraulic_context.csv",
                context_dir / "node_hydraulic_context.csv",
                context_dir / HYDRAULIC_CONTEXT_MANIFEST,
            )
            if not all(path.exists() for path in required_paths):
                raise FileNotFoundError(f"Alias {island_id} lacks reusable hydraulic context: {context_dir}")
            records.append({
                "island_id": island_id,
                "hydraulic_context_action": "reference_existing_independent_state",
                "hydraulic_context_status": "ready_reference",
                "hydraulic_context_dir": str(context_dir),
                "link_hydraulic_context_path": str(context_dir / "link_hydraulic_context.csv"),
                "node_hydraulic_context_path": str(context_dir / "node_hydraulic_context.csv"),
                "hydraulic_context_manifest_path": str(context_dir / HYDRAULIC_CONTEXT_MANIFEST),
            })
            continue
        if backend not in {
            "island_full_raster",
            "island_local_patch",
            "island_panelized",
        }:
            raise ValueError(f"Unsupported island analysis backend: {backend!r}")

        context_dir = state_dir / "hydraulic_context"
        manifest_file = context_dir / HYDRAULIC_CONTEXT_MANIFEST
        refreshed_existing = False
        backup_dir = ""
        backup_files: list[str] = []
        if manifest_file.exists():
            existing = _load_json_file(manifest_file)
            paths = dict(existing.get("output_paths", {}))
            if (
                existing.get("input_paths", {}).get("directed_links") == str(Path(row.analysis_routing_width_links_path).resolve())
                and all(paths.get(key) and Path(paths[key]).exists() for key in ("link_hydraulic_context", "node_hydraulic_context"))
            ):
                if not refresh_existing:
                    records.append({
                        "island_id": island_id,
                        "hydraulic_context_action": "reused_completed_island_context",
                        "hydraulic_context_status": "ready_island_context",
                        "hydraulic_context_dir": str(context_dir),
                        "link_hydraulic_context_path": str(paths["link_hydraulic_context"]),
                        "node_hydraulic_context_path": str(paths["node_hydraulic_context"]),
                        "hydraulic_context_manifest_path": str(manifest_file),
                        "backup_dir": "",
                        "backup_files": "",
                    })
                    continue
            elif not refresh_existing:
                raise FileExistsError(f"Island hydraulic-context directory contains a non-matching result: {context_dir}")
            refreshed_existing = True
            if backup_root is not None:
                destination = backup_root / island_id
                for source in (
                    context_dir / "link_hydraulic_context.csv",
                    context_dir / "node_hydraulic_context.csv",
                    manifest_file,
                ):
                    if source.exists():
                        destination.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, destination / source.name)
                        backup_files.append(source.name)
                backup_dir = str(destination) if backup_files else ""

        outputs = generate_state_hydraulic_context(
            directed_links_path=row.analysis_routing_width_links_path,
            directed_nodes_path=row.analysis_directed_nodes_path,
            hierarchy_output_dir=None,
            output_dir=context_dir,
            state_id=island_id,
            config=slope_config,
            parent_link_context_path=base_link,
            # Island edits are stored as geometry, not as the unit-collapse
            # component CSV consumed by the unit-specific boundary solver.
            # Parent slope inheritance remains available through the normal
            # parent context supplied above.
            collapse_components_path=None,
            node_match_path=None,
            base_link_context_path=None,
            base_node_context_path=None,
        )
        records.append({
            "island_id": island_id,
            "hydraulic_context_action": (
                "refreshed_island_context" if refreshed_existing else "generated_island_context"
            ),
            "hydraulic_context_status": "ready_island_context",
            "hydraulic_context_dir": str(outputs.output_dir),
            "link_hydraulic_context_path": str(outputs.link_context_path),
            "node_hydraulic_context_path": str(outputs.node_context_path),
            "hydraulic_context_manifest_path": str(outputs.manifest_path),
            "backup_dir": backup_dir,
            "backup_files": ",".join(backup_files),
        })

    registry = pd.DataFrame.from_records(records).sort_values("island_id", kind="mergesort").reset_index(drop=True)
    root = Path(output_dir).resolve() if output_dir is not None else manifest_path.parent
    root.mkdir(parents=True, exist_ok=True)
    registry_path = root / "island_hydraulic_context_registry.csv"
    registry.to_csv(registry_path, index=False)
    (root / "island_hydraulic_context_registry.json").write_text(
        json.dumps({
            "schema_version": "island-hydraulic-context-registry-v1",
            "analysis_manifest": str(manifest_path),
            "base_link_context": str(base_link),
            "base_node_context": str(base_node),
            "n_observations": int(len(registry)),
            "n_alias_references": int(registry["hydraulic_context_action"].eq("reference_existing_independent_state").sum()),
            "n_generated_island_contexts": int(registry["hydraulic_context_action"].eq("generated_island_context").sum()),
            "n_refreshed_island_contexts": int(registry["hydraulic_context_action"].eq("refreshed_island_context").sum()),
            "refresh_existing": bool(refresh_existing),
            "backup_existing_dir": str(backup_root) if backup_root is not None else "",
            "n_excluded_topology_review": int(registry["hydraulic_context_action"].eq("excluded_topology_review").sum()),
            "registry_csv": str(registry_path),
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return registry


def materialize_island_mask(
    selection: IslandVariantSelection,
    *,
    base_mask_path: str | Path,
    output_dir: str | Path,
) -> IslandMaskProducts:
    """Materialize one island-only mask using the normal base/origin-label contract.

    Label ``1`` remains original water and ``101`` is the island addition, as
    expected by the two-bank width code.  This intentionally performs no
    topology, direction, hydraulic, or routing work yet.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with rasterio.open(base_mask_path) as src:
        base_water = src.read(1) > 0
        profile = src.profile.copy()
        added = rasterize(
            [(selection.geometry, 1)], out_shape=base_water.shape,
            transform=src.transform, fill=0, all_touched=True, dtype="uint8",
        ).astype(bool) & ~base_water
    if not np.any(added):
        raise ValueError(f"Island {selection.island_id!r} adds no dry raster cells.")
    collapsed = (base_water | added).astype("uint8")
    origin = np.where(base_water, 1, 0).astype("uint16")
    origin[added] = 101
    profile.update(count=1, compress="lzw")
    collapsed_path = output / f"{selection.island_id}_collapsed.tif"
    origin_path = output / "origin_mask.tif"
    with rasterio.open(collapsed_path, "w", **{**profile, "dtype": "uint8"}) as dst:
        dst.write(collapsed, 1)
    with rasterio.open(origin_path, "w", **{**profile, "dtype": "uint16"}) as dst:
        dst.write(origin, 1)
    edits = gpd.GeoDataFrame([{
        "component_id": selection.island_id, "geometry_role": "edit", "action": "add",
        "width_scope_label": 101, "unit_ids": ",".join(map(str, selection.source_unit_ids)),
        "anchor_unit_id": selection.anchor_unit_id, "anchor_bifurcation": selection.anchor_bifurcation,
        "anchor_confluence": selection.anchor_confluence, "geometry": selection.geometry,
    }], geometry="geometry", crs=profile.get("crs"))
    edits_path = output / "collapse_edit_geometries.gpkg"
    edits.to_file(edits_path, driver="GPKG")
    return IslandMaskProducts(collapsed_path, origin_path, edits_path, int(added.sum()))


def run_island_topology(
    selection: IslandVariantSelection,
    *,
    base_mask_path: str | Path,
    output_dir: str | Path,
    exit_sides: str,
    single_thread: bool = False,
) -> IslandTopologyRun:
    """Run the geometry-aware mask and RivGraph stages for one island candidate.

    The recorded anchor remains part of ``selection`` for the following
    direction/segmentation stage; no unit-collapse geometry is regenerated.
    """
    from network_variants.variant_generation import _run_rivgraph

    root = Path(output_dir)
    masks = materialize_island_mask(selection, base_mask_path=base_mask_path, output_dir=root / "mask")
    raw = _run_rivgraph(
        run_name=selection.island_id,
        collapsed_mask_path=masks.collapsed_mask_path,
        rivgraph_output_dir=root / "rivgraph",
        exit_sides=exit_sides,
        single_thread=single_thread,
        export_sword=False,
        metadata={"selection_type": "island", "island_id": selection.island_id,
                  "anchor_unit_id": selection.anchor_unit_id,
                  "anchor_bifurcation": selection.anchor_bifurcation,
                  "anchor_confluence": selection.anchor_confluence},
    )
    return IslandTopologyRun(selection, masks, {key: Path(value) if value is not None else None for key, value in raw.items()})


def island_anchor_unit_ids(selection: IslandVariantSelection) -> tuple[int, ...]:
    """Return the sole hierarchy unit used for anchored island segmentation.

    Shared-island provenance is retained in ``source_unit_ids``; anchoring uses
    the reviewed local unit frozen in the candidate registry.
    """
    return (selection.anchor_unit_id,)


def direct_island_topology(
    topology: IslandTopologyRun,
    *,
    parent_links_path: str | Path,
    parent_nodes_path: str | Path,
    output_dir: str | Path,
    match_tolerance: float,
    sword_node_source_path: str | Path | None = None,
    sword_wse_field: str | None = None,
    sword_match_tolerance: float | None = None,
    example_id: str | int | None = None,
    sword_example_station_source_path: str | Path | None = None,
    sword_station_match_source_path: str | Path | None = None,
    sword_reach_buffer_steps: int = 2,
) -> IslandDirectedProducts:
    """Apply the standard parent matching and anchored segmentation to an island run."""
    from hierarchy_level_definition.unit_detection.bifurcation_confluence_units import analyze_network
    from network_variants.variant_generation import (
        _assign_variant_directions, _infer_parent_node_order, _match_child_links_to_parent_links,
        _match_child_nodes_to_parent_nodes, _resolve_link_lineage, _segment_collapsed_links_to_anchor_nodes,
        _selection_anchor_nodes, _selection_edge_pairs,
    )
    from network_variants.sword_matching import match_variant_nodes_to_sword
    parent_links, parent_nodes = gpd.read_file(parent_links_path), gpd.read_file(parent_nodes_path)
    raw_links = gpd.read_file(topology.rivgraph_paths["links"])
    raw_nodes = gpd.read_file(topology.rivgraph_paths["nodes"])
    order, graph, _ = _infer_parent_node_order(parent_links, parent_nodes)
    node_match = _match_child_nodes_to_parent_nodes(parent_nodes, raw_nodes, match_tolerance=match_tolerance, parent_node_order=order)
    link_match = _match_child_links_to_parent_links(parent_links, raw_links, match_tolerance=match_tolerance)
    # A true island fill can regenerate a small connector group with no
    # spatial parent-overlap match (the skeleton may move substantially within
    # a large filled island).  Use the same boundary-potential fallback as the
    # validated local-patch merge: it is used only after parent matching and
    # local topology cannot orient the remaining component, and must still
    # pass the normal single source/sink DAG validation.
    directed_links, directed_nodes, summary = _assign_variant_directions(
        child_links=raw_links,
        child_nodes=raw_nodes,
        parent_links=parent_links,
        parent_nodes=parent_nodes,
        node_match=node_match,
        link_match=link_match,
        allow_boundary_potential_fallback=True,
    )
    lineage = _resolve_link_lineage(parent_graph=graph, directed_child_links=directed_links, node_match=node_match, link_match=link_match, match_tolerance=match_tolerance)
    _, units, _ = analyze_network(parent_links_path, parent_nodes_path)
    units_by_id = {unit.unit_id: unit for unit in units}
    anchors = island_anchor_unit_ids(topology.selection)
    up, down = _selection_anchor_nodes(resolved_unit_ids=anchors, units_by_id=units_by_id)
    directed_links, directed_nodes, node_match, anchor = _segment_collapsed_links_to_anchor_nodes(
        parent_graph=graph, parent_nodes=parent_nodes, parent_node_order=order, directed_links=directed_links,
        directed_nodes=directed_nodes, node_match=node_match, link_match=link_match, link_lineage=lineage,
        selection_edge_pairs=_selection_edge_pairs(resolved_unit_ids=anchors, units_by_id=units_by_id),
        nominal_upstream_node=up, nominal_downstream_node=down,
    )
    # Anchor segmentation can split/re-ID links. Rebuild matching and lineage
    # afterwards, exactly as the unit-based anchored workflow does.
    link_match = _match_child_links_to_parent_links(parent_links, directed_links, match_tolerance=match_tolerance)
    lineage = _resolve_link_lineage(
        parent_graph=graph, directed_child_links=directed_links, node_match=node_match,
        link_match=link_match, match_tolerance=match_tolerance,
    )
    directed_links = directed_links.drop(
        columns=[column for column in lineage.columns if column in directed_links.columns and column != 'id_link'],
        errors='ignore',
    ).merge(lineage, on='id_link', how='left', validate='1:1')
    # Match the same SWORD WSE fields as normal anchored variants. Anchor
    # nodes are represented in ``node_match`` and therefore inherit reviewed
    # parent context before any spatial fallback is considered.
    directed_nodes, node_sword_match, sword_match_summary = match_variant_nodes_to_sword(
        directed_nodes=directed_nodes,
        parent_nodes=parent_nodes,
        node_match=node_match,
        sword_node_source_path=sword_node_source_path,
        sword_wse_field=sword_wse_field,
        sword_match_tolerance=sword_match_tolerance,
        example_id=example_id,
        sword_example_station_source_path=sword_example_station_source_path,
        sword_station_match_source_path=sword_station_match_source_path,
        sword_reach_buffer_steps=sword_reach_buffer_steps,
    )
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    links_path=root/'directed_links.gpkg'; nodes_path=root/'directed_nodes.gpkg'
    directed_links.to_file(links_path,driver='GPKG'); directed_nodes.to_file(nodes_path,driver='GPKG')
    node_path=root/'node_match.csv'; link_path=root/'link_match.csv'; lineage_path=root/'link_lineage.csv'; sword_path=root/'node_sword_match.csv'; summary_path=root/'anchor_summary.json'
    node_match.to_csv(node_path,index=False); link_match.to_csv(link_path,index=False); lineage.to_csv(lineage_path,index=False)
    node_sword_match.to_csv(sword_path, index=False)
    summary_path.write_text(json.dumps({
        'direction': dict(summary),
        'anchor': anchor,
        'anchor_unit_id': anchors[0],
        'sword_matching': sword_match_summary,
        'node_sword_match_path': str(sword_path),
    }, default=str, indent=2))
    return IslandDirectedProducts(links_path,nodes_path,node_path,link_path,lineage_path,summary_path)


def compute_island_two_bank_widths(
    topology: IslandTopologyRun,
    directed: IslandDirectedProducts,
    *,
    boundary_component_labels_path: str | Path,
    wet_reference_mask_path: str | Path,
    output_dir: str | Path,
) -> IslandWidthProducts:
    """Create island widths with the shared anchored final-width policy.

    Island topology is custom, but its final width treatment must not be: the
    same segmented-stub substitution used by regular anchored full-raster and
    local-patch states is applied before routing-width review.
    """
    from network_variants.width_two_bank import build_state_bank_labels, compute_two_bank_width_families
    from network_variants.variant_generation import _apply_segmented_stub_width_inheritance
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    with rasterio.open(topology.mask_products.origin_mask_path) as src:
        origin = src.read(1).astype(np.uint16); profile = src.profile.copy(); shape = src.shape; transform = src.transform
    with rasterio.open(boundary_component_labels_path) as src:
        base_labels = src.read(1).astype(np.int32)
        if src.shape != shape or src.transform != transform:
            raise ValueError("Boundary component labels must align with the island masks.")
    bank_labels, removed = build_state_bank_labels(base_labels, origin)
    bank_path = root / 'bank_labels.tif'
    profile.update(dtype='int32', count=1, compress='lzw')
    with rasterio.open(bank_path, 'w', **profile) as dst: dst.write(bank_labels.astype(np.int32), 1)
    links = gpd.read_file(directed.directed_links_path)
    enriched, samples = compute_two_bank_width_families(
        links, collapsed_mask_path=topology.mask_products.collapsed_mask_path,
        wet_reference_mask_path=wet_reference_mask_path,
        origin_mask_path=topology.mask_products.origin_mask_path,
        rivgraph_distance_path=topology.rivgraph_paths.get('distance'), bank_labels_path=bank_path,
    )
    metadata_links, enriched, segmented_stub_inheritance = (
        _apply_segmented_stub_width_inheritance(
            directed_links=links,
            enriched_links=enriched,
            link_width_samples=samples,
        )
    )
    for column in metadata_links.columns:
        if column.startswith('family_'):
            enriched[column] = metadata_links[column].to_numpy()
    enriched_path=root/'links_with_width_families.gpkg'; samples_path=root/'link_width_samples.csv'
    enriched.to_file(enriched_path,driver='GPKG'); samples.to_csv(samples_path,index=False)
    pd.DataFrame(enriched.drop(columns=enriched.geometry.name)).to_csv(
        root / 'link_width_families.csv', index=False
    )
    return IslandWidthProducts(
        bank_path,
        enriched_path,
        samples_path,
        tuple(map(int,removed)),
        segmented_stub_inheritance,
    )


def apply_island_routing_widths(
    selection: IslandVariantSelection,
    directed: IslandDirectedProducts,
    widths: IslandWidthProducts,
    *,
    output_dir: str | Path,
    example_id: str,
) -> IslandRoutingWidthProducts:
    """Apply the standard routing-width model to island two-bank products."""
    from network_variants.routing_width_application import apply_routing_width_model, write_routing_width_audit
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    links = gpd.read_file(widths.enriched_links_path)
    nodes = gpd.read_file(directed.directed_nodes_path)
    samples = read_width_samples(widths.width_samples_path)
    enriched, audit, manifest = apply_routing_width_model(
        links, samples, nodes, example_id=str(example_id), variant_id=selection.island_id,
        apply_known_base_exclusions=False,
    )
    paths = write_routing_width_audit(audit, manifest, output_dir=root)
    enriched_path = root / 'links_with_routing_widths.gpkg'
    enriched.to_file(enriched_path, driver='GPKG')
    return IslandRoutingWidthProducts(enriched_path, Path(paths['link_audit']), Path(paths['manifest']))


def refresh_island_full_raster_width_products(
    run_dir: str | Path,
    *,
    example_id: str,
    refresh_dirname: str = "widths_shared_pipeline_v2",
) -> dict[str, object]:
    """Create a non-destructive shared-pipeline width refresh for one full island run.

    The frozen mask and directed topology are deliberately reused.  Only the
    raw two-bank family product is passed through the shared segmented-stub and
    routing stages.  The existing ``variant/widths`` directory is never
    modified; callers can promote the candidate only after paired validation.
    """
    from network_variants.variant_generation import _apply_segmented_stub_width_inheritance

    root = Path(run_dir).resolve()
    source_manifest = json.loads((root / "island_run_manifest.json").read_text())
    if source_manifest.get("status") != "complete":
        raise ValueError(f"Island full-raster run is not complete: {root}")
    island_id = str(source_manifest["island_id"])
    source_width_dir = root / "variant" / "widths"
    source_links = source_width_dir / "links_with_width_families.gpkg"
    source_samples = source_width_dir / "link_width_samples.csv"
    source_bank_labels = source_width_dir / "bank_labels.tif"
    required = (source_links, source_samples, source_bank_labels)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Island width refresh is missing source products: {missing}")

    refresh_dir = root / "variant" / str(refresh_dirname)
    refresh_manifest_path = refresh_dir / "island_width_refresh_manifest.json"
    if refresh_manifest_path.exists():
        existing = json.loads(refresh_manifest_path.read_text())
        if (
            existing.get("source_run_manifest") == str((root / "island_run_manifest.json").resolve())
            and existing.get("shared_pipeline_version") == "island-shared-width-pipeline-v2"
            and all(Path(value).exists() for value in existing.get("final_paths", {}).values())
        ):
            return existing
        raise FileExistsError(
            f"Refresh directory already contains a different candidate: {refresh_dir}"
        )
    refresh_dir.mkdir(parents=True, exist_ok=False)

    links = gpd.read_file(source_links)
    samples = read_width_samples(source_samples)
    metadata_links, enriched, segmented_stub = _apply_segmented_stub_width_inheritance(
        directed_links=links,
        enriched_links=links,
        link_width_samples=samples,
    )
    for column in metadata_links.columns:
        if column.startswith("family_"):
            enriched[column] = metadata_links[column].to_numpy()
    refreshed_widths_path = refresh_dir / "links_with_width_families.gpkg"
    refreshed_samples_path = refresh_dir / "link_width_samples.csv"
    refreshed_bank_labels_path = refresh_dir / "bank_labels.tif"
    enriched.to_file(refreshed_widths_path, driver="GPKG")
    pd.DataFrame(enriched.drop(columns=enriched.geometry.name)).to_csv(
        refresh_dir / "link_width_families.csv", index=False
    )
    samples.to_csv(refreshed_samples_path, index=False)
    shutil.copy2(source_bank_labels, refreshed_bank_labels_path)

    final_paths = source_manifest["final_paths"]
    directed = IslandDirectedProducts(
        directed_links_path=Path(final_paths["directed_links"]),
        directed_nodes_path=Path(final_paths["directed_nodes"]),
        node_match_path=root / "variant" / "directed" / "node_match.csv",
        link_match_path=root / "variant" / "directed" / "link_match.csv",
        lineage_path=root / "variant" / "directed" / "link_lineage.csv",
        anchor_summary_path=root / "variant" / "directed" / "anchor_summary.json",
    )
    selection = IslandVariantSelection(
        island_id=island_id,
        geometry=None,
        anchor_unit_id=int(source_manifest["anchor_unit_id"]),
        anchor_bifurcation=int(source_manifest["anchor_bifurcation"]),
        anchor_confluence=int(source_manifest["anchor_confluence"]),
        source_unit_ids=tuple(int(value) for value in source_manifest["source_unit_ids"]),
    )
    routing = apply_island_routing_widths(
        selection,
        directed,
        IslandWidthProducts(
            refreshed_bank_labels_path,
            refreshed_widths_path,
            refreshed_samples_path,
            tuple(),
            segmented_stub,
        ),
        output_dir=refresh_dir,
        example_id=str(example_id),
    )
    refresh_manifest = {
        "schema_version": "island-full-width-refresh-v2",
        "shared_pipeline_version": "island-shared-width-pipeline-v2",
        "source_run_manifest": str((root / "island_run_manifest.json").resolve()),
        "island_id": island_id,
        "segmented_stub_inheritance": segmented_stub,
        "final_paths": {
            "width_family_links": str(refreshed_widths_path.resolve()),
            "width_samples": str(refreshed_samples_path.resolve()),
            "bank_labels": str(refreshed_bank_labels_path.resolve()),
            "routing_width_links": str(routing.enriched_links_path.resolve()),
            "routing_width_audit": str(routing.audit_path.resolve()),
        },
    }
    refresh_manifest_path.write_text(json.dumps(refresh_manifest, indent=2) + "\n")
    return refresh_manifest


def promote_island_full_raster_width_refresh(
    run_dir: str | Path,
    *,
    validation_report_path: str | Path,
    refresh_dirname: str = "widths_shared_pipeline_v2",
) -> dict[str, object]:
    """Promote a validated, versioned full-raster island width candidate.

    This changes only the run manifest's width-product pointers.  Frozen masks,
    RivGraph products, directed topology, and the original ``variant/widths``
    outputs remain intact.  Promotion is deliberately refused unless the
    paired report identifies this exact candidate and passes its final-width
    checks.
    """
    root = Path(run_dir).resolve()
    run_manifest_path = root / "island_run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    refresh_manifest_path = root / "variant" / str(refresh_dirname) / "island_width_refresh_manifest.json"
    refresh_manifest = json.loads(refresh_manifest_path.read_text(encoding="utf-8"))
    if refresh_manifest.get("source_run_manifest") != str(run_manifest_path):
        raise ValueError(f"Width-refresh candidate does not belong to {root}")

    report_path = Path(validation_report_path).resolve()
    report = pd.read_csv(report_path)
    island_id = str(run_manifest["island_id"])
    selected = report.loc[report["island_id"].astype(str) == island_id]
    if len(selected) != 1:
        raise ValueError(f"Expected one paired-validation row for {island_id}: {report_path}")
    validation = selected.iloc[0]
    expected_source = str(refresh_manifest_path)
    if str(validation.get("full_width_product_source", "")) != expected_source:
        raise ValueError(
            f"Paired validation did not use this refresh candidate for {island_id}: {report_path}"
        )
    if not bool(validation.get("passed", False)) or not bool(
        validation.get("widths_within_relative_tolerance", False)
    ):
        raise ValueError(f"Paired validation did not pass for {island_id}: {report_path}")

    final_paths = dict(run_manifest["final_paths"])
    refreshed_paths = dict(refresh_manifest["final_paths"])
    current_refresh = run_manifest.get("width_refresh_provenance", {})
    if current_refresh.get("promoted_refresh_manifest") == expected_source:
        if final_paths.get("routing_width_links") != refreshed_paths["routing_width_links"]:
            raise ValueError(f"Manifest has inconsistent prior refresh promotion: {run_manifest_path}")
        return run_manifest

    previous_width_paths = {
        key: final_paths[key]
        for key in ("routing_width_links", "routing_width_audit")
        if key in final_paths
    }
    final_paths.update({
        "width_family_links": refreshed_paths["width_family_links"],
        "width_samples": refreshed_paths["width_samples"],
        "bank_labels": refreshed_paths["bank_labels"],
        "routing_width_links": refreshed_paths["routing_width_links"],
        "routing_width_audit": refreshed_paths["routing_width_audit"],
    })
    run_manifest["final_paths"] = final_paths
    run_manifest["final_width_pipeline"] = {
        "shared_pipeline_version": refresh_manifest["shared_pipeline_version"],
        "segmented_stub_inheritance": refresh_manifest["segmented_stub_inheritance"],
        "routing_width_application": "routing-width-application-v2",
    }
    run_manifest["width_refresh_provenance"] = {
        "promoted_refresh_manifest": expected_source,
        "paired_validation_report": str(report_path),
        "previous_final_width_paths": previous_width_paths,
    }
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    return run_manifest


def _island_rerun_is_complete(run_dir: Path, *, geometry_sha256: str) -> bool:
    manifest_path = run_dir / "island_run_manifest.json"
    if not manifest_path.exists():
        return False
    payload = json.loads(manifest_path.read_text())
    if payload.get("status") != "complete" or payload.get("geometry_sha256") != geometry_sha256:
        return False
    final_paths = payload.get("final_paths", {})
    required = ("directed_links", "directed_nodes", "routing_width_links")
    return all(key in final_paths and Path(final_paths[key]).exists() for key in required)


def run_island_full_raster_observations(
    observation_manifest_path: str | Path,
    *,
    base_mask_path: str | Path,
    parent_links_path: str | Path,
    parent_nodes_path: str | Path,
    boundary_component_labels_path: str | Path,
    wet_reference_mask_path: str | Path,
    exit_sides: str,
    example_id: str,
    match_tolerance: float | None = None,
    single_thread: bool = False,
    island_ids: list[str] | None = None,
    report_dir: str | Path | None = None,
    sword_node_source_path: str | Path | None = None,
    sword_wse_field: str | None = None,
    sword_match_tolerance: float | None = None,
    sword_example_station_source_path: str | Path | None = None,
    sword_station_match_source_path: str | Path | None = None,
    sword_reach_buffer_steps: int = 2,
) -> pd.DataFrame:
    """Execute only the non-alias observations in a frozen island manifest.

    Alias observations are validated and reported as references to the normal
    independent unit state; no files are copied. Each rerun produces the
    normal full-raster island mask, RivGraph, directed network, two-bank width
    products, and routing-width product under its manifest-assigned directory.
    A completed run is reused only when its geometry hash and final products
    still match the manifest.
    """
    manifest_path = Path(observation_manifest_path).resolve()
    observations = pd.read_csv(manifest_path, low_memory=False)
    required_columns = {
        "island_id", "execution_action", "source_state_dir", "island_setup_dir",
        "geometry_sha256", "canonical_state_dir", "island_run_dir",
    }
    missing = sorted(required_columns - set(observations.columns))
    if missing:
        raise ValueError(f"Island observation manifest is missing columns: {missing}")
    if island_ids is not None:
        requested = {str(island_id) for island_id in island_ids}
        unknown = requested - set(observations["island_id"].astype(str))
        if unknown:
            raise ValueError(f"Requested island IDs are absent from the manifest: {sorted(unknown)}")
        observations = observations.loc[observations["island_id"].astype(str).isin(requested)].copy()
    if match_tolerance is None:
        with rasterio.open(base_mask_path) as source:
            match_tolerance = abs(float(source.transform.a)) * 1.25

    records: list[dict[str, object]] = []
    report_root = Path(report_dir).resolve() if report_dir is not None else manifest_path.parent
    report_root.mkdir(parents=True, exist_ok=True)
    for row in observations.itertuples(index=False):
        island_id = str(row.island_id)
        action = str(row.execution_action)
        if action == "alias_existing_unit_state":
            reference = Path(str(row.canonical_state_dir))
            if not reference.exists() or not (reference / "variant").exists():
                raise FileNotFoundError(
                    f"Alias {island_id} does not resolve to a canonical independent state: {reference}"
                )
            records.append({
                "island_id": island_id,
                "execution_action": action,
                "outcome": "referenced_existing_state",
                "state_reference_dir": str(reference),
                "run_manifest_path": "",
            })
            continue
        if action != "run_island_state":
            raise ValueError(f"Unsupported island execution action for {island_id}: {action!r}")

        run_dir = Path(str(row.island_run_dir))
        geometry_sha256 = str(row.geometry_sha256)
        run_manifest_path = run_dir / "island_run_manifest.json"
        if _island_rerun_is_complete(run_dir, geometry_sha256=geometry_sha256):
            records.append({
                "island_id": island_id,
                "execution_action": action,
                "outcome": "reused_completed_run",
                "state_reference_dir": str(run_dir),
                "run_manifest_path": str(run_manifest_path),
            })
            continue
        if run_dir.exists() and any(run_dir.iterdir()):
            local_fallback = run_dir / "local_patch" / "fallback_to_full_raster.json"
            run_entries = _non_metadata_directory_entries(run_dir)
            allowed_direct_local_fallback = (
                local_fallback.exists()
                and run_entries == {"local_patch"}
            )
            # A direction-stage failure can leave raw mask/RivGraph products
            # under variant/ before a canonical island manifest exists. They
            # are incomplete derived outputs, never a usable state. Retain a
            # small audit beside the local fallback, then regenerate them.
            retryable_interrupted_full_run = (
                local_fallback.exists()
                and not run_manifest_path.exists()
                and run_entries == {"local_patch", "variant"}
            )
            if retryable_interrupted_full_run:
                retry_audit = run_dir / "local_patch" / "full_raster_retry_cleanup.json"
                retry_audit.write_text(
                    json.dumps(
                        {
                            "schema_version": "island-full-raster-retry-cleanup-v1",
                            "island_id": island_id,
                            "reason": "incomplete_full_raster_direction_stage",
                            "removed_path": str((run_dir / "variant").resolve()),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                shutil.rmtree(run_dir / "variant")
                allowed_direct_local_fallback = True
            if not allowed_direct_local_fallback:
                raise FileExistsError(
                    f"Island rerun directory is non-empty but not a matching completed run: {run_dir}. "
                    "Inspect or remove it explicitly before retrying."
                )
        run_dir.mkdir(parents=True, exist_ok=True)
        selection = load_island_variant_selection(row.island_setup_dir, island_id)
        topology = run_island_topology(
            selection,
            base_mask_path=base_mask_path,
            output_dir=run_dir / "variant",
            exit_sides=exit_sides,
            single_thread=single_thread,
        )
        directed = direct_island_topology(
            topology,
            parent_links_path=parent_links_path,
            parent_nodes_path=parent_nodes_path,
            output_dir=run_dir / "variant" / "directed",
            match_tolerance=float(match_tolerance),
            sword_node_source_path=sword_node_source_path,
            sword_wse_field=sword_wse_field,
            sword_match_tolerance=sword_match_tolerance,
            example_id=str(example_id),
            sword_example_station_source_path=sword_example_station_source_path,
            sword_station_match_source_path=sword_station_match_source_path,
            sword_reach_buffer_steps=sword_reach_buffer_steps,
        )
        widths = compute_island_two_bank_widths(
            topology,
            directed,
            boundary_component_labels_path=boundary_component_labels_path,
            wet_reference_mask_path=wet_reference_mask_path,
            output_dir=run_dir / "variant" / "widths",
        )
        routing = apply_island_routing_widths(
            selection,
            directed,
            widths,
            output_dir=run_dir / "variant" / "widths",
            example_id=str(example_id),
        )
        run_manifest = {
            "schema_version": "island-full-raster-run-v1",
            "status": "complete",
            "island_id": island_id,
            "geometry_sha256": geometry_sha256,
            "source_unit_ids": list(selection.source_unit_ids),
            "anchor_unit_id": selection.anchor_unit_id,
            "anchor_bifurcation": selection.anchor_bifurcation,
            "anchor_confluence": selection.anchor_confluence,
            "execution_backend": "full_raster",
            "final_width_pipeline": {
                "segmented_stub_inheritance": widths.segmented_stub_inheritance,
                "routing_width_application": "routing-width-application-v2",
            },
            "final_paths": {
                "collapsed_mask": str(topology.mask_products.collapsed_mask_path),
                "origin_mask": str(topology.mask_products.origin_mask_path),
                "directed_links": str(directed.directed_links_path),
                "directed_nodes": str(directed.directed_nodes_path),
                "routing_width_links": str(routing.enriched_links_path),
                "routing_width_audit": str(routing.audit_path),
            },
        }
        run_manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
        records.append({
            "island_id": island_id,
            "execution_action": action,
            "outcome": "completed_full_raster_run",
            "state_reference_dir": str(run_dir),
            "run_manifest_path": str(run_manifest_path),
        })

    report = pd.DataFrame(records)
    report.to_csv(report_root / "island_full_raster_execution.csv", index=False)
    (report_root / "island_full_raster_execution_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "island-full-raster-execution-v1",
                "observation_manifest": str(manifest_path),
                "n_observations": int(len(report)),
                "n_alias_references": int(report["outcome"].eq("referenced_existing_state").sum()),
                "n_completed_runs": int(report["outcome"].eq("completed_full_raster_run").sum()),
                "n_reused_runs": int(report["outcome"].eq("reused_completed_run").sum()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return report


def _island_local_patch_is_complete(run_dir: Path, *, geometry_sha256: str) -> bool:
    manifest_path = run_dir / "island_local_patch_run_manifest.json"
    if not manifest_path.exists():
        return False
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or payload.get("geometry_sha256") != geometry_sha256:
        return False
    final_paths = payload.get("final_paths", {})
    required = ("collapsed_mask", "directed_links", "directed_nodes", "routing_width_links")
    return all(key in final_paths and Path(final_paths[key]).exists() for key in required)


def _non_metadata_directory_entries(path: Path) -> set[str]:
    """Return substantive entries, ignoring macOS sidecar metadata.

    Island experiments are commonly written to an external macOS volume.  A
    moved or interrupted state can leave ``._*`` AppleDouble files behind;
    those files are not computational products and must not make an otherwise
    empty/retryable run directory look like a completed state.
    """
    return {
        item.name
        for item in path.iterdir()
        if not item.name.startswith("._") and item.name != ".DS_Store"
    }


def _full_raster_island_cycle_reduction(
    *,
    full_run_dir: Path,
    geometry_sha256: str,
    parent_links_path: str | Path,
) -> int:
    """Use the paired full-raster result as the island-specific topology oracle."""
    if not _island_rerun_is_complete(full_run_dir, geometry_sha256=geometry_sha256):
        raise FileNotFoundError(
            "A completed matching full-raster island run is required before local "
            f"paired validation: {full_run_dir}"
        )
    from network_variants.local_patch_merge import _undirected_cycle_rank

    payload = json.loads((full_run_dir / "island_run_manifest.json").read_text(encoding="utf-8"))
    full_links_path = Path(str(payload["final_paths"]["directed_links"]))
    parent_rank = _undirected_cycle_rank(gpd.read_file(parent_links_path))
    full_rank = _undirected_cycle_rank(gpd.read_file(full_links_path))
    reduction = int(parent_rank - full_rank)
    if reduction < 0:
        raise ValueError(
            "The paired full-raster island result increases cycle rank, so it cannot "
            f"be used as a local-collapse topology expectation: {reduction}."
        )
    return reduction


def island_expected_cycle_reduction(selection: IslandVariantSelection) -> int:
    """Return the topological expectation encoded by one frozen island edit.

    A reviewed island is exactly one enclosed, 8-connected dry-land component.
    Filling it removes one water-mask hole, hence one independent network
    cycle.  This is a property of the frozen island geometry—not of the
    enclosing anchor unit—and lets the local backend run without a full-raster
    topology oracle.
    """
    geometry = selection.geometry
    if geometry is None or geometry.is_empty:
        raise ValueError(f"Island {selection.island_id} has no valid frozen geometry.")
    return 1


def _resolve_parent_width_links_path(parent_links_path: str | Path) -> Path:
    """Resolve the enriched parent link product required for local inheritance."""
    path = Path(parent_links_path).resolve()
    if path.parent.name == "widths":
        return path
    if path.parent.name != "directed":
        raise ValueError(
            "Island local-patch execution requires a parent directed or widths link "
            f"product, received: {path}"
        )
    widths_dir = path.parent.parent / "widths"
    preferred = widths_dir / "links_with_width_families.gpkg"
    if preferred.exists():
        return preferred
    candidates = sorted(
        item for item in widths_dir.glob("*.gpkg") if not item.name.startswith("._")
    )
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        "The local-patch adapter requires an enriched parent width link product for "
        f"unchanged-link inheritance; expected {preferred}."
    )


def run_island_panelized_observations(
    observation_manifest_path: str | Path,
    *,
    base_experiment_dir: str | Path,
    panelized_run_root: str | Path,
    base_mask_path: str | Path,
    parent_links_path: str | Path,
    parent_nodes_path: str | Path,
    boundary_component_labels_path: str | Path,
    exit_sides: str,
    example_id: str,
    island_ids: list[str],
    report_dir: str | Path | None = None,
    sword_node_source_path: str | Path | None = None,
    sword_wse_field: str | None = None,
    sword_match_tolerance: float | None = None,
    sword_example_station_source_path: str | Path | None = None,
    sword_station_match_source_path: str | Path | None = None,
    sword_reach_buffer_steps: int = 2,
) -> pd.DataFrame:
    """Execute local-patch fallbacks with the parent's panelized child backend.

    The frozen island polygon is passed as ``collapse_geometry``.  It therefore
    changes exactly the reviewed island, while panel boundaries and matching
    remain those already validated for the independent experiment.
    """
    from network_variants.collapse_experiment import _make_panelized_child_runner

    manifest_path = Path(observation_manifest_path).resolve()
    observations = pd.read_csv(manifest_path, low_memory=False)
    requested = {str(value) for value in island_ids}
    rows = observations.loc[
        observations["island_id"].astype(str).isin(requested)
        & observations["execution_action"].eq("run_island_state")
    ].copy()
    unknown = requested - set(rows["island_id"].astype(str))
    if unknown:
        raise ValueError(f"Panelized fallback islands are absent or not runnable: {sorted(unknown)}")

    run_root = Path(panelized_run_root).expanduser().resolve()
    required = {
        "panel plan": run_root / "panels" / "panel_plan.gpkg",
        "panel breaks": run_root / "manual_edits" / f"{example_id}_panel_breaks.gpkg",
        "boundary match plan": run_root / "panel_matching" / "boundary_match_plan.json",
        "panel exits": run_root / "panel_matching" / "panel_exit_sides.json",
        "merged parent links": run_root / "merged_network" / "core_boundary" / "merged_links.gpkg",
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Panelized island fallback is missing source products: " + "; ".join(missing))
    exits = _load_json_file(required["panel exits"])
    runner = _make_panelized_child_runner(
        source_panel_root=run_root / "panels",
        panel_plan_path=required["panel plan"],
        panel_breaks_path=required["panel breaks"],
        boundary_match_plan_path=required["boundary match plan"],
        parent_merged_links_path=required["merged parent links"],
        boundary_component_labels_path=Path(boundary_component_labels_path),
        exit_sides_by_panel={key: value["exit_sides"] for key, value in exits.items()},
        run_name=str(example_id),
    )
    metrics_path = (
        Path(base_experiment_dir).expanduser().resolve()
        / "states" / "S000_base" / "hierarchy" / "path_metrics.csv"
    )
    if not metrics_path.exists():
        raise FileNotFoundError(f"Panelized island fallback requires base path metrics: {metrics_path}")
    parent_width_links = _resolve_parent_width_links_path(parent_links_path)
    report_root = Path(report_dir).resolve() if report_dir is not None else manifest_path.parent
    report_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for row in rows.itertuples(index=False):
        island_id = str(row.island_id)
        run_dir = Path(str(row.island_run_dir))
        geometry_sha256 = str(row.geometry_sha256)
        run_manifest_path = run_dir / "island_run_manifest.json"
        if _island_rerun_is_complete(run_dir, geometry_sha256=geometry_sha256):
            records.append({"island_id": island_id, "execution_action": row.execution_action,
                            "outcome": "reused_completed_panelized_run", "state_reference_dir": str(run_dir),
                            "run_manifest_path": str(run_manifest_path)})
            continue
        fallback = run_dir / "local_patch" / "fallback_to_panelized.json"
        if run_dir.exists() and any(run_dir.iterdir()):
            run_entries = _non_metadata_directory_entries(run_dir)
            direct_fallback_only = (
                fallback.exists()
                and run_entries == {"local_patch"}
            )
            retryable_incomplete_panelized_run = (
                fallback.exists()
                and not run_manifest_path.exists()
                and run_entries == {"local_patch", "variant"}
            )
            if retryable_incomplete_panelized_run:
                retry_audit = run_dir / "local_patch" / "panelized_retry_cleanup.json"
                retry_audit.write_text(
                    json.dumps(
                        {
                            "schema_version": "island-panelized-retry-cleanup-v1",
                            "island_id": island_id,
                            "reason": "incomplete_panelized_width_stage",
                            "removed_path": str((run_dir / "variant").resolve()),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                shutil.rmtree(run_dir / "variant")
                direct_fallback_only = True
            if not direct_fallback_only:
                raise FileExistsError(f"Island panelized rerun directory is non-empty: {run_dir}")
        selection = load_island_variant_selection(row.island_setup_dir, island_id)
        outputs = runner(
            cleaned_mask_path=base_mask_path,
            reviewed_links_path=parent_width_links,
            reviewed_nodes_path=parent_nodes_path,
            unit_ids=[selection.anchor_unit_id],
            variant_id=f"{island_id}_panelized",
            example_id=str(example_id),
            output_dir=run_dir / "variant",
            selection_type="island",
            collapse_geometry=selection.geometry,
            collapse_source_unit_ids=selection.source_unit_ids,
            collapse_corridor_mode="anchored",
            panelized_path_metrics=metrics_path,
            panelized_mask_publication="virtual_patch",
            # Match the normal panelized child workflow: retain unchanged
            # parent products and evaluate only the affected patch. The
            # panel-owned full recomputation is an optional diagnostic path,
            # not the canonical routing-width input for these fallbacks.
            compute_panel_width_products=False,
            sword_node_source_path=sword_node_source_path,
            sword_wse_field=sword_wse_field,
            sword_match_tolerance=sword_match_tolerance,
            sword_example_station_source_path=sword_example_station_source_path,
            sword_station_match_source_path=sword_station_match_source_path,
            sword_reach_buffer_steps=sword_reach_buffer_steps,
        )
        paths = outputs.manifest["output_paths"]
        payload = {
            "schema_version": "island-panelized-run-v1", "status": "complete",
            "island_id": island_id, "geometry_sha256": geometry_sha256,
            "source_unit_ids": list(selection.source_unit_ids),
            "anchor_unit_id": selection.anchor_unit_id,
            "anchor_bifurcation": selection.anchor_bifurcation,
            "anchor_confluence": selection.anchor_confluence,
            "execution_backend": "panelized",
            "final_width_pipeline": {"routing_width_application": "routing-width-application-v2"},
            "final_paths": {
                "collapsed_mask": str(outputs.collapsed_mask_path),
                "origin_mask": str(outputs.origin_mask_path),
                "directed_links": str(outputs.directed_links_path),
                "directed_nodes": str(outputs.directed_nodes_path),
                "routing_width_links": str(paths["links_with_width_families"]),
                "routing_width_audit": str(paths["routing_width_application"]),
            },
        }
        run_manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        records.append({"island_id": island_id, "execution_action": row.execution_action,
                        "outcome": "completed_panelized_run", "state_reference_dir": str(run_dir),
                        "run_manifest_path": str(run_manifest_path)})
    report = pd.DataFrame(records)
    report.to_csv(report_root / "island_panelized_execution.csv", index=False)
    return report


def run_island_local_patch_observations(
    observation_manifest_path: str | Path,
    *,
    base_mask_path: str | Path,
    parent_links_path: str | Path,
    parent_nodes_path: str | Path,
    boundary_component_labels_path: str | Path,
    exit_sides: str,
    example_id: str,
    single_thread: bool = False,
    island_ids: list[str] | None = None,
    report_dir: str | Path | None = None,
    local_patch_reference_centerline_path: str | Path | None = None,
    local_patch_mask_publication: str = "materialized",
    fallback_backend: str = "full_raster",
) -> pd.DataFrame:
    """Run the non-alias island observations through the bounded local backend.

    The recorded anchor unit remains the scientific selection.  Its smallest
    hydraulically closed parent domain (a complete containing compound when
    necessary) is used only to build a stable raster crop.  The frozen island
    geometry, rather than a regenerated unit footprint, is the sole raster
    edit.  It runs directly from the independent base state. A safe local
    failure records an audited request for a full-raster fallback, but does not
    require a full-raster island state beforehand.
    """
    from network_variants.local_patch_execution import (
        LocalPatchFallbackRequested,
        materialize_local_patch_network_variant,
    )

    manifest_path = Path(observation_manifest_path).resolve()
    observations = pd.read_csv(manifest_path, low_memory=False)
    required_columns = {
        "island_id", "execution_action", "island_setup_dir", "geometry_sha256",
        "canonical_state_dir", "island_run_dir",
    }
    missing = sorted(required_columns - set(observations.columns))
    if missing:
        raise ValueError(f"Island observation manifest is missing columns: {missing}")
    if island_ids is not None:
        requested = {str(value) for value in island_ids}
        unknown = requested - set(observations["island_id"].astype(str))
        if unknown:
            raise ValueError(f"Requested island IDs are absent from the manifest: {sorted(unknown)}")
        observations = observations.loc[observations["island_id"].astype(str).isin(requested)].copy()

    report_root = Path(report_dir).resolve() if report_dir is not None else manifest_path.parent
    report_root.mkdir(parents=True, exist_ok=True)
    if fallback_backend not in {"full_raster", "panelized"}:
        raise ValueError("fallback_backend must be 'full_raster' or 'panelized'.")
    parent_width_links_path = _resolve_parent_width_links_path(parent_links_path)
    records: list[dict[str, object]] = []
    for row in observations.itertuples(index=False):
        island_id = str(row.island_id)
        action = str(row.execution_action)
        if action == "alias_existing_unit_state":
            reference = Path(str(row.canonical_state_dir))
            if not reference.exists() or not (reference / "variant").exists():
                raise FileNotFoundError(
                    f"Alias {island_id} does not resolve to a canonical independent state: {reference}"
                )
            records.append({
                "island_id": island_id,
                "execution_action": action,
                "outcome": "referenced_existing_state",
                "state_reference_dir": str(reference),
                "run_manifest_path": "",
                "computational_domain_policy": "not_applicable_alias",
            })
            continue
        if action != "run_island_state":
            raise ValueError(f"Unsupported island execution action for {island_id}: {action!r}")

        run_dir = Path(str(row.island_run_dir))
        local_run_dir = run_dir / "local_patch"
        geometry_sha256 = str(row.geometry_sha256)
        local_manifest_path = local_run_dir / "island_local_patch_run_manifest.json"
        if _island_local_patch_is_complete(local_run_dir, geometry_sha256=geometry_sha256):
            records.append({
                "island_id": island_id,
                "execution_action": action,
                "outcome": "reused_completed_local_patch_run",
                "state_reference_dir": str(local_run_dir / "variant"),
                "run_manifest_path": str(local_manifest_path),
                "computational_domain_policy": "recorded_anchor_unit_smallest_closed_domain",
            })
            continue
        fallback_path = local_run_dir / f"fallback_to_{fallback_backend}.json"
        if local_run_dir.exists() and any(local_run_dir.iterdir()):
            if fallback_path.exists():
                records.append({
                    "island_id": island_id,
                    "execution_action": action,
                    "outcome": f"fallback_{fallback_backend}_required",
                    "state_reference_dir": "",
                    "run_manifest_path": str(fallback_path),
                    "computational_domain_policy": "recorded_anchor_unit_smallest_closed_domain",
                })
                continue
            entries = _non_metadata_directory_entries(local_run_dir)
            # A process can be interrupted after it materializes the derived
            # local ``variant/`` tree but before it publishes either its final
            # manifest or a fallback record.  It is not a scientific result:
            # retain an audit trail and regenerate it from the frozen island
            # geometry.  Do not broaden this to arbitrary contents, so a
            # genuinely ambiguous/manual directory still fails closed.
            if entries == {"variant"} and not local_manifest_path.exists():
                retry_audit = local_run_dir / "interrupted_local_patch_retry_cleanup.json"
                retry_audit.write_text(
                    json.dumps(
                        {
                            "schema_version": "island-local-patch-retry-cleanup-v1",
                            "island_id": island_id,
                            "geometry_sha256": geometry_sha256,
                            "reason": "interrupted_derived_local_patch_before_final_manifest",
                            "removed_path": str((local_run_dir / "variant").resolve()),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                shutil.rmtree(local_run_dir / "variant")
            else:
                raise FileExistsError(
                    "Island local-patch directory is non-empty but not a matching completed run: "
                    f"{local_run_dir}. Inspect it before retrying."
                )
        local_run_dir.mkdir(parents=True, exist_ok=True)
        selection = load_island_variant_selection(row.island_setup_dir, island_id)
        expected_cycle_reduction = island_expected_cycle_reduction(selection)
        try:
            outputs = materialize_local_patch_network_variant(
                cleaned_mask_path=base_mask_path,
                reviewed_links_path=parent_width_links_path,
                reviewed_nodes_path=parent_nodes_path,
                exit_sides=exit_sides,
                unit_ids=island_anchor_unit_ids(selection),
                variant_id=f"{island_id}_local_patch",
                example_id=str(example_id),
                output_dir=local_run_dir / "variant",
                selection_type="island",
                boundary_component_labels_path=boundary_component_labels_path,
                collapse_corridor_mode="anchored",
                single_thread=single_thread,
                local_patch_reference_centerline_path=local_patch_reference_centerline_path,
                collapse_geometry=selection.geometry,
                collapse_component_id=selection.island_id,
                collapse_source_unit_ids=selection.source_unit_ids,
                expected_cycle_reduction_override=expected_cycle_reduction,
                local_patch_mask_publication=local_patch_mask_publication,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            full_manifest = run_dir / "island_run_manifest.json"
            full_available = _island_rerun_is_complete(
                run_dir, geometry_sha256=geometry_sha256
            )
            if isinstance(error, LocalPatchFallbackRequested):
                failure = {
                    "stage": error.stage,
                    "code": error.code,
                    "reason": error.reason,
                    "audit_path": str(error.audit_path),
                }
            else:
                retry_summary = local_run_dir / "variant" / "outer_domain_retry_summary.json"
                failure = {
                    "stage": "execution",
                    "code": type(error).__name__,
                    "reason": str(error),
                    "audit_path": str(retry_summary) if retry_summary.exists() else None,
                }
            fallback_path.write_text(
                json.dumps({
                    "schema_version": "island-local-patch-fallback-v1",
                    "island_id": island_id,
                    "geometry_sha256": geometry_sha256,
                    "status": f"fallback_{fallback_backend}_required",
                    "fallback_backend": fallback_backend,
                    "local_failure": failure,
                    "existing_full_raster_run_available": bool(full_available),
                    "existing_full_raster_manifest": str(full_manifest) if full_manifest.exists() else None,
                }, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            records.append({
                "island_id": island_id,
                "execution_action": action,
                "outcome": (
                    f"fallback_to_existing_{fallback_backend}"
                    if full_available else f"fallback_{fallback_backend}_required"
                ),
                "state_reference_dir": str(run_dir) if full_available else "",
                "run_manifest_path": str(fallback_path),
                "computational_domain_policy": "recorded_anchor_unit_smallest_closed_domain",
            })
            continue

        manifest = {
            "schema_version": "island-local-patch-run-v1",
            "status": "complete",
            "island_id": island_id,
            "geometry_sha256": geometry_sha256,
            "source_unit_ids": list(selection.source_unit_ids),
            "anchor_unit_id": selection.anchor_unit_id,
            "anchor_bifurcation": selection.anchor_bifurcation,
            "anchor_confluence": selection.anchor_confluence,
            "scientific_anchor_policy": "recorded_island_anchor_unit",
            "computational_domain_policy": "recorded_anchor_unit_smallest_closed_domain",
            "expected_cycle_reduction": int(expected_cycle_reduction),
            "expected_cycle_reduction_source": "frozen_enclosed_island_geometry",
            "execution_backend": "local_patch",
            "final_paths": {
                "collapsed_mask": str(outputs.collapsed_mask_path),
                "origin_mask": str(outputs.origin_mask_path),
                "directed_links": str(outputs.directed_links_path),
                "directed_nodes": str(outputs.directed_nodes_path),
                "routing_width_links": str(outputs.manifest["output_paths"]["links_with_width_families"]),
                "variant_manifest": str(outputs.output_dir / "summary" / "variant_manifest.json"),
            },
        }
        local_manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        records.append({
            "island_id": island_id,
            "execution_action": action,
            "outcome": "completed_local_patch_run",
            "state_reference_dir": str(outputs.output_dir),
            "run_manifest_path": str(local_manifest_path),
            "computational_domain_policy": "recorded_anchor_unit_smallest_closed_domain",
        })

    report = pd.DataFrame(records)
    report.to_csv(report_root / "island_local_patch_execution.csv", index=False)
    (report_root / "island_local_patch_execution_manifest.json").write_text(
        json.dumps({
            "schema_version": "island-local-patch-execution-v1",
            "observation_manifest": str(manifest_path),
            "n_observations": int(len(report)),
            "n_alias_references": int(report["outcome"].eq("referenced_existing_state").sum()),
            "n_completed_local_runs": int(report["outcome"].eq("completed_local_patch_run").sum()),
            "n_fallbacks": int(report["outcome"].astype(str).str.startswith("fallback").sum()),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
