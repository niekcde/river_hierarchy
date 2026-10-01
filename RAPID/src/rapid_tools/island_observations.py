"""RAPID adapters for the compact island-only observation registry.

The island experiment deliberately has two kinds of observations: an alias to
an already-run independent unit state and a genuinely new island state.  This
module keeps that distinction explicit.  Alias records are provenance-only;
only genuine island states create a new RAPID preparation or routing run.
"""

from __future__ import annotations

from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Collection, Mapping

import pandas as pd

from .engine import (
    RapidResultStorageConfig,
    _publish_compact_result,
    _write_failure_bundle,
    run_prepared_state,
)
from .forcing import ForcingConfig
from .hydrograph import HydrographMetricConfig, write_hydrograph_outputs
from .analysis import compare_hydrographs
from .prep import RapidPrepConfig, prepare_state
from .registry import RapidStateContext


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _selected_unit_id(value: object) -> int | None:
    text = _text(value)
    if not text:
        return None
    try:
        return int(float(text.split(",")[0]))
    except ValueError:
        return None


def _existing_path(value: object, *, label: str) -> Path:
    path = Path(_text(value)).expanduser().resolve()
    if not _text(value) or not path.exists():
        raise FileNotFoundError(f"Island observation is missing {label}: {value!r}")
    return path


def _nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _remove_disposable_prep(prep_dir: Path) -> None:
    """Remove a verified disposable prep directory despite benign APFS races."""
    if not prep_dir.exists():
        return

    def ignore_already_missing(_function: object, _path: str, excinfo: object) -> None:
        # Python 3.12 ``onexc`` supplies the exception instance; retain tuple
        # support for environments using the older ``onerror`` convention.
        error = excinfo if isinstance(excinfo, BaseException) else excinfo[1]  # type: ignore[index]
        if isinstance(error, FileNotFoundError):
            return
        raise error

    shutil.rmtree(prep_dir, onexc=ignore_already_missing)
    if prep_dir.exists():
        raise OSError(f"Disposable RAPID prep directory was not fully removed: {prep_dir}")


def _recover_published_island_run(
    *,
    island_id: str,
    prep_dir: Path,
    run_dir: Path,
    storage: RapidResultStorageConfig,
) -> dict[str, object] | None:
    """Return a completed record when publication survived a cleanup interruption.

    Successful compact publication deliberately removes regenerable preparation.
    A restart must therefore recognise the durable result bundle rather than
    treating the missing prep directory as a routing failure.
    """
    qout_gz = run_dir / "Qout.nc.gz"
    qout_nc = run_dir / "Qout.nc"
    qout = qout_gz if _nonempty_file(qout_gz) else qout_nc
    if not _nonempty_file(qout):
        return None
    if storage.compact_qout and qout != qout_gz:
        return None

    outlet = run_dir / "outlet_hydrograph.csv"
    metrics_csv = run_dir / "hydrograph_metrics.csv"
    metrics_json = run_dir / "hydrograph_metrics.json"
    if not all(_nonempty_file(path) for path in (outlet, metrics_csv, metrics_json)):
        return None

    record: dict[str, object] = {
        "island_id": island_id,
        "status": "ran",
        "rapid_prep_dir": str(prep_dir),
        "rapid_run_dir": str(run_dir),
        "qout_nc": str(qout),
        "hydrograph_status": "computed",
        "outlet_hydrograph_csv": str(outlet),
        "hydrograph_metrics_csv": str(metrics_csv),
        "hydrograph_metrics_json": str(metrics_json),
        "recovered_existing_published_result": True,
        "rapid_prep_retained": prep_dir.exists(),
    }
    mapping = run_dir / "rapid_link_mapping.parquet"
    if storage.write_mapping_parquet:
        if not _nonempty_file(mapping):
            return None
        record["rapid_link_mapping_parquet"] = str(mapping)
    geometry = run_dir / "rapid_reach_attributes.gpkg"
    if not storage.retain_prepared_inputs:
        if not _nonempty_file(geometry):
            return None
        record["rapid_reach_attributes_gpkg"] = str(geometry)
    return record


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _config_from_mapping(config_type: type, values: Mapping[str, object]) -> object:
    allowed = {item.name for item in fields(config_type)}
    return config_type(**{key: value for key, value in values.items() if key in allowed})


def _source_unit_timestep_plan(
    experiment_dir: Path,
    timestep_plan: Mapping[str, object],
) -> dict[int, dict[str, object]]:
    """Map independent source units to their recorded coordinated RAPID clocks."""
    registry_path = experiment_dir / "state_registry.csv"
    state_plans = dict(timestep_plan.get("states") or {})
    if not registry_path.exists() or not state_plans:
        return {}

    state_registry = pd.read_csv(registry_path, low_memory=False)
    result: dict[int, dict[str, object]] = {}
    for _, row in state_registry.iterrows():
        if _text(row.get("selection_type")) != "unit":
            continue
        unit_id = _selected_unit_id(row.get("selected_unit_ids"))
        state_id = _text(row.get("state_id"))
        if unit_id is None or not state_id:
            continue
        state_plan = state_plans.get(f"{experiment_dir.name}::{state_id}")
        if not isinstance(state_plan, Mapping):
            state_plan = state_plans.get(state_id)
        if not isinstance(state_plan, Mapping):
            continue
        routing_dt = state_plan.get("routing_dt_seconds")
        forcing_dt = state_plan.get("forcing_dt_seconds", routing_dt)
        if routing_dt is None or forcing_dt is None:
            continue
        result[unit_id] = {
            "base_state_id": state_id,
            "routing_dt_seconds": int(routing_dt),
            "forcing_dt_seconds": int(forcing_dt),
            "selection_source": _text(state_plan.get("selection_source")),
        }
    return result


def _load_base_rapid_config(base_experiment_dir: str | Path) -> tuple[
    Path,
    str,
    ForcingConfig,
    RapidPrepConfig,
    dict[str, object],
    int | None,
    dict[int, dict[str, object]],
]:
    experiment_dir = Path(base_experiment_dir).expanduser().resolve()
    manifest_path = experiment_dir / "rapid_prep_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "The base RAPID preparation manifest is required to inherit the exact "
            f"forcing and K configuration: {manifest_path}"
        )
    manifest = _read_json(manifest_path)
    forcing_path = _existing_path(manifest.get("forcing_path"), label="base forcing_path")
    forcing_config = _config_from_mapping(
        ForcingConfig, dict(manifest.get("forcing_config") or {})
    )
    prep_config = _config_from_mapping(
        RapidPrepConfig, dict(manifest.get("prep_config") or {})
    )
    reference_kb_summary = dict(manifest.get("reference_kb_summary") or {})
    if str(prep_config.kb_mode) == "based_reference_section" and not reference_kb_summary:
        raise ValueError(
            "The base RAPID manifest has kb_mode='based_reference_section' but no "
            "stored reference_kb_summary."
        )
    timestep_plan = dict(manifest.get("timestep_plan") or {})
    selected = timestep_plan.get("selected_unique_routing_dt_seconds", [])
    planned_dt = int(selected[0]) if len(selected) == 1 else prep_config.preferred_routing_dt_seconds
    return (
        experiment_dir,
        str(forcing_path),
        forcing_config,
        prep_config,
        reference_kb_summary,
        int(planned_dt) if planned_dt is not None else None,
        _source_unit_timestep_plan(experiment_dir, timestep_plan),
    )


def _island_context(row: pd.Series, hydraulic_row: pd.Series, *, work_root: Path) -> RapidStateContext:
    island_id = _text(row.get("island_id"))
    state_dir = _existing_path(row.get("analysis_state_dir"), label="analysis_state_dir")
    hydraulic_dir = _existing_path(
        hydraulic_row.get("hydraulic_context_dir"),
        label="hydraulic_context_dir",
    )
    return RapidStateContext(
        state_id=island_id,
        state_dir=state_dir,
        directed_links_path=_existing_path(
            row.get("analysis_directed_links_path"), label="analysis_directed_links_path"
        ),
        directed_nodes_path=_existing_path(
            row.get("analysis_directed_nodes_path"), label="analysis_directed_nodes_path"
        ),
        link_widths_path=_existing_path(
            row.get("analysis_routing_width_links_path"),
            label="analysis_routing_width_links_path",
        ),
        hierarchy_output_dir=None,
        hydraulic_context_output_dir=hydraulic_dir,
        rapid_prep_dir=(work_root / island_id / "prep").resolve(),
        rapid_run_dir=(state_dir / "rapid" / "run").resolve(),
        status="complete",
        depth=int(row.get("anchor_unit_depth", 0) or 0),
        state_role="island_observation",
    )


def _conceptual_island_prep_records(
    analysis: pd.DataFrame,
    hydraulic: pd.DataFrame,
    *,
    base_dir: Path,
) -> list[dict[str, object]]:
    """Create provenance-only records for aliases and strict exclusions.

    Targeted preparation is also used by the storage-bounded streaming runner.
    It must therefore be able to bootstrap a complete conceptual registry even
    when a preceding all-at-once attempt left a zero-byte registry behind.
    """
    records: list[dict[str, object]] = []
    for _, row in analysis.iterrows():
        island_id = _text(row.get("island_id"))
        backend = _text(row.get("analysis_backend"))
        strict_status = _text(row.get("strict_analysis_status"))
        if strict_status == "exclude_pruning_or_mask_topology_exception":
            records.append({
                "island_id": island_id,
                "execution_backend": backend,
                "status": "excluded_topology_review",
                "analysis_state_dir": _text(row.get("analysis_state_dir")),
                "rapid_prep_dir": "",
                "rapid_run_dir": "",
                "source_state_dir": _text(row.get("canonical_state_dir")),
                "source_unit_id": row.get("source_unit_id"),
                "anchor_bifurcation": row.get("anchor_bifurcation"),
                "anchor_confluence": row.get("anchor_confluence"),
                "hydraulic_context_output_dir": "",
                "config_source_experiment": str(base_dir),
            })
            continue
        if backend != "independent_unit_alias":
            continue
        if island_id not in hydraulic.index:
            raise ValueError(f"Island hydraulic-context registry has no row for {island_id}.")
        hydraulic_row = hydraulic.loc[island_id]
        if isinstance(hydraulic_row, pd.DataFrame):
            raise ValueError(f"Island hydraulic-context registry has duplicate rows for {island_id}.")
        records.append({
            "island_id": island_id,
            "execution_backend": backend,
            "status": "referenced_existing_independent_state",
            "analysis_state_dir": _text(row.get("analysis_state_dir")),
            "rapid_prep_dir": "",
            "rapid_run_dir": "",
            "source_state_dir": _text(row.get("canonical_state_dir")),
            "source_unit_id": row.get("source_unit_id"),
            "anchor_bifurcation": row.get("anchor_bifurcation"),
            "anchor_confluence": row.get("anchor_confluence"),
            "hydraulic_context_output_dir": _text(hydraulic_row.get("hydraulic_context_dir")),
            "config_source_experiment": str(base_dir),
        })
    return records


def prepare_island_observations(
    analysis_manifest_path: str | Path,
    *,
    hydraulic_context_registry_path: str | Path,
    base_experiment_dir: str | Path,
    work_root: str | Path | None = None,
    island_ids: Collection[str] | None = None,
    island_forcing_routing_dt_seconds: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """Prepare only genuine island states and register alias references.

    RAPID parameters, forcing, reference kb and common timestep are inherited
    from the recorded base experiment preparation, rather than reconstructed
    from a notebook or command history.
    """
    analysis_path = Path(analysis_manifest_path).expanduser().resolve()
    hydraulic_path = Path(hydraulic_context_registry_path).expanduser().resolve()
    analysis = pd.read_csv(analysis_path, low_memory=False)
    full_analysis = analysis.copy()
    hydraulic = pd.read_csv(hydraulic_path, low_memory=False).set_index("island_id")
    required = {"island_id", "analysis_backend", "analysis_state_dir"}
    missing = sorted(required - set(analysis.columns))
    if missing:
        raise ValueError(f"Island analysis manifest is missing columns: {missing}")
    if "hydraulic_context_dir" not in hydraulic.columns:
        raise ValueError("Island hydraulic-context registry is missing hydraulic_context_dir.")

    # A targeted retry is intentionally limited to genuine island states. The
    # aliases and topology-review exclusions are conceptual observations, not
    # new RAPID preparations; they are reconstituted below from the full
    # manifest so this also works after a partial, zero-byte registry write.
    requested_island_ids = {_text(value) for value in (island_ids or []) if _text(value)}
    if requested_island_ids:
        available_ids = set(analysis["island_id"].map(_text))
        unknown = sorted(requested_island_ids - available_ids)
        if unknown:
            raise ValueError(f"Requested island IDs are absent from the analysis manifest: {unknown}")
        requested_rows = analysis.loc[analysis["island_id"].map(_text).isin(requested_island_ids)]
        non_genuine = requested_rows.loc[
            requested_rows["analysis_backend"].map(_text).eq("independent_unit_alias")
            | requested_rows.get("strict_analysis_status", pd.Series("", index=requested_rows.index)).map(_text).eq(
                "exclude_pruning_or_mask_topology_exception"
            ),
            "island_id",
        ].map(_text).tolist()
        if non_genuine:
            raise ValueError(
                "Targeted RAPID preparation is only valid for genuine island states; "
                f"these are aliases or exclusions: {sorted(non_genuine)}"
            )
        analysis = requested_rows.copy()

    explicit_timestep_overrides: dict[str, int] = {}
    for raw_island_id, raw_dt in (island_forcing_routing_dt_seconds or {}).items():
        island_id = _text(raw_island_id)
        try:
            dt_seconds = int(raw_dt)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Island forcing/routing timestep override for {island_id!r} is not an integer: {raw_dt!r}"
            ) from exc
        if not island_id or dt_seconds <= 0:
            raise ValueError(
                "Island forcing/routing timestep overrides require a non-empty island ID "
                "and a positive integer number of seconds."
            )
        explicit_timestep_overrides[island_id] = dt_seconds
    available_island_ids = set(analysis["island_id"].map(_text))
    unknown_overrides = sorted(set(explicit_timestep_overrides) - available_island_ids)
    if unknown_overrides:
        raise ValueError(
            "Island forcing/routing timestep overrides are not in the selected preparation set: "
            f"{unknown_overrides}"
        )

    (
        base_dir,
        forcing_path,
        forcing_config,
        prep_config,
        reference_kb_summary,
        fallback_planned_dt,
        source_unit_timestep_plan,
    ) = _load_base_rapid_config(base_experiment_dir)
    resolved_work_root = (
        Path(work_root).expanduser().resolve()
        if work_root is not None
        else (analysis_path.parent / "rapid_work").resolve()
    )
    records: list[dict[str, object]] = []
    for _, row in analysis.iterrows():
        island_id = _text(row["island_id"])
        backend = _text(row.get("analysis_backend"))
        strict_status = _text(row.get("strict_analysis_status"))
        if strict_status == "exclude_pruning_or_mask_topology_exception":
            records.append({
                "island_id": island_id,
                "execution_backend": backend,
                "status": "excluded_topology_review",
                "analysis_state_dir": _text(row.get("analysis_state_dir")),
                "rapid_prep_dir": "",
                "rapid_run_dir": "",
                "source_state_dir": _text(row.get("canonical_state_dir")),
                "source_unit_id": row.get("source_unit_id"),
                "anchor_bifurcation": row.get("anchor_bifurcation"),
                "anchor_confluence": row.get("anchor_confluence"),
                "hydraulic_context_output_dir": "",
                "config_source_experiment": str(base_dir),
            })
            continue
        if island_id not in hydraulic.index:
            raise ValueError(f"Island hydraulic-context registry has no row for {island_id}.")
        hydraulic_row = hydraulic.loc[island_id]
        if isinstance(hydraulic_row, pd.DataFrame):
            raise ValueError(f"Island hydraulic-context registry has duplicate rows for {island_id}.")
        if backend == "independent_unit_alias":
            records.append({
                "island_id": island_id,
                "execution_backend": backend,
                "status": "referenced_existing_independent_state",
                "analysis_state_dir": _text(row.get("analysis_state_dir")),
                "rapid_prep_dir": "",
                "rapid_run_dir": "",
                "source_state_dir": _text(row.get("canonical_state_dir")),
                "source_unit_id": row.get("source_unit_id"),
                "anchor_bifurcation": row.get("anchor_bifurcation"),
                "anchor_confluence": row.get("anchor_confluence"),
                "hydraulic_context_output_dir": _text(hydraulic_row.get("hydraulic_context_dir")),
                "config_source_experiment": str(base_dir),
            })
            continue
        if backend not in {
            "island_full_raster",
            "island_local_patch",
            "island_panelized",
        }:
            raise ValueError(f"Unsupported island analysis backend for {island_id}: {backend!r}")
        context = _island_context(row, hydraulic_row, work_root=resolved_work_root)
        source_unit_id = _selected_unit_id(row.get("source_unit_id"))
        inherited_timestep = source_unit_timestep_plan.get(source_unit_id)
        explicit_timestep = explicit_timestep_overrides.get(island_id)
        source_preferred_dt = explicit_timestep or (
            int(inherited_timestep["routing_dt_seconds"])
            if inherited_timestep is not None
            else fallback_planned_dt
        )
        state_forcing_config = forcing_config
        state_prep_config = prep_config
        if explicit_timestep is not None:
            # This is a documented numerical-clock exception, not a hydraulic
            # recalibration: retain the inherited hydraulic/K configuration,
            # but require the named state to use the explicitly approved clock.
            state_forcing_config = replace(
                forcing_config,
                resample_seconds=int(explicit_timestep),
                resample_minutes=None,
            )
            state_prep_config = replace(
                prep_config,
                preferred_routing_dt_seconds=int(explicit_timestep),
                forcing_dt_search_max_offset_seconds=0,
            )
        elif inherited_timestep is not None:
            state_forcing_config = replace(
                forcing_config,
                resample_seconds=int(inherited_timestep["forcing_dt_seconds"]),
                resample_minutes=None,
            )
            # The source state clock is the preferred island clock, not a
            # blind constraint: the island can change effective routing
            # lengths and hence its own Muskingum stability interval. Leaving
            # the plan unconstrained enables RAPID's existing nearby-forcing
            # fallback when that inherited pair is no longer stable.
            state_prep_config = replace(
                prep_config,
                preferred_routing_dt_seconds=int(inherited_timestep["routing_dt_seconds"]),
            )
        planned_dt = (
            int(explicit_timestep)
            if explicit_timestep is not None
            else (None if inherited_timestep is not None else fallback_planned_dt)
        )
        timestep_selection = {
            "selection_mode": (
                "explicit_island_forcing_routing_timestep_override"
                if explicit_timestep is not None
                else "inherited_base_source_unit_timestep"
                if inherited_timestep is not None
                else "inherited_base_fallback_timestep"
            ),
            "base_experiment_dir": str(base_dir),
            "preferred_routing_dt_seconds": source_preferred_dt,
            "planned_routing_dt_seconds": planned_dt,
            "forcing_dt_seconds": int(state_forcing_config.resample_seconds)
            if state_forcing_config.resample_seconds is not None else None,
            "source_unit_id": source_unit_id,
            "source_unit_timestep": inherited_timestep,
            "island_forcing_routing_dt_override_seconds": explicit_timestep,
        }
        try:
            result = prepare_state(
                context,
                forcing_path=forcing_path,
                forcing_config=state_forcing_config,
                prep_config=state_prep_config,
                reference_kb_summary=reference_kb_summary,
                planned_routing_dt_seconds=planned_dt,
                timestep_selection=timestep_selection,
            )
            result.update({
                "island_id": island_id,
                "execution_backend": backend,
                "source_state_dir": "",
                "source_unit_id": row.get("source_unit_id"),
                "anchor_bifurcation": row.get("anchor_bifurcation"),
                "anchor_confluence": row.get("anchor_confluence"),
                "config_source_experiment": str(base_dir),
                # Keep clock-exception provenance after compact storage removes
                # the regenerable per-state preparation manifest.
                "island_timestep_selection_mode": timestep_selection["selection_mode"],
                "island_forcing_routing_dt_override_seconds": explicit_timestep,
                "source_unit_timestep": json.dumps(inherited_timestep, sort_keys=True)
                if inherited_timestep is not None else "",
            })
        except Exception as exc:  # preserve an actionable state-level record
            result = {
                "island_id": island_id,
                "state_id": island_id,
                "execution_backend": backend,
                "status": "failed",
                "error": str(exc),
                "analysis_state_dir": str(context.state_dir),
                "rapid_prep_dir": str(context.rapid_prep_dir),
                "rapid_run_dir": str(context.rapid_run_dir),
                "source_state_dir": "",
                "source_unit_id": row.get("source_unit_id"),
                "anchor_bifurcation": row.get("anchor_bifurcation"),
                "anchor_confluence": row.get("anchor_confluence"),
                "hydraulic_context_output_dir": str(context.hydraulic_context_output_dir),
                "config_source_experiment": str(base_dir),
            }
        records.append(result)

    output_root = analysis_path.parent
    registry = pd.DataFrame(records)
    registry_path = output_root / "island_rapid_prep_registry.csv"
    if requested_island_ids:
        # An interrupted all-at-once preparation can leave an empty registry;
        # treat that as no usable prior state rather than blocking recovery.
        previous = pd.DataFrame()
        if registry_path.exists() and registry_path.stat().st_size > 0:
            try:
                previous = pd.read_csv(registry_path, low_memory=False)
            except pd.errors.EmptyDataError:
                previous = pd.DataFrame()
        if not previous.empty and "island_id" not in previous.columns:
            raise ValueError(f"Existing island RAPID prep registry has no island_id column: {registry_path}")
        conceptual_records = _conceptual_island_prep_records(
            full_analysis, hydraulic, base_dir=base_dir
        )
        conceptual_ids = {_text(record["island_id"]) for record in conceptual_records}
        if not previous.empty:
            previous = previous.loc[
                ~previous["island_id"].map(_text).isin(requested_island_ids | conceptual_ids)
            ]
        registry = pd.concat(
            [previous, registry, pd.DataFrame(conceptual_records)],
            ignore_index=True,
            sort=False,
        )
        # Restore conceptual-manifest ordering so a targeted retry does not
        # produce a misleading registry order.
        full_order = full_analysis["island_id"].map(_text).tolist()
        order = {island_id: index for index, island_id in enumerate(full_order)}
        registry = (
            registry.assign(_manifest_order=registry["island_id"].map(_text).map(order))
            .sort_values("_manifest_order", kind="stable")
            .drop(columns="_manifest_order")
            .reset_index(drop=True)
        )
    registry.to_csv(registry_path, index=False)
    (output_root / "island_rapid_prep_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "island-rapid-prep-v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "analysis_manifest": str(analysis_path),
                "hydraulic_context_registry": str(hydraulic_path),
                "base_experiment_dir": str(base_dir),
                "forcing_path": forcing_path,
                "forcing_config": asdict(forcing_config),
                "prep_config": asdict(prep_config),
                "reference_kb_summary": reference_kb_summary,
                "planned_routing_dt_seconds": fallback_planned_dt,
                "targeted_island_ids": sorted(requested_island_ids),
                "n_source_unit_timestep_records": len(source_unit_timestep_plan),
                "n_observations": int(len(registry)),
                "n_alias_references": int(registry["status"].eq("referenced_existing_independent_state").sum()),
                "n_prepared": int(registry["status"].eq("prepared").sum()),
                "n_failed": int(registry["status"].eq("failed").sum()),
                "n_excluded_topology_review": int(registry["status"].eq("excluded_topology_review").sum()),
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return registry


def run_prepared_island_observations(
    prep_registry_path: str | Path,
    *,
    hydrograph_config: HydrographMetricConfig | None = None,
    storage_config: RapidResultStorageConfig | None = None,
) -> pd.DataFrame:
    """Route the prepared genuine island states and retain alias references."""
    registry_path = Path(prep_registry_path).expanduser().resolve()
    registry = pd.read_csv(registry_path, low_memory=False)
    storage = storage_config or RapidResultStorageConfig()
    records: list[dict[str, object]] = []
    for _, row in registry.iterrows():
        island_id = _text(row.get("island_id") or row.get("state_id"))
        status = _text(row.get("status"))
        if status == "referenced_existing_independent_state":
            records.append({
                "island_id": island_id,
                "status": "referenced_existing_independent_state",
                "source_state_dir": _text(row.get("source_state_dir")),
            })
            continue
        if status == "excluded_topology_review":
            records.append({"island_id": island_id, "status": "excluded_topology_review"})
            continue
        if status != "prepared":
            records.append({
                "island_id": island_id,
                "status": "not_run",
                "prep_status": status,
                "error": _text(row.get("error")),
            })
            continue
        prep_dir = Path(_text(row.get("rapid_prep_dir"))).expanduser().resolve()
        run_dir = Path(_text(row.get("rapid_run_dir"))).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        recovered = _recover_published_island_run(
            island_id=island_id,
            prep_dir=prep_dir,
            run_dir=run_dir,
            storage=storage,
        )
        if recovered is not None:
            if not storage.retain_prepared_inputs and prep_dir.exists():
                # A prior publication succeeded but its cleanup bookkeeping was
                # interrupted. The recovered bundle has already verified every
                # durable product required by the selected storage policy, so
                # its prep is now safely disposable.
                _remove_disposable_prep(prep_dir)
                recovered["rapid_prep_retained"] = False
            records.append(recovered)
            continue
        if not prep_dir.exists():
            records.append({
                "island_id": island_id,
                "status": "failed",
                "rapid_prep_dir": str(prep_dir),
                "rapid_run_dir": str(run_dir),
                "error": f"Island RAPID prep directory is missing and no complete published result exists: {prep_dir}",
            })
            continue
        try:
            qout_path = run_dir / "Qout.nc"
            qout = run_prepared_state(prep_dir, output_path=qout_path)
            record: dict[str, object] = {
                "island_id": island_id,
                "status": "ran",
                "rapid_prep_dir": str(prep_dir),
                "rapid_run_dir": str(run_dir),
                "qout_nc": str(qout),
            }
            try:
                record.update(write_hydrograph_outputs(prep_dir, qout, run_dir, config=hydrograph_config))
            except Exception as exc:  # retain the valid routing result and diagnostics
                record.update({"hydrograph_status": "failed", "hydrograph_error": str(exc)})
            # Keep island results subject to exactly the same compact-publication
            # policy as ordinary experiment states.  In particular, a requested
            # preparation cleanup happens only after the final result products
            # have been published successfully.
            record.update(
                _publish_compact_result(
                    prep_dir=prep_dir,
                    run_dir=run_dir,
                    qout_path=Path(qout),
                    storage=storage,
                )
            )
            if not storage.retain_prepared_inputs:
                try:
                    _remove_disposable_prep(prep_dir)
                except FileNotFoundError:
                    # A concurrent/restarted cleanup may have already removed
                    # the disposable directory after publication. The durable
                    # result bundle above remains the completion criterion.
                    if prep_dir.exists():
                        raise
                record["rapid_prep_retained"] = False
            else:
                record["rapid_prep_retained"] = True
            records.append(record)
        except Exception as exc:  # pragma: no cover - full engine failure integration
            failure_bundle = _write_failure_bundle(
                prep_dir=prep_dir,
                run_dir=run_dir,
                error=exc,
                failure_dir=run_dir.parent / "failures",
            )
            records.append({
                "island_id": island_id,
                "status": "failed",
                "rapid_prep_dir": str(prep_dir),
                "rapid_run_dir": str(run_dir),
                "error": str(exc),
                "failure_bundle": str(failure_bundle),
            })

    result = pd.DataFrame(records)
    output_root = registry_path.parent
    result.to_csv(output_root / "island_rapid_run_registry.csv", index=False)
    (output_root / "island_rapid_run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "island-rapid-run-v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "prep_registry": str(registry_path),
                "n_observations": int(len(result)),
                "n_alias_references": int(result["status"].eq("referenced_existing_independent_state").sum()),
                "n_ran": int(result["status"].eq("ran").sum()),
                "n_failed": int(result["status"].eq("failed").sum()),
                "storage_config": asdict(storage),
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return result


def compare_island_to_independent_hydrographs(
    analysis_manifest_path: str | Path,
    *,
    base_experiment_dir: str | Path,
) -> pd.DataFrame:
    """Compare real island routing outputs against their independent unit states."""
    analysis_path = Path(analysis_manifest_path).expanduser().resolve()
    base_dir = Path(base_experiment_dir).expanduser().resolve()
    observations = pd.read_csv(analysis_path, low_memory=False)
    state_registry = pd.read_csv(base_dir / "state_registry.csv", low_memory=False)
    independent_by_unit = {
        unit_id: row
        for _, row in state_registry.loc[
            state_registry["selection_type"].astype(str).eq("unit")
            & state_registry["status"].fillna("complete").astype(str).eq("complete")
        ].iterrows()
        if (unit_id := _selected_unit_id(row.get("selected_unit_ids"))) is not None
    }
    records: list[dict[str, object]] = []
    for _, row in observations.iterrows():
        island_id = _text(row.get("island_id"))
        backend = _text(row.get("analysis_backend"))
        if _text(row.get("strict_analysis_status")) == "exclude_pruning_or_mask_topology_exception":
            records.append({
                "island_id": island_id,
                "comparison_status": "excluded_topology_review",
            })
            continue
        unit_id = _selected_unit_id(row.get("source_unit_id"))
        if backend == "independent_unit_alias":
            records.append({
                "island_id": island_id,
                "comparison_status": "identity_reference_existing_independent_state",
                "source_unit_id": unit_id,
                "independent_state_dir": _text(row.get("canonical_state_dir")),
            })
            continue
        independent = independent_by_unit.get(unit_id)
        if independent is None:
            records.append({
                "island_id": island_id,
                "comparison_status": "missing_independent_source_state",
                "source_unit_id": unit_id,
            })
            continue
        island_run = _existing_path(row.get("analysis_state_dir"), label="analysis_state_dir") / "rapid" / "run"
        independent_run = _existing_path(independent.get("state_dir"), label="independent_state_dir") / "rapid" / "run"
        island_hydrograph = island_run / "outlet_hydrograph.csv"
        independent_hydrograph = independent_run / "outlet_hydrograph.csv"
        try:
            comparison = compare_hydrographs(
                pd.read_csv(independent_hydrograph), pd.read_csv(island_hydrograph)
            )
            records.append({
                "island_id": island_id,
                "comparison_status": "compared_island_vs_independent_unit",
                "source_unit_id": unit_id,
                "independent_state_id": _text(independent.get("state_id")),
                "independent_state_dir": str(independent_run.parent),
                "island_run_dir": str(island_run),
                "independent_hydrograph_path": str(independent_hydrograph),
                "island_hydrograph_path": str(island_hydrograph),
                **comparison,
            })
        except Exception as exc:
            records.append({
                "island_id": island_id,
                "comparison_status": "comparison_failed",
                "source_unit_id": unit_id,
                "independent_state_id": _text(independent.get("state_id")),
                "error": str(exc),
            })
    result = pd.DataFrame(records)
    result.to_csv(analysis_path.parent / "island_vs_independent_hydrograph_comparison.csv", index=False)
    (analysis_path.parent / "island_vs_independent_hydrograph_comparison_manifest.json").write_text(
        json.dumps({
            "schema_version": "island-vs-independent-hydrograph-comparison-v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "analysis_manifest": str(analysis_path),
            "base_experiment_dir": str(base_dir),
            "n_observations": int(len(result)),
            "n_compared": int(result["comparison_status"].eq("compared_island_vs_independent_unit").sum()),
            "n_alias_identity_references": int(result["comparison_status"].eq("identity_reference_existing_independent_state").sum()),
            "n_failed": int(result["comparison_status"].eq("comparison_failed").sum()),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def stream_island_observations(
    analysis_manifest_path: str | Path,
    *,
    hydraulic_context_registry_path: str | Path,
    base_experiment_dir: str | Path,
    work_root: str | Path,
    island_ids: Collection[str] | None = None,
    island_forcing_routing_dt_seconds: Mapping[str, int] | None = None,
    storage_config: RapidResultStorageConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare and route genuine island states one at a time.

    Preparation inputs are the large, disposable part of an island RAPID run.
    This adapter bounds peak workspace use to one state: prepare one island,
    publish its compact result, remove its prep inputs, and only then advance
    to the next.  It is restart-safe because each targeted preparation retains
    the complete conceptual alias/exclusion registry.
    """
    analysis_path = Path(analysis_manifest_path).expanduser().resolve()
    analysis = pd.read_csv(analysis_path, low_memory=False)
    genuine = analysis.loc[
        ~analysis["analysis_backend"].map(_text).eq("independent_unit_alias")
        & ~analysis.get("strict_analysis_status", pd.Series("", index=analysis.index)).map(_text).eq(
            "exclude_pruning_or_mask_topology_exception"
        )
    ].copy()
    available = genuine["island_id"].map(_text).tolist()
    requested = [_text(value) for value in (island_ids or available) if _text(value)]
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(
            "Streaming island RAPID accepts only genuine, non-excluded island IDs; "
            f"unknown or conceptual IDs: {unknown}"
        )
    # Preserve manifest order and ignore accidental repeated CLI arguments.
    requested_set = set(requested)
    ordered_ids = [island_id for island_id in available if island_id in requested_set]
    overrides = dict(island_forcing_routing_dt_seconds or {})
    unknown_overrides = sorted(set(overrides) - set(ordered_ids))
    if unknown_overrides:
        raise ValueError(
            "Streaming island timestep overrides must refer to a selected island: "
            f"{unknown_overrides}"
        )

    storage = storage_config or RapidResultStorageConfig(
        compact_qout=True,
        retain_prepared_inputs=False,
        write_mapping_parquet=True,
    )
    iteration_records: list[dict[str, object]] = []
    prep_registry_path = analysis_path.parent / "island_rapid_prep_registry.csv"
    for index, island_id in enumerate(ordered_ids, start=1):
        registry = prepare_island_observations(
            analysis_path,
            hydraulic_context_registry_path=hydraulic_context_registry_path,
            base_experiment_dir=base_experiment_dir,
            work_root=work_root,
            island_ids=[island_id],
            island_forcing_routing_dt_seconds={island_id: overrides[island_id]}
            if island_id in overrides else None,
        )
        prep_row = registry.loc[registry["island_id"].map(_text).eq(island_id)].iloc[0]
        run_registry = run_prepared_island_observations(
            prep_registry_path,
            storage_config=storage,
        )
        run_row = run_registry.loc[run_registry["island_id"].map(_text).eq(island_id)].iloc[0]
        iteration_records.append({
            "sequence": index,
            "island_id": island_id,
            "prep_status": _text(prep_row.get("status")),
            "run_status": _text(run_row.get("status")),
            "error": _text(run_row.get("error")) or _text(prep_row.get("error")),
        })

    comparison = compare_island_to_independent_hydrographs(
        analysis_path,
        base_experiment_dir=base_experiment_dir,
    )
    final_prep = pd.read_csv(prep_registry_path, low_memory=False)
    final_run = pd.read_csv(analysis_path.parent / "island_rapid_run_registry.csv", low_memory=False)
    iteration = pd.DataFrame(iteration_records)
    iteration.to_csv(analysis_path.parent / "island_rapid_stream_execution.csv", index=False)
    (analysis_path.parent / "island_rapid_stream_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "island-rapid-stream-v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "analysis_manifest": str(analysis_path),
                "hydraulic_context_registry": str(Path(hydraulic_context_registry_path).expanduser().resolve()),
                "base_experiment_dir": str(Path(base_experiment_dir).expanduser().resolve()),
                "work_root": str(Path(work_root).expanduser().resolve()),
                "island_ids": ordered_ids,
                "island_forcing_routing_dt_seconds": overrides,
                "storage_config": asdict(storage),
                "n_streamed": len(ordered_ids),
                "n_prepared": int(final_prep["status"].eq("prepared").sum()),
                "n_prep_failed": int(final_prep["status"].eq("failed").sum()),
                "n_ran": int(final_run["status"].eq("ran").sum()),
                "n_run_failed": int(final_run["status"].eq("failed").sum()),
                "n_compared": int(comparison["comparison_status"].eq("compared_island_vs_independent_unit").sum()),
                "n_comparison_failed": int(comparison["comparison_status"].eq("comparison_failed").sum()),
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return final_prep, final_run, comparison
