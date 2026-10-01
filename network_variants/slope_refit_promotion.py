"""Atomically promote validated base slope-refit candidate contexts.

The candidate generator deliberately writes a compact context containing only
the new reference-profile heads and slopes.  Production hydraulic contexts
carry considerably more legacy provenance that downstream code still reads.
This module therefore merges the candidate values into that complete schema,
archives the prior context, and copies the immutable profile beside the base
experiment.  It never changes masks, networks, widths, topology, or child
state contexts.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd


PROMOTION_SCHEMA_VERSION = "slope-refit-base-context-promotion-v1"
_PROFILE_FILES = (
    "reference_profile.parquet",
    "reference_profile_observations.parquet",
    "reference_profile_acquisition_offsets.parquet",
    "reference_profile_cv.csv",
    "reference_profile_selection.json",
    "reference_profile_audit.json",
)
_CANDIDATE_AUDIT_FILES = (
    "candidate_context_manifest.json",
    "candidate_domain_head_loss_audit.csv",
    "candidate_link_ownership_audit.csv",
    "candidate_link_hydraulic_context.parquet",
    "candidate_node_hydraulic_context.parquet",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _canonical_example_id(value: object) -> str:
    return f"{int(float(value)):02d}"


def _require_exact_ids(frame: pd.DataFrame, candidate: pd.DataFrame, *, id_column: str, label: str) -> None:
    if frame[id_column].duplicated().any() or candidate[id_column].duplicated().any():
        raise ValueError(f"{label} contains duplicate {id_column} values.")
    current = set(pd.to_numeric(frame[id_column], errors="raise").astype(int))
    proposed = set(pd.to_numeric(candidate[id_column], errors="raise").astype(int))
    if current != proposed:
        raise ValueError(
            f"{label} candidate IDs differ from production: "
            f"missing={sorted(current - proposed)[:10]} extra={sorted(proposed - current)[:10]}"
        )


def _require_files(directory: Path, names: tuple[str, ...], *, label: str) -> None:
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{label} is incomplete: {missing}")


def _merge_link_context(current: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    _require_exact_ids(current, candidate, id_column="id_link", label="Link context")
    proposed = candidate.set_index("id_link", verify_integrity=True)
    result = current.set_index("id_link", verify_integrity=True).copy()
    # Preserve old schema fields, then retain every compact candidate field
    # except the old slope fields carried there for comparison provenance.
    for column in proposed.columns:
        if column not in {"slope_used", "slope_source_method"}:
            result[column] = proposed[column]
    result["slope_used"] = proposed["candidate_slope_m_per_m"]
    result["slope_source_method"] = proposed["candidate_slope_source_method"]
    result["slope_adjusted"] = True
    result["slope_reason"] = "promoted_frozen_reference_profile_equal_per_link_domain_solver"
    result["slope_minimum_applied"] = proposed["candidate_numerical_floor_active"].astype(bool)
    for prefix, head_field, source_field in (
        ("wse_us", "candidate_head_us_m", "candidate_slope_source_method"),
        ("wse_ds", "candidate_head_ds_m", "candidate_slope_source_method"),
    ):
        result[prefix] = proposed[head_field]
        result[f"{prefix}_source_method"] = proposed[source_field]
        result[f"{prefix}_interpolated"] = proposed["candidate_slope_source_method"].eq(
            "frozen_reference_profile_direct_connector_or_boundary"
        )
        result[f"{prefix}_profile_residual"] = proposed[head_field] - proposed[
            "profile_head_us_m" if prefix == "wse_us" else "profile_head_ds_m"
        ]
    return result.reset_index()


def _merge_node_context(current: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    _require_exact_ids(current, candidate, id_column="id_node", label="Node context")
    proposed = candidate.set_index("id_node", verify_integrity=True)
    result = current.set_index("id_node", verify_integrity=True).copy()
    for column in proposed.columns:
        result[column] = proposed[column]
    result["rapid_wse_for_slope"] = proposed["candidate_head_m"]
    result["rapid_wse_source_method"] = proposed["candidate_head_source_method"]
    result["rapid_wse_interpolated"] = proposed["candidate_head_source_method"].eq(
        "frozen_reference_profile_direct"
    )
    result["rapid_wse_profile_residual"] = proposed["candidate_head_m"] - proposed["profile_head_m"]
    return result.reset_index()


def _context_manifest(
    current: dict[str, Any], *, item: dict[str, Any], promoted_at: str, output_dir: Path
) -> dict[str, Any]:
    profile_audit = _load_json(Path(item["profile_dir"]) / "reference_profile_audit.json")
    profile_summary = dict(profile_audit["summary"])
    result = dict(current)
    result["generated_at"] = promoted_at
    result["output_paths"] = {
        "link_hydraulic_context": str(output_dir / "link_hydraulic_context.csv"),
        "node_hydraulic_context": str(output_dir / "node_hydraulic_context.csv"),
    }
    result["section_slope_summary"] = {
        "slope": float(profile_summary["fitted_global_slope_m_per_m"]),
        "source_method": "frozen_reference_profile_equal_per_link_domain_solver",
        "anchor_count": int(profile_summary["n_profile_nodes"]),
        "profile_mode": "reference_profile_multitemporal_smoothing_spline",
        "reference_section_summary_path": "",
    }
    result["head_concepts"] = {
        "observed_or_matched_node_wse": "sword_wse / node_wse_raw (retained)",
        "derived_profile_wse_for_slope": "rapid_wse_for_slope = promoted candidate_head_m",
        "normalized_solver_heads": "candidate_head_m from profile-boundary graph-potential solve",
    }
    result["slope_refit_promotion"] = {
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "promoted_at": promoted_at,
        "candidate_context_dir": item["candidate_context_dir"],
        "candidate_context_manifest_sha256": item["candidate_context_manifest_sha256"],
        "reference_profile_dir": item["profile_dir"],
        "reference_profile_sha256": item["profile_sha256"],
        "domain_objective": item["domain_objective"],
        "domain_minimum_reference_slope_fraction": item["domain_minimum_reference_slope_fraction"],
        "validation_csv": item["validation_csv"],
        "validation_row_sha256": item["validation_row_sha256"],
        "candidate_audit_dir": str(output_dir / "slope_refit_audit"),
    }
    return result


def build_slope_refit_promotion_plan(
    candidate_root: str | Path, validation_dir: str | Path
) -> dict[str, Any]:
    root = Path(candidate_root).expanduser().resolve()
    validation = Path(validation_dir).expanduser().resolve()
    batch_path = root / "slope_refit_candidate_batch_manifest.csv"
    validation_csv = validation / "slope_refit_candidate_validation.csv"
    batch = pd.read_csv(batch_path)
    checks = pd.read_csv(validation_csv)
    if len(batch) != 16 or batch["example_id"].map(_canonical_example_id).nunique() != 16:
        raise ValueError("Promotion requires the complete 16-example candidate batch.")
    checks["example_id"] = checks["example_id"].map(_canonical_example_id)
    if not checks["validation_passed"].fillna(False).astype(bool).all():
        failed = checks.loc[~checks["validation_passed"].fillna(False).astype(bool), "example_id"].tolist()
        raise ValueError(f"Candidate validation is not fully passing: {failed}")
    validation_by_example = checks.set_index("example_id", verify_integrity=True)
    items: list[dict[str, Any]] = []
    for row in batch.itertuples(index=False):
        example_id = _canonical_example_id(row.example_id)
        if str(row.status) not in {"complete", "already_complete"}:
            raise ValueError(f"Candidate batch is incomplete for Example {example_id}: {row.status}")
        experiment = Path(row.experiment_dir).expanduser().resolve()
        state = experiment / "states" / "S000_base"
        current_context = state / "hydraulic_context"
        candidate_context = Path(row.candidate_context_dir).expanduser().resolve()
        profile = Path(row.profile_dir).expanduser().resolve()
        _require_files(current_context, ("link_hydraulic_context.csv", "node_hydraulic_context.csv", "hydraulic_context_manifest.json"), label=f"Example {example_id} production context")
        _require_files(candidate_context, _CANDIDATE_AUDIT_FILES, label=f"Example {example_id} candidate context")
        _require_files(profile, _PROFILE_FILES, label=f"Example {example_id} frozen profile")
        if (experiment / "slope_refit_refresh_required.json").exists():
            raise FileExistsError(
                f"Example {example_id} already has a slope-refit refresh marker; "
                "refusing a second promotion. Inspect its prior promotion record first."
            )
        candidate_manifest_path = candidate_context / "candidate_context_manifest.json"
        candidate_manifest = _load_json(candidate_manifest_path)
        if Path(str(candidate_manifest["experiment_dir"])).resolve() != experiment:
            raise ValueError(f"Candidate provenance points at a different experiment for Example {example_id}.")
        current_links = pd.read_csv(current_context / "link_hydraulic_context.csv", usecols=["id_link"])
        current_nodes = pd.read_csv(current_context / "node_hydraulic_context.csv", usecols=["id_node"])
        candidate_links = pd.read_parquet(candidate_context / "candidate_link_hydraulic_context.parquet", columns=["id_link"])
        candidate_nodes = pd.read_parquet(candidate_context / "candidate_node_hydraulic_context.parquet", columns=["id_node"])
        _require_exact_ids(current_links, candidate_links, id_column="id_link", label=f"Example {example_id} link context")
        _require_exact_ids(current_nodes, candidate_nodes, id_column="id_node", label=f"Example {example_id} node context")
        candidate_profile = profile / "reference_profile.parquet"
        row_payload = validation_by_example.loc[example_id].to_dict()
        items.append({
            "example_id": example_id,
            "experiment_dir": str(experiment),
            "state_dir": str(state),
            "current_context_dir": str(current_context),
            "candidate_context_dir": str(candidate_context),
            "profile_dir": str(profile),
            "candidate_context_manifest_sha256": _sha256(candidate_manifest_path),
            "profile_sha256": _sha256(candidate_profile),
            "validation_csv": str(validation_csv),
            "validation_row_sha256": hashlib.sha256(
                json.dumps(row_payload, default=str, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "domain_objective": str(candidate_manifest["domain_objective"]),
            "domain_minimum_reference_slope_fraction": float(candidate_manifest["domain_minimum_reference_slope_fraction"]),
            "n_links": int(len(current_links)),
            "n_nodes": int(len(current_nodes)),
        })
    return {
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "candidate_root": str(root),
        "candidate_batch_manifest": str(batch_path),
        "candidate_batch_manifest_sha256": _sha256(batch_path),
        "validation_dir": str(validation),
        "validation_csv": str(validation_csv),
        "validation_csv_sha256": _sha256(validation_csv),
        "n_examples": len(items),
        "items": items,
    }


def build_equivalent_base_adoption_plan(
    candidate_root: str | Path, validation_dir: str | Path
) -> dict[str, Any]:
    """Plan adoption of promoted independent-base contexts by equivalent compound bases.

    A compound experiment has the same realised base network as its independent
    counterpart.  This function proves exact link/node direction identity before
    reusing the already validated independent candidate profile and context. It
    avoids a second profile fit while keeping a separate, recoverable promotion
    record on the compound experiment.
    """
    root = Path(candidate_root).expanduser().resolve()
    validation = Path(validation_dir).expanduser().resolve()
    batch_path = root / "slope_refit_candidate_batch_manifest.csv"
    validation_csv = validation / "slope_refit_candidate_validation.csv"
    batch = pd.read_csv(batch_path)
    checks = pd.read_csv(validation_csv)
    checks["example_id"] = checks["example_id"].map(_canonical_example_id)
    if len(batch) != 16 or not checks["validation_passed"].fillna(False).astype(bool).all():
        raise ValueError("Equivalent-base adoption requires the complete passing 16-example validation.")
    validation_by_example = checks.set_index("example_id", verify_integrity=True)
    items: list[dict[str, Any]] = []
    for row in batch.itertuples(index=False):
        example_id = _canonical_example_id(row.example_id)
        source_experiment = Path(row.experiment_dir).expanduser().resolve()
        target_name = source_experiment.name.replace("_indep", "_compound_bubbles", 1)
        if target_name == source_experiment.name:
            raise ValueError(f"Cannot derive compound counterpart for {source_experiment}.")
        experiment = source_experiment.with_name(target_name).resolve()
        state = experiment / "states" / "S000_base"
        current_context = state / "hydraulic_context"
        candidate_context = Path(row.candidate_context_dir).expanduser().resolve()
        profile = Path(row.profile_dir).expanduser().resolve()
        _require_files(current_context, ("link_hydraulic_context.csv", "node_hydraulic_context.csv", "hydraulic_context_manifest.json"), label=f"Example {example_id} equivalent production context")
        _require_files(candidate_context, _CANDIDATE_AUDIT_FILES, label=f"Example {example_id} candidate context")
        _require_files(profile, _PROFILE_FILES, label=f"Example {example_id} frozen profile")
        if (experiment / "slope_refit_refresh_required.json").exists():
            raise FileExistsError(f"Equivalent base was already promoted: {experiment}")
        target_links = pd.read_csv(current_context / "link_hydraulic_context.csv", usecols=["id_link", "id_us_node", "id_ds_node"])
        candidate_links = pd.read_parquet(
            candidate_context / "candidate_link_hydraulic_context.parquet",
            columns=["id_link", "id_us_node", "id_ds_node"],
        )
        _require_exact_ids(target_links, candidate_links, id_column="id_link", label=f"Example {example_id} equivalent link context")
        comparison = target_links.merge(candidate_links, on="id_link", suffixes=("_target", "_candidate"), validate="1:1")
        same_direction = (
            comparison["id_us_node_target"].eq(comparison["id_us_node_candidate"])
            & comparison["id_ds_node_target"].eq(comparison["id_ds_node_candidate"])
        )
        if not bool(same_direction.all()):
            raise ValueError(f"Equivalent base has non-identical directed links for Example {example_id}.")
        target_nodes = pd.read_csv(current_context / "node_hydraulic_context.csv", usecols=["id_node"])
        candidate_nodes = pd.read_parquet(candidate_context / "candidate_node_hydraulic_context.parquet", columns=["id_node"])
        _require_exact_ids(target_nodes, candidate_nodes, id_column="id_node", label=f"Example {example_id} equivalent node context")
        candidate_manifest_path = candidate_context / "candidate_context_manifest.json"
        candidate_manifest = _load_json(candidate_manifest_path)
        validation_row = validation_by_example.loc[example_id].to_dict()
        items.append({
            "example_id": example_id,
            "experiment_dir": str(experiment),
            "source_experiment_dir": str(source_experiment),
            "state_dir": str(state),
            "current_context_dir": str(current_context),
            "candidate_context_dir": str(candidate_context),
            "profile_dir": str(profile),
            "candidate_context_manifest_sha256": _sha256(candidate_manifest_path),
            "profile_sha256": _sha256(profile / "reference_profile.parquet"),
            "validation_csv": str(validation_csv),
            "validation_row_sha256": hashlib.sha256(json.dumps(validation_row, default=str, sort_keys=True).encode("utf-8")).hexdigest(),
            "domain_objective": str(candidate_manifest["domain_objective"]),
            "domain_minimum_reference_slope_fraction": float(candidate_manifest["domain_minimum_reference_slope_fraction"]),
            "n_links": int(len(target_links)),
            "n_nodes": int(len(target_nodes)),
        })
    return {
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "promotion_kind": "equivalent_compound_base_adoption",
        "candidate_root": str(root),
        "candidate_batch_manifest": str(batch_path),
        "candidate_batch_manifest_sha256": _sha256(batch_path),
        "validation_dir": str(validation),
        "validation_csv": str(validation_csv),
        "validation_csv_sha256": _sha256(validation_csv),
        "n_examples": len(items),
        "items": items,
    }


def _copy_tree_files(source: Path, target: Path, names: tuple[str, ...]) -> None:
    target.mkdir(parents=True, exist_ok=False)
    for name in names:
        shutil.copy2(source / name, target / name)


def _stage_context(item: dict[str, Any], promoted_at: str) -> tuple[Path, Path]:
    state = Path(item["state_dir"])
    current = Path(item["current_context_dir"])
    candidate = Path(item["candidate_context_dir"])
    profile = Path(item["profile_dir"])
    stage_context = state / f".hydraulic_context.slope-refit-{uuid4().hex}"
    stage_profile = state.parent.parent / f".reference_profile.slope-refit-{uuid4().hex}"
    try:
        links = _merge_link_context(
            pd.read_csv(current / "link_hydraulic_context.csv"),
            pd.read_parquet(candidate / "candidate_link_hydraulic_context.parquet"),
        )
        nodes = _merge_node_context(
            pd.read_csv(current / "node_hydraulic_context.csv"),
            pd.read_parquet(candidate / "candidate_node_hydraulic_context.parquet"),
        )
        stage_context.mkdir(parents=True, exist_ok=False)
        links.to_csv(stage_context / "link_hydraulic_context.csv", index=False)
        nodes.to_csv(stage_context / "node_hydraulic_context.csv", index=False)
        _copy_tree_files(candidate, stage_context / "slope_refit_audit", _CANDIDATE_AUDIT_FILES)
        manifest = _context_manifest(
            _load_json(current / "hydraulic_context_manifest.json"),
            item=item,
            promoted_at=promoted_at,
            output_dir=state / "hydraulic_context",
        )
        _atomic_json(manifest, stage_context / "hydraulic_context_manifest.json")
        _copy_tree_files(profile, stage_profile, _PROFILE_FILES)
        return stage_context, stage_profile
    except Exception:
        shutil.rmtree(stage_context, ignore_errors=True)
        shutil.rmtree(stage_profile, ignore_errors=True)
        raise


def apply_slope_refit_promotion_plan(plan: dict[str, Any]) -> Path:
    root = Path(plan["candidate_root"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    promoted_at = datetime.now(timezone.utc).isoformat()
    backup = root / "promotion_backups" / stamp
    record = root / "promotion_records" / f"slope_refit_base_context_promotion_{stamp}.json"
    if backup.exists() or record.exists():
        raise FileExistsError("Promotion timestamp collision; retry.")
    staged: dict[str, tuple[Path, Path]] = {}
    moved: list[tuple[Path, Path, Path, Path | None]] = []
    manifest_backups: list[tuple[Path, Path, Path]] = []
    try:
        for item in plan["items"]:
            staged[item["example_id"]] = _stage_context(item, promoted_at)
        backup.mkdir(parents=True, exist_ok=False)
        for item in plan["items"]:
            experiment = Path(item["experiment_dir"])
            state = Path(item["state_dir"])
            current = Path(item["current_context_dir"])
            production_profile = experiment / "reference_profile"
            state_backup = backup / experiment.name / "states" / state.name
            state_backup.mkdir(parents=True, exist_ok=True)
            context_backup = state_backup / "hydraulic_context"
            profile_backup: Path | None = None
            if production_profile.exists():
                profile_backup = backup / experiment.name / "reference_profile"
                profile_backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(production_profile), str(profile_backup))
            try:
                shutil.move(str(current), str(context_backup))
            except Exception:
                if profile_backup is not None and profile_backup.exists():
                    shutil.move(str(profile_backup), str(production_profile))
                raise
            # Register both archived production paths before installing either
            # staged replacement so a later move failure is fully reversible.
            moved.append((current, context_backup, production_profile, profile_backup))
            stage_context, stage_profile = staged[item["example_id"]]
            shutil.move(str(stage_context), str(current))
            shutil.move(str(stage_profile), str(production_profile))

            experiment_manifest = experiment / "experiment_manifest.json"
            manifest_backup = backup / experiment.name / "experiment_manifest.json"
            shutil.copy2(experiment_manifest, manifest_backup)
            manifest = _load_json(experiment_manifest)
            manifest["slope_refit_base_context_promotion"] = {
                "schema_version": PROMOTION_SCHEMA_VERSION,
                "promoted_at": promoted_at,
                "candidate_batch_manifest_sha256": plan["candidate_batch_manifest_sha256"],
                "candidate_context_dir": item["candidate_context_dir"],
                "reference_profile_dir": str(production_profile),
                "backup_dir": str(backup / experiment.name),
                "requires_downstream_hydraulic_refresh": True,
                "requires_equivalent_resistance_recompute": True,
                "requires_rapid_prepare_and_run": True,
            }
            _atomic_json(manifest, experiment_manifest)
            marker = experiment / "slope_refit_refresh_required.json"
            _atomic_json({
                "schema_version": PROMOTION_SCHEMA_VERSION,
                "status": "required",
                "reason": "base hydraulic heads and slopes were promoted from the validated reference-profile candidate",
                "promoted_at": promoted_at,
                "candidate_batch_manifest_sha256": plan["candidate_batch_manifest_sha256"],
                "required_next_steps": [
                    "refresh child hydraulic contexts",
                    "recompute equivalent resistance",
                    "prepare and run RAPID",
                ],
            }, marker)
            manifest_backups.append((experiment_manifest, manifest_backup, marker))

        payload = {**plan, "promoted_at": promoted_at, "backup_root": str(backup), "is_valid": True}
        record.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(payload, record)
        return record
    except Exception:
        for experiment_manifest, manifest_backup, marker in reversed(manifest_backups):
            shutil.copy2(manifest_backup, experiment_manifest)
            marker.unlink(missing_ok=True)
        for current, context_backup, production_profile, profile_backup in reversed(moved):
            shutil.rmtree(current, ignore_errors=True)
            shutil.move(str(context_backup), str(current))
            shutil.rmtree(production_profile, ignore_errors=True)
            if profile_backup is not None and profile_backup.exists():
                shutil.move(str(profile_backup), str(production_profile))
        raise
    finally:
        for stage_context, stage_profile in staged.values():
            shutil.rmtree(stage_context, ignore_errors=True)
            shutil.rmtree(stage_profile, ignore_errors=True)
