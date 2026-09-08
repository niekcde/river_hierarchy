"""Retention profiles and guarded cleanup planning for state artifacts.

The final directed variant network is always a retained scientific product.
This module only manages regenerable diagnostic products after a successful
state has published its final directed/width outputs. Existing-output cleanup
is deliberately split into a non-destructive manifest and a narrow executor:
the executor can *only* compact approved detailed-width CSV files after it has
verified the corresponding Parquet product. It does not apply any manifest
row that deletes a directory or raw network product.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


ARTIFACT_PROFILES = ("analysis", "reproducible", "debug")


def validate_artifact_profile(profile: str) -> str:
    """Return a normalized supported profile or raise a useful error."""
    normalized = str(profile).strip().lower()
    if normalized not in ARTIFACT_PROFILES:
        raise ValueError(
            f"Unsupported artifact profile {profile!r}. "
            f"Expected one of {ARTIFACT_PROFILES}."
        )
    return normalized


def _allocated_size_bytes(path: Path) -> int:
    if path.is_file():
        return int(path.stat().st_size)
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += int(item.stat().st_size)
            except FileNotFoundError:
                continue
    return total


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _final_state_is_published(variant_dir: Path) -> bool:
    directed = variant_dir / "directed"
    widths = variant_dir / "widths"
    return (
        any(directed.glob("*_directed_links.gpkg"))
        and any(directed.glob("*_directed_nodes.gpkg"))
        and (widths / "links_with_width_families.gpkg").is_file()
    )


def _copy_json_audits(attempt_dir: Path, destination: Path) -> list[dict[str, object]]:
    retained: list[dict[str, object]] = []
    for source in sorted(attempt_dir.rglob("*.json")):
        relative = source.relative_to(attempt_dir)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source_hash = _sha256(source)
        target_preexisted = target.exists()
        if target_preexisted:
            if _sha256(target) != source_hash:
                raise ValueError(
                    f"Retained audit collision has different content: {target}"
                )
        else:
            shutil.copy2(source, target)
        retained.append(
            {
                "relative_path": str(relative),
                "bytes": int(source.stat().st_size),
                "sha256": source_hash,
                "target_preexisted": target_preexisted,
            }
        )
    return retained


def _remove_tree_ignoring_disappeared_children(path: Path) -> None:
    """Remove a directory while tolerating entries removed during its traversal.

    External APFS volumes can report a child in a directory enumeration after
    that child has already disappeared. The parent tree is still safely
    removable; only a non-missing filesystem error must stop cleanup.
    """
    if not path.exists():
        return

    def onexc(_function, _path, error: BaseException) -> None:
        if isinstance(error, FileNotFoundError):
            return
        raise error

    shutil.rmtree(path, onexc=onexc)


def apply_successful_state_retention(
    state_dir: str | Path,
    *,
    profile: str = "analysis",
    compact_width_samples: bool = True,
    drop_regenerable_raw: bool = False,
) -> dict[str, object]:
    """Apply the future-run retention policy after a successful child state.

    ``analysis`` retains the final network and compact JSON safety audits, but
    drops a successful ``local_patch_attempt`` tree. ``reproducible`` and
    ``debug`` retain the complete attempt tree. Failed states never reach this
    function and therefore retain their diagnostic products.
    """
    resolved_profile = validate_artifact_profile(profile)
    root = Path(state_dir)
    variant_dir = root / "variant"
    attempt_dir = variant_dir / "local_patch_attempt"
    summary_dir = variant_dir / "summary"
    payload: dict[str, object] = {
        "schema_version": "artifact-retention-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "state_dir": str(root.resolve()),
        "profile": resolved_profile,
        "final_variant_published": _final_state_is_published(variant_dir),
        "local_patch_attempt": {
            "present_before": attempt_dir.exists(),
            "action": "not_present",
            "bytes_before": _allocated_size_bytes(attempt_dir) if attempt_dir.exists() else 0,
            "retained_json_audits": [],
        },
        "width_samples": {
            "source_csv": str(variant_dir / "widths" / "link_width_samples.csv"),
            "parquet_path": str(variant_dir / "widths" / "link_width_samples.parquet"),
            "action": "not_present",
        },
        "regenerable_raw": {
            "action": "retained_by_profile_or_parent_requirement",
            "preserved_edit_geometry": None,
            "removed_directories": [],
        },
    }
    attempt = payload["local_patch_attempt"]
    assert isinstance(attempt, dict)
    if not attempt_dir.exists():
        pass
    elif any(attempt_dir.rglob("fallback_to_*.json")):
        attempt["action"] = "retained_fallback_diagnostic"
    elif resolved_profile != "analysis":
        attempt["action"] = "retained_by_profile"
    elif not bool(payload["final_variant_published"]):
        attempt["action"] = "retained_final_state_incomplete"
    else:
        audit_dir = summary_dir / "local_patch_attempt_audits"
        attempt["retained_json_audits"] = _copy_json_audits(attempt_dir, audit_dir)
        _remove_tree_ignoring_disappeared_children(attempt_dir)
        attempt["action"] = "removed_after_success"
        attempt["bytes_after"] = 0

    width_samples = payload["width_samples"]
    assert isinstance(width_samples, dict)
    source_csv = Path(str(width_samples["source_csv"]))
    parquet_path = Path(str(width_samples["parquet_path"]))
    if source_csv.exists() and resolved_profile == "analysis" and compact_width_samples:
        samples = pd.read_csv(source_csv, low_memory=False)
        temporary_path = parquet_path.with_name(parquet_path.stem + ".tmp.parquet")
        samples.to_parquet(temporary_path, index=False, compression="zstd")
        source_bytes = int(source_csv.stat().st_size)
        parquet_bytes = int(temporary_path.stat().st_size)
        if parquet_bytes >= source_bytes:
            temporary_path.unlink()
            width_samples.update(
                {
                    "action": "retained_csv_parquet_not_smaller",
                    "source_csv_bytes": source_bytes,
                    "parquet_bytes": parquet_bytes,
                    "net_bytes_reclaimed": 0,
                }
            )
        else:
            temporary_path.replace(parquet_path)
            width_samples.update(
                {
                    "action": "converted_to_parquet_and_removed_csv",
                    "source_csv_bytes": source_bytes,
                    "parquet_bytes": int(parquet_path.stat().st_size),
                    "parquet_sha256": _sha256(parquet_path),
                    "net_bytes_reclaimed": source_bytes - int(parquet_path.stat().st_size),
                }
            )
            source_csv.unlink()
    elif source_csv.exists():
        width_samples["action"] = "retained_by_profile_or_parent_requirement"

    regenerable_raw = payload["regenerable_raw"]
    assert isinstance(regenerable_raw, dict)
    if (
        resolved_profile == "analysis"
        and drop_regenerable_raw
        and bool(payload["final_variant_published"])
    ):
        summary_dir.mkdir(parents=True, exist_ok=True)
        edit_source = variant_dir / "mask" / "collapse_edit_geometries.gpkg"
        if edit_source.is_file():
            edit_target = summary_dir / "collapse_edit_geometries.gpkg"
            shutil.copy2(edit_source, edit_target)
            regenerable_raw["preserved_edit_geometry"] = {
                "path": str(edit_target.resolve()),
                "sha256": _sha256(edit_target),
            }
        removed: list[dict[str, object]] = []
        for name in ("mask", "rivgraph"):
            source = variant_dir / name
            if source.exists():
                removed.append({"name": name, "bytes": _allocated_size_bytes(source)})
                _remove_tree_ignoring_disappeared_children(source)
        regenerable_raw["action"] = "removed_after_success"
        regenerable_raw["removed_directories"] = removed

    summary_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = summary_dir / "artifact_retention_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    payload["manifest_path"] = str(manifest_path.resolve())
    return payload


def apply_successful_state_retention_best_effort(
    state_dir: str | Path,
    *,
    profile: str = "analysis",
    compact_width_samples: bool = True,
    drop_regenerable_raw: bool = False,
) -> dict[str, object]:
    """Apply retention without allowing cleanup to invalidate a published state.

    Retention only manages regenerable diagnostics and storage products. Once the
    final directed network and width-family product have been published, a cleanup
    failure is recorded as a warning and the scientific state remains usable.
    Failures still propagate when the final scientific products are incomplete.
    """
    root = Path(state_dir)
    try:
        payload = apply_successful_state_retention(
            root,
            profile=profile,
            compact_width_samples=compact_width_samples,
            drop_regenerable_raw=drop_regenerable_raw,
        )
        payload["status"] = "complete"
        return payload
    except Exception as exc:
        variant_dir = root / "variant"
        if not _final_state_is_published(variant_dir):
            raise
        summary_dir = variant_dir / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "artifact-retention-v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "state_dir": str(root.resolve()),
            "profile": validate_artifact_profile(profile),
            "status": "cleanup_warning",
            "final_variant_published": True,
            "scientific_state_preserved": True,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "error_traceback": "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
            "local_patch_attempt": {"action": "cleanup_incomplete"},
            "width_samples": {"action": "cleanup_incomplete"},
            "regenerable_raw": {"action": "cleanup_incomplete"},
        }
        manifest_path = summary_dir / "artifact_retention_manifest.json"
        manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        payload["manifest_path"] = str(manifest_path.resolve())
        return payload


@dataclass(frozen=True)
class CleanupManifest:
    rows: pd.DataFrame
    summary: dict[str, object]


@dataclass(frozen=True)
class CleanupExecution:
    """Auditable result of a guarded existing-output cleanup run."""

    rows: pd.DataFrame
    summary: dict[str, object]


def _state_rows(experiment_dir: Path) -> Iterable[tuple[str, Path, str]]:
    registry = pd.read_csv(experiment_dir / "state_registry.csv")
    for _, row in registry.iterrows():
        state_dir = Path(str(row["state_dir"]))
        yield str(row["state_id"]), state_dir, str(row.get("status", ""))


def build_cleanup_manifest(experiment_dir: str | Path) -> CleanupManifest:
    """Inventory existing products and propose safe, non-destructive actions."""
    root = Path(experiment_dir).expanduser().resolve()
    rows: list[dict[str, object]] = []
    for state_id, state_dir, status in _state_rows(root):
        variant = state_dir / "variant"
        final_published = _final_state_is_published(variant)
        candidates = (
            ("local_patch_attempt", variant / "local_patch_attempt", "remove_after_success", "Final directed and width products are retained; copy JSON safety audits first."),
            ("width_samples_csv", variant / "widths" / "link_width_samples.csv", "convert_to_parquet_then_remove_csv", "Retain the detailed samples in a compact columnar format before removing CSV."),
            ("raw_mask", variant / "mask", "remove_after_reproducibility_manifest", "Regenerable from the frozen base/edit contract; preserve the edit geometry and source manifest first."),
            ("raw_rivgraph", variant / "rivgraph", "remove_after_reproducibility_manifest", "Regenerable but potentially expensive; retain it for base, failures, and validation states."),
        )
        for category, path, action, rationale in candidates:
            if not path.exists():
                continue
            state_complete_and_published = bool(status == "complete" and final_published)
            safe_for_action = bool(
                state_complete_and_published and action == "remove_after_success"
            )
            rows.append(
                {
                    "state_id": state_id,
                    "state_status": status,
                    "artifact_category": category,
                    "path": str(path),
                    "bytes": _allocated_size_bytes(path),
                    "recommended_action": action,
                    "eligible_now": safe_for_action,
                    "eligible_for_conversion": bool(
                        state_complete_and_published
                        and action == "convert_to_parquet_then_remove_csv"
                    ),
                    "rationale": rationale,
                }
            )
    frame = pd.DataFrame(rows)
    summary = {
        "schema_version": "artifact-cleanup-manifest-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_dir": str(root),
        "n_rows": int(len(frame)),
        "bytes_by_recommended_action": {
            str(action): int(total)
            for action, total in frame.groupby("recommended_action", dropna=False)["bytes"].sum().items()
        } if not frame.empty else {},
        "eligible_removal_bytes": int(frame.loc[frame["eligible_now"], "bytes"].sum()) if not frame.empty else 0,
        "destructive_action_performed": False,
    }
    return CleanupManifest(rows=frame, summary=summary)


def write_cleanup_manifest(
    experiment_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> CleanupManifest:
    """Write the cleanup plan CSV/JSON. This function never deletes files."""
    result = build_cleanup_manifest(experiment_dir)
    destination = Path(output_dir) if output_dir is not None else Path(experiment_dir)
    destination.mkdir(parents=True, exist_ok=True)
    result.rows.to_csv(destination / "artifact_cleanup_manifest.csv", index=False)
    (destination / "artifact_cleanup_manifest.json").write_text(
        json.dumps(result.summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def write_output_cleanup_manifest(output_root: str | Path) -> CleanupManifest:
    """Write one combined, manifest-only cleanup inventory for all experiments."""
    root = Path(output_root).expanduser().resolve()
    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for registry_path in sorted(root.glob("*/state_registry.csv")):
        result = build_cleanup_manifest(registry_path.parent)
        frame = result.rows.copy()
        if not frame.empty:
            frame.insert(0, "experiment", registry_path.parent.name)
            frames.append(frame)
        summaries.append(result.summary)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    by_action = (
        {
            str(action): int(total)
            for action, total in combined.groupby("recommended_action", dropna=False)["bytes"].sum().items()
        }
        if not combined.empty
        else {}
    )
    summary = {
        "schema_version": "artifact-cleanup-manifest-all-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(root),
        "n_experiments": int(len(summaries)),
        "n_rows": int(len(combined)),
        "bytes_by_recommended_action": by_action,
        "eligible_removal_bytes": int(
            combined.loc[combined["eligible_now"], "bytes"].sum()
        ) if not combined.empty else 0,
        "destructive_action_performed": False,
    }
    combined.to_csv(root / "artifact_cleanup_manifest_all.csv", index=False)
    (root / "artifact_cleanup_manifest_all.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return CleanupManifest(rows=combined, summary=summary)


_WIDTH_SAMPLE_ACTION = "convert_to_parquet_then_remove_csv"
_LOCAL_PATCH_ACTION = "remove_after_success"


def _frames_are_equivalent(source: pd.DataFrame, target: pd.DataFrame) -> None:
    """Raise when a written/read Parquet table differs from its source CSV."""
    pd.testing.assert_frame_equal(
        source.reset_index(drop=True),
        target.reset_index(drop=True),
        check_dtype=True,
        check_like=False,
    )


def _width_conversion_is_safe(row: pd.Series, source: Path) -> tuple[bool, str]:
    """Confirm that a manifest row still describes a completed final state."""
    if str(row.get("artifact_category", "")) != "width_samples_csv":
        return False, "unsupported_artifact_category"
    if str(row.get("recommended_action", "")) != _WIDTH_SAMPLE_ACTION:
        return False, "unsupported_recommended_action"
    if str(row.get("state_status", "")) != "complete":
        return False, "state_not_complete_in_manifest"
    if source.name != "link_width_samples.csv" or source.parent.name != "widths":
        return False, "unexpected_width_samples_path"
    variant_dir = source.parent.parent
    if not _final_state_is_published(variant_dir):
        return False, "final_variant_not_currently_published"
    return True, "eligible"


def _final_output_audit(variant_dir: Path) -> list[dict[str, object]]:
    directed = variant_dir / "directed"
    widths = variant_dir / "widths"
    paths = [
        *sorted(directed.glob("*_directed_links.gpkg")),
        *sorted(directed.glob("*_directed_nodes.gpkg")),
        widths / "links_with_width_families.gpkg",
    ]
    return [
        {"path": str(path.resolve()), "bytes": int(path.stat().st_size), "sha256": _sha256(path)}
        for path in paths
        if path.is_file()
    ]


def _local_patch_removal_is_safe(row: pd.Series, source: Path) -> tuple[bool, str]:
    """Confirm a successful local attempt can be removed without masking fallback evidence."""
    if str(row.get("artifact_category", "")) != "local_patch_attempt":
        return False, "unsupported_artifact_category"
    if str(row.get("recommended_action", "")) != _LOCAL_PATCH_ACTION:
        return False, "unsupported_recommended_action"
    if str(row.get("state_status", "")) != "complete":
        return False, "state_not_complete_in_manifest"
    if source.name != "local_patch_attempt" or source.parent.name != "variant":
        return False, "unexpected_local_patch_attempt_path"
    if any(source.rglob("fallback_to_*.json")):
        return False, "fallback_diagnostic_present"
    if not _final_state_is_published(source.parent):
        return False, "final_variant_not_currently_published"
    return True, "eligible"


def _remove_local_patch_attempt(source: Path, *, execute: bool) -> dict[str, object]:
    """Retain JSON safety evidence and then remove one successful attempt tree."""
    variant_dir = source.parent
    summary_dir = variant_dir / "summary"
    bytes_before = _allocated_size_bytes(source)
    record: dict[str, object] = {
        "attempt_path": str(source),
        "attempt_bytes_before": bytes_before,
        "action": "planned_local_patch_removal" if not execute else "",
        "bytes_reclaimed": 0,
        "attempt_removed": False,
    }
    if not execute:
        return record

    audit_dir = summary_dir / "local_patch_attempt_audits"
    audits = _copy_json_audits(source, audit_dir)
    newly_retained_bytes = sum(
        int(item["bytes"])
        for item in audits
        if not bool(item["target_preexisted"])
    )
    cleanup_manifest = {
        "schema_version": "local-patch-attempt-cleanup-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_attempt": str(source.resolve()),
        "source_attempt_bytes": bytes_before,
        "validation": {
            "final_variant_published": _final_state_is_published(variant_dir),
            "fallback_diagnostic_present": False,
        },
        "retained_final_outputs": _final_output_audit(variant_dir),
        "retained_json_audits": audits,
    }
    summary_dir.mkdir(parents=True, exist_ok=True)
    cleanup_manifest_path = summary_dir / "local_patch_attempt_cleanup_manifest.json"
    cleanup_manifest_path.write_text(
        json.dumps(cleanup_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    _remove_tree_ignoring_disappeared_children(source)
    if source.exists():
        raise RuntimeError(f"Local-patch attempt still exists after cleanup: {source}")
    record.update(
        {
            "action": "retained_json_audits_and_removed_attempt",
            "attempt_removed": True,
            "retained_json_audit_bytes": newly_retained_bytes,
            "cleanup_manifest": str(cleanup_manifest_path.resolve()),
            "bytes_reclaimed": bytes_before - newly_retained_bytes,
        }
    )
    return record


def _convert_width_samples_csv(
    source: Path,
    *,
    execute: bool,
) -> dict[str, object]:
    """Plan or atomically convert one detailed-width table with verification."""
    target = source.with_suffix(".parquet")
    target_existed_before = target.exists()
    record: dict[str, object] = {
        "source_csv": str(source),
        "parquet_path": str(target),
        "source_csv_bytes": int(source.stat().st_size),
        "source_csv_sha256": _sha256(source),
        "action": "planned_conversion" if not execute else "",
        "bytes_reclaimed": 0,
        "source_removed": False,
    }
    if not execute:
        return record

    source_frame = pd.read_csv(source, low_memory=False)
    if target.exists():
        _frames_are_equivalent(source_frame, pd.read_parquet(target))
        record.update(
            {
                "action": "verified_existing_parquet_and_removed_csv",
                "parquet_bytes": int(target.stat().st_size),
                "parquet_sha256": _sha256(target),
            }
        )
    else:
        temporary = target.with_name(
            f".{target.stem}.{uuid.uuid4().hex}.tmp.parquet"
        )
        try:
            source_frame.to_parquet(temporary, index=False, compression="zstd")
            _frames_are_equivalent(source_frame, pd.read_parquet(temporary))
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()
        record.update(
            {
                "action": "converted_verified_and_removed_csv",
                "parquet_bytes": int(target.stat().st_size),
                "parquet_sha256": _sha256(target),
            }
        )

    source_bytes = int(source.stat().st_size)
    parquet_bytes = int(target.stat().st_size)
    if not target_existed_before and parquet_bytes >= source_bytes:
        target.unlink()
        record.update(
            {
                "action": "retained_csv_parquet_not_smaller",
                "parquet_bytes": parquet_bytes,
                "bytes_reclaimed": 0,
                "source_removed": False,
            }
        )
        return record
    source.unlink()
    record["bytes_reclaimed"] = source_bytes if target_existed_before else source_bytes - parquet_bytes
    record["source_removed"] = True
    return record


def execute_cleanup_manifest(
    manifest_path: str | Path,
    *,
    execute: bool = False,
) -> CleanupExecution:
    """Apply only verified width compaction and successful-local-attempt removal.

    ``execute=False`` is a dry run. With ``execute=True``, a source CSV is
    unlinked only after an equivalent Parquet file has been written atomically
    and read back. Local attempts are removed only after their JSON safety
    audits and a final-output manifest have been retained. Raw-mask and
    RivGraph rows are always reported as out of scope and never actioned here.
    """
    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Cleanup manifest was not found: {manifest}")
    rows = pd.read_csv(manifest)
    execution_rows: list[dict[str, object]] = []
    for index, row in rows.iterrows():
        source_value = row.get("path", "")
        source_text = "" if pd.isna(source_value) else str(source_value).strip()
        base = {
            "manifest_path": str(manifest),
            "manifest_row": int(index),
            "state_id": str(row.get("state_id", "")),
            "artifact_category": str(row.get("artifact_category", "")),
            "recommended_action": str(row.get("recommended_action", "")),
            "source_path": source_text,
            "execute_requested": bool(execute),
            "outcome": "",
            "detail": "",
            "bytes_reclaimed": 0,
        }
        action = str(row.get("recommended_action", ""))
        if action not in {_WIDTH_SAMPLE_ACTION, _LOCAL_PATCH_ACTION}:
            execution_rows.append(
                {
                    **base,
                    "outcome": "out_of_scope_not_actioned",
                    "detail": "Raw mask and RivGraph cleanup is intentionally out of scope.",
                }
            )
            continue
        if not source_text:
            execution_rows.append({**base, "outcome": "skipped", "detail": "Missing source path."})
            continue
        source = Path(source_text).expanduser()
        if not source.exists():
            execution_rows.append({**base, "outcome": "skipped", "detail": "Source CSV no longer exists."})
            continue
        safe, reason = (
            _width_conversion_is_safe(row, source)
            if action == _WIDTH_SAMPLE_ACTION
            else _local_patch_removal_is_safe(row, source)
        )
        if not safe:
            execution_rows.append({**base, "outcome": "skipped", "detail": reason})
            continue
        try:
            conversion = (
                _convert_width_samples_csv(source, execute=execute)
                if action == _WIDTH_SAMPLE_ACTION
                else _remove_local_patch_attempt(source, execute=execute)
            )
            source_removed = bool(
                conversion.get("source_removed", conversion.get("attempt_removed", False))
            )
            execution_rows.append(
                {
                    **base,
                    **conversion,
                    "outcome": (
                        "converted"
                        if execute and source_removed
                        and action == _WIDTH_SAMPLE_ACTION
                        else "removed_local_patch_attempt"
                        if execute and source_removed
                        else "retained_not_smaller"
                        if execute
                        else "planned"
                    ),
                    "detail": (
                        "Parquet content verification is required before source removal."
                        if action == _WIDTH_SAMPLE_ACTION
                        else "Final outputs and JSON safety audits were retained before removal."
                    ),
                }
            )
        except Exception as exc:
            execution_rows.append(
                {
                    **base,
                    "outcome": "failed",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )

    execution_frame = pd.DataFrame(execution_rows)
    execution_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_stem = manifest.stem + f"_execution_{execution_stamp}"
    output_csv = manifest.with_name(output_stem + ".csv")
    output_json = manifest.with_name(output_stem + ".json")
    execution_frame.to_csv(output_csv, index=False)
    summary = {
        "schema_version": "artifact-cleanup-execution-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(manifest),
        "execute_requested": bool(execute),
        "n_rows": int(len(execution_frame)),
        "n_planned": int(execution_frame["outcome"].eq("planned").sum()) if not execution_frame.empty else 0,
        "n_converted": int(execution_frame["outcome"].eq("converted").sum()) if not execution_frame.empty else 0,
        "n_local_patch_attempts_removed": int(execution_frame["outcome"].eq("removed_local_patch_attempt").sum()) if not execution_frame.empty else 0,
        "n_retained_not_smaller": int(execution_frame["outcome"].eq("retained_not_smaller").sum()) if not execution_frame.empty else 0,
        "n_failed": int(execution_frame["outcome"].eq("failed").sum()) if not execution_frame.empty else 0,
        "bytes_reclaimed": int(execution_frame["bytes_reclaimed"].sum()) if not execution_frame.empty else 0,
        "execution_csv": str(output_csv),
        "destructive_action_performed": bool(
            execute
            and not execution_frame.empty
            and execution_frame["outcome"].isin(["converted", "removed_local_patch_attempt"]).any()
        ),
        "scope": "verified width_samples_csv conversion and successful local_patch_attempt cleanup only",
    }
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return CleanupExecution(rows=execution_frame, summary=summary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a cleanup plan or safely execute verified width-sample compaction.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--experiment-dir")
    target.add_argument("--output-root")
    target.add_argument("--manifest-path", help="Existing cleanup CSV manifest to inspect or execute.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute only verified width_samples_csv CSV-to-Parquet conversions from --manifest-path.",
    )
    args = parser.parse_args(argv)
    if args.execute and args.manifest_path is None:
        parser.error("--execute requires --manifest-path.")
    if args.manifest_path is not None:
        if args.output_dir is not None:
            parser.error("--output-dir is not supported with --manifest-path.")
        result = execute_cleanup_manifest(args.manifest_path, execute=args.execute)
    elif args.output_root is not None:
        if args.execute:
            parser.error("--execute requires --manifest-path.")
        if args.output_dir is not None:
            parser.error("--output-dir is not supported with --output-root.")
        result = write_output_cleanup_manifest(args.output_root)
    else:
        result = write_cleanup_manifest(args.experiment_dir, output_dir=args.output_dir)
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
