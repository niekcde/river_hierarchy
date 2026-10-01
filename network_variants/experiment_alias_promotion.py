"""Safely expose a corrected experiment under its canonical output name.

Corrected experiment roots often contain absolute paths to their own versioned
directory.  Renaming those roots would invalidate that provenance.  This
module instead archives the superseded canonical roots and creates canonical
directory symlinks to the corrected physical roots.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
from typing import Any


PROMOTION_SCHEMA_VERSION = "experiment-canonical-alias-promotion-v1"


def _names(example_id: int | str, correction_label: str) -> list[tuple[str, str]]:
    example = f"{int(example_id):02d}"
    label = str(correction_label).strip()
    if not label or any(character in label for character in "/\\"):
        raise ValueError("correction_label must be a non-empty directory-name fragment.")
    return [
        (f"{example}_indep__anchored", f"{example}_indep_{label}__anchored"),
        (
            f"{example}_compound_bubbles__anchored",
            f"{example}_compound_bubbles_{label}__anchored",
        ),
        (f"{example}_island__anchored", f"{example}_island_{label}__anchored"),
    ]


def build_experiment_alias_promotion_plan(
    output_root: str | Path,
    *,
    example_id: int | str,
    correction_label: str = "direction130",
    archive_root_name: str = "_archived_pre_correction",
) -> dict[str, Any]:
    """Validate a non-destructive canonical-name alias promotion plan."""
    root = Path(output_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Output root is not a directory: {root}")
    archive_name = str(archive_root_name).strip()
    if not archive_name or Path(archive_name).name != archive_name:
        raise ValueError("archive_root_name must be a single, non-empty directory name.")

    mappings: list[dict[str, Any]] = []
    invalid: list[str] = []
    for canonical_name, corrected_name in _names(example_id, correction_label):
        canonical = root / canonical_name
        corrected = root / corrected_name
        if not corrected.is_dir() or corrected.is_symlink():
            invalid.append(f"Corrected physical root is missing or is not a directory: {corrected}")
            continue
        if canonical.is_symlink():
            try:
                resolved = canonical.resolve(strict=True)
            except FileNotFoundError:
                invalid.append(f"Canonical alias is broken: {canonical}")
                continue
            if resolved != corrected.resolve():
                invalid.append(
                    f"Canonical alias points somewhere unexpected: {canonical} -> {resolved}; "
                    f"expected {corrected}"
                )
                continue
            action = "already_promoted"
        elif canonical.exists():
            if not canonical.is_dir():
                invalid.append(f"Canonical root is not a directory: {canonical}")
                continue
            action = "archive_then_alias"
        else:
            action = "alias_without_archive"
        mappings.append({
            "canonical_name": canonical_name,
            "canonical_path": str(canonical),
            "corrected_name": corrected_name,
            "corrected_path": str(corrected),
            "action": action,
        })

    if invalid:
        raise ValueError("Canonical alias promotion preflight failed:\n- " + "\n- ".join(invalid))
    return {
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "output_root": str(root),
        "example_id": int(example_id),
        "correction_label": str(correction_label),
        "archive_root_name": archive_name,
        "mode": "archive_old_canonical_roots_then_create_canonical_symlink_aliases",
        "rationale": (
            "Corrected physical roots retain their versioned names because experiment manifests and "
            "registries contain absolute paths to those roots. Canonical names are symlink aliases."
        ),
        "mappings": mappings,
    }


def apply_experiment_alias_promotion(plan: dict[str, Any]) -> dict[str, Any]:
    """Archive old roots and atomically create canonical aliases where possible.

    All directory renames happen below the same output root, so no experiment
    artifacts are copied or deleted.  If a later alias cannot be created, the
    function removes newly created aliases and restores moved canonical roots.
    """
    root = Path(str(plan["output_root"])).expanduser().resolve()
    mappings = list(plan.get("mappings") or [])
    if not mappings:
        raise ValueError("Promotion plan has no mappings.")
    archive_root = root / str(plan["archive_root_name"])
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = archive_root / f"{int(plan['example_id']):02d}_{timestamp}"
    manifest_path = root / f"{int(plan['example_id']):02d}_{plan['correction_label']}_canonical_alias_promotion.json"

    to_archive = [item for item in mappings if item["action"] == "archive_then_alias"]
    to_alias = [item for item in mappings if item["action"] != "already_promoted"]
    for item in to_archive:
        source = Path(str(item["canonical_path"]))
        if not source.is_dir() or source.is_symlink():
            raise ValueError(f"Canonical root changed after preflight: {source}")
    for item in to_alias:
        target = Path(str(item["corrected_path"]))
        canonical = Path(str(item["canonical_path"]))
        canonical_is_expected_old_root = item["action"] == "archive_then_alias" and canonical.is_dir()
        if (
            not target.is_dir()
            or target.is_symlink()
            or (canonical.exists() and not canonical_is_expected_old_root)
        ):
            raise ValueError(f"Promotion target changed after preflight: {canonical} -> {target}")
    if archive_dir.exists():
        raise FileExistsError(f"Archive destination already exists: {archive_dir}")
    if manifest_path.exists():
        raise FileExistsError(
            f"A canonical alias promotion manifest already exists: {manifest_path}. "
            "Inspect it before attempting another promotion."
        )

    moved: list[tuple[Path, Path]] = []
    aliases: list[Path] = []
    try:
        if to_archive:
            archive_dir.mkdir(parents=True)
            for item in to_archive:
                canonical = Path(str(item["canonical_path"]))
                archived = archive_dir / canonical.name
                os.replace(canonical, archived)
                moved.append((canonical, archived))
                item["archived_prior_canonical_path"] = str(archived)
        for item in to_alias:
            canonical = Path(str(item["canonical_path"]))
            corrected = Path(str(item["corrected_path"]))
            # Both entries have the same parent. A relative target keeps the
            # alias portable if the whole output root is relocated together.
            canonical.symlink_to(corrected.name, target_is_directory=True)
            aliases.append(canonical)
            item["canonical_alias_target"] = corrected.name
        result = dict(plan)
        result.update({
            "executed_at": datetime.now(UTC).isoformat(),
            "execute_requested": True,
            "archive_directory": str(archive_dir) if to_archive else "",
            "manifest_path": str(manifest_path),
            "destructive_action_performed": False,
            "copied_or_deleted_artifacts": False,
        })
        manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result
    except Exception:
        for alias in reversed(aliases):
            if alias.is_symlink():
                alias.unlink()
        for canonical, archived in reversed(moved):
            if archived.exists() and not canonical.exists():
                os.replace(archived, canonical)
        if archive_dir.exists() and not any(archive_dir.iterdir()):
            archive_dir.rmdir()
        raise
