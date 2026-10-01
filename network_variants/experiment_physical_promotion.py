"""Atomically promote corrected experiment directories to canonical names.

Unlike :mod:`experiment_alias_promotion`, this module performs a physical
rename.  It is deliberately conservative: the former canonical trees are
archived, no artifacts are deleted, and every textual absolute-path reference
to a corrected root is rewritten to the newly canonical root.
"""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


PROMOTION_SCHEMA_VERSION = "experiment-physical-promotion-v1"
FAMILIES = ("indep", "compound_bubbles", "island")
TEXT_SUFFIXES = {".csv", ".json", ".md", ".txt", ".yaml", ".yml", ".log"}


def _root_name(example_id: int | str, family: str, correction_label: str | None) -> str:
    example = f"{int(example_id):02d}"
    if family not in FAMILIES:
        raise ValueError(f"Unknown experiment family: {family}")
    label = f"_{correction_label}" if correction_label else ""
    return f"{example}_{family}{label}__anchored"


def _assert_complete(root: Path, family: str) -> None:
    """Require the lightweight completion evidence emitted by rebuild runners."""
    if family == "island":
        manifest = root / "island_analysis_manifest.csv"
        if not manifest.is_file():
            raise FileNotFoundError(f"Island promotion source lacks analysis manifest: {manifest}")
        return
    record = root / "direction_corrected_rebuild.json"
    if not record.is_file():
        raise FileNotFoundError(f"Corrected promotion source lacks rebuild record: {record}")
    payload = json.loads(record.read_text(encoding="utf-8"))
    if payload.get("completed") is not True:
        raise ValueError(f"Corrected promotion source is not marked complete: {record}")
    if int(payload.get("n_failed_states", -1)) != 0:
        raise ValueError(f"Corrected promotion source has failed states: {record}")


def _text_reference_files(root: Path, replacements: dict[bytes, bytes]) -> list[dict[str, Any]]:
    """Return planned rewrites, without touching files.

    GPKGs and rasters are also byte-scanned. A literal versioned-root path in
    one would be unsafe to rewrite blindly, so it is a hard preflight error.
    """
    changes: list[dict[str, Any]] = []
    binary_hits: list[Path] = []
    for path in root.rglob("*"):
        # macOS may create AppleDouble metadata sidecars (``._name``) beside
        # real artifacts on external volumes. They are binary metadata, not
        # project provenance files.
        if not path.is_file() or path.is_symlink() or path.name.startswith("._"):
            continue
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise OSError(f"Could not inspect promotion artifact {path}: {exc}") from exc
        count = sum(data.count(old) for old in replacements)
        if not count:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            binary_hits.append(path)
            continue
        updated = data
        for old, new in replacements.items():
            updated = updated.replace(old, new)
        changes.append({
            "path": str(path),
            "references_rewritten": count,
            "sha256_before": hashlib.sha256(data).hexdigest(),
            "sha256_after": hashlib.sha256(updated).hexdigest(),
        })
    if binary_hits:
        raise ValueError(
            "Versioned-root references occur in non-text artifacts; refusing an unsafe physical "
            f"promotion: {[str(path) for path in binary_hits]}"
        )
    return changes


def build_experiment_physical_promotion_plan(
    output_root: str | Path,
    *,
    example_ids: list[int] | tuple[int, ...],
    correction_label: str = "directionfix_20260923",
    archive_root_name: str = "_archived_pre_directionfix",
) -> dict[str, Any]:
    """Build and validate a no-write physical-promotion plan."""
    root = Path(output_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Output root is not a directory: {root}")
    if not correction_label or any(char in correction_label for char in "/\\"):
        raise ValueError("correction_label must be a non-empty directory-name fragment.")
    if not archive_root_name or Path(archive_root_name).name != archive_root_name:
        raise ValueError("archive_root_name must be a single directory name.")
    examples = sorted({int(value) for value in example_ids})
    if not examples:
        raise ValueError("At least one example_id is required.")

    mappings: list[dict[str, Any]] = []
    replacements: dict[bytes, bytes] = {}
    for example in examples:
        for family in FAMILIES:
            canonical = root / _root_name(example, family, None)
            corrected = root / _root_name(example, family, correction_label)
            if canonical.is_symlink():
                raise ValueError(f"Canonical root must be a physical directory, not a symlink: {canonical}")
            if not canonical.is_dir():
                raise FileNotFoundError(f"Superseded canonical root is missing: {canonical}")
            if corrected.is_symlink() or not corrected.is_dir():
                raise FileNotFoundError(f"Corrected physical root is missing: {corrected}")
            _assert_complete(corrected, family)
            mappings.append({
                "example_id": example,
                "family": family,
                "canonical_path": str(canonical),
                "corrected_path": str(corrected),
            })
            replacements[str(corrected).encode()] = str(canonical).encode()

    rewrites: list[dict[str, Any]] = []
    for item in mappings:
        corrected = Path(item["corrected_path"])
        for rewrite in _text_reference_files(corrected, replacements):
            rewrite["source_root"] = str(corrected)
            rewrites.append(rewrite)

    return {
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "output_root": str(root),
        "example_ids": examples,
        "correction_label": correction_label,
        "archive_root_name": archive_root_name,
        "mode": "archive_old_physical_roots_then_rename_corrected_roots_to_canonical_names",
        "rationale": (
            "Former canonical roots are retained intact under a dated archive. Corrected roots are "
            "physically renamed and textual provenance references are rewritten to canonical paths."
        ),
        "mappings": mappings,
        "planned_text_reference_rewrites": rewrites,
    }


def _atomic_write(path: Path, data: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(data)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def apply_experiment_physical_promotion(plan: dict[str, Any]) -> dict[str, Any]:
    """Execute a validated physical promotion, restoring all moves on error."""
    root = Path(str(plan["output_root"])).expanduser().resolve()
    mappings = list(plan.get("mappings") or [])
    if not mappings:
        raise ValueError("Promotion plan has no mappings.")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = root / str(plan["archive_root_name"]) / f"physical_promotion_{timestamp}"
    record_dir = root / "_promotion_records"
    record_path = record_dir / f"directionfix_physical_promotion_{timestamp}.json"
    if archive_dir.exists() or record_path.exists():
        raise FileExistsError("Promotion archive or record destination unexpectedly already exists.")

    archived: list[tuple[Path, Path]] = []
    promoted: list[tuple[Path, Path]] = []
    modified: list[tuple[Path, bytes]] = []
    replacements = {
        str(item["corrected_path"]).encode(): str(item["canonical_path"]).encode()
        for item in mappings
    }
    try:
        archive_dir.mkdir(parents=True)
        # First remove every old canonical path; the archive is entirely physical.
        for item in mappings:
            canonical = Path(item["canonical_path"])
            target = archive_dir / canonical.name
            if not canonical.is_dir() or canonical.is_symlink():
                raise ValueError(f"Canonical root changed after preflight: {canonical}")
            os.replace(canonical, target)
            archived.append((canonical, target))
            item["archived_prior_canonical_path"] = str(target)
        # Then expose every corrected tree at its canonical path.
        for item in mappings:
            corrected = Path(item["corrected_path"])
            canonical = Path(item["canonical_path"])
            if not corrected.is_dir() or corrected.is_symlink() or canonical.exists():
                raise ValueError(f"Corrected root changed after preflight: {corrected}")
            os.replace(corrected, canonical)
            promoted.append((corrected, canonical))
        # The plan paths describe pre-move sources. Resolve each rewrite by its
        # relative path within the corresponding moved root.
        for planned in plan.get("planned_text_reference_rewrites", []):
            original_root = Path(planned["source_root"])
            mapping = next(item for item in mappings if Path(item["corrected_path"]) == original_root)
            path = Path(mapping["canonical_path"]) / Path(planned["path"]).relative_to(original_root)
            before = path.read_bytes()
            after = before
            for old, new in replacements.items():
                after = after.replace(old, new)
            if after != before:
                modified.append((path, before))
                _atomic_write(path, after)
        # A post-move scan proves no versioned root reference remains in text.
        residual: list[str] = []
        for item in mappings:
            for path in Path(item["canonical_path"]).rglob("*"):
                if (
                    path.is_file()
                    and not path.is_symlink()
                    and not path.name.startswith("._")
                    and path.suffix.lower() in TEXT_SUFFIXES
                ):
                    data = path.read_bytes()
                    if any(old in data for old in replacements):
                        residual.append(str(path))
        if residual:
            raise ValueError(f"Residual corrected-root references after promotion: {residual}")

        result = dict(plan)
        result.update({
            "executed_at": datetime.now(UTC).isoformat(),
            "execute_requested": True,
            "archive_directory": str(archive_dir),
            "record_path": str(record_path),
            "destructive_action_performed": False,
            "copied_or_deleted_artifacts": False,
            "post_promotion_residual_text_references": 0,
        })
        record_dir.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result
    except Exception:
        for path, before in reversed(modified):
            if path.parent.exists():
                _atomic_write(path, before)
        for corrected, canonical in reversed(promoted):
            if canonical.exists() and not corrected.exists():
                os.replace(canonical, corrected)
        for canonical, target in reversed(archived):
            if target.exists() and not canonical.exists():
                os.replace(target, canonical)
        if archive_dir.exists() and not any(archive_dir.iterdir()):
            archive_dir.rmdir()
        raise
