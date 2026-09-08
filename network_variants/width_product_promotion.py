from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4


PROMOTION_VERSION = "width-product-promotion-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_copy(source: Path, target: Path, expected_sha256: str) -> None:
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    shutil.copy2(source, temporary)
    actual = _sha256(temporary)
    if actual != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"Staged copy hash mismatch for {source}: expected={expected_sha256} actual={actual}"
        )
    os.replace(temporary, target)


def build_width_promotion_plan(
    repair_workspace: str | Path,
    target_experiment: str | Path | None = None,
) -> dict[str, Any]:
    repair = Path(repair_workspace).expanduser().resolve()
    repair_manifest_path = repair / "width_product_repair_manifest.json"
    manifest = _load_json(repair_manifest_path)
    if manifest.get("is_valid") is not True:
        raise ValueError(f"Repair workspace is not valid: {repair}")
    target = (
        Path(target_experiment).expanduser().resolve()
        if target_experiment is not None
        else Path(str(manifest["source_experiment"])).resolve()
    )
    if target != Path(str(manifest["source_experiment"])).resolve():
        raise ValueError("The requested target differs from the repair manifest source experiment.")
    production_states = {
        path.name for path in (target / "states").iterdir()
        if path.is_dir()
    }
    repair_states = {str(state["state_id"]) for state in manifest["states"]}
    if repair_states != production_states:
        raise ValueError(
            "Promotion requires a complete experiment repair; state sets differ: "
            f"missing={sorted(production_states - repair_states)} "
            f"extra={sorted(repair_states - production_states)}"
        )

    items: list[dict[str, Any]] = []
    for state in manifest["states"]:
        if (state.get("validation") or {}).get("is_valid") is not True:
            raise ValueError(f"State repair is invalid: {state['state_id']}")
        state_id = str(state["state_id"])
        output_paths = dict(state["output_paths"])
        hashes = dict(state["file_sha256"])
        files: list[dict[str, str]] = []
        for key, source_text in output_paths.items():
            source = Path(source_text)
            if not source.exists():
                raise FileNotFoundError(f"Staged repair output is missing: {source}")
            actual = _sha256(source)
            if actual != hashes[key]:
                raise ValueError(f"Staged repair output changed after validation: {source}")
            target_path = target / "states" / state_id / "variant" / "widths" / source.name
            files.append({
                "key": key,
                "source": str(source),
                "target": str(target_path),
                "sha256": actual,
            })
        variant_manifest = (
            target / "states" / state_id / "variant" / "summary" / "variant_manifest.json"
        )
        if not variant_manifest.exists():
            raise FileNotFoundError(f"Variant manifest is missing: {variant_manifest}")
        items.append({
            "state_id": state_id,
            "material_change_count": int(state["material_change_count"]),
            "validation": state["validation"],
            "files": files,
            "variant_manifest": str(variant_manifest),
        })
    return {
        "promotion_version": PROMOTION_VERSION,
        "repair_workspace": str(repair),
        "repair_manifest": str(repair_manifest_path),
        "repair_manifest_sha256": _sha256(repair_manifest_path),
        "target_experiment": str(target),
        "state_count": len(items),
        "states": items,
    }


def apply_width_promotion_plan(
    plan: dict[str, Any],
    *,
    backup_root: str | Path,
) -> Path:
    target = Path(plan["target_experiment"])
    backup = Path(backup_root).expanduser().resolve() / target.name
    if backup.exists():
        raise FileExistsError(f"Promotion backup already exists: {backup}")
    backup.mkdir(parents=True)
    promoted_at = datetime.now(timezone.utc).isoformat()

    experiment_manifest_path = target / "experiment_manifest.json"
    promotion_path = target / "width_product_repair_promotion_manifest.json"
    rapid_refresh_path = target / "rapid_width_refresh_required.json"
    shutil.copy2(experiment_manifest_path, backup / "experiment_manifest.json")
    prior_promotion_backup = backup / "width_product_repair_promotion_manifest.json"
    if promotion_path.exists():
        shutil.copy2(promotion_path, prior_promotion_backup)
    prior_rapid_refresh_backup = backup / "rapid_width_refresh_required.json"
    if rapid_refresh_path.exists():
        shutil.copy2(rapid_refresh_path, prior_rapid_refresh_backup)
    promoted_targets: list[tuple[Path, Path | None]] = []
    try:
        for state in plan["states"]:
            state_id = state["state_id"]
            state_backup = backup / "states" / state_id
            width_backup = state_backup / "variant" / "widths"
            manifest_backup = state_backup / "variant" / "summary" / "variant_manifest.json"
            width_backup.mkdir(parents=True)
            manifest_backup.parent.mkdir(parents=True)
            variant_manifest_path = Path(state["variant_manifest"])
            shutil.copy2(variant_manifest_path, manifest_backup)

            for item in state["files"]:
                source = Path(item["source"])
                production = Path(item["target"])
                old_backup: Path | None = None
                if production.exists():
                    old_backup = width_backup / production.name
                    shutil.copy2(production, old_backup)
                _atomic_copy(source, production, item["sha256"])
                promoted_targets.append((production, old_backup))

            routing_manifest = _load_json(
                next(
                    Path(item["target"])
                    for item in state["files"]
                    if item["key"] == "routing_width_application_manifest"
                )
            )
            variant_manifest = _load_json(variant_manifest_path)
            variant_manifest["routing_width_application"] = routing_manifest
            variant_manifest["width_product_repair"] = {
                "promotion_version": PROMOTION_VERSION,
                "promoted_at": promoted_at,
                "repair_manifest_sha256": plan["repair_manifest_sha256"],
                "material_change_count": state["material_change_count"],
                "validation": state["validation"],
                "backup_state_dir": str(state_backup),
            }
            output_paths = variant_manifest.setdefault("output_paths", {})
            for item in state["files"]:
                if item["key"] in {
                    "links_with_width_families",
                    "link_width_families",
                    "link_width_samples",
                    "routing_width_application",
                    "routing_width_application_manifest",
                }:
                    output_paths[item["key"]] = item["target"]
            _atomic_json(variant_manifest, variant_manifest_path)

        experiment_manifest = _load_json(experiment_manifest_path)
        experiment_manifest["width_product_repair"] = {
            "promotion_version": PROMOTION_VERSION,
            "promoted_at": promoted_at,
            "state_count": plan["state_count"],
            "repair_manifest_sha256": plan["repair_manifest_sha256"],
            "backup_dir": str(backup),
            "rapid_refresh_required": bool((target / "rapid_prep_manifest.json").exists()),
        }
        _atomic_json(experiment_manifest, experiment_manifest_path)

        production_manifest = {
            **plan,
            "promoted_at": promoted_at,
            "backup_dir": str(backup),
            "is_valid": True,
        }
        _atomic_json(production_manifest, promotion_path)
        if (target / "rapid_prep_manifest.json").exists():
            _atomic_json(
                {
                    "status": "required",
                    "reason": "routing widths changed after the recorded RAPID preparation",
                    "promotion_version": PROMOTION_VERSION,
                    "promoted_at": promoted_at,
                    "repair_manifest_sha256": plan["repair_manifest_sha256"],
                    "previous_rapid_prep_manifest_sha256": _sha256(
                        target / "rapid_prep_manifest.json"
                    ),
                },
                rapid_refresh_path,
            )

        for state in plan["states"]:
            for item in state["files"]:
                actual = _sha256(Path(item["target"]))
                if actual != item["sha256"]:
                    raise ValueError(f"Promoted output hash mismatch: {item['target']}")
        return backup
    except Exception:
        for production, old_backup in reversed(promoted_targets):
            if old_backup is None:
                production.unlink(missing_ok=True)
            elif old_backup.exists():
                shutil.copy2(old_backup, production)
        for state in plan["states"]:
            original = backup / "states" / state["state_id"] / "variant" / "summary" / "variant_manifest.json"
            if original.exists():
                shutil.copy2(original, Path(state["variant_manifest"]))
        shutil.copy2(backup / "experiment_manifest.json", experiment_manifest_path)
        if prior_promotion_backup.exists():
            shutil.copy2(prior_promotion_backup, promotion_path)
        else:
            promotion_path.unlink(missing_ok=True)
        if prior_rapid_refresh_backup.exists():
            shutil.copy2(prior_rapid_refresh_backup, rapid_refresh_path)
        else:
            rapid_refresh_path.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Promote validated staged width products safely.")
    parser.add_argument("repair_workspace", type=Path)
    parser.add_argument("--target-experiment", type=Path)
    parser.add_argument("--backup-root", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    plan = build_width_promotion_plan(args.repair_workspace, args.target_experiment)
    print(json.dumps({
        "repair_workspace": plan["repair_workspace"],
        "target_experiment": plan["target_experiment"],
        "state_count": plan["state_count"],
        "file_count": sum(len(state["files"]) for state in plan["states"]),
        "repair_manifest_sha256": plan["repair_manifest_sha256"],
    }, indent=2))
    if not args.apply:
        print("Dry run only. Re-run with --apply and --backup-root to promote.")
        return 0
    if args.backup_root is None:
        parser.error("--backup-root is required with --apply")
    backup = apply_width_promotion_plan(plan, backup_root=args.backup_root)
    print(f"Promotion complete. Recoverable backup: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
