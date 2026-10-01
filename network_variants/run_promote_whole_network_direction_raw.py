"""Transactionally replace only a raw RivGraph orientation after preflight.

The command preserves the former ``rivgraph/`` directory under the same run
root, reruns RivGraph on the unchanged cleaned mask with the corrected exits,
and verifies the permanent width-review continuity record.  It deliberately
does not touch any experiment output: those must be rebuilt into versioned
roots before the canonical experiment aliases can be switched.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RIVGRAPH_SOURCE = ROOT / "rivgraph_centerline" / "src"
if str(RIVGRAPH_SOURCE) not in sys.path:
    sys.path.insert(0, str(RIVGRAPH_SOURCE))

from rivgraph_centerline.workflow import run_rivgraph_for_run
from network_variants.width_review_archive import review_archive_dir
from network_variants.width_review_fingerprint import (
    DIRECTION_ONLY_CURRENT_STATUS,
    validate_direction_only_review_continuity,
)


SCHEMA_VERSION = "whole-network-direction-raw-promotion-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plan(
    run_dir: Path,
    *,
    example_id: str,
    prior_exit_sides: str,
    corrected_exit_sides: str,
    allow_interrupted_recovery: bool = False,
) -> dict[str, object]:
    raw_dir = run_dir / "rivgraph"
    links = raw_dir / f"{example_id}_links.gpkg"
    nodes = raw_dir / f"{example_id}_nodes.gpkg"
    mask = run_dir / "masks_cleaned" / f"{example_id}_cleaned.tif"
    archive = review_archive_dir(links)
    for label, path in {"cleaned mask": mask}.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    raw_missing = not links.exists() or not nodes.exists()
    if raw_missing and not allow_interrupted_recovery:
        raise FileNotFoundError(
            "Raw links/nodes are missing. Refuse to overwrite an interrupted promotion without "
            "--allow-interrupted-recovery."
        )
    continuity = archive / "direction_only_continuity.json"
    if not continuity.exists():
        raise FileNotFoundError(
            "Direction promotion requires the explicit review-continuity record: "
            f"{continuity}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "example_id": example_id,
        "run_dir": str(run_dir),
        "raw_rivgraph_dir": str(raw_dir),
        "cleaned_mask": str(mask),
        "cleaned_mask_sha256": _sha256(mask),
        "prior_exit_sides": prior_exit_sides,
        "corrected_exit_sides": corrected_exit_sides,
        "review_archive": str(archive),
        "direction_continuity_record": str(continuity),
        "source_raw_links_sha256": _sha256(links) if links.exists() else None,
        "source_raw_nodes_sha256": _sha256(nodes) if nodes.exists() else None,
        "interrupted_recovery": bool(raw_missing),
        "action": "archive_old_raw_rivgraph_then_rerun_from_unchanged_cleaned_mask",
        "does_not": [
            "modify the cleaned mask",
            "delete the prior raw RivGraph output",
            "modify, rename, or alias any experiment output root",
        ],
    }


def _validated_candidate_paths(candidate_dir: Path, example_id: str) -> dict[str, Path]:
    """Resolve the standard raw products from a validated trial RivGraph directory."""
    suffixes = {
        "links": "_links.gpkg",
        "nodes": "_nodes.gpkg",
        "centerline": "_centerline.gpkg",
        "skeleton": "_skel.tif",
        "directions": "_link_directions.tif",
        "fixlinks": "_fixlinks.csv",
        "log": "_log.log",
    }
    resolved: dict[str, Path] = {}
    for label, suffix in suffixes.items():
        matches = sorted(candidate_dir.glob(f"*{suffix}"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Validated candidate must contain exactly one {label} matching *{suffix}: {candidate_dir}"
            )
        resolved[label] = matches[0]
    return resolved


def _execute(plan: dict[str, object], *, validated_candidate_rivgraph_dir: Path | None = None) -> dict[str, object]:
    run_dir = Path(str(plan["run_dir"]))
    raw_dir = Path(str(plan["raw_rivgraph_dir"]))
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = run_dir / "direction_promotion_backups" / timestamp / "rivgraph"
    record = run_dir / f"direction_raw_promotion_{timestamp}.json"
    if backup.exists() or record.exists():
        raise FileExistsError("Direction promotion backup/record already exists; retry to create a new plan.")
    backup.parent.mkdir(parents=True, exist_ok=False)
    moved = False
    try:
        if raw_dir.exists():
            os.replace(raw_dir, backup)
            moved = True
        if validated_candidate_rivgraph_dir is None:
            _, rivgraph_summary = run_rivgraph_for_run(
                name=str(plan["example_id"]),
                run_dir=run_dir,
                mask=Path(str(plan["cleaned_mask"])),
                mask_stage="cleaned",
                exit_sides=str(plan["corrected_exit_sides"]),
                assign_flow=False,
                verbose=True,
                clean_target=False,
            )
        else:
            candidate = validated_candidate_rivgraph_dir.expanduser().resolve()
            products = _validated_candidate_paths(candidate, str(plan["example_id"]))
            raw_dir.mkdir(parents=True, exist_ok=False)
            destinations = {
                "links": raw_dir / f"{plan['example_id']}_links.gpkg",
                "nodes": raw_dir / f"{plan['example_id']}_nodes.gpkg",
                "centerline": raw_dir / f"{plan['example_id']}_centerline.gpkg",
                "skeleton": raw_dir / f"{plan['example_id']}_skel.tif",
                "directions": raw_dir / f"{plan['example_id']}_link_directions.tif",
                "fixlinks": raw_dir / f"{plan['example_id']}_fixlinks.csv",
                "log": raw_dir / f"{plan['example_id']}_log.log",
            }
            for label, source in products.items():
                shutil.copy2(source, destinations[label])
            rivgraph_summary = {
                "raw_publication_method": "validated_trial_copy",
                "validated_candidate_rivgraph_dir": str(candidate),
                "validated_candidate_product_sha256": {
                    label: _sha256(source) for label, source in products.items()
                },
                "exit_sides": str(plan["corrected_exit_sides"]),
            }
        links = raw_dir / f"{plan['example_id']}_links.gpkg"
        nodes = raw_dir / f"{plan['example_id']}_nodes.gpkg"
        validation = validate_direction_only_review_continuity(
            Path(str(plan["review_archive"])),
            continuity_path=Path(str(plan["direction_continuity_record"])),
            mask_path=Path(str(plan["cleaned_mask"])),
            links_path=links,
            nodes_path=nodes,
        )
        if validation.get("status") != DIRECTION_ONLY_CURRENT_STATUS:
            raise ValueError(
                "Corrected raw run failed the direction-only continuity proof: "
                f"status={validation.get('status')}; issues={validation.get('issues')}; "
                f"continuity_issues={validation.get('direction_continuity_issues')}"
            )
        result = {
            **plan,
            "executed_at": datetime.now(UTC).isoformat(),
            "backup_raw_rivgraph_dir": str(backup),
            "corrected_raw_links_sha256": _sha256(links),
            "corrected_raw_nodes_sha256": _sha256(nodes),
            "rivgraph_summary": rivgraph_summary,
            "continuity_validation": validation["direction_continuity_audit"],
            "next_required_step": (
                "Rebuild independent, compound, and island experiment roots into versioned corrected "
                "directories, validate them, then archive old canonical roots and alias the corrected roots."
            ),
        }
        record.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {**result, "promotion_record": str(record)}
    except Exception:
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        if moved and backup.exists() and not raw_dir.exists():
            os.replace(backup, raw_dir)
        if backup.parent.exists() and not any(backup.parent.iterdir()):
            backup.parent.rmdir()
            parent = backup.parent.parent
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Canonical rivgraph_centerline output run directory.")
    parser.add_argument("--example-id", required=True)
    parser.add_argument("--prior-exit-sides", required=True)
    parser.add_argument("--corrected-exit-sides", required=True)
    parser.add_argument(
        "--validated-candidate-rivgraph-dir",
        type=Path,
        default=None,
        help=(
            "Optional already-validated trial raw RivGraph directory. When supplied, "
            "its canonical products are copied under the standard raw names instead of rerunning RivGraph."
        ),
    )
    parser.add_argument(
        "--allow-interrupted-recovery",
        action="store_true",
        help=(
            "Recover a missing raw rivgraph/ directory only from a supplied validated candidate. "
            "The record will explicitly retain this exceptional recovery status."
        ),
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    plan = _plan(
        Path(args.run_dir).expanduser().resolve(),
        example_id=str(args.example_id).zfill(2),
        prior_exit_sides=str(args.prior_exit_sides).upper(),
        corrected_exit_sides=str(args.corrected_exit_sides).upper(),
        allow_interrupted_recovery=bool(args.allow_interrupted_recovery),
    )
    result = (
        _execute(plan, validated_candidate_rivgraph_dir=args.validated_candidate_rivgraph_dir)
        if args.execute
        else {"execute_requested": False, **plan}
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
