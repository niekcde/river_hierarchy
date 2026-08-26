"""Safely promote targeted repair states into a complete experiment.

Filtered ``--unit-id`` experiments deliberately renumber their children.  A
repair state therefore cannot be copied into a complete experiment verbatim:
state/transition identifiers and absolute output paths must be rebased, root
registries must be updated, and derived experiment-level resistance tables
must be refreshed.  This module performs that operation transactionally and
uses dry-run mode by default through its CLI wrapper.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable, Sequence
from uuid import uuid4

import pandas as pd


TEXT_SUFFIXES = {".csv", ".json", ".log", ".md", ".txt", ".vrt", ".xml"}

STATE_IDENTITY_COLUMNS = {
    "state_id", "parent_state_id", "depth", "mode", "state_role",
    "selection_type", "collapsed_selection_label", "selected_unit_ids",
    "selected_group_label", "selected_group_size", "selected_rank_start",
    "selected_rank_end", "selected_collapse_order",
    "selected_collapse_priority_score", "is_compound_state",
    "compound_bubble_id", "bubble_root_unit_id", "total_unit_nodes_selected",
}

TRANSITION_IDENTITY_COLUMNS = {
    "transition_id", "step_index", "parent_state_id", "child_state_id",
    "depth", "mode", "selection_type", "collapsed_selection_label",
    "selected_unit_ids", "selected_group_label", "selected_group_size",
    "selected_rank_start", "selected_rank_end", "selected_collapse_order",
    "selected_collapse_priority_score", "is_compound_state",
    "compound_bubble_id", "bubble_root_unit_id", "total_unit_nodes_selected",
}

ROOT_METADATA_FILES = (
    "state_registry.csv",
    "transition_registry.csv",
    "equivalent_transition_registry.csv",
    "experiment_manifest.json",
)

EQUIVALENT_AGGREGATE_FILES = (
    "base_nominal_targets.csv",
    "child_realized_objects.csv",
    "matched_base_corridors.csv",
    "equivalent_resistance_comparison.csv",
    "equivalent_resistance_manifest.json",
)


@dataclass(frozen=True)
class PromotionItem:
    unit_id: int
    repair_state_id: str
    target_state_id: str
    repair_transition_id: str
    target_transition_id: str
    repair_state_dir: Path
    target_state_dir: Path


@dataclass(frozen=True)
class PromotionPlan:
    repair_experiment: Path
    target_experiment: Path
    items: tuple[PromotionItem, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "repair_experiment": str(self.repair_experiment),
            "target_experiment": str(self.target_experiment),
            "n_states": len(self.items),
            "items": [
                {
                    **asdict(item),
                    "repair_state_dir": str(item.repair_state_dir),
                    "target_state_dir": str(item.target_state_dir),
                }
                for item in self.items
            ],
        }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _unit_id(value: Any) -> int:
    if pd.isna(value):
        raise ValueError("A derived-state registry row has no selected unit ID.")
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        values = json.loads(text)
        if len(values) != 1:
            raise ValueError(f"Repair promotion requires one independent unit, got {values}.")
        return int(values[0])
    return int(float(text))


def _rows_by_unit(frame: pd.DataFrame, *, state_registry: bool) -> dict[int, pd.Series]:
    rows: dict[int, pd.Series] = {}
    for _, row in frame.iterrows():
        if state_registry and str(row.get("state_role", "")) == "base":
            continue
        unit = _unit_id(row["selected_unit_ids"])
        if unit in rows:
            raise ValueError(f"Registry contains duplicate rows for unit {unit}.")
        rows[unit] = row
    return rows


def _validate_local_repair_state(state_dir: Path) -> None:
    manifest_path = state_dir / "variant" / "summary" / "variant_manifest.json"
    direction_path = state_dir / "variant" / "directed" / "direction_validation_report.json"
    timing_path = state_dir / "stage_timings.json"
    for path in (manifest_path, direction_path, timing_path):
        if not path.exists():
            raise ValueError(f"Repair state is incomplete; required file is missing: {path}")
    manifest = _load_json(manifest_path)
    if manifest.get("execution_mode_used") != "local_patch":
        raise ValueError(
            f"Repair state {state_dir.name} used {manifest.get('execution_mode_used')!r}, "
            "not 'local_patch'."
        )
    if manifest.get("ready_for_graph_matching") is not True or manifest.get("ready_for_rapid") is not True:
        raise ValueError(f"Repair state {state_dir.name} is not ready for graph matching and RAPID.")
    local = manifest.get("local_patch") or {}
    required_statuses = {
        "planning": "safe",
        "raster": "safe",
        "realized_assembly": "complete_local_realized_graph",
        "shared_anchoring": "complete_shared_anchored_graph",
        "products": "complete_width_length_products",
    }
    for section, expected in required_statuses.items():
        actual = (local.get(section) or {}).get("status")
        if actual != expected:
            raise ValueError(
                f"Repair state {state_dir.name} has {section} status {actual!r}; expected {expected!r}."
            )
    if (local.get("collapse_topology_validation") or {}).get("passed") is not True:
        raise ValueError(f"Repair state {state_dir.name} failed collapse-topology validation.")
    if _load_json(direction_path).get("is_valid") is not True:
        raise ValueError(f"Repair state {state_dir.name} failed direction validation.")
    if _load_json(timing_path).get("hierarchy_status") != "complete":
        raise ValueError(f"Repair state {state_dir.name} has incomplete hierarchy processing.")
    missing = [
        key for key, value in (manifest.get("output_paths") or {}).items()
        if isinstance(value, str) and value and not Path(value).exists()
    ]
    if missing:
        raise ValueError(f"Repair state {state_dir.name} is missing manifest outputs: {missing}.")


def build_promotion_plan(
    repair_experiment: str | Path,
    target_experiment: str | Path,
    *,
    unit_ids: Sequence[int] | None = None,
) -> PromotionPlan:
    repair = Path(repair_experiment).expanduser().resolve()
    target = Path(target_experiment).expanduser().resolve()
    if repair == target:
        raise ValueError("Repair and target experiments must be different directories.")
    for root in (repair, target):
        for name in ROOT_METADATA_FILES:
            if not (root / name).exists():
                raise FileNotFoundError(f"Experiment metadata is missing: {root / name}")
    repair_manifest = _load_json(repair / "experiment_manifest.json")
    target_manifest = _load_json(target / "experiment_manifest.json")
    for key in ("mode", "example_id", "root_example_id", "exit_sides"):
        if str(repair_manifest.get(key)) != str(target_manifest.get(key)):
            raise ValueError(
                f"Repair experiment is incompatible for {key}: "
                f"repair={repair_manifest.get(key)!r}, target={target_manifest.get(key)!r}."
            )

    repair_states = pd.read_csv(repair / "state_registry.csv")
    target_states = pd.read_csv(target / "state_registry.csv")
    repair_transitions = pd.read_csv(repair / "transition_registry.csv")
    target_transitions = pd.read_csv(target / "transition_registry.csv")
    repair_equivalent = pd.read_csv(repair / "equivalent_transition_registry.csv")
    repair_state_rows = _rows_by_unit(repair_states, state_registry=True)
    target_state_rows = _rows_by_unit(target_states, state_registry=True)
    repair_transition_rows = _rows_by_unit(repair_transitions, state_registry=False)
    target_transition_rows = _rows_by_unit(target_transitions, state_registry=False)
    repair_equivalent_rows = _rows_by_unit(repair_equivalent, state_registry=False)

    requested = set(int(value) for value in unit_ids) if unit_ids is not None else None
    available = set(repair_state_rows)
    if requested is not None:
        unknown = sorted(requested - available)
        if unknown:
            raise ValueError(f"Requested repair units are absent: {unknown}.")
        candidates = sorted(requested)
    else:
        candidates = sorted(available)

    items: list[PromotionItem] = []
    for unit in candidates:
        repair_row = repair_state_rows[unit]
        repair_state_dir = Path(str(repair_row["state_dir"])).resolve()
        try:
            _validate_local_repair_state(repair_state_dir)
        except ValueError:
            if requested is not None:
                raise
            continue
        if unit not in target_state_rows or unit not in target_transition_rows:
            raise ValueError(f"Target experiment has no unique state/transition for unit {unit}.")
        if unit not in repair_transition_rows or unit not in repair_equivalent_rows:
            raise ValueError(f"Repair experiment has incomplete transition products for unit {unit}.")
        eq = repair_equivalent_rows[unit]
        if (
            str(eq.get("selected_domain_status")) != "complete"
            or eq.get("parent_solver_success") is not True
            or eq.get("child_solver_success") is not True
        ):
            raise ValueError(f"Repair unit {unit} has an incomplete equivalent transition.")
        target_state_row = target_state_rows[unit]
        target_transition_row = target_transition_rows[unit]
        repair_transition_row = repair_transition_rows[unit]
        target_state_id = str(target_state_row["state_id"])
        target_state_dir = target / "states" / target_state_id
        if not target_state_dir.exists():
            raise FileNotFoundError(f"Target state directory is missing: {target_state_dir}")
        items.append(
            PromotionItem(
                unit_id=unit,
                repair_state_id=str(repair_row["state_id"]),
                target_state_id=target_state_id,
                repair_transition_id=str(repair_transition_row["transition_id"]),
                target_transition_id=str(target_transition_row["transition_id"]),
                repair_state_dir=repair_state_dir,
                target_state_dir=target_state_dir,
            )
        )
    if not items:
        raise ValueError("No validated local-patch repair states are available for promotion.")
    return PromotionPlan(repair, target, tuple(items))


def _replace_text(value: str, replacements: Sequence[tuple[str, str]]) -> str:
    result = value
    for old, new in replacements:
        result = result.replace(old, new)
    return result


def _rebase_frame(frame: pd.DataFrame, replacements: Sequence[tuple[str, str]]) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if result[column].dtype == object:
            result[column] = result[column].map(
                lambda value: _replace_text(value, replacements) if isinstance(value, str) else value
            )
    return result


def _rewrite_tree(root: Path, replacements: Sequence[tuple[str, str]]) -> None:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rewritten = _replace_text(text, replacements)
        if rewritten != text:
            path.write_text(rewritten, encoding="utf-8")


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.promotion-{uuid4().hex}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_write_json(value: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.promotion-{uuid4().hex}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _replace_registry_row(
    target: pd.DataFrame,
    repair: pd.DataFrame,
    *,
    target_index: int,
    repair_index: int,
    identity_columns: Iterable[str],
    replacements: Sequence[tuple[str, str]],
) -> None:
    identity = {column: target.at[target_index, column] for column in identity_columns if column in target}
    repaired = _rebase_frame(repair.loc[[repair_index]], replacements).iloc[0]
    for column in target.columns:
        if column in repaired.index:
            target.at[target_index, column] = repaired[column]
    for column, value in identity.items():
        target.at[target_index, column] = value


def _refresh_equivalent_resistance(target: Path) -> None:
    runner = Path(__file__).with_name("run_equivalent_resistance_experiment.py")
    subprocess.run(
        [sys.executable, str(runner), str(target), "--reuse-manifest-config"],
        check=True,
    )


def apply_promotion_plan(
    plan: PromotionPlan,
    *,
    backup_root: str | Path | None = None,
    refresh_equivalent_resistance: bool = True,
) -> Path:
    """Apply a validated plan and return its persistent backup directory."""
    repair = plan.repair_experiment
    target = plan.target_experiment
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = (
        Path(backup_root).expanduser().resolve()
        if backup_root is not None
        else target / ".repair_backups" / stamp
    )
    if backup.exists():
        raise FileExistsError(f"Promotion backup already exists: {backup}")
    backup.mkdir(parents=True)
    (backup / "states").mkdir()
    (backup / "root_metadata").mkdir()
    (backup / "equivalent_resistance").mkdir()

    repair_states = pd.read_csv(repair / "state_registry.csv")
    target_states = pd.read_csv(target / "state_registry.csv")
    repair_transitions = pd.read_csv(repair / "transition_registry.csv")
    target_transitions = pd.read_csv(target / "transition_registry.csv")
    repair_equivalent = pd.read_csv(repair / "equivalent_transition_registry.csv")
    target_equivalent = pd.read_csv(target / "equivalent_transition_registry.csv")

    staged: dict[int, Path] = {}
    moved_targets: list[PromotionItem] = []
    try:
        for name in ROOT_METADATA_FILES:
            shutil.copy2(target / name, backup / "root_metadata" / name)
        equivalent_dir = target / "equivalent_resistance"
        for name in EQUIVALENT_AGGREGATE_FILES:
            source = equivalent_dir / name
            if source.exists():
                shutil.copy2(source, backup / "equivalent_resistance" / name)

        repair_state_rows = _rows_by_unit(repair_states, state_registry=True)
        target_state_rows = _rows_by_unit(target_states, state_registry=True)
        repair_transition_rows = _rows_by_unit(repair_transitions, state_registry=False)
        target_transition_rows = _rows_by_unit(target_transitions, state_registry=False)
        repair_equivalent_rows = _rows_by_unit(repair_equivalent, state_registry=False)
        target_equivalent_rows = _rows_by_unit(target_equivalent, state_registry=False)

        for item in plan.items:
            staging = target / "states" / f".{item.target_state_id}.promotion-{uuid4().hex}"
            shutil.copytree(item.repair_state_dir, staging, ignore=shutil.ignore_patterns("._*"))
            replacements = (
                (str(item.repair_state_dir), str(item.target_state_dir)),
                (str(repair), str(target)),
                (item.repair_state_id, item.target_state_id),
                (item.repair_transition_id, item.target_transition_id),
            )
            _rewrite_tree(staging, replacements)
            staged[item.unit_id] = staging

            repair_state_index = repair_state_rows[item.unit_id].name
            target_state_index = target_state_rows[item.unit_id].name
            _replace_registry_row(
                target_states, repair_states,
                target_index=target_state_index, repair_index=repair_state_index,
                identity_columns=STATE_IDENTITY_COLUMNS, replacements=replacements,
            )
            target_states.at[target_state_index, "state_dir"] = str(item.target_state_dir)

            repair_transition_index = repair_transition_rows[item.unit_id].name
            target_transition_index = target_transition_rows[item.unit_id].name
            _replace_registry_row(
                target_transitions, repair_transitions,
                target_index=target_transition_index, repair_index=repair_transition_index,
                identity_columns=TRANSITION_IDENTITY_COLUMNS, replacements=replacements,
            )

            repair_equivalent_index = repair_equivalent_rows[item.unit_id].name
            target_equivalent_index = target_equivalent_rows[item.unit_id].name
            repaired_eq = _rebase_frame(
                repair_equivalent.loc[[repair_equivalent_index]], replacements
            ).iloc[0]
            for column in target_equivalent.columns:
                if column in repaired_eq.index:
                    target_equivalent.at[target_equivalent_index, column] = repaired_eq[column]
            target_equivalent.at[target_equivalent_index, "transition_id"] = item.target_transition_id
            target_equivalent.at[target_equivalent_index, "parent_state_id"] = str(
                target_transitions.at[target_transition_index, "parent_state_id"]
            )
            target_equivalent.at[target_equivalent_index, "child_state_id"] = item.target_state_id

        for item in plan.items:
            backup_state = backup / "states" / item.target_state_id
            os.replace(item.target_state_dir, backup_state)
            os.replace(staged[item.unit_id], item.target_state_dir)
            moved_targets.append(item)

        _atomic_write_csv(target_states, target / "state_registry.csv")
        _atomic_write_csv(target_transitions, target / "transition_registry.csv")
        _atomic_write_csv(target_equivalent, target / "equivalent_transition_registry.csv")

        manifest = _load_json(target / "experiment_manifest.json")
        history = list(manifest.get("repair_promotions") or [])
        history.append(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "repair_experiment": str(repair),
                "backup_dir": str(backup),
                "unit_ids": [item.unit_id for item in plan.items],
                "state_mapping": {
                    item.repair_state_id: item.target_state_id for item in plan.items
                },
            }
        )
        manifest["repair_promotions"] = history
        manifest["last_repair_promotion_at"] = history[-1]["generated_at"]
        _atomic_write_json(manifest, target / "experiment_manifest.json")

        if refresh_equivalent_resistance:
            _refresh_equivalent_resistance(target)

        for item in plan.items:
            _validate_local_repair_state(item.target_state_dir)
        (backup / "promotion_plan.json").write_text(
            json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return backup
    except Exception:
        for staging in staged.values():
            if staging.exists():
                shutil.rmtree(staging)
        for item in reversed(moved_targets):
            backup_state = backup / "states" / item.target_state_id
            if item.target_state_dir.exists():
                shutil.rmtree(item.target_state_dir)
            if backup_state.exists():
                os.replace(backup_state, item.target_state_dir)
        for name in ROOT_METADATA_FILES:
            source = backup / "root_metadata" / name
            if source.exists():
                shutil.copy2(source, target / name)
        equivalent_dir = target / "equivalent_resistance"
        for name in EQUIVALENT_AGGREGATE_FILES:
            current = equivalent_dir / name
            saved = backup / "equivalent_resistance" / name
            if saved.exists():
                shutil.copy2(saved, current)
            elif current.exists():
                current.unlink()
        raise

