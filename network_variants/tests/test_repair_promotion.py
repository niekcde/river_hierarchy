from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import network_variants.repair_promotion as repair_promotion
from network_variants.repair_promotion import apply_promotion_plan, build_promotion_plan


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_experiment(root: Path, *, state_id: str, unit_id: int, local: bool) -> None:
    state = root / "states" / state_id
    state.mkdir(parents=True)
    output = state / "variant" / "directed" / f"99__{state_id}_directed_links.gpkg"
    output.parent.mkdir(parents=True)
    output.write_text("fixture", encoding="utf-8")
    if local:
        local_patch = {
            "planning": {"status": "safe"},
            "raster": {"status": "safe"},
            "realized_assembly": {"status": "complete_local_realized_graph"},
            "shared_anchoring": {"status": "complete_shared_anchored_graph"},
            "products": {"status": "complete_width_length_products"},
            "collapse_topology_validation": {"passed": True},
        }
        mode = "local_patch"
    else:
        local_patch = {}
        mode = "full_raster"
    _write_json(
        state / "variant" / "summary" / "variant_manifest.json",
        {
            "execution_mode_used": mode,
            "ready_for_graph_matching": True,
            "ready_for_rapid": True,
            "local_patch": local_patch,
            "output_paths": {"directed_links": str(output.resolve())},
        },
    )
    _write_json(state / "variant" / "directed" / "direction_validation_report.json", {"is_valid": True})
    _write_json(state / "stage_timings.json", {"hierarchy_status": "complete"})
    (state / "old_marker.txt").write_text("repair" if local else "target", encoding="utf-8")

    pd.DataFrame(
        [
            {
                "state_id": "S000_base", "parent_state_id": "", "depth": 0,
                "mode": "independent-units", "state_role": "base",
                "selected_unit_ids": pd.NA, "state_dir": str(root / "states" / "S000_base"),
                "status": "complete",
            },
            {
                "state_id": state_id, "parent_state_id": "S000_base", "depth": 1,
                "mode": "independent-units", "state_role": "derived",
                "selected_unit_ids": unit_id, "state_dir": str(state.resolve()),
                "variant_output_dir": str((state / "variant").resolve()), "status": "complete",
            },
        ]
    ).to_csv(root / "state_registry.csv", index=False)
    transition_id = f"T{int(state_id[1:4]):03d}"
    pd.DataFrame(
        [{
            "transition_id": transition_id, "step_index": int(state_id[1:4]),
            "parent_state_id": "S000_base", "child_state_id": state_id,
            "mode": "independent-units", "selected_unit_ids": unit_id,
            "child_variant_output_dir": str((state / "variant").resolve()), "status": "complete",
        }]
    ).to_csv(root / "transition_registry.csv", index=False)
    pd.DataFrame(
        [{
            "transition_id": transition_id, "parent_state_id": "S000_base",
            "child_state_id": state_id, "selected_unit_ids": unit_id,
            "selected_domain_status": "complete", "parent_solver_success": True,
            "child_solver_success": True, "Lambda_K": 1.25 if local else 9.0,
        }]
    ).to_csv(root / "equivalent_transition_registry.csv", index=False)
    _write_json(
        root / "experiment_manifest.json",
        {
            "experiment_id": root.name, "mode": "independent-units",
            "example_id": "99", "root_example_id": "99", "exit_sides": "NS",
        },
    )


def test_build_promotion_plan_maps_filtered_state_to_original_state(tmp_path) -> None:
    repair = tmp_path / "repair"
    target = tmp_path / "target"
    repair.mkdir(); target.mkdir()
    _write_experiment(repair, state_id="S001_unit_7", unit_id=7, local=True)
    _write_experiment(target, state_id="S042_unit_7", unit_id=7, local=False)

    plan = build_promotion_plan(repair, target)

    assert len(plan.items) == 1
    assert plan.items[0].repair_state_id == "S001_unit_7"
    assert plan.items[0].target_state_id == "S042_unit_7"
    assert plan.items[0].repair_transition_id == "T001"
    assert plan.items[0].target_transition_id == "T042"


def test_apply_promotion_rebases_state_and_registries_with_backup(tmp_path) -> None:
    repair = tmp_path / "repair"
    target = tmp_path / "target"
    repair.mkdir(); target.mkdir()
    _write_experiment(repair, state_id="S001_unit_7", unit_id=7, local=True)
    _write_experiment(target, state_id="S042_unit_7", unit_id=7, local=False)
    # Failed legacy states may have no authoritative equivalent-transition row.
    empty_equivalent = pd.read_csv(target / "equivalent_transition_registry.csv").iloc[0:0]
    empty_equivalent.to_csv(target / "equivalent_transition_registry.csv", index=False)
    plan = build_promotion_plan(repair, target)

    backup = apply_promotion_plan(
        plan,
        backup_root=tmp_path / "backup",
        refresh_equivalent_resistance=False,
    )

    promoted = target / "states" / "S042_unit_7"
    assert (promoted / "old_marker.txt").read_text() == "repair"
    assert (backup / "states" / "S042_unit_7" / "old_marker.txt").read_text() == "target"
    manifest = json.loads((promoted / "variant" / "summary" / "variant_manifest.json").read_text())
    assert manifest["execution_mode_used"] == "local_patch"
    assert str(target.resolve()) in manifest["output_paths"]["directed_links"]
    assert "S042_unit_7" in manifest["output_paths"]["directed_links"]
    assert Path(manifest["output_paths"]["directed_links"]).exists()
    assert not list(promoted.rglob("*S001_unit_7*"))
    state_registry = pd.read_csv(target / "state_registry.csv")
    child = state_registry.loc[state_registry["state_role"].eq("derived")].iloc[0]
    assert child["state_id"] == "S042_unit_7"
    assert str(target.resolve()) in child["variant_output_dir"]
    transitions = pd.read_csv(target / "transition_registry.csv")
    assert transitions.iloc[0]["transition_id"] == "T042"
    assert transitions.iloc[0]["child_state_id"] == "S042_unit_7"
    equivalent = pd.read_csv(target / "equivalent_transition_registry.csv")
    assert equivalent.iloc[0]["transition_id"] == "T042"
    assert equivalent.iloc[0]["child_state_id"] == "S042_unit_7"
    assert equivalent.iloc[0]["Lambda_K"] == 1.25
    root_manifest = json.loads((target / "experiment_manifest.json").read_text())
    assert root_manifest["repair_promotions"][-1]["unit_ids"] == [7]


def test_apply_promotion_rolls_back_state_and_metadata_when_refresh_fails(
    tmp_path, monkeypatch
) -> None:
    repair = tmp_path / "repair"
    target = tmp_path / "target"
    repair.mkdir(); target.mkdir()
    _write_experiment(repair, state_id="S001_unit_7", unit_id=7, local=True)
    _write_experiment(target, state_id="S042_unit_7", unit_id=7, local=False)
    original_registry = (target / "state_registry.csv").read_bytes()
    original_manifest = (target / "experiment_manifest.json").read_bytes()
    plan = build_promotion_plan(repair, target)

    def fail_refresh(_target):
        raise RuntimeError("synthetic aggregate refresh failure")

    monkeypatch.setattr(repair_promotion, "_refresh_equivalent_resistance", fail_refresh)
    with pytest.raises(RuntimeError, match="synthetic aggregate refresh failure"):
        apply_promotion_plan(plan, backup_root=tmp_path / "backup")

    restored = target / "states" / "S042_unit_7"
    assert (restored / "old_marker.txt").read_text() == "target"
    assert (target / "state_registry.csv").read_bytes() == original_registry
    assert (target / "experiment_manifest.json").read_bytes() == original_manifest
