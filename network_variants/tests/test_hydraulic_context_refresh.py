from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import network_variants.hydraulic_context_refresh as refresh_module
from network_variants.hydraulic_context import SlopeConfig


def _state_row(root: Path, state_id: str, parent_state_id: str | None, depth: int) -> dict[str, object]:
    state = root / "states" / state_id
    variant = state / "variant"
    widths = variant / "widths"
    directed = variant / "directed"
    widths.mkdir(parents=True)
    directed.mkdir(parents=True)
    (widths / "links_with_width_families.gpkg").touch()
    (directed / f"{state_id}_nodes.gpkg").touch()
    context = state / "hydraulic_context"
    context.mkdir()
    (context / "link_hydraulic_context.csv").write_text("old_link\n", encoding="utf-8")
    (context / "node_hydraulic_context.csv").write_text("old_node\n", encoding="utf-8")
    (context / "hydraulic_context_manifest.json").write_text(
        json.dumps({"slope_config": {"slope_profile_mode": "network_dist"}}),
        encoding="utf-8",
    )
    return {
        "state_id": state_id,
        "parent_state_id": parent_state_id,
        "depth": depth,
        "status": "complete",
        "state_dir": str(state),
        "variant_output_dir": str(variant),
        "analysis_links_path": str(widths / "links_with_width_families.gpkg"),
        "directed_nodes_path": str(directed / f"{state_id}_nodes.gpkg"),
        "hydraulic_context_output_dir": str(context),
        "hierarchy_output_dir": "",
    }


def test_refresh_is_dry_by_default_and_preserves_parent_provenance(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "15_indep_direction130__anchored"
    rows = [
        _state_row(root, "S000_base", None, 0),
        _state_row(root, "S001_unit_1", "S000_base", 1),
    ]
    pd.DataFrame(rows).to_csv(root / "state_registry.csv", index=False)

    calls: list[dict[str, object]] = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "link_hydraulic_context.csv").write_text("new_link\n", encoding="utf-8")
        (output / "node_hydraulic_context.csv").write_text("new_node\n", encoding="utf-8")
        (output / "hydraulic_context_manifest.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(output_dir=output)

    monkeypatch.setattr(refresh_module, "generate_state_hydraulic_context", fake_generate)
    config = SlopeConfig(slope_profile_mode="reference_section", max_slope_for_k=0.009, section_slope_ratio_max=5.0)

    plan = refresh_module.refresh_experiment_hydraulic_contexts(root, slope_config=config)
    assert plan["status"].tolist() == ["planned", "planned"]
    assert not calls

    result = refresh_module.refresh_experiment_hydraulic_contexts(root, slope_config=config, execute=True)
    assert result["status"].tolist() == ["refreshed", "refreshed"]
    assert len(calls) == 2
    assert "parent_link_context_path" not in calls[0]
    assert calls[1]["parent_link_context_path"] == (
        root / "states" / "S000_base" / "hydraulic_context" / "link_hydraulic_context.csv"
    )
    assert (Path(result.loc[0, "backup_dir"]) / "link_hydraulic_context.csv").read_text(encoding="utf-8") == "old_link\n"
    assert (root / "states" / "S001_unit_1" / "hydraulic_context" / "link_hydraulic_context.csv").read_text(encoding="utf-8") == "new_link\n"


def test_refresh_can_exclude_promoted_base_state(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "02_indep__anchored"
    rows = [
        _state_row(root, "S000_base", None, 0),
        _state_row(root, "S001_unit_1", "S000_base", 1),
    ]
    pd.DataFrame(rows).to_csv(root / "state_registry.csv", index=False)
    calls: list[dict[str, object]] = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "link_hydraulic_context.csv").write_text("new_link\n", encoding="utf-8")
        (output / "node_hydraulic_context.csv").write_text("new_node\n", encoding="utf-8")
        (output / "hydraulic_context_manifest.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(output_dir=output)

    monkeypatch.setattr(refresh_module, "generate_state_hydraulic_context", fake_generate)
    report = refresh_module.refresh_experiment_hydraulic_contexts(
        root,
        slope_config=SlopeConfig(),
        execute=True,
        include_base_state=False,
    )
    assert report["state_id"].tolist() == ["S001_unit_1"]
    assert len(calls) == 1


def test_refresh_can_select_a_bounded_state_batch(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "02_indep__anchored"
    rows = [
        _state_row(root, "S000_base", None, 0),
        _state_row(root, "S001_unit_1", "S000_base", 1),
        _state_row(root, "S002_unit_2", "S000_base", 1),
    ]
    pd.DataFrame(rows).to_csv(root / "state_registry.csv", index=False)
    calls: list[dict[str, object]] = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(output_dir=output)

    monkeypatch.setattr(refresh_module, "generate_state_hydraulic_context", fake_generate)
    report = refresh_module.refresh_experiment_hydraulic_contexts(
        root,
        slope_config=SlopeConfig(),
        execute=True,
        include_base_state=False,
        state_ids={"S002_unit_2"},
    )
    assert report["state_id"].tolist() == ["S002_unit_2"]
    assert [call["state_id"] for call in calls] == ["S002_unit_2"]
    assert calls[0]["parent_link_context_path"] == (
        root / "states" / "S000_base" / "hydraulic_context" / "link_hydraulic_context.csv"
    )


def test_refresh_parallelizes_only_within_parent_depth_wave(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "02_indep__anchored"
    rows = [
        _state_row(root, "S000_base", None, 0),
        _state_row(root, "S001_unit_1", "S000_base", 1),
        _state_row(root, "S002_unit_2", "S000_base", 1),
        _state_row(root, "S003_unit_3", "S001_unit_1", 2),
    ]
    pd.DataFrame(rows).to_csv(root / "state_registry.csv", index=False)
    parents: dict[str, str | None] = {}

    def fake_generate(**kwargs):
        state_id = str(kwargs["state_id"])
        parent = kwargs.get("parent_link_context_path")
        parents[state_id] = str(parent) if parent is not None else None
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(output_dir=output)

    monkeypatch.setattr(refresh_module, "generate_state_hydraulic_context", fake_generate)
    report = refresh_module.refresh_experiment_hydraulic_contexts(
        root, slope_config=SlopeConfig(), execute=True, max_workers=4
    )
    assert report["status"].tolist() == ["refreshed"] * 4
    assert parents["S003_unit_3"] == str(
        root / "states" / "S001_unit_1" / "hydraulic_context" / "link_hydraulic_context.csv"
    )
