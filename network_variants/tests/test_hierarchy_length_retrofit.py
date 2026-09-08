from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import network_variants.hierarchy_length_retrofit as retrofit


def _target(tmp_path: Path) -> tuple[Path, Path]:
    hierarchy = tmp_path / "56_compound_bubbles__anchored" / "states" / "S000_base" / "hierarchy"
    hierarchy.mkdir(parents=True)
    links = hierarchy.parent / "links.gpkg"; nodes = hierarchy.parent / "nodes.gpkg"
    links.touch(); nodes.touch()
    (hierarchy / "unit_metrics.csv").write_text("unit_id,equivalent_length,elongation\n1,10,2\n")
    (hierarchy / "path_metrics.csv").write_text("unit_id,path_id,path_length\n1,1,10\n")
    (hierarchy / "unit_workflow_manifest.json").write_text(json.dumps({"input_links_path": str(links), "input_nodes_path": str(nodes)}))
    return hierarchy, tmp_path


def _results() -> SimpleNamespace:
    empty = pd.DataFrame()
    return SimpleNamespace(
        unit_summary=empty, path_metrics=pd.DataFrame({"unit_id": [1], "path_id": [1], "path_length": [20.0]}),
        unit_metrics=pd.DataFrame({"unit_id": [1], "equivalent_length": [20.0], "elongation": [4.0]}),
        hierarchy_level_metrics=empty, collapse_ranking=pd.DataFrame({"unit_id": [1], "equivalent_length": [20.0]}),
        merge_tree=empty, group_count_summary=empty, bubble_summary=empty, selected_groups=empty,
    )


def test_dry_run_discovers_invalid_hierarchy_products_and_sources(tmp_path: Path) -> None:
    hierarchy, root = _target(tmp_path)
    manifest = retrofit.build_dry_run_manifest(root)
    assert manifest["n_targets"] == 1
    target = manifest["targets"][0]
    assert target["hierarchy_dir"] == str(hierarchy)
    assert target["safe_to_recompute"] is True
    assert "path_metrics.csv" in target["invalid_artifacts"]
    assert target["replacement_dir"].endswith("backfills/physical-centreline-length-v1")
    assert "equivalent-resistance" in manifest["policy"]["excluded_workflows"]


def test_discovery_ignores_hidden_promotion_state_directories(tmp_path: Path) -> None:
    _target(tmp_path)
    hidden = tmp_path / "22_indep__anchored" / "states" / ".S216_unit_10.promotion-a" / "hierarchy"
    hidden.mkdir(parents=True)
    (hidden / "unit_metrics.csv").write_text("unit_id\n1\n")
    assert len(retrofit.discover_targets(tmp_path)) == 1


def test_execute_writes_sidecar_and_provenance_without_replacing_original(tmp_path: Path, monkeypatch) -> None:
    hierarchy, root = _target(tmp_path)
    monkeypatch.setattr(retrofit, "run_unit_workflow", lambda *_args, **_kwargs: _results())
    result = retrofit.execute_retrofit(root)
    destination = hierarchy / "backfills" / retrofit.RETROFIT_VERSION
    assert result["n_written"] == 1
    assert pd.read_csv(hierarchy / "unit_metrics.csv").loc[0, "equivalent_length"] == 10
    assert pd.read_csv(destination / "unit_metrics.csv").loc[0, "equivalent_length"] == 20
    provenance = json.loads((destination / "provenance.json").read_text())
    assert provenance["changed_fields"]["unit_metrics.csv"] == ["elongation", "equivalent_length"]
    assert provenance["equivalent_resistance"] == "retained; not recomputed"


def test_execute_skips_existing_matching_provenanced_sidecar(tmp_path: Path, monkeypatch) -> None:
    hierarchy, root = _target(tmp_path)
    destination = hierarchy / "backfills" / retrofit.RETROFIT_VERSION
    destination.mkdir(parents=True)
    (destination / "provenance.json").write_text(json.dumps({
        "retrofit_version": retrofit.RETROFIT_VERSION,
        "source_links": str(hierarchy.parent / "links.gpkg"),
        "source_nodes": str(hierarchy.parent / "nodes.gpkg"),
    }))
    monkeypatch.setattr(retrofit, "run_unit_workflow", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not run")))
    result = retrofit.execute_retrofit(root)
    assert result["targets"][0]["status"] == "already_written"


def test_execute_records_failure_and_does_not_abort_other_targets(tmp_path: Path, monkeypatch) -> None:
    _hierarchy, root = _target(tmp_path)
    monkeypatch.setattr(retrofit, "run_unit_workflow", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad input")))
    result = retrofit.execute_retrofit(root)
    assert result["n_failed"] == 1
    assert result["targets"][0]["status"] == "failed"
