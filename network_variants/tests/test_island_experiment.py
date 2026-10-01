from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

import network_variants.island_experiment as island_module
from network_variants.island_experiment import (
    build_island_execution_plan,
    build_island_observation_manifest,
    create_island_experiment_root,
    generate_island_hydraulic_contexts,
    run_island_experiment,
    resolve_island_backend_policy,
)


def test_panelized_parent_uses_virtual_local_patch_and_panelized_fallback(tmp_path: Path) -> None:
    experiment = tmp_path / "41_indep__anchored"
    run_root = tmp_path / "rivgraph_run"
    (run_root / "panels").mkdir(parents=True)
    (run_root / "panels" / "panel_plan.gpkg").touch()
    cleaned = run_root / "masks_cleaned" / "41_cleaned.tif"
    cleaned.parent.mkdir()
    cleaned.touch()
    (experiment / "experiment_manifest.json").parent.mkdir()
    (experiment / "experiment_manifest.json").write_text(json.dumps({
        "execution_backends": {"base_backend": "panelized", "fallback_backend": "panelized"},
        "root_paths": {"cleaned_mask": str(cleaned)},
    }), encoding="utf-8")

    policy = resolve_island_backend_policy(experiment)

    assert policy.local_patch_mask_publication == "virtual_patch"
    assert policy.fallback_backend == "panelized"
    assert policy.panelized_run_root == run_root


def _write_observation_inputs(tmp_path: Path) -> tuple[Path, Path]:
    setup = tmp_path / "island_setup"
    experiment = tmp_path / "56_indep__anchored"
    setup.mkdir()
    (experiment / "states" / "S001_unit_1").mkdir(parents=True)
    (experiment / "states" / "S002_unit_2").mkdir(parents=True)
    (experiment / "states" / "S001_unit_1" / "variant").mkdir()
    (experiment / "states" / "S002_unit_2" / "variant").mkdir()
    (setup / "island_experiment_manifest.json").write_text(
        json.dumps({"approved_island_ids": ["island_56_001", "island_56_002"]}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "island_id": "island_56_001",
                "proposed_anchor_unit_id": 1,
                "current_unit_ids": "1",
                "anchor_bifurcation": 10,
                "anchor_confluence": 11,
            },
            {
                "island_id": "island_56_002",
                "proposed_anchor_unit_id": 2,
                "current_unit_ids": "2",
                "anchor_bifurcation": 20,
                "anchor_confluence": 21,
            },
        ]
    ).to_csv(setup / "island_candidate_registry.csv", index=False)
    gpd.GeoDataFrame(
        {"island_id": ["island_56_001", "island_56_002"]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
        ],
        crs="EPSG:3857",
    ).to_file(setup / "island_candidate_geometries.gpkg", driver="GPKG")
    pd.DataFrame(
        [
            {
                "island_id": "island_56_001",
                "source_unit_id": 1,
                "mask_equal": True,
                "anchor_equal": True,
                "bank_labels_equal": True,
                "equivalence_status": "alias_existing_unit_state",
            },
            {
                "island_id": "island_56_002",
                "source_unit_id": 2,
                "mask_equal": False,
                "anchor_equal": True,
                "bank_labels_equal": False,
                "equivalence_status": "requires_full_island_state",
            },
        ]
    ).to_csv(setup / "island_equivalence_registry.csv", index=False)
    return setup, experiment


def test_observation_manifest_references_aliases_without_creating_alias_state_dirs(tmp_path: Path) -> None:
    setup, experiment = _write_observation_inputs(tmp_path)
    output = tmp_path / "island_observations"

    observations = build_island_observation_manifest(
        setup_dir=setup,
        experiment_dir=experiment,
        output_dir=output,
    ).set_index("island_id")

    alias = observations.loc["island_56_001"]
    rerun = observations.loc["island_56_002"]
    assert alias["execution_action"] == "alias_existing_unit_state"
    assert alias["observation_status"] == "ready_reference"
    assert Path(alias["canonical_state_dir"]) == experiment / "states" / "S001_unit_1"
    assert alias["island_run_dir"] == ""
    assert Path(alias["analysis_state_dir"]) == experiment / "states" / "S001_unit_1"
    assert rerun["execution_action"] == "run_island_state"
    assert rerun["observation_status"] == "pending_island_execution"
    assert rerun["canonical_state_dir"] == ""
    assert Path(rerun["island_run_dir"]) == output / "island_reruns" / "island_56_002"
    assert Path(rerun["analysis_state_dir"]) == output / "island_reruns" / "island_56_002"
    assert len(str(alias["geometry_sha256"])) == 64
    assert not (output / "island_reruns" / "island_56_001").exists()
    assert not (output / "island_reruns" / "island_56_002").exists()

    written = pd.read_csv(output / "island_observation_manifest.csv").set_index("island_id")
    assert written.loc["island_56_001", "canonical_state_dir"] == alias["canonical_state_dir"]
    manifest = json.loads((output / "island_observation_manifest.json").read_text())
    assert manifest["n_conceptual_islands"] == 2
    assert manifest["n_aliases"] == 1
    assert manifest["n_required_island_runs"] == 1


def test_legacy_execution_plan_builder_returns_observation_manifest(tmp_path: Path) -> None:
    setup, experiment = _write_observation_inputs(tmp_path)

    observations = build_island_execution_plan(
        setup_dir=setup,
        experiment_dir=experiment,
    )

    assert set(observations["execution_action"]) == {
        "alias_existing_unit_state",
        "run_island_state",
    }
    assert "canonical_state_dir" in observations.columns


def test_full_raster_adapter_runs_only_reruns_and_reuses_completed_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    setup, experiment = _write_observation_inputs(tmp_path)
    output = tmp_path / "island_observations"
    build_island_observation_manifest(
        setup_dir=setup,
        experiment_dir=experiment,
        output_dir=output,
    )
    calls: list[str] = []

    def touch(path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("artifact", encoding="utf-8")
        return path

    def fake_selection(_setup, island_id):
        return island_module.IslandVariantSelection(
            island_id=island_id,
            geometry=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            anchor_unit_id=2,
            anchor_bifurcation=20,
            anchor_confluence=21,
            source_unit_ids=(2,),
        )

    def fake_topology(selection, *, output_dir, **_kwargs):
        calls.append(selection.island_id)
        root = Path(output_dir)
        return island_module.IslandTopologyRun(
            selection=selection,
            mask_products=island_module.IslandMaskProducts(
                touch(root / "mask" / "collapsed.tif"),
                touch(root / "mask" / "origin.tif"),
                touch(root / "mask" / "edits.gpkg"),
                1,
            ),
            rivgraph_paths={},
        )

    def fake_directed(_topology, *, output_dir, **_kwargs):
        root = Path(output_dir)
        return island_module.IslandDirectedProducts(
            touch(root / "directed_links.gpkg"), touch(root / "directed_nodes.gpkg"),
            touch(root / "node_match.csv"), touch(root / "link_match.csv"),
            touch(root / "lineage.csv"), touch(root / "anchor.json"),
        )

    def fake_widths(_topology, _directed, *, output_dir, **_kwargs):
        root = Path(output_dir)
        return island_module.IslandWidthProducts(
            touch(root / "bank_labels.tif"), touch(root / "widths.gpkg"),
            touch(root / "samples.csv"), (1,),
        )

    def fake_routing(_selection, _directed, _widths, *, output_dir, **_kwargs):
        root = Path(output_dir)
        return island_module.IslandRoutingWidthProducts(
            touch(root / "links_with_routing_widths.gpkg"),
            touch(root / "routing_audit.csv"), touch(root / "routing_manifest.json"),
        )

    monkeypatch.setattr(island_module, "load_island_variant_selection", fake_selection)
    monkeypatch.setattr(island_module, "run_island_topology", fake_topology)
    monkeypatch.setattr(island_module, "direct_island_topology", fake_directed)
    monkeypatch.setattr(island_module, "compute_island_two_bank_widths", fake_widths)
    monkeypatch.setattr(island_module, "apply_island_routing_widths", fake_routing)
    kwargs = {
        "base_mask_path": tmp_path / "base.tif",
        "parent_links_path": tmp_path / "parent_links.gpkg",
        "parent_nodes_path": tmp_path / "parent_nodes.gpkg",
        "boundary_component_labels_path": tmp_path / "labels.tif",
        "wet_reference_mask_path": tmp_path / "wet.tif",
        "exit_sides": "south",
        "example_id": "56",
        "match_tolerance": 1.0,
    }
    # AppleDouble metadata on an external macOS volume must not prevent the
    # full-raster fallback from consuming its otherwise clean local fallback.
    local_fallback = (
        output / "island_reruns" / "island_56_002" / "local_patch"
        / "fallback_to_full_raster.json"
    )
    local_fallback.parent.mkdir(parents=True, exist_ok=True)
    local_fallback.write_text("{}", encoding="utf-8")
    (local_fallback.parent.parent / "._local_patch").write_text("metadata", encoding="utf-8")
    report = island_module.run_island_full_raster_observations(
        output / "island_observation_manifest.csv", **kwargs
    )
    assert calls == ["island_56_002"]
    assert set(report["outcome"]) == {"referenced_existing_state", "completed_full_raster_run"}
    assert not (output / "island_reruns" / "island_56_001").exists()
    rerun_manifest = output / "island_reruns" / "island_56_002" / "island_run_manifest.json"
    assert rerun_manifest.exists()

    second = island_module.run_island_full_raster_observations(
        output / "island_observation_manifest.csv", **kwargs
    )
    assert calls == ["island_56_002"]
    assert "reused_completed_run" in set(second["outcome"])


def test_local_patch_adapter_uses_anchor_unit_but_frozen_island_geometry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    setup, experiment = _write_observation_inputs(tmp_path)
    output = tmp_path / "island_observations"
    build_island_observation_manifest(
        setup_dir=setup,
        experiment_dir=experiment,
        output_dir=output,
    )
    calls: list[dict] = []

    def touch(path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("artifact", encoding="utf-8")
        return path

    def fake_local_patch_runner(**kwargs):
        calls.append(kwargs)
        variant = Path(kwargs["output_dir"])
        collapsed = touch(variant / "mask" / "collapsed.tif")
        origin = touch(variant / "mask" / "origin.tif")
        directed_links = touch(variant / "directed" / "links.gpkg")
        directed_nodes = touch(variant / "directed" / "nodes.gpkg")
        routing = touch(variant / "widths" / "links_with_width_families.gpkg")
        return SimpleNamespace(
            output_dir=variant,
            collapsed_mask_path=collapsed,
            origin_mask_path=origin,
            directed_links_path=directed_links,
            directed_nodes_path=directed_nodes,
            manifest={"output_paths": {"links_with_width_families": str(routing)}},
        )

    import network_variants.local_patch_execution as local_execution

    monkeypatch.setattr(
        local_execution, "materialize_local_patch_network_variant", fake_local_patch_runner
    )
    monkeypatch.setattr(
        island_module, "_full_raster_island_cycle_reduction", lambda **_kwargs: 1
    )
    parent_variant = tmp_path / "parent_variant"
    parent_links = parent_variant / "directed" / "parent_links.gpkg"
    parent_links.parent.mkdir(parents=True)
    parent_width_links = parent_variant / "widths" / "links_with_width_families.gpkg"
    parent_width_links.parent.mkdir()
    parent_width_links.write_text("artifact", encoding="utf-8")
    report = island_module.run_island_local_patch_observations(
        output / "island_observation_manifest.csv",
        base_mask_path=tmp_path / "base.tif",
        parent_links_path=parent_links,
        parent_nodes_path=tmp_path / "parent_nodes.gpkg",
        boundary_component_labels_path=tmp_path / "labels.tif",
        exit_sides="S",
        example_id="56",
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["selection_type"] == "island"
    assert call["unit_ids"] == (2,)
    assert call["collapse_component_id"] == "island_56_002"
    assert call["collapse_source_unit_ids"] == (2,)
    assert call["expected_cycle_reduction_override"] == 1
    assert call["collapse_geometry"].equals(Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]))
    assert set(report["outcome"]) == {
        "referenced_existing_state", "completed_local_patch_run"
    }
    local_manifest = (
        output / "island_reruns" / "island_56_002" / "local_patch"
        / "island_local_patch_run_manifest.json"
    )
    payload = json.loads(local_manifest.read_text())
    assert payload["scientific_anchor_policy"] == "recorded_island_anchor_unit"
    assert payload["computational_domain_policy"] == (
        "recorded_anchor_unit_smallest_closed_domain"
    )


def test_local_patch_adapter_recovers_interrupted_derived_only_attempt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    setup, experiment = _write_observation_inputs(tmp_path)
    output = tmp_path / "island_observations"
    build_island_observation_manifest(
        setup_dir=setup,
        experiment_dir=experiment,
        output_dir=output,
    )
    local_run = output / "island_reruns" / "island_56_002" / "local_patch"
    stale_variant = local_run / "variant"
    stale_variant.mkdir(parents=True)
    (stale_variant / "partial_product.txt").write_text("interrupted", encoding="utf-8")
    # Typical external-volume AppleDouble sidecars must not block recovery.
    (local_run / "._variant").write_text("metadata", encoding="utf-8")

    def touch(path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("artifact", encoding="utf-8")
        return path

    def fake_local_patch_runner(**kwargs):
        variant = Path(kwargs["output_dir"])
        return SimpleNamespace(
            output_dir=variant,
            collapsed_mask_path=touch(variant / "mask" / "collapsed.tif"),
            origin_mask_path=touch(variant / "mask" / "origin.tif"),
            directed_links_path=touch(variant / "directed" / "links.gpkg"),
            directed_nodes_path=touch(variant / "directed" / "nodes.gpkg"),
            manifest={"output_paths": {
                "links_with_width_families": str(touch(variant / "widths" / "links.gpkg"))
            }},
        )

    import network_variants.local_patch_execution as local_execution
    monkeypatch.setattr(
        local_execution, "materialize_local_patch_network_variant", fake_local_patch_runner
    )
    parent_variant = tmp_path / "parent_variant"
    parent_links = parent_variant / "directed" / "parent_links.gpkg"
    parent_links.parent.mkdir(parents=True)
    widths = parent_variant / "widths" / "links_with_width_families.gpkg"
    widths.parent.mkdir()
    widths.write_text("artifact", encoding="utf-8")

    report = island_module.run_island_local_patch_observations(
        output / "island_observation_manifest.csv",
        base_mask_path=tmp_path / "base.tif",
        parent_links_path=parent_links,
        parent_nodes_path=tmp_path / "parent_nodes.gpkg",
        boundary_component_labels_path=tmp_path / "labels.tif",
        exit_sides="S",
        example_id="56",
    )

    assert "completed_local_patch_run" in set(report["outcome"])
    cleanup = json.loads((local_run / "interrupted_local_patch_retry_cleanup.json").read_text())
    assert cleanup["reason"] == "interrupted_derived_local_patch_before_final_manifest"
    assert not (stale_variant / "partial_product.txt").exists()


def test_island_hydraulic_context_accepts_panelized_realisation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state = tmp_path / "island_panelized"
    state.mkdir()
    links = state / "links.gpkg"
    nodes = state / "nodes.gpkg"
    widths = state / "widths.gpkg"
    for path in (links, nodes, widths):
        path.touch()
    analysis_path = tmp_path / "island_analysis_manifest.csv"
    pd.DataFrame([
        {
            "island_id": "island_41_062",
            "analysis_backend": "island_panelized",
            "analysis_state_dir": str(state),
            "analysis_directed_links_path": str(links),
            "analysis_directed_nodes_path": str(nodes),
            "analysis_routing_width_links_path": str(widths),
        },
        {
            "island_id": "island_46_038",
            "analysis_backend": "island_panelized",
            "analysis_state_dir": str(state),
            "analysis_directed_links_path": str(links),
            "analysis_directed_nodes_path": str(nodes),
            "analysis_routing_width_links_path": str(widths),
            "strict_analysis_status": "exclude_pruning_or_mask_topology_exception",
        }
    ]).to_csv(analysis_path, index=False)
    base_link = tmp_path / "base_link_context.csv"
    base_node = tmp_path / "base_node_context.csv"
    base_link.touch()
    base_node.touch()
    calls: list[dict[str, object]] = []

    def fake_context(**kwargs):
        calls.append(kwargs)
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True)
        link_context = output / "link_hydraulic_context.csv"
        node_context = output / "node_hydraulic_context.csv"
        manifest = output / "hydraulic_context_manifest.json"
        for path in (link_context, node_context, manifest):
            path.touch()
        return SimpleNamespace(
            output_dir=output,
            link_context_path=link_context,
            node_context_path=node_context,
            manifest_path=manifest,
        )

    import network_variants.hydraulic_context as hydraulic_context
    monkeypatch.setattr(hydraulic_context, "generate_state_hydraulic_context", fake_context)

    registry = generate_island_hydraulic_contexts(
        analysis_path,
        base_link_context_path=base_link,
        base_node_context_path=base_node,
    )

    assert registry.loc[0, "hydraulic_context_status"] == "ready_island_context"
    assert registry.loc[1, "hydraulic_context_status"] == "excluded_from_primary_analysis"
    assert calls[0]["directed_links_path"] == str(widths)


def test_create_island_experiment_root_references_without_copying(tmp_path: Path) -> None:
    base = tmp_path / "56_indep__anchored"
    base.mkdir()
    (base / "experiment_manifest.json").write_text("{}", encoding="utf-8")
    independent_state = base / "states" / "S001_unit_1"
    independent_state.mkdir(parents=True)
    island_state = tmp_path / "pilot_setup" / "island_reruns" / "island_56_002"
    island_state.mkdir(parents=True)
    analysis_path = tmp_path / "pilot_setup" / "island_analysis_manifest.csv"
    pd.DataFrame([
        {
            "island_id": "island_56_001",
            "analysis_backend": "independent_unit_alias",
            "analysis_state_dir": str(independent_state),
        },
        {
            "island_id": "island_56_002",
            "analysis_backend": "island_full_raster",
            "analysis_state_dir": str(island_state),
        },
    ]).to_csv(analysis_path, index=False)

    result = create_island_experiment_root(
        analysis_path,
        base_experiment_dir=base,
        output_dir=tmp_path / "56_island__anchored",
    )

    registry = pd.read_csv(result["observation_registry"])
    assert registry["observation_role"].tolist() == [
        "alias_existing_independent_state", "genuine_island_state",
    ]
    assert registry["artifact_policy"].str.contains("no_copy").all()
    manifest = json.loads(result["experiment_manifest"].read_text())
    assert manifest["n_alias_references"] == 1
    assert manifest["n_genuine_island_states"] == 1
    assert not (result["experiment_dir"] / "states").exists()


def test_run_island_experiment_owns_complete_manifest_but_only_runs_reruns(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reviewed_setup, base = _write_observation_inputs(tmp_path)
    (base / "experiment_manifest.json").write_text("{}", encoding="utf-8")

    def fail_equivalence_recompute(**_kwargs):
        raise AssertionError("execution must consume the reviewed equivalence registry")

    def fake_observations(*, output_dir, **_kwargs):
        root = Path(output_dir)
        frame = pd.DataFrame([
            {"island_id": "island_56_001", "execution_action": "alias_existing_unit_state"},
            {"island_id": "island_56_002", "execution_action": "run_island_state"},
        ])
        frame.to_csv(root / "island_observation_manifest.csv", index=False)
        return frame

    def fake_full(*_args, report_dir, **_kwargs):
        frame = pd.DataFrame([
            {"outcome": "referenced_existing_state"},
            {"outcome": "completed_full_raster_run"},
        ])
        frame.to_csv(Path(report_dir) / "island_full_raster_execution.csv", index=False)
        return frame

    def fake_analysis(_path, *, output_dir):
        frame = pd.DataFrame({"analysis_execution_status": ["ready_alias", "ready_full"]})
        frame.to_csv(Path(output_dir) / "island_analysis_manifest.csv", index=False)
        return frame

    monkeypatch.setattr(island_module, "build_island_equivalence_registry", fail_equivalence_recompute)
    monkeypatch.setattr(island_module, "build_island_observation_manifest", fake_observations)
    monkeypatch.setattr(island_module, "run_island_full_raster_observations", fake_full)
    monkeypatch.setattr(island_module, "build_island_analysis_manifest", fake_analysis)

    output_root = tmp_path / "56_island__anchored"
    output_root.mkdir()
    (output_root / "experiment_manifest.json").write_text(
        json.dumps({"schema_version": "island-experiment-root-v1"}),
        encoding="utf-8",
    )
    result = run_island_experiment(
        reviewed_setup_dir=reviewed_setup,
        base_experiment_dir=base,
        output_dir=output_root,
        base_mask_path=tmp_path / "base.tif",
        parent_links_path=tmp_path / "links.gpkg",
        parent_nodes_path=tmp_path / "nodes.gpkg",
        boundary_component_labels_path=tmp_path / "labels.tif",
        wet_reference_mask_path=tmp_path / "wet.tif",
        exit_sides="S",
        example_id="56",
        adopt_metadata_root=True,
    )

    manifest = json.loads(result["experiment_manifest"].read_text())
    assert manifest["n_conceptual_islands"] == 2
    assert manifest["n_alias_references"] == 1
    assert manifest["n_genuine_island_states"] == 1
    assert (result["setup_dir"] / "island_candidate_geometries.gpkg").exists()
    assert (result["experiment_dir"] / "island_observation_manifest.csv").exists()
    assert (output_root / "promotion_manifest.json").exists()


def test_run_island_experiment_local_mode_does_not_call_full_raster(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reviewed_setup, base = _write_observation_inputs(tmp_path)
    (base / "experiment_manifest.json").write_text("{}", encoding="utf-8")
    calls: list[str] = []

    def fail_equivalence_recompute(**_kwargs):
        raise AssertionError("execution must consume the reviewed equivalence registry")

    def fake_observations(*, output_dir, **_kwargs):
        frame = pd.DataFrame([
            {"island_id": "island_56_001", "execution_action": "alias_existing_unit_state"},
            {"island_id": "island_56_002", "execution_action": "run_island_state"},
        ])
        frame.to_csv(Path(output_dir) / "island_observation_manifest.csv", index=False)
        return frame

    def fake_full(*_args, **_kwargs):
        calls.append("full")
        raise AssertionError("local mode must not invoke the full-raster backend")

    def fake_local(*_args, report_dir, **_kwargs):
        calls.append("local")
        frame = pd.DataFrame([
            {"outcome": "referenced_existing_state"},
            {"outcome": "completed_local_patch_run"},
        ])
        frame.to_csv(Path(report_dir) / "island_local_patch_execution.csv", index=False)
        return frame

    def fake_analysis(_path, *, output_dir):
        frame = pd.DataFrame({"analysis_execution_status": ["ready_alias", "ready_direct_local_patch"]})
        frame.to_csv(Path(output_dir) / "island_analysis_manifest.csv", index=False)
        return frame

    monkeypatch.setattr(island_module, "build_island_equivalence_registry", fail_equivalence_recompute)
    monkeypatch.setattr(island_module, "build_island_observation_manifest", fake_observations)
    monkeypatch.setattr(island_module, "run_island_full_raster_observations", fake_full)
    monkeypatch.setattr(island_module, "run_island_local_patch_observations", fake_local)
    monkeypatch.setattr(island_module, "build_island_analysis_manifest", fake_analysis)

    result = run_island_experiment(
        reviewed_setup_dir=reviewed_setup,
        base_experiment_dir=base,
        output_dir=tmp_path / "56_island__anchored",
        base_mask_path=tmp_path / "base.tif",
        parent_links_path=tmp_path / "links.gpkg",
        parent_nodes_path=tmp_path / "nodes.gpkg",
        boundary_component_labels_path=tmp_path / "labels.tif",
        wet_reference_mask_path=tmp_path / "wet.tif",
        exit_sides="S",
        example_id="56",
        execution_mode="local_patch",
    )

    assert calls == ["local"]
    manifest = json.loads(result["experiment_manifest"].read_text())
    assert manifest["n_full_raster_completed"] == 0
    assert manifest["n_local_patch_completed"] == 1


def test_run_island_experiment_resumes_frozen_setup_without_rebuilding(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reviewed_setup, base = _write_observation_inputs(tmp_path)
    (base / "experiment_manifest.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "56_island__anchored"
    setup = island_module.freeze_reviewed_island_setup(
        reviewed_setup, output_dir=output / "setup"
    )
    pd.DataFrame({"island_id": ["island_56_001", "island_56_002"]}).to_csv(
        setup / "island_equivalence_registry.csv", index=False
    )
    pd.DataFrame([
        {"island_id": "island_56_001", "execution_action": "alias_existing_unit_state"},
        {"island_id": "island_56_002", "execution_action": "run_island_state"},
    ]).to_csv(output / "island_observation_manifest.csv", index=False)

    def fail_rebuild(**_kwargs):
        raise AssertionError("a resumed run must reuse the frozen setup")

    def fake_local(*_args, report_dir, **_kwargs):
        frame = pd.DataFrame([
            {"outcome": "referenced_existing_state"},
            {"outcome": "completed_local_patch_run"},
        ])
        frame.to_csv(Path(report_dir) / "island_local_patch_execution.csv", index=False)
        return frame

    def fake_analysis(_path, *, output_dir):
        frame = pd.DataFrame({"analysis_execution_status": ["ready_alias", "ready_direct_local_patch"]})
        frame.to_csv(Path(output_dir) / "island_analysis_manifest.csv", index=False)
        return frame

    monkeypatch.setattr(island_module, "build_island_equivalence_registry", fail_rebuild)
    monkeypatch.setattr(island_module, "build_island_observation_manifest", fail_rebuild)
    monkeypatch.setattr(island_module, "run_island_local_patch_observations", fake_local)
    monkeypatch.setattr(island_module, "build_island_analysis_manifest", fake_analysis)

    result = run_island_experiment(
        reviewed_setup_dir=reviewed_setup,
        base_experiment_dir=base,
        output_dir=output,
        base_mask_path=tmp_path / "base.tif",
        parent_links_path=tmp_path / "links.gpkg",
        parent_nodes_path=tmp_path / "nodes.gpkg",
        boundary_component_labels_path=tmp_path / "labels.tif",
        wet_reference_mask_path=tmp_path / "wet.tif",
        exit_sides="S",
        example_id="56",
        execution_mode="local_patch",
        resume_existing_experiment=True,
    )

    manifest = json.loads(result["experiment_manifest"].read_text())
    assert manifest["resumed_existing_experiment"] is True
    assert result["setup_dir"] == setup
