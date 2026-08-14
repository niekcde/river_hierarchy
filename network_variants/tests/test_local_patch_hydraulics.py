from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from network_variants.hydraulic_context import SlopeConfig
from network_variants.local_patch_hydraulics import (
    materialize_local_patch_hydraulic_products,
)
from network_variants.local_patch_merge import LocalPatchMergeResult
from network_variants.local_patch_products import LocalPatchProductResult


def test_phase6b_reuses_parent_context_and_localizes_solver_cache(tmp_path: Path) -> None:
    experiment = tmp_path / "example_indep__anchored"
    experiment.mkdir()
    (experiment / "reference_section").mkdir()
    reference_summary = experiment / "reference_section/reference_section_summary.json"
    reference_summary.write_text("{}\n", encoding="utf-8")
    (experiment / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "options": {
                    "hierarchy": {
                        "max_path_cutoff": 77,
                        "max_paths": 321,
                        "pixel_width_fields": ["wid_adj_routing"],
                        "pixel_width_percentiles": [50],
                        "use_pixel_widths_for_extremes": False,
                        "debug_hierarchy": True,
                        "equivalent_resistance_config": {
                            "resistance_law": "chezy",
                            "solver_backend": "energy_sparse",
                            "width_field_candidates": ["wid_adj_routing", "wid_adj_wet"],
                            "solver_cache_dir": "/stale/machine/path",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    links_path = tmp_path / "links.gpkg"
    nodes_path = tmp_path / "nodes.gpkg"
    links_path.touch()
    nodes_path.touch()
    node_match_path = tmp_path / "node_merge_audit.csv"
    node_match_path.write_text(
        "child_id_node,final_node_id,matched_parent_node_id,match_within_tolerance\n9,1,1,True\n",
        encoding="utf-8",
    )
    collapse_components_path = tmp_path / "collapse_components.csv"
    collapse_components_path.write_text(
        "component_id,link_ids,node_ids\nC1,1,1\n",
        encoding="utf-8",
    )
    parent_state = experiment / "states/S000_base"
    parent_context = parent_state / "hydraulic_context/link_hydraulic_context.csv"
    parent_context.parent.mkdir(parents=True)
    parent_context.write_text("id_link,slope_used\n1,0.001\n", encoding="utf-8")
    (parent_context.parent / "node_hydraulic_context.csv").write_text(
        "id_node,rapid_wse_for_slope\n1,1.0\n", encoding="utf-8"
    )

    products = LocalPatchProductResult(
        status="complete_width_length_products",
        output_paths={"links_with_width_families": str(links_path)},
        summary={"recalculated_link_ids": [9, 10]},
    )
    merge = LocalPatchMergeResult(
        status="complete_conservative_merge",
        output_paths={
            "merged_nodes": str(nodes_path),
            "node_merge_audit": str(node_match_path),
        },
        summary={},
    )
    calls: dict[str, object] = {}

    def fake_workflow_runner(links, nodes, **kwargs):
        calls["workflow"] = (Path(links), Path(nodes), kwargs)
        return object()

    def fake_workflow_writer(output_dir, workflow, *, manifest_overrides):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "unit_summary.csv").write_text("unit_id\n1\n", encoding="utf-8")
        calls["writer"] = (Path(output_dir), workflow, manifest_overrides)

    def fake_hydraulic_runner(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        link_path = output_dir / "link_hydraulic_context.csv"
        node_path = output_dir / "node_hydraulic_context.csv"
        manifest_path = output_dir / "hydraulic_context_manifest.json"
        link_path.write_text("id_link\n1\n", encoding="utf-8")
        node_path.write_text("id_node\n1\n", encoding="utf-8")
        manifest_path.write_text("{}\n", encoding="utf-8")
        calls["hydraulic"] = kwargs
        return SimpleNamespace(
            link_context_path=link_path,
            node_context_path=node_path,
            manifest_path=manifest_path,
        )

    result = materialize_local_patch_hydraulic_products(
        products_result=products,
        merge_result=merge,
        collapse_components_path=collapse_components_path,
        experiment_dir=experiment,
        parent_state_dir=parent_state,
        output_dir=tmp_path / "phase6b",
        state_id="S007_unit_7_local",
        slope_config=SlopeConfig(
            min_slope=2e-6,
            slope_profile_mode="reference_section",
            reference_section_summary_path="/stale/reference.json",
        ),
        unit_workflow_runner=fake_workflow_runner,
        unit_workflow_writer=fake_workflow_writer,
        hydraulic_context_runner=fake_hydraulic_runner,
    )

    workflow_kwargs = calls["workflow"][2]
    assert workflow_kwargs["max_path_cutoff"] == 77
    assert workflow_kwargs["max_paths"] == 321
    assert workflow_kwargs["use_pixel_widths_for_extremes"] is False
    equivalent = workflow_kwargs["equivalent_resistance_config"]
    assert equivalent.solver_backend == "energy_sparse"
    assert equivalent.width_field_candidates == ("wid_adj_routing", "wid_adj_wet")
    assert Path(equivalent.solver_cache_dir) == experiment / "equivalent_resistance/solve_cache"

    hydraulic_kwargs = calls["hydraulic"]
    assert Path(hydraulic_kwargs["parent_link_context_path"]) == parent_context
    assert Path(hydraulic_kwargs["collapse_components_path"]) == collapse_components_path
    normalized_match_path = Path(hydraulic_kwargs["node_match_path"])
    assert normalized_match_path.name == "final_node_match.csv"
    assert "1,1,1,True" in normalized_match_path.read_text(encoding="utf-8")
    assert Path(hydraulic_kwargs["hierarchy_output_dir"]) == tmp_path / "phase6b/hierarchy"
    assert hydraulic_kwargs["config"].min_slope == 2e-6
    assert Path(hydraulic_kwargs["config"].reference_section_summary_path) == reference_summary
    assert result.status == "complete_hierarchy_hydraulic_products"
    assert result.summary["recalculated_width_link_ids"] == [9, 10]
    assert Path(result.output_paths["manifest"]).exists()
