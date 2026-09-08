from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "RAPID" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rapid_tools.engine import RapidResultStorageConfig
from rapid_tools.island_observations import (
    _remove_disposable_prep,
    prepare_island_observations,
    run_prepared_island_observations,
)


@pytest.mark.parametrize("backend", ["island_full_raster", "island_local_patch", "island_panelized"])
def test_prepare_island_observations_only_prepares_real_islands(
    tmp_path: Path,
    monkeypatch,
    backend: str,
) -> None:
    forcing = tmp_path / "forcing.parquet"
    forcing.touch()
    base = tmp_path / "base_experiment"
    base.mkdir()
    (base / "rapid_prep_manifest.json").write_text(json.dumps({
        "forcing_path": str(forcing),
        "forcing_config": {"resample_seconds": 300},
        "prep_config": {"kb_mode": "fixed", "preferred_routing_dt_seconds": 300},
        "reference_kb_summary": {},
        "timestep_plan": {"selected_unique_routing_dt_seconds": [300]},
    }))

    real_state = tmp_path / "island_real"
    alias_state = tmp_path / "alias_state"
    for directory in (real_state, alias_state):
        directory.mkdir()
    real_paths = []
    for name in ("links.gpkg", "nodes.gpkg", "widths.gpkg"):
        path = real_state / name
        path.touch()
        real_paths.append(path)
    hydraulic_dir = real_state / "hydraulic_context"
    hydraulic_dir.mkdir()
    alias_hydraulic_dir = alias_state / "hydraulic_context"
    alias_hydraulic_dir.mkdir()

    analysis = pd.DataFrame([
        {
            "island_id": "island_real",
            "analysis_backend": backend,
            "analysis_state_dir": str(real_state),
            "analysis_directed_links_path": str(real_paths[0]),
            "analysis_directed_nodes_path": str(real_paths[1]),
            "analysis_routing_width_links_path": str(real_paths[2]),
            "source_unit_id": 4,
            "anchor_bifurcation": 1,
            "anchor_confluence": 2,
        },
        {
            "island_id": "island_alias",
            "analysis_backend": "independent_unit_alias",
            "analysis_state_dir": str(alias_state),
            "canonical_state_dir": str(alias_state),
            "source_unit_id": 5,
            "anchor_bifurcation": 3,
            "anchor_confluence": 4,
        },
        {
            "island_id": "island_excluded",
            "analysis_backend": "island_panelized",
            "analysis_state_dir": str(real_state),
            "strict_analysis_status": "exclude_pruning_or_mask_topology_exception",
        },
    ])
    analysis_path = tmp_path / "island_analysis_manifest.csv"
    analysis.to_csv(analysis_path, index=False)
    hydraulic_path = tmp_path / "island_hydraulic_context_registry.csv"
    pd.DataFrame([
        {"island_id": "island_real", "hydraulic_context_dir": str(hydraulic_dir)},
        {"island_id": "island_alias", "hydraulic_context_dir": str(alias_hydraulic_dir)},
    ]).to_csv(hydraulic_path, index=False)

    calls = []

    def fake_prepare(context, **kwargs):
        calls.append((context, kwargs))
        return {
            "state_id": context.state_id,
            "status": "prepared",
            "rapid_prep_dir": str(context.rapid_prep_dir),
            "rapid_run_dir": str(context.rapid_run_dir),
        }

    monkeypatch.setattr("rapid_tools.island_observations.prepare_state", fake_prepare)
    result = prepare_island_observations(
        analysis_path,
        hydraulic_context_registry_path=hydraulic_path,
        base_experiment_dir=base,
    )

    assert [context.state_id for context, _ in calls] == ["island_real"]
    assert calls[0][1]["planned_routing_dt_seconds"] == 300
    assert result["status"].value_counts().to_dict() == {
        "prepared": 1,
        "referenced_existing_independent_state": 1,
        "excluded_topology_review": 1,
    }
    assert (tmp_path / "island_rapid_prep_registry.csv").exists()


def test_prepare_island_observations_inherits_source_unit_timestep_pair(
    tmp_path: Path,
    monkeypatch,
) -> None:
    forcing = tmp_path / "forcing.parquet"
    forcing.touch()
    base = tmp_path / "base_experiment"
    base.mkdir()
    (base / "rapid_prep_manifest.json").write_text(json.dumps({
        "forcing_path": str(forcing),
        "forcing_config": {"resample_seconds": 980},
        "prep_config": {"kb_mode": "fixed", "preferred_routing_dt_seconds": 980},
        "reference_kb_summary": {},
        "timestep_plan": {
            "selected_unique_routing_dt_seconds": [705, 740],
            "states": {
                "base_experiment::S001_unit_7": {
                    "routing_dt_seconds": 740,
                    "forcing_dt_seconds": 740,
                    "selection_source": "coordinated_state_specific",
                },
            },
        },
    }))
    pd.DataFrame([{
        "state_id": "S001_unit_7",
        "selection_type": "unit",
        "selected_unit_ids": "7",
    }]).to_csv(base / "state_registry.csv", index=False)

    state = tmp_path / "island_real"
    state.mkdir()
    products = []
    for name in ("links.gpkg", "nodes.gpkg", "widths.gpkg"):
        path = state / name
        path.touch()
        products.append(path)
    hydraulic = state / "hydraulic_context"
    hydraulic.mkdir()
    analysis_path = tmp_path / "island_analysis_manifest.csv"
    pd.DataFrame([{
        "island_id": "island_real",
        "analysis_backend": "island_local_patch",
        "analysis_state_dir": str(state),
        "analysis_directed_links_path": str(products[0]),
        "analysis_directed_nodes_path": str(products[1]),
        "analysis_routing_width_links_path": str(products[2]),
        "source_unit_id": 7,
    }]).to_csv(analysis_path, index=False)
    hydraulic_path = tmp_path / "island_hydraulic_context_registry.csv"
    pd.DataFrame([{"island_id": "island_real", "hydraulic_context_dir": str(hydraulic)}]).to_csv(
        hydraulic_path, index=False
    )
    calls = []

    def fake_prepare(context, **kwargs):
        calls.append((context, kwargs))
        return {"state_id": context.state_id, "status": "prepared"}

    monkeypatch.setattr("rapid_tools.island_observations.prepare_state", fake_prepare)
    prepare_island_observations(
        analysis_path,
        hydraulic_context_registry_path=hydraulic_path,
        base_experiment_dir=base,
    )

    assert calls[0][1]["planned_routing_dt_seconds"] is None
    assert calls[0][1]["forcing_config"].resample_seconds == 740
    assert calls[0][1]["prep_config"].preferred_routing_dt_seconds == 740
    assert calls[0][1]["timestep_selection"]["selection_mode"] == "inherited_base_source_unit_timestep"


def test_targeted_island_preparation_preserves_non_target_registry_records(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A failed island can be retried without re-preparing completed peers."""
    forcing = tmp_path / "forcing.parquet"
    forcing.touch()
    base = tmp_path / "base_experiment"
    base.mkdir()
    (base / "rapid_prep_manifest.json").write_text(json.dumps({
        "forcing_path": str(forcing),
        "forcing_config": {"resample_seconds": 300},
        "prep_config": {"kb_mode": "fixed", "preferred_routing_dt_seconds": 300},
        "reference_kb_summary": {},
        "timestep_plan": {"selected_unique_routing_dt_seconds": [300]},
    }))
    state = tmp_path / "island_state"
    state.mkdir()
    products = []
    for name in ("links.gpkg", "nodes.gpkg", "widths.gpkg"):
        product = state / name
        product.touch()
        products.append(product)
    hydraulic = state / "hydraulic"
    hydraulic.mkdir()
    analysis_path = tmp_path / "island_analysis_manifest.csv"
    pd.DataFrame([
        {
            "island_id": "retry_me",
            "analysis_backend": "island_local_patch",
            "analysis_state_dir": str(state),
            "analysis_directed_links_path": str(products[0]),
            "analysis_directed_nodes_path": str(products[1]),
            "analysis_routing_width_links_path": str(products[2]),
        },
        {
            "island_id": "already_an_alias",
            "analysis_backend": "independent_unit_alias",
            "analysis_state_dir": str(state),
            "canonical_state_dir": str(state),
        },
    ]).to_csv(analysis_path, index=False)
    hydraulic_path = tmp_path / "island_hydraulic_context_registry.csv"
    pd.DataFrame([
        {"island_id": "retry_me", "hydraulic_context_dir": str(hydraulic)},
        {"island_id": "already_an_alias", "hydraulic_context_dir": str(hydraulic)},
    ]).to_csv(hydraulic_path, index=False)
    pd.DataFrame([
        {"island_id": "retry_me", "status": "failed", "rapid_prep_dir": "old-prep"},
        {"island_id": "already_an_alias", "status": "referenced_existing_independent_state"},
    ]).to_csv(tmp_path / "island_rapid_prep_registry.csv", index=False)
    calls = []

    def fake_prepare(context, **kwargs):
        calls.append(context.state_id)
        return {"state_id": context.state_id, "status": "prepared"}

    monkeypatch.setattr("rapid_tools.island_observations.prepare_state", fake_prepare)
    result = prepare_island_observations(
        analysis_path,
        hydraulic_context_registry_path=hydraulic_path,
        base_experiment_dir=base,
        island_ids=["retry_me"],
    )

    assert calls == ["retry_me"]
    assert result[["island_id", "status"]].to_dict("records") == [
        {"island_id": "retry_me", "status": "prepared"},
        {"island_id": "already_an_alias", "status": "referenced_existing_independent_state"},
    ]


def test_run_prepared_island_observations_applies_compact_storage_policy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prep_dir = tmp_path / "prep"
    prep_dir.mkdir()
    (prep_dir / "rapid_prep_manifest.json").write_text("{}")
    run_dir = tmp_path / "run"
    registry_path = tmp_path / "island_rapid_prep_registry.csv"
    pd.DataFrame([
        {
            "island_id": "island_real",
            "status": "prepared",
            "rapid_prep_dir": str(prep_dir),
            "rapid_run_dir": str(run_dir),
        },
        {"island_id": "island_alias", "status": "referenced_existing_independent_state"},
        {"island_id": "island_excluded", "status": "excluded_topology_review"},
    ]).to_csv(registry_path, index=False)

    qout = run_dir / "Qout.nc"
    compact_calls = []

    def fake_run(prep, *, output_path):
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"qout")
        return output

    def fake_hydrograph(prep, output, run, *, config):
        return {"hydrograph_status": "computed"}

    def fake_publish(*, prep_dir, run_dir, qout_path, storage):
        compact_calls.append((prep_dir, run_dir, qout_path, storage))
        return {"compact_qout": storage.compact_qout}

    monkeypatch.setattr("rapid_tools.island_observations.run_prepared_state", fake_run)
    monkeypatch.setattr("rapid_tools.island_observations.write_hydrograph_outputs", fake_hydrograph)
    monkeypatch.setattr("rapid_tools.island_observations._publish_compact_result", fake_publish)

    result = run_prepared_island_observations(
        registry_path,
        storage_config=RapidResultStorageConfig(compact_qout=True),
    )

    assert result["status"].tolist() == [
        "ran", "referenced_existing_independent_state", "excluded_topology_review",
    ]
    assert len(compact_calls) == 1
    assert compact_calls[0][3].compact_qout is True


def test_run_prepared_island_observations_recovers_published_compact_result(
    tmp_path: Path,
) -> None:
    """A restart must not relabel a compact completed result as failed."""
    prep_dir = tmp_path / "removed-prep"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for name in (
        "Qout.nc.gz",
        "outlet_hydrograph.csv",
        "hydrograph_metrics.csv",
        "hydrograph_metrics.json",
        "rapid_link_mapping.parquet",
        "rapid_reach_attributes.gpkg",
    ):
        (run_dir / name).write_bytes(b"published")
    registry_path = tmp_path / "island_rapid_prep_registry.csv"
    pd.DataFrame([{
        "island_id": "island_completed",
        "status": "prepared",
        "rapid_prep_dir": str(prep_dir),
        "rapid_run_dir": str(run_dir),
    }]).to_csv(registry_path, index=False)

    result = run_prepared_island_observations(
        registry_path,
        storage_config=RapidResultStorageConfig(
            compact_qout=True,
            retain_prepared_inputs=False,
            write_mapping_parquet=True,
        ),
    )

    row = result.iloc[0]
    assert row["status"] == "ran"
    assert bool(row["recovered_existing_published_result"])
    assert row["rapid_prep_retained"] == False


def test_recovered_published_result_drops_stale_prep_when_requested(tmp_path: Path) -> None:
    prep_dir = tmp_path / "stale-prep"
    prep_dir.mkdir()
    (prep_dir / "regenerable_input.txt").write_text("safe to remove after verified publication")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for name in (
        "Qout.nc.gz",
        "outlet_hydrograph.csv",
        "hydrograph_metrics.csv",
        "hydrograph_metrics.json",
        "rapid_link_mapping.parquet",
        "rapid_reach_attributes.gpkg",
    ):
        (run_dir / name).write_bytes(b"published")
    registry_path = tmp_path / "island_rapid_prep_registry.csv"
    pd.DataFrame([{
        "island_id": "island_completed",
        "status": "prepared",
        "rapid_prep_dir": str(prep_dir),
        "rapid_run_dir": str(run_dir),
    }]).to_csv(registry_path, index=False)

    result = run_prepared_island_observations(
        registry_path,
        storage_config=RapidResultStorageConfig(
            compact_qout=True,
            retain_prepared_inputs=False,
            write_mapping_parquet=True,
        ),
    )

    assert result.loc[0, "status"] == "ran"
    assert result.loc[0, "rapid_prep_retained"] == False
    assert not prep_dir.exists()


def test_remove_disposable_prep_is_idempotent_for_missing_directory(tmp_path: Path) -> None:
    """A completed prior cleanup is not an error during restart recovery."""
    _remove_disposable_prep(tmp_path / "already_removed")


def test_remove_disposable_prep_ignores_missing_appledouble_sidecar(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """APFS may remove Finder sidecars while Python is walking a directory."""
    prep = tmp_path / "prep"
    prep.mkdir()

    def fake_rmtree(path, *, onexc):
        onexc(None, str(Path(path) / "._rat.csv"), FileNotFoundError("sidecar vanished"))
        Path(path).rmdir()

    monkeypatch.setattr("rapid_tools.island_observations.shutil.rmtree", fake_rmtree)
    _remove_disposable_prep(prep)
    assert not prep.exists()
