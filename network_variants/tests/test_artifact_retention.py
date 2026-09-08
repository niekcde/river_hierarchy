from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from network_variants.artifact_retention import (
    _remove_tree_ignoring_disappeared_children,
    apply_successful_state_retention,
    apply_successful_state_retention_best_effort,
    build_cleanup_manifest,
    execute_cleanup_manifest,
    validate_artifact_profile,
)


def _published_state(root: Path) -> Path:
    state = root / "states" / "S001_unit_1"
    variant = state / "variant"
    directed = variant / "directed"
    widths = variant / "widths"
    directed.mkdir(parents=True)
    widths.mkdir(parents=True)
    (directed / "01__S001_unit_1_directed_links.gpkg").write_text("final links")
    (directed / "01__S001_unit_1_directed_nodes.gpkg").write_text("final nodes")
    (widths / "links_with_width_families.gpkg").write_text("final widths")
    return state


def test_analysis_profile_removes_successful_local_patch_tree_but_keeps_audits(
    tmp_path: Path,
) -> None:
    state = _published_state(tmp_path)
    attempt = state / "variant" / "local_patch_attempt"
    (attempt / "planning").mkdir(parents=True)
    (attempt / "planning" / "planning.json").write_text(json.dumps({"safe": True}))
    (attempt / "variant").mkdir()
    (attempt / "variant" / "raw_network.pkl").write_bytes(b"raw")

    result = apply_successful_state_retention(state, profile="analysis")

    assert result["local_patch_attempt"]["action"] == "removed_after_success"
    assert not attempt.exists()
    assert (state / "variant" / "directed" / "01__S001_unit_1_directed_links.gpkg").exists()
    assert (
        state
        / "variant"
        / "summary"
        / "local_patch_attempt_audits"
        / "planning"
        / "planning.json"
    ).exists()


def test_debug_profile_keeps_successful_local_patch_tree(tmp_path: Path) -> None:
    state = _published_state(tmp_path)
    attempt = state / "variant" / "local_patch_attempt"
    attempt.mkdir(parents=True)
    (attempt / "planning.json").write_text("{}")

    result = apply_successful_state_retention(state, profile="debug")

    assert result["local_patch_attempt"]["action"] == "retained_by_profile"
    assert attempt.exists()


def test_analysis_profile_compacts_final_width_samples_to_parquet(tmp_path: Path) -> None:
    state = _published_state(tmp_path)
    samples_path = state / "variant" / "widths" / "link_width_samples.csv"
    pd.DataFrame({"id_link": [1] * 5_000, "width_wet": [10.0] * 5_000}).to_csv(
        samples_path, index=False
    )

    result = apply_successful_state_retention(state, profile="analysis")

    parquet_path = samples_path.with_suffix(".parquet")
    assert result["width_samples"]["action"] == "converted_to_parquet_and_removed_csv"
    assert not samples_path.exists()
    compacted = pd.read_parquet(parquet_path)
    assert len(compacted) == 5_000
    assert compacted["width_wet"].eq(10.0).all()


def test_analysis_profile_removes_regenerable_raw_products_after_preserving_edit(
    tmp_path: Path,
) -> None:
    state = _published_state(tmp_path)
    variant = state / "variant"
    (variant / "mask").mkdir()
    (variant / "mask" / "collapse_edit_geometries.gpkg").write_bytes(b"frozen edit")
    (variant / "mask" / "collapsed.tif").write_bytes(b"mask")
    (variant / "rivgraph").mkdir()
    (variant / "rivgraph" / "network.pkl").write_bytes(b"raw")

    result = apply_successful_state_retention(
        state, profile="analysis", drop_regenerable_raw=True
    )

    assert result["regenerable_raw"]["action"] == "removed_after_success"
    assert not (variant / "mask").exists()
    assert not (variant / "rivgraph").exists()
    assert (variant / "summary" / "collapse_edit_geometries.gpkg").read_bytes() == b"frozen edit"


def test_fallback_attempt_is_retained_even_in_analysis_profile(tmp_path: Path) -> None:
    state = _published_state(tmp_path)
    attempt = state / "variant" / "local_patch_attempt"
    attempt.mkdir(parents=True)
    (attempt / "fallback_to_full_raster.json").write_text("{}")

    result = apply_successful_state_retention(state, profile="analysis")

    assert result["local_patch_attempt"]["action"] == "retained_fallback_diagnostic"
    assert attempt.exists()


def test_cleanup_manifest_is_non_destructive_and_marks_only_successful_attempts(tmp_path: Path) -> None:
    state = _published_state(tmp_path)
    variant = state / "variant"
    (variant / "local_patch_attempt").mkdir()
    (variant / "local_patch_attempt" / "planning.json").write_text("{}")
    (variant / "widths" / "link_width_samples.csv").write_text("id_link,width\n1,10\n")
    (variant / "mask").mkdir()
    (variant / "mask" / "collapsed.tif").write_bytes(b"mask")
    (variant / "rivgraph").mkdir()
    (variant / "rivgraph" / "network.pkl").write_bytes(b"raw")
    pd.DataFrame(
        [{"state_id": "S001_unit_1", "state_dir": str(state), "status": "complete"}]
    ).to_csv(tmp_path / "state_registry.csv", index=False)

    result = build_cleanup_manifest(tmp_path)

    attempt_row = result.rows.loc[
        result.rows["artifact_category"].eq("local_patch_attempt")
    ].iloc[0]
    assert bool(attempt_row["eligible_now"])
    assert attempt_row["recommended_action"] == "remove_after_success"
    assert (variant / "local_patch_attempt").exists()
    assert result.summary["destructive_action_performed"] is False


def test_cleanup_executor_dry_run_is_non_destructive(tmp_path: Path) -> None:
    state = _published_state(tmp_path)
    samples = state / "variant" / "widths" / "link_width_samples.csv"
    pd.DataFrame({"id_link": [1, 2], "width_wet": [10.0, 12.0]}).to_csv(samples, index=False)
    pd.DataFrame(
        [{"state_id": "S001_unit_1", "state_dir": str(state), "status": "complete"}]
    ).to_csv(tmp_path / "state_registry.csv", index=False)
    manifest = build_cleanup_manifest(tmp_path)
    manifest_path = tmp_path / "artifact_cleanup_manifest.csv"
    manifest.rows.to_csv(manifest_path, index=False)

    result = execute_cleanup_manifest(manifest_path)

    assert result.summary["destructive_action_performed"] is False
    assert result.rows.loc[result.rows["artifact_category"].eq("width_samples_csv"), "outcome"].iloc[0] == "planned"
    assert samples.exists()
    assert not samples.with_suffix(".parquet").exists()


def test_cleanup_executor_converts_only_verified_width_samples(tmp_path: Path) -> None:
    state = _published_state(tmp_path)
    variant = state / "variant"
    samples = variant / "widths" / "link_width_samples.csv"
    expected = pd.DataFrame({"id_link": [1] * 5_000, "width_wet": [10.0] * 5_000})
    expected.to_csv(samples, index=False)
    (variant / "mask").mkdir()
    (variant / "mask" / "collapsed.tif").write_bytes(b"raw")
    pd.DataFrame(
        [{"state_id": "S001_unit_1", "state_dir": str(state), "status": "complete"}]
    ).to_csv(tmp_path / "state_registry.csv", index=False)
    manifest = build_cleanup_manifest(tmp_path)
    manifest_path = tmp_path / "artifact_cleanup_manifest.csv"
    manifest.rows.to_csv(manifest_path, index=False)

    result = execute_cleanup_manifest(manifest_path, execute=True)

    width_row = result.rows.loc[result.rows["artifact_category"].eq("width_samples_csv")].iloc[0]
    assert width_row["outcome"] == "converted"
    assert not samples.exists()
    assert pd.read_parquet(samples.with_suffix(".parquet")).equals(expected)
    assert (variant / "mask" / "collapsed.tif").exists()
    assert result.summary["destructive_action_performed"] is True


def test_cleanup_executor_keeps_csv_when_existing_parquet_does_not_match(tmp_path: Path) -> None:
    state = _published_state(tmp_path)
    samples = state / "variant" / "widths" / "link_width_samples.csv"
    pd.DataFrame({"id_link": [1], "width_wet": [10.0]}).to_csv(samples, index=False)
    pd.DataFrame({"id_link": [1], "width_wet": [99.0]}).to_parquet(
        samples.with_suffix(".parquet"), index=False
    )
    pd.DataFrame(
        [{"state_id": "S001_unit_1", "state_dir": str(state), "status": "complete"}]
    ).to_csv(tmp_path / "state_registry.csv", index=False)
    manifest = build_cleanup_manifest(tmp_path)
    manifest_path = tmp_path / "artifact_cleanup_manifest.csv"
    manifest.rows.to_csv(manifest_path, index=False)

    result = execute_cleanup_manifest(manifest_path, execute=True)

    width_row = result.rows.loc[result.rows["artifact_category"].eq("width_samples_csv")].iloc[0]
    assert width_row["outcome"] == "failed"
    assert samples.exists()
    assert result.summary["destructive_action_performed"] is False


def test_cleanup_executor_retains_csv_when_new_parquet_is_not_smaller(tmp_path: Path) -> None:
    state = _published_state(tmp_path)
    samples = state / "variant" / "widths" / "link_width_samples.csv"
    pd.DataFrame({"id_link": [1], "width_wet": [10.0]}).to_csv(samples, index=False)
    pd.DataFrame(
        [{"state_id": "S001_unit_1", "state_dir": str(state), "status": "complete"}]
    ).to_csv(tmp_path / "state_registry.csv", index=False)
    manifest = build_cleanup_manifest(tmp_path)
    manifest_path = tmp_path / "artifact_cleanup_manifest.csv"
    manifest.rows.to_csv(manifest_path, index=False)

    result = execute_cleanup_manifest(manifest_path, execute=True)

    row = result.rows.loc[result.rows["artifact_category"].eq("width_samples_csv")].iloc[0]
    assert row["outcome"] == "retained_not_smaller"
    assert samples.exists()
    assert not samples.with_suffix(".parquet").exists()


def test_cleanup_executor_removes_successful_local_patch_after_retaining_audits(tmp_path: Path) -> None:
    state = _published_state(tmp_path)
    variant = state / "variant"
    attempt = variant / "local_patch_attempt"
    (attempt / "planning").mkdir(parents=True)
    (attempt / "planning" / "planning.json").write_text(json.dumps({"safe": True}))
    (attempt / "raw_network.pkl").write_bytes(b"raw")
    (variant / "mask").mkdir()
    (variant / "mask" / "collapsed.tif").write_bytes(b"mask")
    pd.DataFrame(
        [{"state_id": "S001_unit_1", "state_dir": str(state), "status": "complete"}]
    ).to_csv(tmp_path / "state_registry.csv", index=False)
    manifest = build_cleanup_manifest(tmp_path)
    manifest_path = tmp_path / "artifact_cleanup_manifest.csv"
    manifest.rows.to_csv(manifest_path, index=False)

    result = execute_cleanup_manifest(manifest_path, execute=True)

    row = result.rows.loc[result.rows["artifact_category"].eq("local_patch_attempt")].iloc[0]
    assert row["outcome"] == "removed_local_patch_attempt"
    assert not attempt.exists()
    assert (variant / "summary" / "local_patch_attempt_audits" / "planning" / "planning.json").exists()
    assert (variant / "summary" / "local_patch_attempt_cleanup_manifest.json").exists()
    assert (variant / "mask" / "collapsed.tif").exists()


def test_cleanup_executor_retains_fallback_local_patch_attempt(tmp_path: Path) -> None:
    state = _published_state(tmp_path)
    attempt = state / "variant" / "local_patch_attempt"
    attempt.mkdir(parents=True)
    (attempt / "fallback_to_full_raster.json").write_text("{}")
    pd.DataFrame(
        [{"state_id": "S001_unit_1", "state_dir": str(state), "status": "complete"}]
    ).to_csv(tmp_path / "state_registry.csv", index=False)
    manifest = build_cleanup_manifest(tmp_path)
    manifest_path = tmp_path / "artifact_cleanup_manifest.csv"
    manifest.rows.to_csv(manifest_path, index=False)

    result = execute_cleanup_manifest(manifest_path, execute=True)

    row = result.rows.loc[result.rows["artifact_category"].eq("local_patch_attempt")].iloc[0]
    assert row["outcome"] == "skipped"
    assert row["detail"] == "fallback_diagnostic_present"
    assert attempt.exists()


def test_tree_removal_tolerates_an_already_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "already_removed"
    _remove_tree_ignoring_disappeared_children(missing)


def test_tree_removal_tolerates_child_disappearing_during_traversal(
    tmp_path: Path, monkeypatch,
) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()

    def disappearing_child(_path, *, onexc):
        onexc(os.unlink, tree / "vanished", FileNotFoundError("vanished"))
        tree.rmdir()

    monkeypatch.setattr("network_variants.artifact_retention.shutil.rmtree", disappearing_child)

    _remove_tree_ignoring_disappeared_children(tree)

    assert not tree.exists()


def test_best_effort_retention_preserves_published_state_on_cleanup_error(
    tmp_path: Path, monkeypatch,
) -> None:
    state = _published_state(tmp_path)

    def fail_cleanup(*_args, **_kwargs):
        raise OSError("simulated cleanup failure")

    monkeypatch.setattr(
        "network_variants.artifact_retention.apply_successful_state_retention",
        fail_cleanup,
    )

    result = apply_successful_state_retention_best_effort(state, profile="analysis")

    assert result["status"] == "cleanup_warning"
    assert result["scientific_state_preserved"] is True
    assert (state / "variant" / "directed" / "01__S001_unit_1_directed_links.gpkg").exists()
    manifest = json.loads(
        (state / "variant" / "summary" / "artifact_retention_manifest.json").read_text()
    )
    assert manifest["error_type"] == "OSError"


def test_best_effort_retention_propagates_when_state_is_not_published(
    tmp_path: Path, monkeypatch,
) -> None:
    state = tmp_path / "states" / "S001_unit_1"
    state.mkdir(parents=True)

    def fail_cleanup(*_args, **_kwargs):
        raise OSError("simulated cleanup failure")

    monkeypatch.setattr(
        "network_variants.artifact_retention.apply_successful_state_retention",
        fail_cleanup,
    )

    try:
        apply_successful_state_retention_best_effort(state, profile="analysis")
    except OSError as exc:
        assert "simulated cleanup failure" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Cleanup failure was hidden for an unpublished state")


def test_invalid_profile_is_rejected() -> None:
    assert validate_artifact_profile("analysis") == "analysis"
    try:
        validate_artifact_profile("tiny")
    except ValueError as exc:
        assert "Unsupported artifact profile" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Invalid profile was accepted")
