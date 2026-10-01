from __future__ import annotations

import json
from pathlib import Path

from network_variants.experiment_alias_promotion import (
    apply_experiment_alias_promotion,
    build_experiment_alias_promotion_plan,
)


def _write_roots(root: Path) -> None:
    for name in (
        "15_indep__anchored",
        "15_compound_bubbles__anchored",
        "15_island__anchored",
        "15_indep_direction130__anchored",
        "15_compound_bubbles_direction130__anchored",
        "15_island_direction130__anchored",
    ):
        path = root / name
        path.mkdir()
        (path / "identity.txt").write_text(name)


def test_build_experiment_alias_promotion_plan_is_non_mutating(tmp_path: Path) -> None:
    _write_roots(tmp_path)
    plan = build_experiment_alias_promotion_plan(tmp_path, example_id=15)

    assert [item["action"] for item in plan["mappings"]] == [
        "archive_then_alias",
        "archive_then_alias",
        "archive_then_alias",
    ]
    assert (tmp_path / "15_indep__anchored").is_dir()
    assert not (tmp_path / "_archived_pre_correction").exists()


def test_apply_experiment_alias_promotion_archives_old_roots_and_creates_aliases(tmp_path: Path) -> None:
    _write_roots(tmp_path)
    plan = build_experiment_alias_promotion_plan(tmp_path, example_id=15)
    result = apply_experiment_alias_promotion(plan)

    archive = Path(result["archive_directory"])
    assert archive.is_dir()
    assert (archive / "15_indep__anchored" / "identity.txt").read_text() == "15_indep__anchored"
    canonical = tmp_path / "15_indep__anchored"
    assert canonical.is_symlink()
    assert canonical.resolve() == (tmp_path / "15_indep_direction130__anchored").resolve()
    assert (canonical / "identity.txt").read_text() == "15_indep_direction130__anchored"
    manifest = Path(result["manifest_path"])
    assert json.loads(manifest.read_text())["mode"] == (
        "archive_old_canonical_roots_then_create_canonical_symlink_aliases"
    )

    repeat = build_experiment_alias_promotion_plan(tmp_path, example_id=15)
    assert [item["action"] for item in repeat["mappings"]] == ["already_promoted"] * 3
