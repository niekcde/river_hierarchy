"""Backward-compatible helpers for the legacy RivGraph smoke workflow."""

from __future__ import annotations

from pathlib import Path

from rivgraph_centerline.workflow import (
    PROJECT_ROOT,
    WorkflowPaths,
    apply_edits_for_run as _apply_edits_for_run,
    choose_projected_crs,
    load_json,
    prepare_binary_projected_mask,
    prepare_mask_for_run as _prepare_mask_for_run,
    resolve_input_mask,
    run_all_for_run as _run_all_for_run,
    run_rivgraph_for_run as _run_rivgraph_for_run,
    write_json,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "rivgraph_centerline" / "outputs" / "smoke_tests"


class SmokePaths(WorkflowPaths):
    """Conventional paths for a single smoke-test run."""

    @classmethod
    def from_name(cls, name: str, run_dir: Path | None = None) -> "SmokePaths":
        base = WorkflowPaths.from_name(name, run_dir, output_root=DEFAULT_OUTPUT_ROOT)
        return cls(name=base.name, run_dir=base.run_dir)

    @property
    def smoke_summary(self) -> Path:
        return self.run_dir / "smoke_summary.json"


def _write_legacy_smoke_summary(paths: SmokePaths, summary: dict[str, object]) -> None:
    write_json(paths.smoke_summary, summary)


def prepare_mask_for_run(
    *,
    name: str,
    source_mask: Path,
    run_dir: Path | None = None,
    threshold: float = 0.5,
    dst_crs: str | None = None,
    force: bool = False,
) -> tuple[SmokePaths, dict[str, object]]:
    _, summary = _prepare_mask_for_run(
        name=name,
        source_mask=source_mask,
        run_dir=run_dir,
        output_root=DEFAULT_OUTPUT_ROOT,
        threshold=threshold,
        dst_crs=dst_crs,
        force=force,
    )
    paths = SmokePaths.from_name(name, run_dir)
    return paths, summary


def apply_edits_for_run(
    *,
    name: str,
    run_dir: Path | None = None,
    edits: Path | None = None,
    base_mask: Path | None = None,
    output_mask: Path | None = None,
    layer: str | None = None,
    action_field: str = "action",
    all_touched: bool = True,
) -> tuple[SmokePaths, dict[str, object]]:
    _, summary = _apply_edits_for_run(
        name=name,
        run_dir=run_dir,
        output_root=DEFAULT_OUTPUT_ROOT,
        edits=edits,
        base_mask=base_mask,
        output_mask=output_mask,
        layer=layer,
        action_field=action_field,
        all_touched=all_touched,
    )
    paths = SmokePaths.from_name(name, run_dir)
    return paths, summary


def run_rivgraph_for_run(
    *,
    name: str,
    exit_sides: str,
    run_dir: Path | None = None,
    mask: Path | None = None,
    mask_stage: str = "auto",
    vector_format: str = "gpkg",
    assign_flow: bool = False,
    verbose: bool = True,
) -> tuple[SmokePaths, dict[str, object]]:
    _, summary = _run_rivgraph_for_run(
        name=name,
        exit_sides=exit_sides,
        run_dir=run_dir,
        output_root=DEFAULT_OUTPUT_ROOT,
        mask=mask,
        mask_stage=mask_stage,
        vector_format=vector_format,
        assign_flow=assign_flow,
        verbose=verbose,
    )
    paths = SmokePaths.from_name(name, run_dir)
    _write_legacy_smoke_summary(paths, summary)
    return paths, summary


def run_all_for_run(
    *,
    name: str,
    source_mask: Path,
    exit_sides: str,
    run_dir: Path | None = None,
    threshold: float = 0.5,
    dst_crs: str | None = None,
    edits: Path | None = None,
    vector_format: str = "gpkg",
    assign_flow: bool = False,
    force_prepare: bool = False,
    verbose: bool = True,
) -> dict[str, object]:
    summary = _run_all_for_run(
        name=name,
        source_mask=source_mask,
        exit_sides=exit_sides,
        run_dir=run_dir,
        output_root=DEFAULT_OUTPUT_ROOT,
        threshold=threshold,
        dst_crs=dst_crs,
        edits=edits,
        vector_format=vector_format,
        assign_flow=assign_flow,
        force_prepare=force_prepare,
        verbose=verbose,
    )
    paths = SmokePaths.from_name(name, run_dir)
    _write_legacy_smoke_summary(paths, summary["rivgraph"])
    return summary
