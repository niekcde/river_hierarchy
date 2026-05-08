# Regular Synthetic

This folder is the operational area for the regular synthetic workflow.

Current step order:

1. Build geometry recipes.
2. Sample widths on those geometries.
3. Keep the explicit single-edge control separate.
4. Select routing targets and run RAPID.
5. Build post-run K metrics.
6. Do notebook-based inspection on top if needed.

The config-driven entry points are:

- `python -m synthetic_runs.pipelines.regular build-geometry --config ...`
- `python -m synthetic_runs.pipelines.regular sample-widths --config ...`
- `python -m synthetic_runs.pipelines.regular route --config ...`
- `python -m synthetic_runs.pipelines.regular k-metrics --config ...`

Use [run_all.sh](/Users/6256481/Code/river-hierarchy/synthetic_runs/regular/run_all.sh) to chain the current extracted steps.

Important: the shell wrapper expects the active `python` environment to include
the RAPID runtime dependencies for the routing step.
