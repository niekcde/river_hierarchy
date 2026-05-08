# Sensitivity

This folder is the operational area for the sensitivity workflow inside
`synthetic_runs/`.

Current extracted steps:

1. Build the 7 structural recipes.
2. Keep the explicit single-edge control separate.
3. Run the sensitivity grid.
4. Continue with notebook-based downstream analysis until that tail is
   extracted into code.

The config-driven entry points are:

- `python -m synthetic_runs.pipelines.sensitivity build-recipes --config ...`
- `python -m synthetic_runs.pipelines.sensitivity run-grid --config ...`

Use [run_all.sh](/Users/6256481/Code/river-hierarchy/synthetic_runs/sensitivity/run_all.sh) to chain the current extracted steps.

Generated outputs should remain outside Git. The downstream analysis notebook is
still the preserved legacy notebook until that part is extracted.
