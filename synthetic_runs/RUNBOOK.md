# Synthetic Runbook

This is the current operational runbook for the extracted synthetic pipeline.
It documents the working step order only. The final shared analysis notebook is
still intentionally out of scope.

## Regular Synthetic

Current extracted steps:

1. Build geometry recipes.
2. Sample widths on those geometries.
3. Keep the explicit single-edge control separate.
4. Select routing targets and run RAPID.
5. Build post-run K metrics.

Run the whole extracted path with:

```bash
cd /Users/6256481/Code/river-hierarchy
./synthetic_runs/regular/run_all.sh
```

Step-specific entry points:

```bash
PYTHONPATH="synthetic_runs/src:RAPID/src" \
python -m synthetic_runs.pipelines.regular build-geometry \
  --config synthetic_runs/regular/configs/geometry.example.json
```

```bash
PYTHONPATH="synthetic_runs/src:RAPID/src" \
python -m synthetic_runs.pipelines.regular sample-widths \
  --config synthetic_runs/regular/configs/sampling.example.json
```

```bash
PYTHONPATH="synthetic_runs/src:RAPID/src" \
/opt/anaconda3/envs/UNC/bin/python -m synthetic_runs.pipelines.regular route \
  --config synthetic_runs/regular/configs/routing.example.json
```

```bash
PYTHONPATH="synthetic_runs/src:RAPID/src" \
/opt/anaconda3/envs/UNC/bin/python -m synthetic_runs.pipelines.regular k-metrics \
  --config synthetic_runs/regular/configs/k_metrics.example.json
```

Notes:

- the routing and K-metrics steps should use the environment that has the RAPID
  dependencies installed
- the explicit `single_edge` control is separate from the structural network
  recipes and is included during routing

## Sensitivity

Current extracted steps:

1. Build the 7 structural recipes.
2. Keep the explicit single-edge control separate.
3. Run the sensitivity grid.

Run the whole extracted path with:

```bash
cd /Users/6256481/Code/river-hierarchy
./synthetic_runs/sensitivity/run_all.sh
```

Step-specific entry points:

```bash
PYTHONPATH="synthetic_runs/src:RAPID/src" \
python -m synthetic_runs.pipelines.sensitivity build-recipes \
  --config synthetic_runs/sensitivity/configs/recipes.example.json
```

```bash
PYTHONPATH="synthetic_runs/src:RAPID/src" \
/opt/anaconda3/envs/UNC/bin/python -m synthetic_runs.pipelines.sensitivity run-grid \
  --config synthetic_runs/sensitivity/configs/grid.example.json
```

Notes:

- the 7 structural recipes stay distinct from the explicit `single_edge`
  baseline control
- downstream analysis still lives outside this runbook until the final notebook
  structure is decided
