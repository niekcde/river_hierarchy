# Running controlled synthetic experiments

Run commands from the repository root with the existing scientific environment:
`/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python`.
The entry point adds the new package, repository, and shared RAPID to its own
import path. It never imports the legacy synthetic package.

## Selected hydraulic workflow (current)

The selected synthetic setup is BASED reference kb, local width–depth alpha=.65,
n=.035, X=.1 and 200m target/floor. It is defined in
`configs/selected_hydraulic_reference.json`. The versioned runner preserves the
legacy outputs and deduplicates exact routing inputs:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python \
  synthetic_experiments/investigations/selected_hydraulic_workflow.py all --workers 4
```

The output root is `synthetic_experiments/outputs/hydraulic_alpha065_20260930`.
`plan`, `run`, and `analyse` can also be invoked separately; rerunning reuses
complete jobs with the same plan identity. Implementation/selection changes
require a new version directory. Each job records its own BASED kb and complete
routing settings. Do not use the scalar kb in older run manifests for this run.

Rebuild the analysis under that output root, in this order:

```bash
export SYNTHETIC_RESULTS_ROOT="$PWD/synthetic_experiments/outputs/hydraulic_alpha065_20260930"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python synthetic_experiments/investigations/current_synthetic_analysis.py
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python synthetic_experiments/investigations/compact_predictor_selection.py
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python synthetic_experiments/investigations/omega_numerical_check.py
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python synthetic_experiments/investigations/embedded_predictor_validation.py
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python synthetic_experiments/investigations/selected_hydraulic_report.py
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python synthetic_experiments/investigations/selected_hydraulic_applicability.py
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python synthetic_experiments/investigations/timing_response_comparison.py
```

The main notebooks resolve `configs/active_analysis.json`; an explicit
`SYNTHETIC_RESULTS_ROOT` overrides it. Update the registry only after the version
has complete routing, derived analysis and QA. The response definitions and
original validation partitions are unchanged. Existing geometry is reused only
where invariant, with descriptor/split parity checked. Real-network results are
not rerun or used to choose a real-case exponent here.

## Legacy hydraulic commands

The commands below reproduce the earlier constant-kb experiment, not the
selected alpha=.65 setup. Retain its outputs for matched historical comparison.

## Baseline and checks

```bash
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python synthetic_experiments/run_experiment.py all \
  --config synthetic_experiments/configs/baseline.json \
  --output synthetic_experiments/outputs/baseline_island

/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python synthetic_experiments/run_experiment.py validate \
  --config synthetic_experiments/configs/baseline.json \
  --output synthetic_experiments/outputs/baseline_island

/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python synthetic_experiments/run_experiment.py all \
  --config synthetic_experiments/configs/extremes.json \
  --output synthetic_experiments/outputs/extremes_island
```

`validate` compares the baseline with twice the buffer, a coarser stable timestep,
and 100 m segmentation (with a correspondingly 100 m effective-length floor).
It checks spin-up sensitivity; the other differences are reported rather than
being silently declared converged.

## Full grid

```bash
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python synthetic_experiments/run_experiment.py plan \
  --config synthetic_experiments/configs/grid.json \
  --output synthetic_experiments/outputs/full_grid_island

/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python synthetic_experiments/run_experiment.py run \
  --output synthetic_experiments/outputs/full_grid_island

/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python synthetic_experiments/run_experiment.py analyse \
  --output synthetic_experiments/outputs/full_grid_island
```

`plan` performs no routing. Inspect `manifest.json` for the exact counts.
`run --limit 100` attempts up to 100 unfinished jobs; running again resumes.
Completed results are reused only when the job ID and all required result files
are present. Failures are recorded separately and retried on the next run.
Do not run multiple processes against the same output directory.

Configuration or scientific-code changes require a new output directory. Plans
contain a fingerprint of the implementation, relevant shared modules, and
scientific-library versions; stale plans are rejected. The directory cannot be
silently repurposed for a changed configuration. Outputs are ignored by Git.

## Reuse the completed six-comparison grid

The island extension uses new output directories and leaves the original
`baseline`, `extremes`, and `full_grid` outputs intact. Add `--reuse-from` to
`plan` or `all`, or run the explicit reuse command after planning:

```bash
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python synthetic_experiments/run_experiment.py reuse \
  --output synthetic_experiments/outputs/full_grid_island \
  --reuse-from synthetic_experiments/outputs/full_grid
```

Reuse requires exact canonical network, routing settings, timestep and window
matches, plus a matching execution-only routing fingerprint. The old complete
source fingerprint was verified before this extension; its routing provenance
is recorded in `configs/routing_compatibility.json`. Unrecognized code revisions
are rejected. Newer manifests record routing provenance directly. Files are
copied, not linked, and each reused metrics file records its source. The
`reuse_status.json` report lists reused jobs and the remaining routing work.
If the extra state changes a configuration's coordinated clock, its previous
jobs are not reused under a different clock.

The notebook defaults point to the extended full-grid and baseline outputs.
All four notebooks should be executed after the extended run and analysis finish.

## Products

- `manifest.json`: settings, code provenance, exact counts and unique run IDs.
- `configurations.json`: grid parameters and coordinated numerical settings.
- `states.csv`: original/child state references mapped to unique routing jobs.
- `comparisons.csv`: the seven parent–child pairs for every configuration.
- `jobs/*.json`: complete canonical hydraulic network and routing job definition.
- `runs/<id>/hydrograph.parquet`: prescribed inlet and routed outlet discharge.
- `runs/<id>/segments.csv`: lengths, widths, K, celerity, and active safeguards.
- `runs/<id>/metrics.json`: timing, peaks, conservation, runtime, and window checks.
- `runs/<id>/failure.json`: failure reason/traceback, if a job failed.
- `results.csv`: completed paired comparisons with resistance predictors.
- `pending_comparisons.csv`: pairs whose routing jobs have not completed.
- `figures/`: signed/log lag and speed views, raw-ratio alternatives, baseline
  parameter slices, and representative paired hydrographs.
- `numerical_validation.json`: baseline resolution/window diagnostics.

The runner retains inlet/outlet hydrographs and segment attributes, not full
reach-by-time discharge arrays. Resistance remains geometry-only. Predicted
speed changes use modeled lag and reference extent, not observed gauge speeds.

## Validation

```bash
PYTHONPATH=synthetic_experiments/src:RAPID/src:. \
  /opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python -m pytest synthetic_experiments/tests -q
```

Tests include topology after collapse, width budgets, path head drops, shared
segmentation parity, timestep-invariant forcing volume, closed resistance
comparison domains, uniform-width resistance-ratio invariance, routing parity
with the shared RAPID file engine, and resumable end-to-end comparisons.
