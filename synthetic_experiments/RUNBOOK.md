# Running controlled synthetic experiments

> **26 October 2026 restart:** the selected September results below remain historical fixed-X results. The current-method Cunge restart protocol and isolated pilot are documented in [CUNGE_PROTOCOL.md](CUNGE_PROTOCOL.md). The active analysis selector has not been promoted.

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

## Current three-method comparison (27 October 2026)

Use `run_depth_comparison.py` for the 2:1 forcing, branch-width factors and conservative variable-depth method. The old `run_experiment.py` still represents historical routing.

```bash
# Plan the representative configurations and all three methods (does not route).
python synthetic_experiments/run_depth_comparison.py \
  --config synthetic_experiments/configs/depth_comparison_20261027.json \
  --output synthetic_experiments/outputs/my_depth_comparison

# Execute/resume that identical plan.
python synthetic_experiments/run_depth_comparison.py \
  --config synthetic_experiments/configs/depth_comparison_20261027.json \
  --output synthetic_experiments/outputs/my_depth_comparison --run

# A selected case/method (use a separate directory for a different plan).
python synthetic_experiments/run_depth_comparison.py \
  --config synthetic_experiments/configs/depth_comparison_20261027.json \
  --output synthetic_experiments/outputs/my_variable_check \
  --case narrow_long --method variable_manning --run

# Explicit full-factorial plan; add --run only when intending the full batch.
python synthetic_experiments/run_depth_comparison.py \
  --config synthetic_experiments/configs/depth_comparison_20261027.json \
  --output synthetic_experiments/outputs/my_factorial_comparison --factorial
```

Run with the river-hierarchy-rivgraph environment. For predictable CPU usage set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. Variable routing requires a C++ compiler, compiled automatically into the temporary cache. No external-drive outputs are written by these commands. Failed/nonconverged runs are not complete. Do not reuse an output directory after changing its inputs or source code; the manifest check rejects mixed provenance.

Review `responses.csv`, `groups/*/state_metrics.csv`, `parameters.csv`, `hydrographs.parquet` and `completion.json`. Geometry and method controls are in the new bottom sections of the existing notebooks. Baseline comparison output is `outputs/depth_methods_20261027`; long-narrow conservative checks are `outputs/depth_methods_stress_20261027`. Point `COMPARISON_OUTPUT` at another completed run to inspect it. The core factorial has not been run or fitted.

## Two local workers — 27 October 2026

The current depth-comparison runner supports `--workers 2` (two local processes,
not a distributed multi-node scheduler). It caches identical state/tail calculations
under an exclusive filesystem lock, saves compressed reporting-resolution traces
and native-derived metrics, and aggregates responses every 100 groups and at the
end. Numerical settings, tail doubling and convergence checks are unchanged.
Cache keys include source provenance, prepared frame hashes, hydraulic settings
and forcing. Raw native traces are not retained in the persistent cache.

```bash
cd /Users/6256481/Code/river-hierarchy
caffeinate -i env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python synthetic_experiments/run_depth_comparison.py \
  --config synthetic_experiments/configs/depth_comparison_20261027.json \
  --output /Volumes/PhD/river_hierarchy/synthetic_experiments/outputs/depth_methods_factorial_parallel_20261027 \
  --factorial --workers 2 --run
```

Requires at least 40 GiB free on the output filesystem to enter the full factorial
runner. Before each group, a 5 GiB minimum-free-space check stops additional work
if necessary. This is a storage guard, not a guaranteed upper bound on storage.
Free space or choose an output directory on a larger writable volume first.
Do not run two independent CLI commands into the same output directory: use one
command with two workers. Resume with the identical command and unchanged source;
completed groups are reused. If changing output volume, use a new directory or
copy the entire existing output directory before resuming. Earlier manifests
from before the runner changes have different provenance and must not be mixed.

Validation: 11 hydraulic tests passed, including exact cached/uncached comparison
and source-key invalidation. A four-group/two-worker integration run and resume
passed; all saved hydrographs were bit-for-bit identical to the pre-change
benchmark. Measured peak RSS: 219–231 MiB per worker for those 20-km cases. Leave
2–4 GB available for the overall job; these measurements do not bound every
long-tail configuration. Smoke outputs: `outputs/two_worker_smoke_20261027/run`.
The full grid was not launched by this validation.


### Output location agreed 27 October 2026

Save the full factorial outputs on the PhD drive:
`/Volumes/PhD/river_hierarchy/synthetic_experiments/outputs/depth_methods_factorial_parallel_20261027`.
The directory has been created. The manifest, shared cache, group hydrographs,
parameters, metrics, resource audits, aggregate responses and completion markers
all follow `--output`. Source/configuration remain in the repository. No full run
has been started, and earlier local benchmark outputs have not been migrated.

The drive had approximately 105 GiB free at this check. Its allocation unit is
512 KiB (verified with statvfs and allocated size of an existing small file).
Consequently, logical-size projections understate space used by the many small
files; 105 GiB is not a verified sufficient-space guarantee for the whole batch.
Keep the per-group 5 GiB free-space safeguard. Terminal output can also be saved
on the drive by piping the command through `tee -a <output-directory>/run.log`.

### Storage recovery: APFS container on the PhD drive

After 13,773 completed groups, the external drive's 512 KiB allocation units
expanded 18.1 GiB of file contents into approximately 99 GiB of allocated files
and directories. The initial 40 GiB launch threshold was insufficient for this
layout. Recovery uses an 80 GiB maximum, expandable APFS sparsebundle physically
stored at:

`/Volumes/PhD/river_hierarchy/synthetic_experiments/synthetic_results_20261027.sparsebundle`

The container mounts at `/Volumes/SyntheticResults20261027`. macOS refused
mounting it inside the original external-drive directory. The backing sparsebundle
and all data remain on the PhD drive; only the mounted access path changes. It grows with stored data rather than allocating 80 GiB immediately.
The storage migration reads back and SHA256-verifies every copied file, flushes
an audit record, then removes only that verified original. Model code, manifest,
job identities, source hashes, cache entries and existing results are preserved.
Migration hashes are in `storage_migration_sha256.jsonl` inside the container.

Use this launcher to mount the container if needed and resume two local workers:

```bash
cd /Users/6256481/Code/river-hierarchy
bash synthetic_experiments/resume_depth_grid_phd.sh
```

The launcher mounts the named container if needed and checks its manifest. Do not
delete the sparsebundle: it contains the actual saved results. Keep the PhD drive
connected during routing. The original raw command does not mount the container;
use the launcher after reconnecting the drive or rebooting. No scientific source
file was changed for this storage recovery, so the existing provenance remains
valid and completed groups can resume without recalculation.
