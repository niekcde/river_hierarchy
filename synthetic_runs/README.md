# Synthetic Runs

This subproject is being reorganized from the legacy synthetic-network code
copied from the old `river_hierarchy` repository.

The active package extraction is now in place. Archive-only legacy material has
been moved into `legacy/`, while active wrapper/provenance files that still
anchor the working pipeline remain at the top level of this folder.

## Current Refactor Rule

- Treat `legacy/` as the archive area for retired prototype and backup code.
- Treat the remaining top-level wrapper/provenance files as the active bridge
  layer until they are retired.
- Extract code into `src/` in small steps.
- Do not delete preserved legacy material; move it into `legacy/` once it is no
  longer needed as an active top-level bridge and record the move in
  `MIGRATION_MAP.md`.

## Target Structure

```text
synthetic_runs/
  README.md
  MIGRATION_MAP.md
  RUNBOOK.md
  configs/
  legacy/
    reference/
  notebooks/
  regular/
    configs/
  sensitivity/
    configs/
  src/
    synthetic_runs/
      core/
      enumerate/
      runners/
      analysis/
      pipelines/
```

## Main Intent

- `core/`: shared network classes, recipe loading, shared metrics
- `enumerate/`: geometry enumeration and width sampling
- `runners/`: regular sampled runs and sensitivity runs
- `analysis/`: post-run metrics and modeling
- `pipelines/`: config-driven operational entry points for workflow steps
- `regular/`: configs and shell wrappers for the regular synthetic workflow
- `sensitivity/`: configs and run-specific docs for the sensitivity workflow
- `legacy/`: archive-only notebooks, prototypes, backups, preserved reference
  implementations, and preserved RAPID snapshots
- `RUNBOOK.md`: current operational commands for the extracted regular and
  sensitivity workflows

The shared RAPID engine and RAPID file-prep code are being split into the
top-level `../RAPID/` subproject so they can be reused by both synthetic runs
and RivGraph-derived workflows.
