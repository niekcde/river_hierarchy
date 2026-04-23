# Synthetic Runs

This subproject is being reorganized from the legacy synthetic-network code
copied from the old `river_hierarchy` repository.

For the first refactor pass, the legacy source files remain preserved at the
top level of this folder. No legacy file has been moved yet. New code should be
placed under `src/` and extracted incrementally from those preserved files.

## Current Refactor Rule

- Treat the top-level legacy files in this folder as the provenance source.
- Extract code into `src/` in small steps.
- Do not delete or rename legacy files until the extracted module has been
  validated and the provenance is recorded in `MIGRATION_MAP.md`.

## Target Structure

```text
synthetic_runs/
  README.md
  MIGRATION_MAP.md
  configs/
  notebooks/
  sensitivity/
    configs/
  src/
    synthetic_runs/
      core/
      enumerate/
      runners/
      analysis/
```

## Main Intent

- `core/`: shared network classes, recipe loading, shared metrics
- `enumerate/`: geometry enumeration and width sampling
- `runners/`: regular sampled runs and sensitivity runs
- `analysis/`: post-run metrics and modeling
- `sensitivity/`: configs and run-specific docs for the sensitivity workflow

The shared RAPID engine and RAPID file-prep code are being split into the
top-level `../RAPID/` subproject so they can be reused by both synthetic runs
and RivGraph-derived workflows.
