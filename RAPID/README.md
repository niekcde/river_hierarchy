# RAPID

This subproject will hold the shared RAPID routing engine and RAPID input-file
preparation code used by multiple workflows in this repository.

## Scope

- shared RAPID engine
- shared RAPID CSV/NetCDF input preparation
- adapters that convert workflow-specific network objects into RAPID-ready
  reach graphs

## Planned Relationship To Other Subprojects

- `synthetic_runs/` will use this package for synthetic routing runs
- `rivgraph_centerline/` can later use the same RAPID prep/engine layer for
  RivGraph-derived networks

## Current Refactor Rule

No legacy RAPID code has been moved here yet. The preserved source of truth is
still:

- `synthetic_runs/rapid_run.py`
- the RAPID prep helpers embedded in `synthetic_runs/synthetic_runs`
- the RAPID prep helpers embedded in `synthetic_runs/synthetic_runs_sensitivity`

Those preserved files will be extracted into this subproject incrementally.
