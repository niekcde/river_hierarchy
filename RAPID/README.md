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

The first shared RAPID extraction pass is now in place:

- `src/rapid_tools/engine.py`
- `src/rapid_tools/prep.py`
- `src/rapid_tools/adapters/synthetic.py`

The preserved legacy copies still remain authoritative for backtracking during
the migration:

- `synthetic_runs/legacy/rapid/rapid_run.py`
- the RAPID prep helpers embedded in `synthetic_runs/synthetic_runs`
- the RAPID prep helpers embedded in `synthetic_runs/synthetic_runs_sensitivity`

The current runner paths in `synthetic_runs/` now call the shared modules in
this subproject, while the embedded legacy helper definitions remain in place as
reference during the transition.
