# Synthetic Legacy Archive

This folder holds preserved legacy material that is no longer part of the
active synthetic pipeline, but is still kept for provenance and backtracking.

Structure:

- `prototypes/`
  Early synthetic generators and exploratory admissibility scripts.
- `notebooks/`
  Legacy notebooks kept as historical workflow references.
- `backups/`
  Backup copies and archived snapshots.
- `rapid/`
  Preserved RAPID-era source snapshots that were replaced by the shared
  top-level `RAPID/` package.

Active code should now live in:

- `src/synthetic_runs/`
- the top-level active wrapper files that still bridge legacy entry points to
  extracted modules
- `notebooks/test_phase_*.ipynb`
