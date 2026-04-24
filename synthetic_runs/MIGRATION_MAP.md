# Synthetic Migration Map

This file records where the copied legacy synthetic code came from and where it
is intended to land during the refactor.

## Provenance Rule

Until extraction is complete, the authoritative preserved legacy copies are the
archive material under `synthetic_runs/legacy/` plus the remaining top-level
wrapper/provenance files that still bridge active workflows to extracted code.

Their historical source was:

```text
/Users/6256481/Documents/GitHub/river_hierarchy/
```

Each legacy file should only be removed after:

1. the replacement module exists in the new structure,
2. imports or runners have been updated,
3. the behavior has been checked against the preserved legacy file,
4. this map has been updated if the target changed.

## Frozen External Outputs To Keep Available

These are not repository source files, but they are required to reproduce and
validate the first refactor pass:

- `/Volumes/PhD/river_hierarchy/output/synthetic_network/synthetic_run_5_geom/sampled_realizations/networks.jsonl.gz`
- `/Volumes/PhD/river_hierarchy/output/synthetic_network/synthetic_run_5_geom/sampled_realizations/summary.parquet`
- `/Volumes/PhD/river_hierarchy/output/synthetic_network/synthetic_run_5_geom/sampled_realizations/edges.parquet`
- `/Volumes/PhD/river_hierarchy/output/synthetic_network/synthetic_run_5_geom/sampled_realizations/run_meta_sampling.json`
- `/Volumes/PhD/river_hierarchy/output/synthetic_network/synthetic_run_sensitivity/networks_sensitivity.jsonl.gz`
- `/Volumes/PhD/river_hierarchy/output/synthetic_network/synthetic_run_sensitivity/grid_manifest.csv`
- `/Volumes/PhD/river_hierarchy/output/synthetic_network/synthetic_run_sensitivity/q_outlet.parquet`
- `/Volumes/PhD/river_hierarchy/output/synthetic_network/synthetic_run_sensitivity/edge_velocity_tc.parquet`
- `/Volumes/PhD/river_hierarchy/output/synthetic_network/synthetic_run_sensitivity/k_q_metrics.csv`
- `/Volumes/PhD/river_hierarchy/output/synthetic_network/synthetic_run_sensitivity/peak_q_metrics.csv`
- `/Volumes/PhD/river_hierarchy/output/synthetic_network/synthetic_run_sensitivity/run_errors.csv`
- `/Volumes/PhD/river_hierarchy/output/synthetic_network/synthetic_run_sensitivity/run_regime.csv`
- `/Volumes/PhD/river_hierarchy/output/synthetic_network/synthetic_run_sensitivity/edge_full2.csv`

Recipe fixture anchors:

- `synthetic_run_sensitivity/networks_sensitivity.jsonl.gz`
  rows: `7`
  sha256: `45a0dc374916ee92809c79645afbf2d4cdbd3394d92a8f4f6858726fd13e15b2`
- `synthetic_run_5_geom/sampled_realizations/networks.jsonl.gz`
  rows: `4399771`
  sha256: `6c5edd9c7bddda0f7845a32736872d098cc0026f963de2baad02e64c267caf18`

## Sensitivity Provenance

The first sensitivity recipe set is notebook-derived.

- Recipe construction notebook source:
  `synthetic_runs/legacy/notebooks/synthetic_admissable_networkx.ipynb`
- Output recipe file:
  `/Volumes/PhD/river_hierarchy/output/synthetic_network/synthetic_run_sensitivity/networks_sensitivity.jsonl.gz`
- Current preserved count:
  `7` recipes
- Those `7` recipes are the structural sensitivity set only:
  `2` loop, `3` cross, `2` no-break
- Execution script:
  `synthetic_runs/synthetic_runs_sensitivity`
- Post-analysis notebook:
  `synthetic_runs/legacy/notebooks/synthetic_sensitivty.ipynb`
- Explicit baseline control artifact:
  `synthetic_runs/configs/single_edge_control.json`

Important: the notebook contains stale cell outputs, so provenance should be
checked against the generated files above, not notebook output displays.

Phase 1 note:

- helper, metric, and recipe IO functions are being extracted into
  `src/synthetic_runs/core/`
- `RiverNetworkNX`, `Params`, and `canonical_signature` remain compatibility
  exports backed by the preserved legacy implementation during the first pass
- `src/synthetic_runs/runners/sensitivity_recipes.py` reproduces the preserved
  `networks_sensitivity.jsonl.gz` fixture exactly
- the separate single-edge baseline is now treated as a shared explicit control
  artifact for both sampled runs and sensitivity runs, not as an extra hidden
  recipe or a hidden runner-only special case

Phase 2 note:

- shared RAPID engine extracted to `../RAPID/src/rapid_tools/engine.py`
- shared RAPID prep helpers extracted to `../RAPID/src/rapid_tools/prep.py`
- synthetic RAPID adapters extracted to
  `../RAPID/src/rapid_tools/adapters/synthetic.py`
- the active runner paths in `synthetic_runs` and `synthetic_runs_sensitivity`
  now call the shared RAPID layer
- sampled routing outputs now write `run_meta_routing.json`
- sensitivity outputs now write `run_meta_sensitivity.json` and include
  `single_edge_control_path` in `grid_manifest.csv`

Phase 3 note:

- sampled runner logic is now extracted to
  `src/synthetic_runs/runners/sampled.py`
- sensitivity runner logic is now extracted to
  `src/synthetic_runs/runners/sensitivity.py`
- shared runner helpers live in `src/synthetic_runs/runners/shared.py`
- `src/synthetic_runs/runners/__init__.py` now exposes lazy package entry
  points for `run_sampled_realizations` and `run_sensitivity_grid`
- the preserved legacy top-level files `synthetic_runs` and
  `synthetic_runs_sensitivity` still contain the old implementations for
  provenance, but their active entry points are now aliased to the extracted
  package modules
- Phase 3 smoke checks succeeded through the extracted modules for:
  - a tiny sampled run using `synthetic_runs/outputs/test_phase_1/sampled_small`
  - a tiny sensitivity run using the preserved
    `synthetic_run_sensitivity/networks_sensitivity.jsonl.gz`

Phase 4 note:

- `src/synthetic_runs/core/network.py` now contains the extracted
  `Params`, `RiverNetworkNX`, and `canonical_signature` implementation
  instead of importing them from the preserved legacy file
- the extracted core now directly supports:
  - corridor instantiation
  - loop/cross break insertion
  - width recomputation
  - recipe round-trips via `to_recipe()` / `from_recipe()`
  - A/B-invariant canonical signatures
- Phase 4 smoke checks succeeded for:
  - a direct `RiverNetworkNX` recipe round-trip with matching signatures
  - a tiny geometry-realization check through
    `synthetic_geometric_enumeration.iter_realized_networks_from_geom_recipe`
  - a tiny width-plan realization through
    `synthetic_width_perc_splits.realize_geom_with_plan`

Phase 5 note:

- K-path analysis is now extracted to
  `src/synthetic_runs/analysis/k_metrics.py`
- run-level Random Forest analysis is now extracted to
  `src/synthetic_runs/analysis/run_level_rf.py`
- `src/synthetic_runs/analysis/__init__.py` now exposes the extracted
  analysis entry points alongside the plotting helpers
- the preserved legacy top-level analysis files
  `synthetic_network_k_metrics.py` and `synthetic_run_level_rf.py` still
  contain the old implementations for provenance, but their active entry
  points are now aliased to the extracted package modules
- Phase 5 smoke checks succeeded for:
  - `compute_metrics(...)` on the tiny sampled fixture in
    `synthetic_runs/outputs/test_phase_1/sampled_small`
  - `run_random_forest_regression(...)` on a small synthetic run-level
    dataset with realized edge summaries in the `UNC` environment

Phase 6 note:

- archive-only prototype scripts have been moved under
  `synthetic_runs/legacy/prototypes/`
- legacy notebooks have been moved under
  `synthetic_runs/legacy/notebooks/`
- backup copies and snapshots have been moved under
  `synthetic_runs/legacy/backups/`
- preserved RAPID source snapshots have been moved under
  `synthetic_runs/legacy/rapid/`
- active wrapper/provenance files remain at the top level of
  `synthetic_runs/` so the working pipeline and smoke tests are unchanged

## File-To-Target Map

### Active Extraction Targets

| Preserved legacy file | Intended target | Notes |
| --- | --- | --- |
| `synthetic_admissable_networkx_part_save.py` | `src/synthetic_runs/core/network.py`, `src/synthetic_runs/core/recipes.py`, `src/synthetic_runs/core/metrics.py` | Main keeper for `Params`, `RiverNetworkNX`, streamed summaries, recipe IO |
| `synthetic_geometric_enumeration.py` | `src/synthetic_runs/enumerate/geometry.py` | Geometry-first enumeration; currently depends on private helpers from the core file |
| `synthetic_width_perc_splits.py` | `src/synthetic_runs/enumerate/sample_widths.py` | Width realization and sampled-network generation |
| `synthetic_runs` | `src/synthetic_runs/runners/sampled.py` | Keep only runner logic here; extract RAPID prep/helpers out |
| `synthetic_runs_sensitivity` | `src/synthetic_runs/runners/sensitivity.py` | Keep only sensitivity runner logic here; extract shared code out |
| shared single-edge control in sampled/sensitivity runners | `configs/single_edge_control.json`, `src/synthetic_runs/runners/controls.py` | Explicit baseline control artifact; remains distinct from the 7 structural sensitivity recipes |
| `synthetic_network_k_metrics.py` | `src/synthetic_runs/analysis/k_metrics.py` | Post-run K-path analysis |
| `synthetic_run_level_rf.py` | `src/synthetic_runs/analysis/run_level_rf.py` | Run-level Random Forest analysis |

### Shared RAPID Extraction Targets

| Preserved legacy file | Intended target | Notes |
| --- | --- | --- |
| `legacy/rapid/rapid_run.py` | `../RAPID/src/rapid_tools/engine.py` | Core RAPID routing engine |
| RAPID prep helpers inside `synthetic_runs` and `synthetic_runs_sensitivity` | `../RAPID/src/rapid_tools/prep.py`, `../RAPID/src/rapid_tools/io.py` | `create_conn_file`, `create_riv_file`, `compute_reach_ratios`, `compute_area_csv`, `create_runoff`, `create_routing_parameters`, `compute_dt_from_K` |
| `rivernetwork_to_rapid_graph` inside runner files | `../RAPID/src/rapid_tools/adapters/synthetic.py` | Synthetic-specific adapter; keep separate from generic RAPID prep |

### Legacy Archive Only

These should be retained for provenance during the refactor, but they are not
the intended long-term implementation.

| Preserved legacy file | Planned status | Notes |
| --- | --- | --- |
| `legacy/prototypes/synthetic_admissable_networkx.py` | Archive only | Older non-streaming predecessor of `part_save` version |
| `legacy/prototypes/synthetic.py` | Archive only | Early random generator prototype |
| `legacy/prototypes/synthetic_det.py` | Archive only | Deterministic prototype |
| `legacy/prototypes/synthetic_det_reach.py` | Archive only | Reach-based deterministic prototype |
| `legacy/prototypes/synthetic_reach_width.py` | Archive only | Later reach-width prototype |
| `legacy/prototypes/admissable_k_one_bif.py` | Archive only | Exploratory predecessor |
| `legacy/prototypes/admissable_k_1_2_bif.py` | Archive only | Exploratory predecessor |
| `legacy/prototypes/admissable_multi_bif.py` | Archive only | Exploratory predecessor |
| `legacy/prototypes/admissable_multi_bif_2.py` | Archive only | Exploratory predecessor |
| `legacy/backups/synthetic_runs_backup` | Archive only | Backup copy |
| `legacy/backups/synthetic_runs.backup_20260206_113807` | Archive only | Backup copy |
| `legacy/backups/synthetic_runs.bak` | Archive only | Broken backup copy |
| `legacy/backups/synthetic_network_k_metrics_backup.py` | Archive only | Backup copy |
| `legacy/rapid/rapid_run_back_up.py` | Archive only | Backup copy |
| `legacy/backups/synthetic_runs_20260210_092628.py` | Archive only | Backup snapshot |
| `legacy/backups/synthetic_network_k_metrics_20260210_085741.py` | Archive only | Backup snapshot |

### Legacy Notebook Targets

| Preserved legacy file | Planned status | Notes |
| --- | --- | --- |
| `legacy/notebooks/synthetic_admissable_networkx.ipynb` | Archive only | Contains sensitivity recipe provenance |
| `legacy/notebooks/synthetic_sensitivty.ipynb` | Archive only | Sensitivity post-analysis |
| `legacy/notebooks/synthetic_data.ipynb` | Archive only | Legacy exploration |
| `legacy/notebooks/distribution_synth_network.ipynb` | Archive only | Legacy exploration |

## Extraction Order

1. Extract `Params`, `RiverNetworkNX`, recipe IO, and shared metrics from
   `synthetic_admissable_networkx_part_save.py`.
2. Extract geometry and width-enumeration modules.
3. Extract RAPID engine and RAPID prep into `../RAPID/`.
4. Rebuild the sampled and sensitivity runners on top of those shared modules.
5. Move analysis utilities.
6. Reorganize archive-only files into an explicit `legacy/` area while keeping
   the active wrapper/provenance layer stable.
