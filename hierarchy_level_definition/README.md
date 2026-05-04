# Hierarchy Level Definition

This area is split into the stages you described:

- `manual_review/`
  Review link direction and node inlet/outlet roles, then write corrected GeoPackages.
- `graph_building/`
  Build the directed graph from the reviewed files and validate that the network has one source, one sink, and no extra local minima/maxima.
- `unit_detection/`
  Detect bifurcation-confluence units and their nesting relationships from the validated directed network.
- `metrics/`
  Compute direct, hierarchy-aware unit metrics from the detected units so the same logic can be used from a notebook or batch scripts. The current metrics are computed from each unit's detected paths in the present graph; recursive collapsed geometry is future work.
- `collapse_decisions/`
  Build two decision layers from the unit metrics: a one-at-a-time collapse ranking and global ordered contiguous group partitions for multi-unit collapse candidates. Bubble IDs are retained as annotations, but they do not constrain grouping.
- `notebooks/`
  Interactive notebooks for stepping through the workflow on smoke-test examples such as `sarl_03`.

Recommended order:

1. Run `manual_review/direction_review_gui.py`.
2. Run `graph_building/directed_network_checks.py`.
3. Run `unit_detection/bifurcation_confluence_units.py`.
4. Compute metrics in `metrics/unit_metrics.py`.
5. Build collapse rankings / ordered group partitions in `collapse_decisions/unit_collapse_decisions.py`.
6. Inspect the same steps in `notebooks/level_detection_workbench.ipynb`.

Metrics entry points:

- From the notebook or another Python module:
  `from hierarchy_level_definition.metrics import compute_unit_metrics, compute_unit_metrics_from_units, summarize_by_hierarchy_level`
- From the command line:
  `python hierarchy_level_definition/metrics/unit_metrics.py <links.gpkg> <nodes.gpkg> --output-dir <dir>`

The metrics module writes:

- `unit_summary.csv`
- `unit_metrics.csv`
- `path_metrics.csv`
- `hierarchy_level_metrics.csv`
- `metrics_manifest.json`

Metric definitions:

- See `metrics/metrics_definition.md` for the full metric reference, including equations, width definitions, and a shorter "important metrics" overview.

Collapse decision entry points:

- From the notebook or another Python module:
  `from hierarchy_level_definition.collapse_decisions import compute_collapse_decisions_from_unit_metrics, rank_unit_collapse_priority, build_constrained_merge_tree, summarize_group_count_selection`
- From the command line:
  `python hierarchy_level_definition/collapse_decisions/unit_collapse_decisions.py <links.gpkg> <nodes.gpkg> --output-dir <dir>`

The collapse-decision module writes:

- `collapse_ranking.csv`
- `constrained_merge_tree.csv`
- `ordered_group_partitions.csv`
- `group_count_selection.csv`
- `bubble_summary.csv`
- `collapse_manifest.json`
