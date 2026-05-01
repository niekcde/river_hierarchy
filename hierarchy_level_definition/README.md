# Hierarchy Level Definition

This area is split into the stages you described:

- `manual_review/`
  Review link direction and node inlet/outlet roles, then write corrected GeoPackages.
- `graph_building/`
  Build the directed graph from the reviewed files and validate that the network has one source, one sink, and no extra local minima/maxima.
- `unit_detection/`
  Detect bifurcation-confluence units and their nesting relationships from the validated directed network.
- `metrics/`
  Compute geometry-based unit metrics from the detected units so the same logic can be used from a notebook or batch scripts.
- `notebooks/`
  Interactive notebooks for stepping through the workflow on smoke-test examples such as `sarl_03`.

Recommended order:

1. Run `manual_review/direction_review_gui.py`.
2. Run `graph_building/directed_network_checks.py`.
3. Run `unit_detection/bifurcation_confluence_units.py`.
4. Compute metrics in `metrics/unit_metrics.py`.
5. Inspect the same steps in `notebooks/level_detection_workbench.ipynb`.

Metrics entry points:

- From the notebook or another Python module:
  `from hierarchy_level_definition.metrics import compute_unit_metrics, compute_unit_metrics_from_units`
- From the command line:
  `python hierarchy_level_definition/metrics/unit_metrics.py <links.gpkg> <nodes.gpkg> --output-dir <dir>`

The metrics module writes:

- `unit_summary.csv`
- `unit_metrics.csv`
- `path_metrics.csv`
- `metrics_manifest.json`
