# Hierarchy Level Definition

This area is split into the stages you described:

- `manual_review/`
  Review link direction and node inlet/outlet roles, then write corrected GeoPackages.
- `graph_building/`
  Build the directed graph from the reviewed files and validate that the network has one source, one sink, and no extra local minima/maxima.
- `unit_detection/`
  Detect bifurcation-confluence units and their nesting relationships from the validated directed network.
- `notebooks/`
  Interactive notebooks for stepping through the workflow on smoke-test examples such as `sarl_03`.

Recommended order:

1. Run `manual_review/direction_review_gui.py`.
2. Run `graph_building/directed_network_checks.py`.
3. Run `unit_detection/bifurcation_confluence_units.py`.
4. Inspect the same steps in `notebooks/level_detection_workbench.ipynb`.
