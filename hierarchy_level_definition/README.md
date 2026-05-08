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

If you want one end-to-end runner instead of separate stages, use:

- `run_unit_workflow.py`
  This starts from reviewed graph-like links/nodes files and writes the full output bundle for one input network.

Metrics entry points:

- From the notebook or another Python module:
  `from hierarchy_level_definition.metrics import compute_unit_metrics, compute_unit_metrics_from_units, summarize_by_hierarchy_level`
- From the command line:
  `/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python hierarchy_level_definition/metrics/unit_metrics.py <links.gpkg> <nodes.gpkg> --output-dir <dir>`

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
  `/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python hierarchy_level_definition/collapse_decisions/unit_collapse_decisions.py <links.gpkg> <nodes.gpkg> --output-dir <dir>`

The collapse-decision module writes:

- `collapse_ranking.csv`
- `constrained_merge_tree.csv`
- `ordered_group_partitions.csv`
- `group_count_selection.csv`
- `bubble_summary.csv`
- `collapse_manifest.json`

## Full Workflow Runner

The end-to-end runner is:

- `hierarchy_level_definition/run_unit_workflow.py`

It is designed for reviewed graph-like network files:

- links GeoPackage with the directed edge fields already fixed
- nodes GeoPackage with node IDs

This is the same input style used in the notebook for `sarl_03`.

### Python entry point

From a notebook or another Python module:

```python
from hierarchy_level_definition.run_unit_workflow import run_unit_workflow

results = run_unit_workflow(
    links_path,
    nodes_path,
    pixel_width_fields=["wid_pix"],
)

results.unit_metrics
results.collapse_ranking
results.selected_groups
```

### Command line entry point

The examples below use the Python interpreter from the `river-hierarchy-rivgraph`
conda environment directly:

```bash
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python
```

Example for `sarl_03`:

```bash
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python \
  hierarchy_level_definition/run_unit_workflow.py \
  hierarchy_level_definition/manual_review/outputs/sarl_03/directed_links.gpkg \
  hierarchy_level_definition/manual_review/outputs/sarl_03/reviewed_nodes.gpkg
```

You can also set an explicit output directory:

```bash
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python \
  hierarchy_level_definition/run_unit_workflow.py \
  <links.gpkg> \
  <nodes.gpkg> \
  --output-dir <output_dir>
```

If no output directory is given, the runner writes to:

- `hierarchy_level_definition/outputs/<network_name>/`

### Output files

The full runner writes:

- `unit_summary.csv`
- `path_metrics.csv`
- `unit_metrics.csv`
- `hierarchy_level_metrics.csv`
- `collapse_ranking.csv`
- `merge_tree.csv`
- `ordered_group_partitions.csv`
- `group_count_summary.csv`
- `bubble_summary.csv`
- `selected_groups.csv`
- `unit_workflow_manifest.json`

### How the workflow is assembled

This full runner is just a thin wrapper around the steps already developed in the notebook:

1. **Unit detection and path creation**
   The reviewed links/nodes files are loaded, a directed `MultiDiGraph` is built, and bifurcation-confluence units plus their paths are detected.
2. **Direct unit metrics**
   Path metrics and unit metrics are computed directly from the detected unit paths in the current graph.
3. **Collapse ranking**
   Units are ranked from more collapsible to less collapsible using the rank-based collapse score built from:
   - `path_disparity_width`
   - `elongation`
   - `equivalent_length`
   - `topologic_complexity_score`
4. **Ordered group partitions**
   Alternative non-overlapping contiguous groupings are built along the global collapse ranking.
5. **Natural group-count selection**
   The total partition cost is summarized by `n_groups`, and an elbow-style rule selects the current `optimal_n_groups`.
6. **Final selected grouping**
   `selected_groups.csv` contains only the rows for that selected `n_groups`.

So the final selected groups are:

- derived from the collapse ranking
- contiguous in that ranking
- non-overlapping within the selected partition

This runner should work for any reviewed graph-like files that follow the same links/nodes schema used here.
