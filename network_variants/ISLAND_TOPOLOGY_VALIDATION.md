# Island topology validation and strict analytical selection

## Scientific criterion

The primary island analysis evaluates the routing response to one local,
enclosed island simplification. A retained observation must therefore preserve
the intended one-cycle change from the binary water mask through the final
routing graph.

RivGraph's raw skeleton is a useful diagnostic stage, but its final river graph
also applies pruning rules intended to retain an inlet--outlet-connected river
network. In particular, it removes self-loop links in `remove_all_spurs()` and
bridge-attached subnetworks in `remove_disconnected_bridge_links()`. Those
operations are appropriate for the original river-network objective, but are
not guaranteed to preserve every valid side-channel cycle after a local island
fill.

The strict analysis manifest therefore excludes observations when either the
working mask is not a single clean island edit or RivGraph pruning changes the
graph by more than the one cycle present in the raw skeleton. Exclusion does
not delete the realised network, mask, or audit evidence: the raw analysis
manifest remains the complete observational inventory.

## Reviewed exclusions

`island_topology_review_registry.csv` is the authoritative, machine-readable
record. It excludes five observations:

- `15 / island_15_031`: one valid self-loop is deleted by RivGraph's
  self-loop pruning after a correct one-cycle skeleton change.
- `22 / island_22_095` and `37 / island_37_151`: each edit also reaches a
  tiny detached water remnant (33 and 16 pixels) and then loses an additional
  self-loop during pruning.
- `22 / island_22_289`: the edit reaches the main water body plus 77 detached
  water components and three dry components; it is not a singleton island
  simplification in the working mask and also triggers self-loop pruning.
- `46 / island_46_038`: the mask and skeleton are correct, but bridge pruning
  removes a connected 17-link side subnetwork containing six cycles.

`41 / island_41_099` remains included: its reviewed composite fill with the
16-pixel `island_41_100` raster fragment yields the expected one-cycle final
graph reduction. `island_41_100` is excluded earlier as a raster artifact and
is not an island observation.

## Reproducible selection

Run `run_build_island_analysis_selection.py` with the review registry and all
island experiment roots. It writes `island_analysis_manifest_strict.csv` and a
JSON summary beside each source manifest. These derived files are the inputs
to hydraulic-context and RAPID preparation for the primary analysis; exception
outputs remain available for sensitivity analysis.
