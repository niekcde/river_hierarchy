# Island-only simplification workflow

## Scientific purpose

The island-only experiment is a morphology-first sensitivity analysis of the
same question as the existing unit-collapse experiment: how does network
simplification affect routing?  It differs only in the feature selected for
removal.  Each conceptual observation removes one enclosed dry-land island;
the directed network is then regenerated, parent-matched, anchored, and passed
through the existing width and routing workflows.

The existing bifurcation--confluence-unit experiment remains the primary,
topology-first framework.  Island-only results are a paired alternative
representation, not a replacement or a mixing of treatments.

## Frozen identification contract

For each example:

1. Start from the cleaned base water mask and retain its largest
   **8-connected** water component. Detached water is excluded.
2. Label dry land with **8-connectivity**. A dry component is a provisional
   island only when it does not touch the raster edge.
3. Review and freeze the resulting candidate registry, geometries, source-unit
   provenance, and a single reviewed anchor unit.
4. Rasterize only the island geometry as the edit.  A compound unit boundary
   must never become the island fill geometry.

The use of eight-connected dry components is essential: diagonally touching
dry cells belong to the same island.  In example 56 this correctly merges the
former one-pixel `island_56_002` with `island_56_003`.

## Implemented building blocks

`network_variants/island_experiment.py` currently provides:

- approved-candidate loading and provenance validation;
- exact island-mask materialisation with the normal base/origin-mask contract;
- full-raster RivGraph topology generation;
- parent matching, direction assignment, anchor segmentation, and refreshed
  post-segmentation link lineage;
- standard two-bank widths and routing-width application;
- a strict equivalence registry and compact 15-row observation manifest;
- a manifest-consuming full-raster adapter that references aliases without
  copying products and runs only non-equivalent islands;
- a manifest-consuming local-patch adapter that uses the anchor unit's
  smallest closed parent domain only as computational context, rasterizes the
  frozen island geometry as the sole edit, and records a full-raster fallback
  audit when a safe local realization is unavailable.

The shared collapse workflow now also has an `analysis` artifact profile. For
successful independent/compound child states it retains final directed variant
networks and provenance, converts final width samples to Zstandard-compressed
Parquet, preserves JSON safety audits and edit geometry, and removes
regenerable local-patch/mask/raw-RivGraph products. A local-patch fallback is
always retained as a complete diagnostic. `reproducible` and `debug` profiles
retain the raw products. RAPID preparation accepts both Parquet and legacy CSV
width samples.

The example-56 pilot has 15 conceptual islands.  With the corrected
eight-connected land labels, 12 are exact aliases of an existing unit state
and 3 require an island-specific regenerated state (`island_56_005`,
`island_56_011`, and `island_56_013`).

## Remaining implementation sequence

### 1. Execute and validate the full-raster manifest adapter

The observation manifest and full-raster adapter are implemented. Run its
three example-56 reruns and confirm the execution report contains 12
references, three completed runs, and no alias output directories. Re-running
the same manifest must reuse matching completed reruns rather than overwrite
them.

### 2. Execute and validate the local-patch island adapter

The adapter is implemented. It builds the local computational domain from the
reviewed anchor unit. For an island in compound context, the smallest
hydraulically closed compound is raster/graph support only; the frozen island
geometry remains the sole fill and the recorded anchor unit remains the
scientific replacement interface. A local safety failure records and uses the
existing matching full-raster island state when available.

**Acceptance:** local and full-raster island runs agree on anchored topology,
parent lineage, widths, and final routing products within the established
comparison tolerances.

### 3. Run paired validation

For each rerun island, compare full-raster versus local-patch state outputs.
For each alias island, compare the frozen edit mask, anchor, bank-label
removal set, and manifest reference against its canonical unit state.
Publish a compact paired-comparison table; do not require duplicated full
state directories for aliases.

### 4. Extend to further examples

Run the 8-connected identification and equivalence preflight first.  Only
approve/recompute a state after its geometry, anchor, and base-bank context
are frozen.  Report the number of conceptual islands, aliases, reruns, and
full-raster fallbacks per example.

## Storage and export policy

State products are divided into retained analysis products and reproducible
debug products.  The normal experiment runner should default to the former.

Retain per final regenerated state:

- state/selection manifest and geometry hash;
- directed links/nodes, anchor summary, matching and lineage tables;
- final width and routing-width link product plus compact audit/summary;
- hierarchy/path metrics and any final hydraulic or routing result actually
  used in analysis.

Make optional (debug/profile export):

- collapsed/origin/bank-label rasters after their hashes and edit geometry are
  recorded;
- raw RivGraph skeletons, pickles, logs, centreline exports, and raw
  intermediate links/nodes;
- local-patch crop windows, attempt products, splice intermediates, and
  duplicate pre-anchor/pre-width outputs;
- RAPID preparation inputs (`inflow.nc`, reach/node attribute copies) and raw
  time-series results, unless that state is explicitly selected for hydraulic
  analysis or reproducibility archiving.

Aliases must never copy any of these products.  They retain only a registry
row referencing the canonical existing state.

Before deleting a debug product family, write a compact manifest with source
hashes, command/configuration, retained final outputs, and a validation
status.  Cleanup is then a separate, explicit operation; the runner never
deletes products automatically.
