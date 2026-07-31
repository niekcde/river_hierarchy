# Panelized collapse implementation plan

## Problem

Run 46's cleaned mask is 47,990 by 40,267 pixels (1.93 billion pixels). The
current collapse materializer reads whole rasters and creates several full-size
boolean and `int32` arrays for boundary labels, bank labels, distance
transforms, and width measurements. Even `--base-only` therefore exceeds local
memory.

The RivGraph workflow already has a panel plan, per-panel RivGraph outputs,
core ownership, boundary matching, and global link/node reconstruction. It now
also publishes both a canonical single reference centerline and a distinct
multipart network-centerlines product. The collapse workflow must reuse those
panel capabilities instead of materializing whole-raster states.

## Design principles

- Panel cores own outputs; overlap/halo pixels are processing context only.
- Unchanged panels are reused between states.
- Collapses touching a panel boundary process the required neighboring panels
  together.
- Panelized state outputs retain the existing experiment output schema.
- No full-size raster array is loaded in panelized mode.

## Phase 1 — Correct width aggregation at panel boundaries

**Status: complete**

Replace the current length-weighted aggregation of already-aggregated source
link widths when publishing the merged panel network.

1. Identify and preserve the per-sample RivGraph width values and the metadata
   required to locate samples along each source link.
2. Restrict samples to each source link's core-owned segment, so panel overlap
   is never counted twice.
3. Order retained samples along every merged global link.
4. Recompute `wid`, `wid_med`, `wid_adj`, quantiles, and diagnostics using the
   same adjustment rules as RivGraph, once per merged sample series.
5. Persist merged raw width samples for provenance and downstream use.
6. Add a cross-panel regression test against an equivalent unpanelized link.

**Acceptance criteria**

- A merged link does not use a weighted mean of source `wid_adj` values.
- Every width sample is owned by exactly one panel core.
- A merged link has raw-sample provenance and reproducible final summaries.

## Phase 2 — Panelized state materializer

**Status: in progress**

Create a panel-aware alternative to the current full-raster state materializer.
It will select affected panels from a collapse footprint, reuse untouched panel
outputs, regenerate only affected panels, and use the existing core matching,
node rebuilding, direction assignment, and publication steps.

Initial implementation contract:

- inputs: the panel plan/root, the published panelized parent handoff, and a
  state selection;
- base state: reuse all published panel outputs and the phase-1 merged width
  samples without reading a full raster;
- child state: classify panels as unchanged, affected, or halo-only; only
  affected panels are regenerated;
- outputs: the same directed links/nodes and variant-manifest paths consumed
  by hierarchy, hydraulic-context, and experiment metadata code;
- guardrail: panelized mode rejects a child selection until phase 3 provides
  windowed mask edits, bank labels, and two-bank width calculation.

## Phase 3 — Windowed raster and width processing

**Status: in progress**

- Build or reuse boundary-component labels without full-raster allocation.
- Read masks, bank labels, and distance-transform context by panel/window plus
  a documented halo.
- Keep boundary-component identifiers consistent across panel boundaries.
- Compute width samples per panel and retain only core-owned samples before
  global aggregation.

Implemented foundation: blockwise bank-label writing and panel-local collapse
edits, each operating only on an existing panel mask/window. Remaining work is
halo-aware regeneration, merging the mixed panel workspace into a state, and
panelwise two-bank width sampling.

## Phase 4 — Incremental rollout

**Status: pending**

1. Panelized `--base-only` for run 46, reusing the existing panel outputs.
2. A collapse fully contained in one panel.
3. A collapse crossing one panel boundary.
4. Independent and sequential experiment modes.

## Global acceptance criteria

- Panelized mode never calls a full-raster `read(1)` for run-scale masks.
- Unchanged panels are reused.
- Overlap samples are not duplicated.
- Merged links/nodes/centerlines and manifests remain compatible with existing
  downstream workflows.

## Follow-up audit TODOs (2026-07-31)

- **Completed source preparation:** width review is complete for examples 41
  and 46. Example 37's directions were repaired, its base-only analysis was
  rerun, and its width review was completed. All requested base networks now
  have reviewed source-width values.
- Fix base-state direction validation so a reused base graph is validated from
  its actual directed links instead of receiving a successful inherited or
  synthesized report.
- Recreate `S000_base` as part of each new full collapse experiment. This
  imports the now-reviewed legacy fields from the source links and computes
  hierarchy products with the current `wid_adj_wet`/`wid_pix_wet` precedence;
  old experiment directories are not updated retroactively.
- Rerun the complete example-45 experiment if its child states are required.
  Its existing 12-state run uses the former hierarchy-width definition, and
  its anchored base also predates nine legacy GUI width corrections.
