# Width Method Overhaul Specification

This document specifies the intended replacement for the current regenerated-link
width workflow in `network_variants`.

It supersedes the current transect + `width_scope_mask` concept as the target
methodology, but it is still a design/specification document rather than an
implemented code path.

Related discussion note:

- [WIDTH_METHOD_IMPLEMENTATION_PLAN.md](/Users/6256481/Code/river-hierarchy/network_variants/WIDTH_METHOD_IMPLEMENTATION_PLAN.md)

Current code to be replaced eventually:

- [compute_width_families](/Users/6256481/Code/river-hierarchy/network_variants/variant_generation.py:768)

## 1. Design Goal

Use one consistent geometry-based width method for:

- `width_total`
- `width_wet`
- `width_dry`
- `n_channels`

The new method must:

- stay closer to RivGraph's distance-transform intuition than the current
  explicit transect-intersection workflow
- be less sensitive to local centerline-angle changes than pure perpendicular
  transect intersection
- distinguish original wet width from collapse-added width
- support simple, nested, and compound collapses
- avoid per-island full-scene EDT blow-up
- produce interpretable QA/QC diagnostics

## 2. Core Decision

The width represented by a sample is the **local controlling two-boundary
section** at the collapsed-state centerline pixel, not an outer-envelope width.

That means:

- we intentionally use the most relevant left/right obstacle boundaries near the
  centerline sample
- we do **not** force width to span the full outer collapsed corridor if a more
  local two-boundary section is what controls the centerline-scale width

This is the key methodological choice.

Rationale:

- in real examples, the meaningful width is usually the local corridor width
- this matches the spirit of a RivGraph-like local-width estimate more closely
  than an outer-envelope rule
- this avoids over-widening compound sections by forcing the width across
  irrelevant outer banks

## 3. RivGraph Relationship

Native RivGraph representative width uses:

- one binary mask
- one distance transform
- skeleton/link pixels
- `wid_pix = 2 * Idt[pixel] * pixlen`
- end trimming to produce `wid_adj`

Relevant native RivGraph function:

- `/opt/anaconda3/envs/river-hierarchy-rivgraph/lib/python3.12/site-packages/rivgraph/ln_utils.py`
- `link_widths_and_lengths(...)`
- currently line `452` in the installed environment

The new method is **not** native RivGraph width.

It is a RivGraph-inspired extension that adds:

- boundary-component identity
- left/right bank selection
- collapse-aware width decomposition
- wet/dry/channel diagnostics

## 4. Terminology

### 4.1 Obstacle boundary component

An `obstacle boundary component` is any connected land-water interface boundary
after artificial open-end boundaries are removed.

This includes:

- outer banks
- island boundaries

Implementation-level rule:

- banks and islands are treated as the same class
- the only difference is whether a component remains active or is removed during
  a collapse

### 4.2 Active obstacle set

The `active obstacle set` for a state is the set of obstacle boundary
components that remain blocking obstacles in that collapse state.

### 4.3 Origin mask

The `origin mask` is the label mask used for decomposition of the final
bank-pair geometry:

- `0` = blocking land/background
- `1` = original wet water
- `100+` = collapse-added traversable pixels for the selected collapse object

### 4.4 Local controlling pair

The `local controlling pair` is the selected left/right obstacle boundary
component pair that defines the local width for a centerline sample.

## 5. Raster Semantics

The method uses two state-level raster views.

### 5.1 Collapsed total mask

Binary mask:

- `1` = active water/traversable collapsed corridor
- `0` = active blocking obstacle or background

Purpose:

- compute a global DT for the collapsed state
- define the geometry from which total width is derived

### 5.2 Origin mask

Label mask:

- `0` = blocking land/background
- `1` = original wet water
- `100+` = collapse-added traversable pixels for the selected object

Purpose:

- decompose the final section into:
  - wet width
  - dry/fill width
  - number of wet channels

Important:

- `100+` is assigned only to collapse-added pixels for the selected collapse
  object
- it is **not** assigned to the whole filled domain

## 6. Example-Level Preprocessing

This stage is performed once per example.

### 6.1 Boundary extraction

From the example binary water mask:

1. Extract land-water boundary pixels.
2. Detect where the river intersects the image border.
3. Remove or ignore those artificial open-end boundary pixels.

Goal:

- ensure that the upstream/downstream open boundaries are not treated as banks

### 6.2 Boundary-component labeling

After open-end filtering:

1. Label connected boundary components.
2. Assign a stable `component_id` to each.

Outputs:

- boundary-component raster or pixel table
- per-component metadata:
  - `component_id`
  - bbox
  - centroid
  - pixel coordinates
  - optional vector geometry

### 6.3 Spatial index

Build one spatial lookup over boundary locations, e.g.:

- KD-tree over boundary pixels, or
- spatial index over component geometry/pixel bbox

Purpose:

- local candidate selection without full-scene per-component EDT

## 7. State-Level Inputs

Each hierarchy state must provide:

- collapsed-state regenerated centerline/link pixels
- collapsed total mask
- origin mask
- active obstacle set for that state

The centerline used for width sampling must be the **collapsed-state** centerline,
not the parent/base centerline.

## 8. Active Obstacle Set Update Under Collapse

Each collapse state removes one or more obstacle components from the active
obstacle set.

Interpretation:

- a collapse removes selected blocking island/obstacle components
- those removed pixels become traversable in the collapsed total mask
- those same pixels are marked `100+` in the origin mask

This is the key state update.

## 9. Distance-Transform Strategy

### 9.1 Chosen first implementation

Use:

- one **global DT** per state on the collapsed total mask
- local candidate selection from the example-level boundary-component index

Do **not** use:

- one full-scene EDT per obstacle component

Reason:

- full-scene per-component EDT scales poorly with many islands/components

### 9.2 Role of the global DT

The global DT is used for:

- a RivGraph-like baseline width estimate:
  - `width_dt_global = 2 * DT(centerline_pixel)`
- adaptive search-radius seeding
- QC comparison against the final two-bank width

The global DT is **not** sufficient by itself because it does not distinguish:

- left vs right bank
- original wet vs collapse-added width
- number of wet threads/channels

## 10. Candidate Boundary Selection

For each centerline sample:

1. Compute a local tangent from neighboring centerline pixels.
2. Compute the local normal.
3. Use `width_dt_global` or another local width estimate to define an initial
   candidate search radius.
4. Query nearby boundary components using the spatial index.
5. Restrict to active obstacle components only.

This candidate stage exists only to keep the problem local and computationally
cheap.

It does **not** define the final width by itself.

## 11. Bank-Pair Selection

For each candidate component:

1. Retrieve the nearest boundary point or nearest boundary pixel to the
   centerline sample.
2. Form vector:
   - `v_i = bank_point_i - center_point`
3. Classify side by the sign of:
   - `dot(v_i, normal)`

Then:

- choose the nearest valid candidate on one side
- choose the nearest valid candidate on the opposite side

This remains a DT-guided method:

- the local normal is only a left/right classifier
- it does not replace distance-based ranking

### Why not just choose the two nearest components overall?

Because that can select:

- two components on the same side
- the wrong island pair in a bend

### Methodological choice

This selection intentionally represents the **local controlling pair**, not the
outer envelope of the whole collapsed corridor.

## 12. Width Geometry

For each sample, after selecting left and right obstacle boundaries:

- left distance: `d_left`
- right distance: `d_right`
- left nearest point: `p_left`
- right nearest point: `p_right`
- center point: `c`

### Primary total-width definition

Use:

- `width_total = d_left + d_right`

This is the official section width.

### Secondary chord-width definition

Also compute:

- `width_chord = ||p_right - p_left||`

This is a QA/QC diagnostic only.

### Shared decomposition geometry

The decomposition geometry is the two-leg path:

- `p_left -> c -> p_right`

This geometry is used to derive:

- `width_wet`
- `width_dry`
- `n_channels`

Important:

- the polyline is preferred over the direct chord for decomposition
- because the two nearest-bank vectors are not guaranteed to define a perfect
  straight bank-to-bank section in bends or asymmetric situations

## 13. Decomposition Rules

Sample the polyline `p_left -> c -> p_right` against the state `origin_mask`.

### 13.1 width_total

- `width_total = d_left + d_right`

### 13.2 width_wet

- count only segments crossing label `1`
- `width_wet = total length of all label-1 segments on the two-leg path`

### 13.3 width_dry

- count only segments crossing the selected `100+` label
- `width_dry = total length of all selected-100+ segments on the two-leg path`

### 13.4 n_channels

- `n_channels = number of contiguous runs of label 1 on the sampled path`

Interpretation:

- for a simple two-thread collapse, this should typically be `2`
- for compound/multi-thread collapses, it may be larger

## 14. Quality Checks

Each sample should carry QC diagnostics.

### 14.1 DT consistency

Compare:

- `width_total`
- `width_dt_global = 2 * DT(centerline_pixel)`

Flag large relative discrepancies.

### 14.2 Left/right opposition

Let:

- `u_left = p_left - c`
- `u_right = p_right - c`

Check cosine similarity of normalized vectors.

Target:

- vectors should point in roughly opposite directions

Flag if not approximately opposite.

### 14.3 Chord consistency

Compare:

- `width_total = d_left + d_right`
- `width_chord = ||p_right - p_left||`

Large mismatch indicates a problematic or highly oblique bank pair.

### 14.4 Search-radius escalation

Record whether the initial radius was sufficient or had to be expanded.

Flag if:

- no pair found
- or a very large search expansion was needed

### 14.5 Collapse-type expectation

For simple two-path unit collapses:

- expect `n_channels` usually near `2`

Flag anomalies, but treat them as diagnostics, not hard failures.

## 15. Link-Level Aggregation

For each regenerated link:

1. Compute sample-wise:
   - `width_total`
   - `width_wet`
   - `width_dry`
   - `n_channels`
   - QC flags
2. Apply RivGraph-style end trimming.
3. Aggregate to link-level representative values.

Outputs:

- `wid_adj_total`, `wid_med_total`
- `wid_adj_wet`, `wid_med_wet`
- `wid_adj_dry`, `wid_med_dry`
- p05/p50/p95 for each width family
- `n_channels_mean`, `n_channels_max`
- QC summaries:
  - fraction flagged
  - max discrepancy
  - mean discrepancy

## 16. Output Artifacts

### Example-level cached artifacts

- `boundary_component_labels.tif` or equivalent indexed boundary store
- `boundary_component_metadata.csv`
- spatial-index cache if persisted

### State-level artifacts

- collapsed total mask
- origin mask
- sample-level width diagnostics
- link-level width families

### Sample-level attributes

At minimum:

- centerline sample coordinate
- selected left/right `component_id`
- selected left/right nearest-point coordinates
- `d_left`, `d_right`
- `width_total`
- `width_chord`
- `width_wet`
- `width_dry`
- `n_channels`
- QC flags and discrepancy metrics

## 17. Scope Limits and Assumptions

This first version assumes:

- the width mask has no junctions for the local width problem
- open-end border artifacts can be identified and removed reliably
- collapse can be represented as removing one or more whole blocking obstacle
  components from the active obstacle set
- the local controlling pair is the intended notion of width

This first version does **not** try to solve:

- partial removal of only part of an obstacle component
- ambiguous highly open compound sections where no stable local left/right pair
  exists
- full outer-envelope width of the entire collapsed corridor

## 18. Main Risks

### Risk 1: wrong local pair

The chosen pair may still be locally wrong in complex geometry.

Mitigation:

- normal-side classification
- QC with vector opposition
- QC with DT and chord comparisons
- local search-radius control

### Risk 2: over-local width in compound settings

Choosing the local controlling pair may under-represent the full compound
corridor width.

Status:

- accepted as a methodological choice for this specification

### Risk 3: border/open-end handling

If open-end boundary filtering is wrong, the whole method can fail.

Mitigation:

- treat this as a first-class preprocessing product with inspection outputs

## 19. Why This Is the Chosen Direction

This method was chosen because it:

- keeps one geometric method for all width quantities
- stays closer to the RivGraph DT concept than explicit perpendicular
  transect-mask intersection
- is less sensitive to centerline direction than the current transect method
- cleanly separates:
  - total width geometry
  - wet/dry decomposition
  - channel counting
- avoids full-scene per-component EDT blow-up

## 20. Implementation Principle

The implementation should proceed in this order:

1. example-level obstacle boundary preprocessing
2. state-level active obstacle set construction
3. sample-level local candidate lookup
4. left/right component selection
5. `width_total` computation from the two-bank geometry
6. path-based `width_wet`, `width_dry`, `n_channels` decomposition
7. QC diagnostics
8. link-level aggregation with RivGraph-style trimming

That sequencing should be preserved even if some engineering details change.
