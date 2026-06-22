# Width Method Implementation Plan

This note captures the current design discussion for regenerated-link width
calculation in `network_variants`, especially the question of whether the
collapse-aware `width_wet` / `width_total` workflow should stay with the
current domain-mask logic or move to a simpler "filled-islands only" rule.

Status note:

- this document is now mainly historical context
- the repository has since migrated to the implemented `two_bank_dt` workflow
- current user-facing behavior is described in
  [README.md](/Users/6256481/Code/river-hierarchy/network_variants/README.md)
- where this note conflicts with the code or README, the code and README should
  be treated as authoritative

## 1. Current State

Regenerated variant widths are currently computed in
[variant_generation.py](/Users/6256481/Code/river-hierarchy/network_variants/variant_generation.py).

Relevant functions:

- [compute_width_families](/Users/6256481/Code/river-hierarchy/network_variants/variant_generation.py:768)
- [_build_transect](/Users/6256481/Code/river-hierarchy/network_variants/variant_generation.py:661)
- [_apply_collapse_components_to_mask](/Users/6256481/Code/river-hierarchy/network_variants/variant_generation.py:1032)

Current outputs:

- `variant/mask/*_collapsed.tif`
  - binary realized collapsed water mask
- `variant/mask/width_scope_mask.tif`
  - label mask used only for width scoping
- `variant/widths/link_width_families.csv`
- `variant/widths/link_width_samples.csv`
- `variant/widths/links_with_width_families.gpkg`

Current label semantics:

- `0` = outside scope / dry / stop
- `1` = base wet domain
- `101+` = selected collapsed component domain

Current behavior:

- the binary `*_collapsed.tif` is used for the realized collapsed geometry
- `width_scope_mask.tif` is used to decide which transect-intersection geometry
  is allowed for a sample
- the current implementation labels the full `component_filled` domain with
  `101+`, not only the newly added fill pixels

## 2. What RivGraph Actually Does

Native RivGraph representative width is not based on explicit transect-mask
intersection in the way the current `width_wet` method is.

The relevant source in the installed RivGraph environment is:

- `/opt/anaconda3/envs/river-hierarchy-rivgraph/lib/python3.12/site-packages/rivgraph/ln_utils.py`
- function `link_widths_and_lengths(...)`
- line `452` in the currently installed package

Native RivGraph logic:

1. Compute a distance transform `Idt` of the binary mask.
2. For each skeleton/link pixel:
   - `wid_pix = 2 * Idt[pixel] * pixlen`
3. Compute distances along the link.
4. Trim link ends by about half an endpoint width.
5. Aggregate:
   - `wid`
   - `wid_adj`
   - `wid_med`
   - `len`
   - `len_adj`

So RivGraph width is:

- distance-transform based
- skeleton-pixel based
- trimmed at the ends
- not unit-aware
- not component-aware
- not explicitly split into `wet`, `dry`, and `total`
- not counting channels/threads

Important correction:

- RivGraph does **not** create a unit-specific corridor for width
- RivGraph does **not** build the same mask-intersection workflow now used by
  `network_variants`

## 3. Why We Did Not Just Reuse Native RivGraph Width

We need quantities that RivGraph does not natively provide.

Required collapse-aware outputs:

- `width_total`
  - effective full collapsed cross-section
- `width_wet`
  - original wet-only contribution
- `width_dry`
  - filled/land contribution by difference
- `n_wet_threads`
  - number of original wet channels represented at a sample

Why plain RivGraph `wid_adj` is not enough:

- it only sees one binary mask at a time
- if filled islands are turned into water in the collapsed mask, native RivGraph
  cannot distinguish:
  - original wet channel width
  - newly bridged collapse width
- it provides no direct channel-count diagnostic
- it cannot selectively include only the filled islands from the chosen
  collapse object
- it does not answer the scientific question:
  "what width effect did the collapse create?"

So the current custom method exists because we need:

- a decomposition into wet vs total width
- collapse-aware inclusion rules
- a channel-count style diagnostic

## 4. Current Custom Method

Current `compute_width_families(...)` does:

1. Use regenerated link pixels as sample locations.
2. Build a perpendicular transect at each sample.
3. Use either:
   - local corridor logic
   - full-mask logic
   - or `width_scope_mask` label logic
4. Intersect the transect with the allowed geometry.
5. Keep the contiguous segment containing the sample point.
6. Measure:
   - `width_total`
   - `width_wet`
   - `width_dry`
   - `n_wet_threads`
7. Trim ends in a RivGraph-like way and aggregate to:
   - `wid_adj_total`
   - `wid_adj_wet`
   - `wid_adj_dry`
   - medians / percentiles / thread summaries

This is RivGraph-like only in the sense that:

- sampling happens densely along links
- there is an end-trimming rule similar to RivGraph

It is not native RivGraph width.

## 5. Current Concern

The current `101+` domain-mask approach is more complicated than necessary.

Main concern:

- `101+` is currently assigned to the whole filled collapse domain
- but conceptually we may only want to mark the actual filled islands
- then let the transect logic decide what interval to measure

The simpler proposal is:

- keep `0` as hard stop
- keep `1` as original wet
- assign `100+` only to pixels added by the selected collapse object
- count:
  - `width_total` from `1 + selected 100+`
  - `width_wet` from `1` only
  - channel count from contiguous `1` runs

## 6. Proposed Simpler Measurement Rule

This proposal changes the interpretation of the label mask:

- `0` = stop
- `1` = original wet
- `100+` = filled pixels belonging only to the selected collapse object

For a collapsed-object sample:

1. Build the perpendicular transect.
2. Identify the selected object label, e.g. `103` for `unit 3`.
3. Walk left and right from the sample point.
4. Allow only values in `{1, selected_label}`.
5. Stop at:
   - first `0`
   - or a foreign `100+` label
6. The center-connected interval is the measured cross-section.

Then define:

- `width_total`
  - length of the full interval containing values `{1, selected_label}`
- `width_wet`
  - sum of contiguous `1` runs inside that interval
- `n_channels`
  - number of contiguous `1` runs in that interval

This is the intended simplification.

## 7. Concrete Cases

### 7.1 Base / non-collapsed

Target behavior:

- recover one-channel widths
- do not inflate width from nearby connected water

Recommended rule:

- still restrict sampling to a local corridor around the link
- inside that corridor, walk until first `0`
- measure only the center-connected `1` interval

This keeps outside-collapse widths close to a single-channel interpretation.

### 7.2 Simple single-unit collapse

Example transect:

```text
0  1  1  103 103  1  1  0
```

Interpretation:

- `width_total` includes `1 + 103 + 1`
- `width_wet` includes only the two `1` sections
- `n_channels = 2`

This is the desired behavior for a simple bifurcation-confluence collapse.

### 7.3 Compound / multi-unit collapse

Example transect:

```text
0  1  204  1  204  1  0
```

Interpretation:

- all fill pixels for the selected collapsed object share a common label
- `width_total` includes `1 + 204`
- `width_wet` includes only `1`
- `n_channels` may be greater than `2`

This is acceptable for compound or multi-unit collapses.

## 8. Design Choice Still Open

There are now two plausible approaches:

### Option A: Keep current whole-domain `101+` ownership mask

Pros:

- already implemented
- works with current label-based geometry selection
- explicit ownership of the full collapsed domain

Cons:

- conceptually heavier
- less transparent
- labels more than the actual changed pixels

### Option B: Switch to filled-islands-only `100+` labels plus a simpler transect walk

Pros:

- closer to the physical collapse edit
- easier to explain
- cleaner distinction:
  - `1` = original wet
  - `100+` = added by collapse
  - `0` = stop

Cons:

- requires changing the current measurement rule
- no longer fits the current "sample-point label selects geometry" approach

## 9. Recommended Next Decision

Before any code change, decide explicitly:

1. Do we want the width method to represent:
   - full collapsed domain ownership
   - or only actual fill edits?
2. Should `width_wet` count:
   - only original wet pixels (`1`)?
   - yes, current preference in this discussion
3. Should non-collapsed links still use a local corridor rule?
   - likely yes
4. For compound selections:
   - should one collapsed object use one shared `100+` label?
   - likely yes, unless disconnected objects must be distinguishable

## 10. Working Conclusion

Current conclusion from the discussion:

- native RivGraph width is not solving the same problem as `width_wet`
- we should not treat `wid_adj_wet` as "just RivGraph with a different mask"
- the simpler `0 / 1 / 100+` rule is scientifically attractive
- the strongest version is:
  - `width_total` counts `1 + selected 100+`
  - `width_wet` counts only `1`
  - channels are counted from contiguous `1` runs
- but this requires a deliberate switch in measurement logic, not just a relabeling of the current mask
