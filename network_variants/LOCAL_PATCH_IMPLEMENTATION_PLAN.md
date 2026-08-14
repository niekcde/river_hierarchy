# Topology-defined local collapse patches

## Objective

Avoid full-raster RivGraph regeneration for every collapse.  Regenerate a
bounded topology-defined neighbourhood, merge only genuinely changed local
topology into the parent graph, and inherit the complete parent product for
strict unchanged one-to-one links.

This optimization changes the computational extent, not the scientific
hydraulic domain.  A small local raster patch may still require a complete
enclosing compound to be recalculated for equivalent resistance.

## Phase 1: local-patch contract

Phase 1 is implemented in `network_variants/local_patch.py`.  It defines the
planner output consumed by later raster, RivGraph, and merge stages.  It does
not change any existing experiment execution path.

### Selected core

- A hydraulically closed standalone unit uses its complete unit links.
- A hydraulically open compound member expands to its complete containing
  compound.
- A selected compound uses the complete selected compound.
- The initial contract supports one upstream and one downstream interface.

### Collars and splice anchors

The first directed links outside the core with no unit membership are matching
collars.  Their outer endpoints are splice-anchor nodes.  For example 15 unit
131, link 421 is the upstream collar (`280 -> 279`) and link 48 is the
downstream collar (`33 -> 32`), giving protected splice anchors 280 and 32.

Collars are **mutable matching context**, not immutable boundaries.  The local
result should preferentially retain a collar when it is a strict unchanged
one-to-one match.  Small edge changes may instead cause a collar to be changed,
split, or merged.  This is allowed.  The protected interface is the pair of
outer splice-anchor nodes, and a later merge stage must prove that the final
child graph reconnects to the unchanged parent network at those nodes.

## Phase 2: topology planner

Phase 2 is implemented by `plan_local_patch` in
`network_variants/local_patch.py`.  It is read-only and does not create a crop
or invoke RivGraph.

The planner reuses the production closed-hydraulic-domain selection:

- a closed unit retains its unit-link membership as the core;
- an open member expands to every directed link in its complete containing
  compound;
- an explicitly selected compound uses its complete closed domain.

Starting at the core source and sink, the planner walks outward along a unique
directed chain.  Links that still have unit membership are recorded as
connector context.  The first link with zero unit memberships becomes the
collar.  An absent continuation, ambiguous branch, search-limit exceedance, or
invalid domain produces a structured `fallback_full_raster` result with a
machine-readable reason; it does not guess an interface.

The safe planner result records the domain source and sink, whether the
selected unit itself was closed, links that violated the smaller unit domain,
core and connector counts, collars, and outer splice anchors.  Connector links
join core and collars in `local_context_link_ids`, ensuring the future crop
contains the complete topology between splice anchors.

## Phase 3: adaptive node-centred base-raster crop

Phase 3 is implemented in `network_variants/local_patch_raster.py`.  It reads
only candidate windows from the cleaned mask and the existing global
land-component labels.  It does not edit the mask or run RivGraph.

The original complete-perimeter crossing rule was replaced after real-mask
inspection showed that an axis-aligned crop could intersect neighbouring units
even when the selected hydraulic domain itself was valid. The production rule
now treats explicit bank-to-bank cuts as the hydraulic interfaces.

For each selected domain, the builder:

1. resolves the two largest global land components as the two main banks;
2. finds the independently nearest pixel on each main bank from each protected
   splice-anchor node;
3. creates a bent cut from the first bank through the exact anchor node to the
   second bank and extends both endpoints four pixels into land;
4. bounds every `local_context_link_id` and expands by the requested halo;
5. rasterizes both cuts as barriers and retains only water connected to the
   selected core between them;
6. requires both complete cuts to lie inside the crop and contact retained
   water;
7. requires retained water to have zero contacts with every other part of the
   rectangular crop perimeter.

An unsafe attempt expands the halo.  If no attempt is safe by the configured
maximum, the result records `fallback_full_raster`.  A safe result writes the
unaltered source window, seeded relevant-water mask, global label window, crop
geometry, context links, crossing points, and a JSON attempt audit.

The first production checks of the new rule are:

- example 56 unit 6 is safe at 500 m and uses 1.17% of the full raster;
- example 56 unit 10 is safe at 500 m and uses 1.84% of the full raster;
- example 56 unit 14 is safe at 500 m and uses 1.57% of the full raster;
- all six cuts use global main-bank raster values 2 and 3, pass through the
  exact protected nodes, contact retained water, and leave zero unauthorized
  perimeter contacts.

### Raster safety policy

The initial raster halo is at least 500 m and may expand in 250 m increments.
The relevant water is selected from connected components seeded by the core
after both node-centred bank cuts have been applied. A safe crop has two
prescribed hydraulic cut interfaces and no other connected-water contact with
the rectangular raster edge. Internal island components are allowed. A crop
that cannot meet these requirements falls back to full-raster execution.

### Conservative merge policy

- Outside the splice domain, retain the base network.
- For a strict match in geometry, directed endpoints, and topology, retain the
  base link and all inheritable products.
- Insert unmatched local links.
- Remove or replace conflicting base links.
- Do not require collars themselves to remain unchanged.
- Preserve the outer splice-anchor nodes as the parent-child interface.

## Remaining phases

### Phase 4: local collapse and RivGraph execution

Implemented in `network_variants/local_patch_variant.py`. It applies the
selected collapse inside an accepted crop, re-enforces both bank-cut barriers,
runs RivGraph on the bounded mask, and retains raw local outputs plus a
manifest for the Phase-5 merge. RivGraph's ordered local exit sides are derived
from the accepted upstream and downstream cut centres. The two distinct crop
sides minimizing the joint cut-to-edge distance are used; the global network
exit sides are recorded only as provenance. Parent matches and local topology
remain authoritative for final direction assignment during the merge. Example
56 unit 6 completed this stage using a 1.17%-of-full-raster crop.

### Phase 5: conservative parent/local graph merge

Implemented in `network_variants/local_patch_merge.py`. Cut-created terminals
are identified from the source and sink cut geometries. These computational
cuts provide at least 1,000 m of reference-centerline context from each inner
unit boundary when a collar is too short. For anchored output the extra
context is trimmed, the original parent collar links are retained, and the
regenerated core endpoints are snapped to the original inner boundary nodes.
If an inner boundary already exists as a local RivGraph node, pure cut-to-node
context is pruned. If it falls inside a regenerated link, that link is split at
the boundary and only the source-to-sink subgraph between both anchored
boundaries is retained.
For a hydraulically open member of a compound domain, raster-domain links and
anchored replacement links are intentionally different. All selected/unselected
interface nodes are restored; this can produce multiple upstream anchored
boundaries. For example 56 unit 7, only selected links 19--21 are replaced,
while parent links 22 and 23 remain exact one-to-one inputs to the regenerated
three-link core.
Strict unchanged matches may retain parent IDs and attributes, while genuinely
changed core links receive IDs above the parent maximum. The merge writes
node/link decision audits and refuses ambiguous cut bindings, duplicate node
mappings, or a disconnected or multi-inlet/outlet result.

### Phase 6: changed products and inheritance

Phase 6A is implemented in `network_variants/local_patch_products.py` for
widths and lengths. It reconstructs ordered raster indices from the final
trimmed geometries, measures only changed links against the bounded local
raster, and inherits complete parent width samples and reviewed routing widths
for strict unchanged one-to-one links. Scientific lineage uses the same
1.25-pixel tolerance as the full-raster workflow; anchored segmented-stub
inheritance and routing-width evaluation then run through the existing shared
implementations. The output manifest records inherited/recalculated IDs and
per-link local sample counts.

The investigation notebook includes mechanical inheritance assertions and a
geometry-based comparison against the completed full-raster state.

Phase 6B is implemented in `network_variants/local_patch_hydraulics.py`. It
rebuilds topology-derived hierarchy tables from the complete merged network,
passes explicit collapse-component and final child-to-parent node matches to
the existing collapsed-unit head solver, and supplies the base hydraulic table
for exact slope inheritance on strict unchanged links. Persisted SWORD source
paths can be relocated without mutating the merged node artifact. Equivalent
resistance uses the normal solver with the experiment-level cache, so unchanged
subgraphs reuse cached solutions while affected bubbles are recalculated.

For example 56 unit 7, all 43 unchanged slopes are inherited exactly. The
three changed links enter the same path-relative head solver as the full-raster
state and agree with its slopes to better than 1e-5 relative. All hierarchy
table row counts match. Only two hydraulically affected equivalent-resistance
rows differ, with a maximum relative difference below 0.1%, consistent with
the bounded-raster width sampling differences measured in Phase 6A.

### Phase 7: reference validation

Validate example 56 units 6, 10, and 14 against their completed full-raster
states, followed by example 15 unit 131 against `S093_unit_131`.

The investigation notebook now contains a controlled Phase-7 batch runner for
the three example-56 cases. It records explicit topology, inheritance, changed
width, head-solver, slope, hierarchy, and equivalent-resistance acceptance
checks and plots local versus full-raster changed geometry. Expensive execution
is guarded by `RUN_PHASE7_BATCH`. Example 15 unit 131 remains a separate
follow-up call using its reviewed `SN` global exit-side provenance.

### Phase 8: fallback and integration

Implemented in `network_variants/local_patch_execution.py` and the collapse
experiment CLI. `--execution-mode local_patch` is an opt-in, anchored-only mode
for independent-unit and compound-bubble experiments. It runs Phases 1--6A for
each child, publishes the result in the normal `variant/` layout, and then lets
the existing experiment driver rebuild hierarchy, heads/slopes, and equivalent
resistance.

The full child mask is written blockwise: the parent mask is copied without a
full-array allocation and only pixels that differ between the local base and
local collapsed masks are patched. Temporary bank-to-bank crop cuts are
explicitly excluded. Thus downstream tools receive a real full-extent child
mask rather than a parent-mask proxy.

Planning refusal, an unsafe raster crop, ambiguous interface binding, or any
other local execution failure triggers the unchanged full-raster variant
runner for that child. The reason is retained at
`variant/local_patch_attempt/fallback_to_full_raster.json` and in the normal
variant manifest. Keyboard interrupts and process exits are never swallowed.

Regression coverage includes successful dispatch, an audited ambiguous-merge
fallback, and full-mask publication that applies a scientific collapse edit
while excluding the computational cut barrier.

## Initial reference cases

### Example 56, unit 14 — positive simple case

- unit links: 40 and 41;
- upstream collar: link 42 (`29 -> 28`);
- downstream collar: link 39 (`27 -> 26`);
- splice anchors: 29 and 26;
- safe node-centred-cut crop: 500 m halo, 1.57% of the full raster;
- existing reference: `S002_unit_14`.

### Example 56, unit 6 — adjacent-unit regression case

- unit links: 16 and 17;
- upstream collar: link 18 (`13 -> 12`);
- downstream collar: link 14 (`11 -> 9`);
- splice anchors: 13 and 9;
- the old axis-aligned rule found no safe crop through 5,000 m because adjacent
  connected branches crossed the rectangle;
- the node-centred nearest-main-bank rule is safe at 500 m and uses 1.17% of
  the full raster;
- existing full-raster reference: `S008_unit_6`.

### Example 15, unit 131

- selected unit belongs to compound bubble 4;
- core is the complete containing compound;
- upstream collar: link 421 (`280 -> 279`);
- downstream collar: link 48 (`33 -> 32`);
- splice anchors: 280 and 32;
- safe base-raster crop: 1,500 m halo, 32.21% of the full raster;
- existing reference: `S093_unit_131`.
