# Topology-defined local collapse patches

## Objective

Avoid full-raster RivGraph regeneration for every collapse.  Regenerate a
bounded topology-defined neighbourhood, merge only genuinely changed local
topology into the parent graph, and inherit the complete parent product for
strict unchanged one-to-one links.

This optimization changes the raster extent without changing the selected
collapse.  RivGraph always regenerates the smallest complete hydraulically
closed domain: the selected unit when it is closed, otherwise its containing
compound.  The final anchored merge still replaces only the explicitly
selected unit links.

## Phase 1: local-patch contract

Phase 1 is implemented in `network_variants/local_patch.py`.  It defines the
planner output consumed by later raster, RivGraph, and merge stages.  It does
not change any existing experiment execution path.

### Computational domain and selected replacement

- A hydraulically closed selected unit uses its own complete link set as the
  computational domain.
- An open compound member uses the complete smallest containing compound as
  the computational domain.
- A selected compound uses the complete selected compound.
- `core_link_ids`/`computational_domain_link_ids` record what RivGraph must
  regenerate, while `selected_link_ids` record what the anchored merge may
  scientifically replace.

### Collars and splice anchors

The immediate directed links outside the closed computational domain are
computational collars. They are temporary context, not selected morphology.

Collars are **mutable computational context**, not scientific boundaries. The
protected interfaces are the selected unit's exact inner boundary nodes. The
merge trims the local graph to those nodes before any parent matching, so the
unchanged parent wins everywhere outside the selected morphology.

## Phase 2: topology planner

Phase 2 is implemented by `plan_local_patch` in
`network_variants/local_patch.py`.  It is read-only and does not create a crop
or invoke RivGraph.

The planner evaluates the production closed hydraulic domain and uses its full
link set as the raster core. At the closed domain's source and sink, the unique
immediate external directed link becomes temporary collar context. The
explicitly selected unit links remain a separate anchored replacement set. A
genuinely absent or ambiguous immediate continuation produces a structured
`fallback_full_raster` result; it does not guess an interface.

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
2. projects each closed-domain boundary and its external collar direction onto
   the reference centerline, then places the computational cut at the bounded
   1,000 m context distance; the collar determines direction, not crop length;
3. finds the independently nearest point on each main bank and creates a bent
   cut from the first bank through the exact cut centre to the
   second bank and extends both endpoints four pixels into land;
4. bounds every `local_context_link_id` and expands by the requested halo;
5. rasterizes both cuts as barriers and retains only water connected to the
   selected core between them;
6. requires both complete cuts to lie inside the crop and contact retained
   water, except for a bounded land-overlap tail clipped by a true source-raster
   edge;
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
from the accepted upstream and downstream cut centres. The dominant horizontal
or vertical component of the upstream-to-downstream reference-centreline
displacement selects the ordered opposite cardinal sides; this makes local
direction independent of crop shape and halo. After RivGraph, one inlet must
lie near the upstream cut, one outlet must lie near the downstream cut, and a
directed path must connect them. The global network exit sides are recorded
only as provenance. Parent matches and local topology remain authoritative for
final direction assignment during the merge. Example 56 unit 6 completed this
stage using a 1.17%-of-full-raster crop.

### Phase 5: conservative parent/local graph merge

Implemented in `network_variants/local_patch_realized.py`,
`network_variants/local_patch_anchored.py`, and
`network_variants/local_patch_merge.py`. The bank-to-bank cuts 1,000 m beyond
the closed domain are raster context barriers only. They are never graph
splice interfaces. Raw local RivGraph is trimmed at the unique source and sink
of the complete hydraulically closed computational domain, and every parent
link outside `core_link_ids` is restored with exact parent topology and
geometry.

The resulting complete realized child is matched to the parent and passed to
the same `derive_anchored_graph_from_realized` transformation used by the
full-raster workflow. Exact selected-unit boundary points are allocated there:
an existing local node is snapped or a regenerated link is split. This avoids
maintaining a second local-only scientific anchoring algorithm.

Raw RivGraph `id_nodes` values are treated as unordered endpoint membership,
not flow direction. The complete hydraulically closed computational graph is
directed using parent matches before shared scientific anchoring. The shared
merge retains the union of directed paths from any anchored upstream interface
to any anchored downstream interface.
This matters for open compound members: one nominal upstream interface can lie
downstream of another inside the selected unit and is therefore not necessarily
a topological source of the selected subgraph. The final merged global graph,
not the temporary local interval, must have exactly one global source and sink.
Any additional raw degree-one branch that is neither prescribed cut terminal
is iteratively pruned because it cannot belong to a cut-to-cut path. If the
parent-guided solver cannot produce a valid directed acyclic graph, a bounded
fallback solves a harmonic topological potential with the computational
upstream cut fixed to one and downstream cut fixed to zero, then directs every
retained edge down that potential. This fallback changes direction only; it
does not calculate or substitute hydraulic head, slope, or resistance.
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

After shared anchoring, exact parent geometry and direction are assigned only
to links with authoritative **pre-anchor unchanged context provenance** and to
outer-context pieces split away from a changed link at the selected scientific
boundary. A post-segmentation one-to-one geometric match is not sufficient:
a changed collapsed core can resemble one surviving parent branch after the
cut and must retain its raw collapsed geometry and changed-width status. An
eligible unchanged child node is placed at the corresponding exact parent
endpoint, and an incident changed link follows that node by moving only its
terminal coordinate. When RivGraph compresses two nearby scientific
interfaces into one junction, the short exact parent connector is restored
and its unique non-strict collapsed continuation is trimmed and transferred
to the connector's downstream interface. This separate exceptional repair
prevents a reversed one-to-one connector from serving as a false continuation
of the collapse. The correction is recorded in
`compressed_interface_continuation_corrections`. This exceptional correction
is bounded to five canonical matching tolerances: both strict endpoint matches
must lie within one matching tolerance, while the complete parent-connector
length, continuation trim distance, and attachment offset must each lie within
the five-tolerance ceiling. The exception is activated only when all of those
conditions and exactly one non-strict collapsed continuation are present. If
they are not, the changed raw segment is preserved; the ordinary final graph
gate then accepts it only when the result remains a valid single-source,
single-sink DAG. The final production gate
requires a one-source/one-sink DAG and the undirected cycle-rank reduction to
equal the number of selected dry components actually filled in the local
raster edit.

### Component ownership and bounded recovery rules

The width-buffer footprint defines where a collapse may be evaluated, but it
does not define which dry components belong to the selected unit. Proposed
hole fills are labelled independently and compared with the polygonal interior
enclosed by the explicitly selected link set. A candidate is accepted only
when at least half of its pixels overlap that interior. Rejected neighbouring
components are written as `candidate_edit_rejected` audit geometries and remain
dry. This protects adjacent units that share a link; computational context can
therefore never broaden the scientific collapse selection.

For a single nested unit, a failed local attempt is retried with progressively
larger primary-parent units as computational context. The selected unit IDs,
selected link IDs, and anchored replacement interfaces remain unchanged during
every retry. Attempts and the selected context unit are recorded in
`outer_domain_retry_summary.json`. Multi-unit selections do not use this retry.

If the requested 1,000 m context cut extends beyond the reference centerline,
the cut is clamped to the corresponding global source or sink cross-section.
A zero-length context remains invalid. Duplicate child-to-parent node matches
are resolved only when every discarded candidate is an internal degree-two-or-
higher node; unique external terminals and forced anchored boundaries take
precedence, and ambiguous external duplicates still fail safely.

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

The strict inheritance classifier, rather than lineage labels alone, defines
the measurement set before pixel indices are attached. Consequently, a
lineage candidate rejected for non-identical geometry receives newly generated
local-raster indices and can never send inherited full-raster indices into a
bounded-mask width calculation.

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

The investigation notebook contains a controlled Phase-7 batch runner for
the three example-56 cases. It records explicit topology, inheritance, changed
width, head-solver, slope, hierarchy, and equivalent-resistance acceptance
checks and plots local versus full-raster changed geometry. Expensive execution
is guarded by `RUN_PHASE7_BATCH`. Example 15 unit 131 remains a separate
follow-up call using its reviewed `SN` global exit-side provenance.

The reset Phase-5 path was additionally exercised on the seven former
example-15 anchoring failures (units 55, 74, 115, 116, 120, 186, and 190).
Every case completed without full-raster fallback, produced a valid single-
source/single-sink DAG, and passed the final cycle-reduction gate. A simple
example-56 unit-14 run inherited 44 of 45 links; example-15 compound bubble 9
inherited 616 of 617 links. The complete network-variant test suite passes.

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

## Realized-to-anchored unification reset

The first local-patch implementation performed scientific anchor trimming in
`local_patch_merge.py`. Review of example 15 units 55, 74, 116, 120, and 186
showed that this duplicated the established full-raster realized-to-anchored
algorithm and could retain local collar geometry even when topology-only gates
passed. The replacement architecture is therefore:

1. assemble and validate a full-network local **realized** child;
2. pass that graph through the same realized-to-anchored transformation used
   by conventional full-raster execution;
3. inherit or recalculate widths and hydraulic products only after final
   anchored lineage has been resolved.

### Shared anchoring extraction — complete

`derive_anchored_graph_from_realized(...)` is now the graph-only contract for
the established anchoring operation. It accepts realized links, nodes, and
matching/lineage products together with the reviewed parent graph and selected
units. It returns anchored links, nodes, refreshed node/link matches, refreshed
lineage, and the existing anchor/direction summaries. Raster, width, and file
serialization remain outside this function.

The conventional `derive_anchored_variant_from_realized(...)` path now calls
this shared transform. Synthetic equivalence coverage compares the extracted
function with the former direct helper sequence. A read-only replay of example
15 unit 120 from the saved realized state reproduced the existing conventional
anchored output exactly: identical link and node IDs, endpoints, geometries,
and anchor-segmentation summary.

The next implementation phase is the local realized assembler. The current
local scientific anchoring path remains in place only until that assembler and
its validation gates are ready; it must not be treated as the final method.

### Local realized assembler — complete and production-wired

`assemble_local_patch_realized_graph(...)` in `local_patch_realized.py` now
builds the intermediate graph required by the reset architecture:

- raw RivGraph uses the bank cuts only as bounded raster context;
- the local graph is trimmed and bound at the unique source and sink of the
  closed computational domain, not at the 1,000 m raster cuts and not at the
  inner scientific anchored boundaries;
- every parent link outside the complete computational context is retained
  with its original ID, directed endpoints, and exact geometry;
- parent/local matches inside the splice interval are classified as
  `unchanged_parent`, `partial_parent_overlap_interface`, or `changed_local`;
- the result is a complete, validated full-network realized child and its
  manifest explicitly records `scientific_anchoring_applied: false`.

The conservative merge engine accepts explicit replacement-link and
boundary-node sets. Regression coverage proves that realized assembly
preserves outside-parent geometry and records partial local lineage. The
adapter then feeds this graph and refreshed lineage into
`derive_anchored_graph_from_realized(...)`. Production dispatch uses this
composed path and applies the final cycle-rank gate before publishing products.

### Shared local realized-to-anchored adapter — complete and production-wired

`anchor_local_patch_realized_graph(...)` in `local_patch_anchored.py` composes
the first two reset stages without introducing another anchoring algorithm. It:

1. loads the complete Phase-3 realized graph;
2. recomputes canonical node matching, link matching, and link lineage against
   the reviewed parent network;
3. calls `derive_anchored_graph_from_realized(...)` directly;
4. refuses an invalid final direction report;
5. persists the pre-anchor matches/lineage, final anchored graph, final
   matches/lineage, segmentation summary, and direction report.

The Phase-3 `partial_parent_overlap_interface` classification remains an
explicit splice audit; it is not passed off as scientific lineage. This is
important for collar links that are only partial matches before anchoring but
become unchanged one-to-one parent links after scientific segmentation.

The composed regression begins with a cropped raw graph, assembles a realized
full network at outer interfaces, and anchors it at distinct inner scientific
interfaces. It verifies exact preservation outside the computational splice,
presence of both scientific boundary matches, a valid final directed graph,
and exact equality between the persisted adapter result and a direct call to
the shared transformation. The following Phase-5 wiring promotes that tested
composition into production and moves width/length inheritance after final
anchored lineage.

### Production composition and post-anchor products — complete

The `local_patch` production dispatcher now executes the reset architecture in
this order for every RivGraph exit-side attempt:

1. create the cropped raw local RivGraph result;
2. assemble a validated full-network realized child at the outer computational
   splice interfaces;
3. reconstruct canonical realized matching and lineage;
4. apply the shared scientific realized-to-anchored transformation;
5. calculate/inherit width and length products from the **final anchored**
   graph and its canonical post-anchor lineage;
6. publish the normal child variant layout for hierarchy and hydraulics.

The adapter reconstructs only the selected `StructuralUnit` records from the
saved base `unit_summary.csv` and `unit_link_membership.csv`. It does not rerun
network-wide unit detection for each child. The stored edge sets use the exact
parent link IDs and directed endpoints from the reviewed parent graph.

The bank-to-bank crop cuts may fall inside the two outer parent collar links.
The realized assembler temporarily splits those parent links to obtain exact
computational splice nodes, performs the local merge there, and then dissolves
each temporary degree-two cut node into the adjacent local reach. Consequently
the graph passed to shared anchoring again spans the canonical outer parent
nodes; crop-cut nodes never become scientific topology and cannot suppress the
normal parent-node-path anchoring logic.

`materialize_local_patch_width_length_products(...)` accepts either the legacy
local merge result (for backward-compatible tests and saved diagnostics) or the
new shared anchored result. In production it consumes
`anchored_link_lineage.csv`; the Phase-3 realized splice audit can therefore
never decide width inheritance. Strict unchanged final links inherit all
parent width/length products, while changed or segmented final links receive
local-raster pixel indices and are recalculated as before.

The child manifest records `realized_assembly`, `shared_anchoring`, and
`products` as separate stages. Full-raster fallback and progressively larger
outer-domain retries remain unchanged. The former local scientific merge is no
longer called by production dispatch, although its implementation remains
temporarily available for regression comparison and historical artifacts.

### Selected-interface identity normalization

The realized splice deliberately distinguishes three concepts that earlier
versions conflated:

1. a raw node can be spatially near a parent node;
2. a raw node can inherit that parent node's identity during the computational
   splice; and
3. a scientific interface can be allocated later by shared anchoring.

All parent nodes incident to a selected-unit link are protected during the
local-realized splice. A spatial match to one of those nodes is retained in
`node_merge_audit.csv` as a candidate, but the raw node receives a child-local
ID and keeps its raw coordinate. The only exception is a hard outer
computational-boundary binding. Non-selected parent identity is accepted only
within the normal 1.25-pixel matching tolerance; the wider cut-terminal
tolerance remains available for locating raster barriers but cannot authorize
node identity.

Shared anchoring then processes every graph-cut interface, including lateral
inflow and outflow nodes as well as the nominal source and sink. It projects
each interface onto the appropriate realized child reach and splits that reach
where necessary. This makes path-internal lateral interfaces scientifically
equivalent to nominal end interfaces without forcing their coordinates onto
unrelated raw junctions.

The old bounded same-endpoint reconstruction remains available only for
unselected computational context. A parent connector belonging to the
selected unit is ineligible because restoring it would reintroduce parent
morphology before anchoring. Audits record both deferred node identities and
any rejected selected-connector reconstruction candidate.

The rollback point for the preceding behavior is commit `a03d0cb`
(`feat(local-patch): checkpoint validated shared-anchoring baseline`). That
checkpoint still permits broad realized identity matching and bounded selected
parent-connector reconstruction; the following interface-normalization commit
removes those two behaviors.
