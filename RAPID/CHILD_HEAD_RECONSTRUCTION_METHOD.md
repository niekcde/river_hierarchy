# Shared-head child reconstruction

Decision reference: **CHILD-SHARED-HEADS-2026-10-03**.

This replaces the provisional child-head transfer in the October 2 signed-routing
investigation. It changes child hydraulic reconstruction, not the fitted reference
profile, roughness, depth–width exponent, routing caps or segmentation.

## Rule

Match child nodes injectively to base nodes by coordinates. Coincident child ports
remain distinct graph nodes: prefer the port retaining unchanged direct-profile
continuation, then the number and length of unchanged incident links. Ambiguous
ties fail explicitly. Node-ID equality alone is not sufficient evidence.

When topology or length changes inside an owned parent unit, recompute the entire
affected interior jointly. Keep exterior/shared-profile heads fixed. Unchanged
domains retain their heads exactly. Width-only changes do not trigger a slope
reallocation. Every link uses `(upstream_head - downstream_head) / actual_length`.

The objective is the existing equal-per-link squared relative slope error against
the patch reference slope, with the existing `0.01 * S_ref` unit floor. Here
`S_ref = outer_anchor_head_span / reference_section_distance_span`. Sparse convex
solves use explicit feasibility and convergence checks; no hydraulic parameter
is adjusted to satisfy routing stability. There is no anchor-relaxation fallback.

## Implementation and rerun

- `network_variants/shared_head_reconstruction.py`: reusable matching and solve.
- `RAPID/investigations/reconstruct_child_heads.py`: all-case isolated contexts.
- `RAPID/investigations/check_reconstructed_head_invariants.py`: independent checks.
- `RAPID/investigations/route_reconstructed_child_heads.py`: fixed-K two-clock screen.
- `RAPID/investigations/report_reconstructed_child_heads.py`: dated notebook report.

Run materialization before the invariant checks and routing, then generate the
report. The batch scripts accept example identifiers where applicable. They use
the frozen bounded-robust base heads and stored reference-depth anchors. Their
output is `RAPID/assets/bounded_robust_slopes_20261002/child_head_reconstruction_20261003`.

`inventory.csv` maps each included entry to its candidate manifest; its `input_paths`
contains source geometry only. Candidate link/node CSVs retain native child IDs.
Old context pointers are separate provenance and are not refreshed parent inputs.
Intentional exclusions remain `excluded_topology_review`, not missing-input errors.

## Validation and scope

All 3,551 included entries reconstruct: 2,230 unique hydraulic states and 2,214
base–child comparisons. Five pre-existing topology exclusions remain excluded.
All 16 base frames are exactly unchanged. Shared heads, fixed boundary heads and
inlet–outlet drops pass the independent checks. Eleven focused/regression tests
pass. See the [dated results](assets/bounded_robust_slopes_20261002/child_head_reconstruction_20261003/README.md)
for the two-clock screen, pulse-only cases, celerity ranges and cap sensitivity.

These are validated candidate contexts and investigation routing results. No
production defaults, production outputs or equivalent-resistance caches have been
overwritten. Production preparation must consume the new contexts and local Kb
consistently before old resistance results can be replaced or compared.
