# Restarting candidate selection from mathematical coverage

Reference: SYN-POPULATION-RESTART-2026-11-02.

Requested to revisit the full 205 candidates before predictive selection. The previous 19 are a benchmark, not an eligibility filter. No candidate definitions, routing outputs or prediction models were changed.

## Verified population

Rechecked `inventory.csv`, `coverage_by_type.csv`, `undefined_reasons.csv`, `finite_but_undefined_audit.csv`, and `affine_equalities.csv` in `/Volumes/SyntheticResults20261027/analysis/candidate_reduction_20261030/`, together with the audit implementation. These are the existing roundoff-cleaned audit results, not a new routing run or a new real-network coverage audit.

Of 205 representations, zero are undefined everywhere, 165 are fully defined, and 40 are partly defined. Of the 165 complete representations, 139 vary within at least one treatment, 22 vary only between treatments, and four are invariant on this design. There are 102 wholly undefined candidate-by-treatment cells; this is not 102 wholly undefined candidates. Across 45,927 geometry comparisons, 784,890 of 9,415,035 candidate entries are undefined. Geometry is shared across hydraulic methods and is counted once.

The 40 partial candidates separate into mutually exclusive categories:

| Category | Candidates | Transfer to real networks |
|---|---:|---|
| Child bifurcation mean/SD or its difference has no bifurcation population | 24 | Structural issue also occurs after real full collapses |
| Ratios inherit undefined bifurcation summaries | 6 | Same structural issue; three also encounter zero denominators |
| Other ratios encounter zero parent denominators | 7 | Possible for zero spread or uniform widths; frequency depends on geometry |
| Route-length skewness requires nonzero route spread | 3 | Possible for equal-length or single-route states, including real collapsed children |

No unresolved missingness reason remains in the saved audit. This does not establish the frequency of each problem in real cases. Exact equalities are particularly common in the factorial design; near-zero denominators can still cause numerical fragility in real data. A meaningful zero SD or entropy must not be confused with an undefined skewness, ratio, or mean over an empty set.

## Affine equivalence

The earlier screen considered 161 complete nonconstant representations. It found 71 pairwise affine links forming 12 non-singleton connected components containing 39 representations. Keeping one per component would remove 27 representations and leave 134 empirical equivalence classes. This is descriptive, not an instruction to discard all 27.

Separate formula-level identities (including unit conversions and duplicate implementations) from design-specific equivalences. Verified duplicate examples are mean route-length change in metres versus kilometres, duplicate route-entropy differences, and duplicate route-spread differences. Prefer a single canonical definition and unit, with aliases documented; do not select between identical information using response correlation.

Other components include preserved parent/child minimum route length, head loss and maximum slope; several topology descriptors; exposure summaries; and length/area/sinuosity ratios. These require formula and domain-assumption review before selecting representatives. Components can mix universal or domain-specific graph identities with incidental grid equalities. Equality on this grid alone is insufficient to assert equivalence on real networks.

## Working decision

Restart from all 205: 165 complete candidates and 40 conditional candidates. Keep the four design constants visible as unassessable on this design. Do not use the earlier shortlist or response scores to resolve eligibility. For each affine component, record equations, scope, validity assumptions and a canonical representative only when the equivalence is established. Keep distinct mechanisms available for real-network checks when the synthetic design cannot distinguish them. No final reduced count is claimed yet.
