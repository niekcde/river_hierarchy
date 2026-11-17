# Original selection repeated without summed equivalent width

Reference: SYN-FORWARD-WIDTH-REVISION-2026-11-03.

Repeat the 2026-11-02 quadratic-ridge selection exactly, changing only candidate eligibility: remove parent and delta equivalent path width and parent and delta path elongation (which divides by equivalent path width). All four occur in the original 74-candidate pool, leaving 70, or 69 without Manning ΔG. Mean-width ratio was already eligible; no variables are added or forced.

Preserve all three depth approaches, centroid and peak responses, four search variants, three outer and three inner case-group folds, up to ten inputs, two seed swap passes and the 0.005 parsimony tolerance. Fixed-depth Manning centroid is the primary interpretation; other outputs retain comparability with the original experiment. No RF selection or routing reruns.

Results are in `results/`; `provenance.json` records completed runtime and source hashes. `comparison.csv` compares held-out scores with the original run; `width_selection.csv` records width representatives. This is a post-hoc sensitivity analysis, motivated by real-network equivalent-width failures. Nested scores remain conditional on the previously screened candidate pool. RF evaluation of selected sets is a separate next step.

## Completed result

Completed in 22.5 seconds. Fold assignments match the original exactly; direct-prediction and scikit-learn numerical checks passed in all three outer folds.

For fixed-depth Manning centroid, both coverage searches select four inputs: change in allocated mean route length, parent slope descriptor, child width P90, and a topology descriptor. Topology is parent bifurcation count in folds 0 and 2 and change in topologic complexity score in fold 1. The with-G and without-G selections coincide: G is eligible, not compulsory. Pooled outer R² is 0.995316 versus 0.995297 before exclusion; MAE is 0.924 minutes. The 0.005 parsimony tolerance explains why higher-scoring longer paths need not be retained.

Mean-width ratio does NOT emerge as the width representative in either Manning-depth approach (centroid or peak coverage searches); child width P90 is selected in every fold. Mean-width ratio does enter all three fixed-BASED centroid coverage selections, together with other width terms. Thus automatic recovery of the recent mean-width RF replacement is not supported by this matched ridge rerun. This remains a model- and hydraulic-approach-dependent selection result, not evidence against the separate real-network replacement test.
