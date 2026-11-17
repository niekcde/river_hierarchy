# Matched synthetic selection excluding width P90

Reference SYN-FORWARD-NO-P90-2026-11-03. User requested excluding P90 on interpretability grounds after its predictive comparison. Exclude all P90 representations present in the working pool (parent, child, difference and ratio), to avoid reintroducing the same descriptor through another transformation. Retain the four prior equivalent-width/elongation exclusions.

Everything else matches the previous quadratic-ridge procedure: original standalone eligibility filter plus topology reserves, all three depth approaches, centroid and peak responses, identical nested case-group folds, four search variants, coverage rules, up to ten predictors and 0.005 compactness tolerance. Mean-width ratio is eligible but not required. P10 remains eligible: this experiment excludes P90, not every width quantile.

This is a post-hoc interpretability sensitivity test, not evidence of weak P90 prediction or independent feature discovery. No RF or routing reruns. Results and source hashes are in `results/`.

## Completed findings

Completed in 19.8 seconds with 67 candidates (66 without G). Parent, child and delta P90 were removed; ratio P90 was already outside the eligible pool. Original case folds match exactly; numerical SSE and scikit-learn Ridge checks passed in all outer folds.

For fixed-Manning centroid, all three outer folds select the same four inputs in both coverage variants:

- `B_delta_route_length_mean`: change in allocated mean route length.
- `parent_slope_slow`: parent slope descriptor.
- `B_parent_width_mean`: parent length-weighted arithmetic mean reach width, sum(L_e B_e)/sum(L_e), in metres.
- `B_parent_bifurcations`: parent bifurcation count.

Pooled outer R² is 0.995180 versus 0.995316 with P90 eligible; MAE is 0.994839 versus 0.923656 minutes. Both coverage searches return the same four-input model because ΔG is eligible rather than mandatory, and the 0.005 compactness tolerance selects the four-variable model. This does not imply G has no information: the unconstrained G-only baseline remains almost exact for fixed-Manning centroid.

Parent mean width is also the sole width representative in all coverage folds for both Manning-depth approaches and both responses. It is an absolute parent width, not the child/parent mean-width ratio tested previously in RF. Those real RF scores must not be assigned to this newly selected descriptor or four-input panel. Fixed-BASED centroid still selects mean-width ratio together with other width terms. Real-network RF evaluation of the new choice has not been performed in this investigation.

Interpretation: excluding P90 barely changes synthetic centroid explanatory performance and yields a readily interpretable width scale. Prior real results remain archived; this is an explicitly documented interpretability decision after inspecting results, not independent confirmation or a target-driven claim of superiority.
