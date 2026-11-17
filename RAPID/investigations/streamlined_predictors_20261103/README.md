# Streamlined predictor reconciliation — 3 November 2026

Reference: **STREAMLINED-PREDICTORS-2026-11-03**.

Purpose: test whether a compact, interpretable set informed by synthetic
experiments and explicit complexity descriptors can approach the previous
nine-input set. The earlier synthetic selection used quadratic ridge and
variable-depth Manning; this experiment aligns the primary depth approach and
prediction model before drawing conclusions about predictor choice.

An existing response-definition distinction remains explicit: synthetic targets
are full converged excess-volume centroids (`depth_pipeline.py`), while real
targets use the finite 0.5% event window. Comparisons between predictor panels
use identical responses within each dataset. Cross-dataset performance gaps
must not be attributed solely to geometry complexity: integration windows,
forcing and hydraulic parameter distributions also differ. A full-pulse
centroid travel-time identity does not automatically apply to a finite-window
centroid. This run preserves the existing targets and does not reroute events.

## Frozen comparison

Primary response: child-minus-parent centroid shift from **fixed-depth Manning**
routing. Synthetic and real models are fitted separately. Variable-depth
Manning remains a sensitivity analysis; this run does not select on it.

Every panel uses the same existing `fine_squared` RF configuration: 500 trees,
maximum depth 16, minimum leaf size 2, squared-error criterion, all predictors
available per split, seed 20261014. Training-only preprocessing, target scaling
and weighting use the existing fitting code. This is a fixed-configuration
diagnostic, **not** another hyperparameter search. Its old-nine refit is the
proper comparator; previous nested-tuned scores remain historical context.

| Panel | Inputs | Change tested |
|---|---:|---|
| `old9` | 9 | Previous normalized-spread nine-input benchmark |
| `revised9` | 9 | Replace width-CV ratio with child-minus-parent width CV |
| `five_with_G` | 5 | Frozen synthetic-selected set including Manning geometry ΔG |
| `five_without_G` | 5 | Frozen synthetic-selected set using area change instead |
| `seven_with_G` | 7 | Five with ΔG plus entropy change and parent dominance |
| `seven_without_G` | 7 | Five without ΔG plus entropy change and parent dominance |
| `revised_without_allocation7` | 7 | Revised nine without entropy change or dominance |
| `revised_without_spread8` | 8 | Revised nine without normalized route-spread change |

No additional candidates will be added based on interim scores. These tests
assess predefined sets and joint removals; they do not prove that a feature is
causal or individually necessary in the presence of correlated predictors.

## Coverage and validation

- Synthetic: 45,927 comparisons, seven states per parameter combination, three
  existing outer case-group folds. All seven states stay together. No routing
  reruns or synthetic subsampling.
- The original width-CV ratio is undefined in 2,187 simple/full comparisons.
  Original and revised nine are both trained and evaluated on the **same 43,740
  defined rows** for their direct synthetic comparison. The seven remaining
  primary panels, including revised nine, also run on the complete population.
- No epsilon denominator or zero imputation makes an undefined ratio appear
  defined. Zero CV itself is valid. Parent CV at or below 1e-12 is treated as
  numerical zero for ratio validity.
- Real: all 2,185 canonical comparisons; exact existing 15 leave-river-out folds,
  25 network-grouped folds and 25 compound-family-grouped folds. All eight panels
  use identical splits. The original nine are defined throughout this population.
- Weighting: equal total per real river or synthetic parameter combination.
- Lengths are in metres. Equivalent path width retains the order-preserving
  log1p representation needed for RF numerical resolution.

Normalized route-spread change:

\[
\Delta\sigma_L/\mu_{L,p}=(\sigma_{L,c}-\sigma_{L,p})/\mu_{L,p}.
\]

Revised width heterogeneity descriptor:

\[
\Delta CV_W=CV_{W,c}-CV_{W,p},\qquad CV_W=\sigma_W/\overline W.
\]

Width moments retain the existing physical-length weighting. Real differences
were reconstructed using recomputed parent CV and the previously verified
child/parent ratios; 30 direct parent/child checks agreed within 1.2e-16.

## Execution and outputs

```bash
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python \
  RAPID/investigations/streamlined_predictors_20261103/run_comparison.py --run
```

The launcher waits for `matched_centroid_all_strategies_20261103/completion.json`
before starting two single-threaded fit workers. There are 27 synthetic fits and
520 real fits (547 total), with no inner tuning fits. Completed prediction files
are hashed and checkpointed. The same command resumes completed jobs. Keep the
PhD drive and its synthetic-results container mounted for provenance checks.

Reporting repair: all 547 fits completed, but the unqualified report import
resolved to another investigation's `summarize` module. Use the explicit local
finalizer below to validate saved predictions and generate the report without
refitting. The frozen fitting source and checkpoint fingerprints are preserved.

```bash
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python \
  RAPID/investigations/streamlined_predictors_20261103/finish.py
```

Outputs: `RAPID/outputs/streamlined_predictors_20261103/`.

- `design.json`, `tasks.json`: frozen inputs, sources, model and splits.
- `synthetic_coverage.csv`, `width_cv_reconstruction_checks.csv`: coverage checks.
- `progress.json`, `completion.json`: actual run status and final receipt.
- `summary.csv`, `comparison_matrix.png`: validation-strategy × predictor-set
  comparison, with synthetic support explicitly identified.
- `paired_changes.csv`: direct additions/replacements/removals on matched rows.
- `river_error_contributions.csv`, `largest_errors.csv`: where real errors occur.
- `synthetic_control_residuals.csv`, `length_allocation_residuals.csv`: synthetic
  residual behaviour as controls change.
- `report.md`: complete results and limitations, written automatically at finish.

Interpretation is exploratory: previous synthetic screens and real outcomes
have already informed this design. A favourable result supports a candidate
within the tested domain; it is not independent confirmation of the entire
selection procedure. Assess error concentration and consistency across all
three validation strategies before recommending a final set. Do not select
solely on the combined ordinary/full-network R².
