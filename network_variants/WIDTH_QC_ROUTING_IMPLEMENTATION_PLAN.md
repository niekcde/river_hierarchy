# Width QC and routing-width implementation plan

## Objective

Use the completed manual width reviews for the available base networks to
build a reproducible supervised quality-control model for new links created by
collapse variants. Preserve measured wet, dry, and total widths while adding a
separate effective width for routing and hierarchy calculations.

The implementation must:

- protect the existing manual review data from experiment overwrites;
- detect when a review has become stale after a mask or network change;
- train and validate a supervised classifier from the manual good/bad labels;
- apply width corrections only when a safe upstream-continuity target exists;
- inherit reviewed/effective values exactly for unchanged links;
- preserve the semantic difference between measured geometry and inferred
  routing width;
- behave consistently in regular and panelized execution modes.

## Phase 1 — Permanently archive existing reviews

**Status: complete (2026-07-31).** All 16 reviews were copied and verified
before full experiment reruns.

Create the following durable directory for every reviewed example:

```text
rivgraph_centerline/outputs/runs/<example>/width_review/
```

Store:

```text
base_width_review_decisions.csv
base_width_review_apply_summary.json
reviewed_width_targets.csv
reviewed_links_snapshot.gpkg
base_width_review_manifest.json
```

The archive must retain:

- every manual good/bad decision;
- all candidate metrics shown by the GUI;
- legacy correction targets and scale factors as provenance;
- link geometry and upstream/downstream nodes;
- original and corrected legacy width values;
- the raw two-bank width used during review;
- review date and code revision.

Existing artifacts are copied, not moved, during initial migration.

### Acceptance criteria

- All reviewed examples have complete archives.
- Archive contents survive RivGraph and collapse `--overwrite` operations.
- Archived decisions agree with review metadata in the source-link files.

## Phase 2 — Review fingerprints and stale-review protection

**Status: complete (2026-07-31).** All 16 archives have current mask,
network, link-identity, and width-method fingerprints. Validation distinguishes
mask, topology, ID-only, width-method, missing, and corrupt-archive states.

Record the following in every review manifest:

- cleaned-mask fingerprint;
- structural link-network fingerprint;
- node and link counts;
- sorted link identifiers;
- upstream/downstream node pairs;
- link geometry signatures;
- CRS;
- width-method and model-input version;
- source paths;
- code revision.

The structural network fingerprint excludes mutable width-review attributes
and includes link ID, upstream node, downstream node, and geometry.

Supported review states:

```text
current
missing
stale_mask
stale_topology
stale_link_ids
stale_width_method
```

Rules:

- A mask or topology change prevents automatic reuse of the archived review.
- A link-ID-only change produces a geometry-based remapping proposal, but the
  proposal requires confirmation.
- A stale review is never silently joined by link ID.
- An explicit override may exist for debugging, but not as the normal
  production path.

### Acceptance criteria

- Changing a mask invalidates its review.
- Renumbering links is detected.
- Changing only mutable width/review fields does not invalidate topology.
- Stale reviews cannot silently affect an experiment.

## Phase 3 — Permanent GUI save contract

**Status: complete (2026-07-31).** The GUI now defaults to the durable
run-level archive, writes decisions and summaries atomically, resumes partial
reviews with an explicit `incomplete_review` status, validates freshness before
reusing link IDs, and versions existing archive files before replacement.
RivGraph overwrite behavior is covered by a regression test that preserves the
top-level `width_review/` directory.

Change the GUI default review location to:

```text
rivgraph_centerline/outputs/runs/<example>/width_review/
```

The GUI must:

- create the directory automatically with `parents=True`;
- resume incomplete reviews from the permanent decision CSV;
- write decisions atomically;
- update fingerprints and the archive manifest;
- preserve previous completed review versions;
- continue updating reviewed legacy fields in the source links;
- warn before opening a stale review;
- never delete a previous archive when starting over.

Keep `--review-output-dir` for explicit debugging or export use. Add a
regression test proving that RivGraph `--overwrite` does not remove the
top-level `width_review` directory.

### Acceptance criteria

- A new example receives its review directory automatically.
- Review archives survive recreation of `network_variants/output`.
- The GUI detects and reports stale reviews.

## Phase 4 — Consolidated supervised-learning dataset

**Status: complete (2026-07-31).** A deterministic, versioned dataset builder
now reads only permanent run-level archives, validates their Phase 2 freshness,
maps every good/bad decision to one canonical example/link row, derives the
agreed primary and secondary predictors, and writes a dataset, validation
report, and content manifest. Missing predictors remain explicit rather than
causing silent row removal.

Build or refresh the dataset with:

```bash
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python -m network_variants.width_review_dataset
```

Combine all permanent review archives into one reproducible training table.
Each row represents one manually classified candidate link.

### Response variable

```text
manual_decision: good / bad
```

Encode the modelling target explicitly, for example:

```text
requires_correction: 0 / 1
```

### Primary predictors

- length divided by raw two-bank wet width;
- `split_sensitive`;
- trimmed mean opposition cosine;
- trimmed median opposition cosine;
- invalid-sample fraction;
- upstream width-continuity ratio and deviation.

### Secondary diagnostic predictors

- downstream continuity deviation;
- bank-to-bank chord difference;
- distance-transform width difference;
- number of wet threads;
- wet fraction;
- lineage type, when available;
- raw wet width and link length;
- example ID for grouping and validation only, not as a predictive feature.

The stored legacy numeric target is retained as provenance but is not used as
the target wet/routing width.

### Acceptance criteria

- Every archived decision maps to exactly one modelling row.
- No example/link duplicates exist.
- Missing values and exclusions are reported explicitly.
- Dataset construction is deterministic and versioned.

## Phase 5 — Supervised algorithm development

**Status: complete (2026-07-31).** Candidate logistic-regression, linear-SVM,
and exploratory RBF-SVM pipelines now use fold-local median imputation and
standardization with leave-one-complete-example-out validation. The generated
artifacts include held-out probabilities, per-example metrics, threshold
trade-offs, calibration bins, disagreement/routing-impact tables, fitted
candidate bundles, dataset fingerprints, coefficients, parameters, and
software/code provenance. Logistic regression is the provisional review
candidate; neither it nor its threshold is production-authorized.

Build or refresh the candidate-model comparison with:

```bash
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python -m network_variants.width_qc_model
```

The width-QC algorithm is a supervised probabilistic classifier, not a
hand-written stepwise decision tree.

Develop and compare at least:

1. regularized logistic regression;
2. support-vector classification, initially with a linear kernel and, if
   justified, an RBF kernel.

Logistic regression is the preferred baseline because it provides calibrated
probabilities and interpretable coefficient direction. The SVM comparison
tests whether a nonlinear boundary materially improves out-of-example
performance. Apply feature scaling within the cross-validation pipeline to
avoid leakage.

The agreed predictors remain the scientific basis of the model:

- length/width ratio;
- opposition-cosine QC;
- invalid-sample fraction;
- upstream continuity;
- `split_sensitive`;
- selected secondary diagnostics if they improve held-out performance.

Length/width remains the candidate-population definition used by the GUI. It
does not prescribe the internal form of the classifier. The model estimates
the probability that a candidate requires correction.

### Validation design

Use grouped leave-one-example-out validation:

- hold out one complete river/example;
- fit preprocessing and the model on the remaining examples;
- predict the excluded example;
- repeat for all examples.

Report:

- ROC-AUC and precision-recall AUC;
- confusion matrices at candidate thresholds;
- precision, recall, and false-correction rate;
- probability calibration;
- metrics specifically for `split_sensitive=True` links;
- results by held-out example;
- manual/algorithm disagreements;
- highest-routing-impact false positives and false negatives.

Select the production probability threshold based on the cost of incorrectly
correcting a manually good, routing-sensitive link. Do not select it from
in-sample accuracy.

### Model outputs

For every candidate, retain:

```text
width_model_probability
width_model_prediction
width_model_threshold
width_model_name
width_model_version
manual_model_disagreement
```

The trained model bundle must include preprocessing, feature order, model
parameters, threshold, training-data fingerprint, code revision, and
validation summary.

### Acceptance criteria

- The model is reproducible from the archived reviews.
- Preprocessing is fitted inside grouped cross-validation folds.
- Generalization is evaluated by complete held-out examples.
- Logistic regression and SVM results are compared transparently.
- No model becomes authoritative before disagreement review and approval.

## Phase 6 — Algorithm-review notebook

**Status: complete (2026-07-31).** The review notebook reads only the
versioned Phase 4/5 artifacts and covers review balance, missingness,
predictor distributions, collinearity, linear coefficients and odds ratios,
model and per-example performance, ROC/PR curves, calibration, threshold and
split-sensitive trade-offs, partial SVM/logistic decision surfaces,
disagreements, routing-impact ranking, and archived-link context maps. It has
been executed successfully from a clean project kernel with all 17 code cells
completing and no errors.
Portable coefficient-based diagnostic surfaces allow the notebook to run in
other analysis environments while warning about scikit-learn bundle-version
mismatches; direct candidate-bundle inference remains restricted to the
training environment.
The ranked disagreement tables expose every classifier input alongside
interpretive width/QC context, with color-coded false-correction and
missed-correction predictor plots.
Pairwise matrices cover all classifier inputs for the complete reviewed
population and for disagreements, plus the important interpretation-only
variables for disagreements.
They use interactive Plotly scatter panels with network/link hover labels.
The matrices are descriptive projections and intentionally omit a
two-variable decision line because the classifier operates in seven
dimensions and the held-out points come from different fitted folds.

### Disagreement adjudication GUI

**Status: implemented and human review complete (2026-08-03).**

`network_variants/width_qc_disagreement_gui.py` creates a focused 63-link
review queue: 54 provisional-model held-out disagreements plus 10 cross-window
junction-audit priorities, with link 43 belonging to both sets. The resulting
union adds nine links that the provisional model agrees with but whose historic
label or interpretation may change under near-junction continuity. It
shows the local directed network, cleaned-mask land context, optional two-bank
transects, all classifier inputs, interpretation variables, the held-out
probability, and the routing-impact proxy. It also loads the provisional
10%/250–1,000 m near-junction audit, displays the plain incoming/outgoing
whole-link and junction width sums, incoming reach length and length/width,
and endpoint-locality flags, and colors upstream-junction incoming and
outgoing links separately. It never modifies archived manual labels, source
links, or width fields.

Adjudications are written separately under:

```text
network_variants/output/width_qc_disagreement_review/
```

Every case records whether it was triggered by a model disagreement, the
junction audit, or both. The review CSV preserves both original decisions and all displayed evidence,
and records one of:
`manual_upheld`, `model_upheld`, `ambiguous`,
`measurement_valid_but_routing_unsuitable`, `data_or_geometry_problem`, or
`exclude_from_training`. A manifest binds the review to exact dataset,
prediction, validation-summary, and near-junction-audit hashes. Existing saves
created before junction evidence remain resumable and are upgraded on save;
superseded saves are copied into `versions/`.

Launch it with:

```bash
MPLCONFIGDIR=/tmp/matplotlib-width-qc-disagreements \
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python -m \
  network_variants.width_qc_disagreement_gui \
  --reviewer "<reviewer name>"
```

### Near-junction width audit

**Status: exploratory implementation complete (2026-08-03); window approval
pending.**

`network_variants/width_junction_audit.py` calculates upstream- and
downstream-node wet widths for every link using medians of valid, trimmed
two-bank samples inside physical along-link windows. Sample order is checked
against directed node geometry link-by-link, so reversed sample sequences are
handled explicitly. The audit compares plain incoming/outgoing whole-link
width sums with incoming/outgoing near-junction width sums for all 1,269
reviewed links. It does not alter the classifier, review archives, or widths.

The exploratory defaults compare three percentage windows with physical
minimum and maximum lengths:

```text
5%,  100–500 m
10%, 250–1000 m
15%, 250–1500 m
```

Run with:

```bash
MPLCONFIGDIR=/tmp/matplotlib-width-junction-audit \
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python -m \
  network_variants.width_junction_audit
```

Outputs under `network_variants/output/width_qc_junction_audit/` include all
endpoint widths, reviewed-link comparisons for every candidate window, an
aggregate window profile, a cross-window targeted-review ranking, and a
manifest containing exact source hashes. Automatic model integration remains
blocked until one window definition and its use conditions are approved.

### Hybrid near-junction rule audit

**Status: implemented and executed (2026-08-03); production approval pending.**

`network_variants/width_junction_hybrid_audit.py` tests the selected
10%/250–1,000 m endpoint widths without modifying the completed adjudication,
training data, model, or any network width. At every upstream junction, the
incoming reach uses its downstream endpoint and each outgoing reach uses its
upstream endpoint. Endpoint substitution is decided independently per adjacent
reach.

The audit compares whole-link means, endpoint widths for every link, and
length/wet-width thresholds of 3, 5, and 10, each with and without requiring
non-overlapping endpoint windows. Resolved adjudications become exploratory
effective labels, while all 11 ambiguous adjudications remain unlabeled and
are written to the follow-up set. The outputs include 1,269 links by eight
variants, a variant profile, all ambiguous cases, a targeted follow-up list,
and a manifest binding the exact dataset, endpoint-audit, and completed-review
hashes.

Run with:

```bash
MPLCONFIGDIR=/tmp/matplotlib-width-junction-hybrid \
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python -m \
  network_variants.width_junction_hybrid_audit
```

Outputs are stored under
`network_variants/output/width_qc_junction_hybrid_audit/`. This remains an
audit stage: none of the candidate hybrid rules is yet authorized for
production width correction.

### Frozen adjudicated hybrid model

**Status: implemented and executed (2026-08-03); routing-width integration
pending.**

`network_variants/width_qc_final_model.py` is the terminal model-development
step. It freezes `lw_ge_10_nonoverlap`, adds its absolute upstream-continuity
deviation to the original seven predictors, applies the completed adjudication
once, and excludes the 11 ambiguous cases. It then repeats leave-one-complete-
example-out validation for logistic regression, linear SVM, and RBF SVM using
the same predefined threshold constraint. No new disagreement review feeds
back into this dataset.

The selected interpretable model is logistic regression at probability 0.40.
Among 1,258 included reviewed candidates, its held-out predictions contain 533
true corrections, 701 true accepts, 17 false corrections, and seven missed
corrections. Relative to the original held-out logistic model, 27 decisions
change: 26 accept-to-correct and one correct-to-accept. These are validation
predictions, not edits to source networks.

Run with:

```bash
MPLCONFIGDIR=/tmp/matplotlib-width-qc-final \
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python -m \
  network_variants.width_qc_final_model
```

The workflow writes the final dataset, excluded cases, grouped-validation
artifacts, candidate bundles, old/new prediction comparison, per-example
decision summary, validation comparison, and exact source hashes under
`network_variants/output/width_qc_final_model/`. The model specification is
frozen, but `production_authorized` remains false until Phases 7–9 implement
and validate the separate numeric correction solver. Numeric correction
magnitudes are deliberately not inferred independently per link because
multiple flagged outgoing links must be solved jointly.

Execute a versioned copy with:

```bash
MPLCONFIGDIR=/tmp/matplotlib-width-qc-review \
IPYTHONDIR=/tmp/ipython-width-qc \
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/jupyter nbconvert \
  --to notebook --execute \
  network_variants/notebooks/width_qc_model_review.ipynb \
  --output-dir network_variants/output/width_qc_model_review \
  --output width_qc_model_review_executed_v5.ipynb \
  --ExecutePreprocessor.timeout=300 \
  --ExecutePreprocessor.kernel_name=python3
```

Create a reproducible notebook, suggested location:

```text
network_variants/notebooks/width_qc_model_review.ipynb
```

The notebook reads generated datasets and model artifacts; it must not contain
hidden manual data preparation that is absent from production code.

The notebook should provide:

1. review counts by example and class;
2. missing-data and predictor distributions;
3. good/bad distributions for each agreed predictor;
4. predictor correlations and collinearity checks;
5. logistic-regression coefficients and odds ratios;
6. SVM decision-boundary diagnostics where meaningful;
7. leave-one-example-out performance tables;
8. ROC, precision-recall, and calibration plots;
9. threshold trade-off plots;
10. results restricted to `split_sensitive=True`;
11. manual/algorithm disagreement tables;
12. maps or local link-context plots for selected disagreements;
13. high-impact false-positive and false-negative inspection;
14. final production-model and threshold recommendation.

Notebook execution should write or reference versioned outputs rather than
silently replacing production models.

### Acceptance criteria

- The notebook runs from a clean kernel using documented inputs.
- Numerical model results match the production training script.
- Disagreements can be traced back to example and link geometry.
- Model choice and threshold are reviewable before deployment.

## Phase 7 — Measured versus effective routing-width contract

**Status: standalone contract implemented (2026-08-03); variant-output wiring
remains Phase 9.**

Keep wet, dry, and total width families as measured geometric products. Add a
separate canonical effective width:

```text
wid_adj_routing
```

Add provenance and QC fields such as:

```text
width_qc_candidate
width_qc_split_sensitive
width_qc_opposition_trimmed
width_qc_invalid_fraction
width_qc_us_ratio_before
width_qc_us_ratio_after
width_qc_decision
width_qc_reason
width_qc_flag
width_correction_applied
width_correction_source
width_algorithm_version
```

Semantics:

- `wid_adj_total`: measured controlling-bank corridor width;
- `wid_adj_wet`: measured original wet component;
- `wid_adj_dry`: measured newly filled/dry component;
- `wid_adj_routing`: accepted or corrected effective routing width.

Do not overwrite raw wet, dry, or total sample measurements.

### Acceptance criteria

- Measurement and inference remain distinguishable.
- Every correction retains its raw value and provenance.
- Maps can still display the physical wet/dry/total decomposition.

## Phase 8 — Upstream-continuity correction solver

**Status: selected adaptive joint solver implemented and evaluated
(2026-08-05); Phase 9 base wiring is implemented and awaits a user-run base
validation.**

`network_variants/width_routing_solver.py` preserves `wid_adj_wet` and writes a
separate `wid_adj_routing`. It processes directed junctions in topological
order so an upstream correction propagates into the downstream endpoint
context. At each junction it keeps accepted outgoing context fixed, allocates
the remaining incoming context jointly across all predicted-bad outgoing
links in proportion to their current local context, and converts those context
targets to routing-width scale factors. The frozen `lw_ge_10_nonoverlap`
endpoint rule supplies local US/DS context; all other links use whole-link wet
width.

The first evaluation uses explicit provisional scale bounds of 0.25–4.0. A
whole sibling group is retained raw and flagged if the remaining width is
nonpositive, the allocation lies outside those bounds, the source has no
incoming context, or the directed graph is cyclic. Ambiguous training cases
are also retained raw and flagged. No measured fields or source networks are
edited.

Run the evaluation with:

```bash
MPLCONFIGDIR=/tmp/matplotlib-width-routing-solver \
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python -m \
  network_variants.width_routing_solver
```

Across the 16 current base examples, 550 held-out candidate decisions request
correction. The provisional solver applies 393 bounded corrections and retains
157 predicted corrections raw with a safety flag; the 11 excluded ambiguous
links add 11 further flags. Of the 157 unresolved predicted corrections, 101
exceed the scale bounds and 56 have no positive width remaining after accepted
outgoing links. These evaluation results are diagnostic and remain
`production_authorized: false`.

For direct inspection, the evaluation also writes
`width_routing_solver_rejected_corrections_v1.csv`: one row for each of the 157
rejected predicted corrections. It combines link and junction evidence and
retains the rejected candidate routing width, candidate scale factor, violated
bound, sibling IDs, incoming/accepted/remaining context widths, and continuity
diagnostics. No link/junction join is required.

Classification and numeric correction are separate operations. The supervised
model decides whether a candidate should be corrected. The correction solver
then calculates a safe effective routing width.

At an upstream split:

```text
incoming effective routing width
    approximately equals
sum(outgoing effective routing widths)
```

For one flagged outgoing link:

```text
target = incoming routing width - sum(other accepted outgoing widths)
```

For multiple flagged outgoing links:

- solve them jointly;
- preserve their raw relative proportions where possible;
- allocate only the remaining upstream width;
- apply neighborhood-derived bounds;
- flag infeasible or extreme targets.

The selected adaptive solver uses upstream and downstream continuity jointly,
with frozen weights of 2:1. Accepted sibling links stay fixed. Classification
confidence, upstream deviation, and opposition QC determine how far the soft
scale bounds may relax; corrections beyond the normal 0.25–4 range are
applied only with an explicit flag.

Possible outcomes:

```text
accept
correct + flag
retain raw + flag
```

### Acceptance criteria

- Corrections improve upstream continuity.
- Multiple outgoing corrections are solved jointly.
- Effective widths remain positive.
- Unsafe corrections are never applied silently.

## Phase 9 — Base-state integration

**Status: complete; routing behavior and the final provenance/path contract
were validated by rerunning example 56 (2026-08-05).**

Both regular and panelized base materializers now execute the same frozen
logistic classifier and adaptive two-ended solver. They write
`wid_adj_routing`, the predictor/model/solver fields,
and a per-link CSV plus manifest in the state's `widths/` directory. The
routing input is the freshly measured `wid_adj_wet`. Permanent review archives
provide human labels and audit provenance only: their historical numeric
targets are never reapplied to measured width-family fields. Realized and
anchored copies use relative nested artifact references and are independently
portable.

After raw base width-family calculation:

1. validate the permanent review archive;
2. attach manual labels for training/audit provenance;
3. calculate the agreed model predictors;
4. predict correction probabilities for all candidates;
5. calculate fresh two-bank/upstream-continuity routing targets for predicted
   corrections;
6. do not reuse the legacy numeric target as the wet/routing target;
7. assign `wid_adj_routing`;
8. write manual/model disagreement and base width-QC reports before hierarchy
   analysis.

The versioned model and solver determine `wid_adj_routing`; manual decisions
remain permanently available as evidence and never overwrite fresh measured
width families.

### Acceptance criteria

- Every base link has an effective routing width.
- Targets are recalculated from current two-bank values.
- Base disagreement reports are reproducible.
- Collapse runs never modify archived review files.

## Phase 10 — Variant inheritance and changed-link evaluation

**Status: complete and validated with synthetic/integration tests plus the
final example-56 realized and anchored single-unit rerun (2026-08-05). The
real validation exposed and led to fixes for missing regenerated-child
`n_units` and ambiguous parallel-edge lineage. Current-state structural
membership is derived directly from child topology; exact geometry overrides
an ambiguous multi-edge parent-node path.**

### Unchanged links

For `unchanged_1to1` lineage, inherit exactly from the immediate parent:

- measured width fields;
- raw samples;
- `wid_adj_routing`;
- QC, model, and correction provenance.

Regular, panelized, and realized-to-anchored child paths now use one shared
inheritance contract. `unchanged_1to1` is only a candidate classification:
inheritance additionally requires topologically equal parent/child geometries
and a unique child-to-parent mapping. This prevents anchored segments or other
shortened links from inheriting whole-parent samples. Non-eligible links keep
their newly measured physical widths, have stale routing/model values cleared,
and receive `changed_requires_evaluation` status.

### Changed links

For split, merged, regenerated, or collapsed links:

1. calculate new measured wet/dry/total widths;
2. calculate model predictors;
3. predict correction probability;
4. apply accept/correct/flag logic;
5. calculate and assign `wid_adj_routing`;
6. record parent lineage and complete provenance.

The child evaluator now runs the frozen classifier only for links marked
`changed_requires_evaluation`. Inherited links remain fixed: neither their
model decision nor their `wid_adj_routing` is recalculated. Their accepted
effective widths (including scaled endpoint context for long links) still
participate in the joint two-ended continuity objective. After evaluation,
changed links receive `changed_evaluated`; the routing audit contains both
`evaluated_changed_link` and `preserved_inherited_link` rows so every final
child value has an explicit source.

Sequential states inherit from their immediate parent state.

### Acceptance criteria

- Unchanged links retain identical measured and effective widths.
- Changed links do not inherit obsolete parent measurements.
- Every child link is classified exactly once.
- Regular and panelized modes implement the same semantic contract.

## Phase 11 — Dry collapse-corridor handling

**Status: complete in code and validated with focused regression tests and a
shadow audit of the final example-56 single-unit products (2026-08-05). The
existing example-56 products contain no links above the meaningful-dry
threshold; future runs will write the new provenance fields to their width
products.**

When dry width is negligible, measured wet width and effective routing width
describe nearly the same cross-section. A correction updates only the routing
width; measured and legacy compatibility fields remain unchanged.

When dry width is meaningful:

- never scale total, wet, and dry measurements together;
- retain the measured decomposition unchanged;
- calculate `wid_adj_routing` independently from upstream continuity;
- do not interpret routing correction as a physical change to island or
  corridor geometry;
- flag extreme routing targets relative to measured wet/total widths.

The implemented audit records dry fraction, routing-to-wet and
routing-to-total ratios, and separate physical-consistency flags. Dry width is
meaningful at a fraction of 0.05 or greater. Measured wet or effective routing
width exceeding measured total corridor width by more than 5% is flagged;
extreme routing-to-wet corrections remain governed by the existing adaptive
solver bounds.

### Acceptance criteria

- Routing correction does not alter measured total or dry width.
- Maps retain the physical collapse-corridor decomposition.
- Dry-corridor cases have explicit correction provenance.

## Phase 12 — Downstream consumer migration

**Status: complete and validated by the final example-56 single-unit rerun
(2026-08-05). Written hierarchy path widths agree with independent
`wid_adj_routing` harmonic-mean calculations to numerical precision and differ
from the corresponding wet-width calculations where corrections apply.**

Use `wid_adj_routing` for:

- routing fractions;
- hierarchy equivalent path widths;
- collapse prioritization where effective flow width is intended;
- equivalent-resistance calculations;
- collapse-footprint buffering derived from effective pre-collapse flow width.
- RAPID branch allocation and K/celerity preparation when link-level widths
  are used (physical subreach-sample mode remains an explicit alternative).

Continue using measured fields for:

- total-corridor maps;
- wet/dry decomposition maps;
- geometric QA;
- RivGraph compatibility exports.

Temporary fallback precedence during migration:

```text
wid_adj_routing
wid_adj_wet
wid_adj
wid
wid_med
```

The hierarchy and resistance manifests record candidate precedence; variant
manifests record the preferred footprint field; RAPID outputs record the
resolved `rapid_width_source_field` and `rapid_branch_width_source_field`.

### Acceptance criteria

- Routing consumers do not silently use total corridor width.
- Maps retain all measured width families.
- Manifests state the width field and model version used.

## Phase 13 — Testing and staged rollout

### Unit tests

Test:

- archive migration and GUI paths;
- atomic writes and version preservation;
- stale mask/topology/link-ID detection;
- dataset construction;
- grouped model preprocessing without leakage;
- model serialization and reproducibility;
- probability-threshold application;
- single and multiple outgoing corrections;
- infeasible correction behavior;
- dry-corridor semantics;
- unchanged-lineage inheritance;
- regular/panelized parity.

### Staged integration

1. Archive and verify every existing review.
2. Train models and execute the review notebook.
3. Review manual/model disagreements.
4. Approve a production model, threshold, and version.
5. Run a small non-panelized base in shadow mode.
6. Run a small panelized base in shadow mode.
7. Validate run 41 base.
8. Validate run 46 base.
9. Validate one contained unit collapse.
10. Validate one cross-panel unit collapse.
11. Validate one compound collapse.
12. Compare regular and panelized behavior on a manageable example.
13. Enable authoritative automatic corrections only after approval.
14. Run complete independent-unit experiments.

## Related audit cleanup

Retain these separate tasks:

- fix base-state direction validation so it validates actual reused links;
- let new full runs recreate stale base outputs with current width precedence;
- rerun example 45's complete experiment if its existing child states are
  required.

## Immediate implementation sequence

Implement Phases 1–3 first:

1. migrate and verify all existing reviews;
2. add fingerprints and stale-review validation;
3. change the GUI default save location and add overwrite-safety tests.

Then implement Phases 4–6 together:

1. build the reproducible modelling dataset;
2. train and compare logistic-regression and SVM pipelines;
3. create and execute the model-review notebook;
4. review held-out performance and manual/model disagreements.

Do not integrate automatic corrections into base or child variants until the
review archive is safe and a production model and threshold have been
explicitly approved.
