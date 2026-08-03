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
been executed successfully from a clean project kernel with all 14 code cells
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
They use interactive Plotly scatter panels with network/link hover labels;
classifier-input panels include the partial logistic decision boundary with
all remaining classifier inputs fixed at their medians.

Execute a versioned copy with:

```bash
MPLCONFIGDIR=/tmp/matplotlib-width-qc-review \
IPYTHONDIR=/tmp/ipython-width-qc \
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/jupyter nbconvert \
  --to notebook --execute \
  network_variants/notebooks/width_qc_model_review.ipynb \
  --output-dir network_variants/output/width_qc_model_review \
  --output width_qc_model_review_executed_v4.ipynb \
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
wid_adj_routing_raw
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

Downstream continuity is recorded as a diagnostic and does not drive the
primary correction.

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

During shadow rollout, predictions and suggested corrections are recorded but
do not replace the approved routing width. After model review and approval,
the versioned production model becomes authoritative while manual decisions
remain permanently available as evidence.

### Acceptance criteria

- Every base link has an effective routing width.
- Targets are recalculated from current two-bank values.
- Base disagreement reports are reproducible.
- Collapse runs never modify archived review files.

## Phase 10 — Variant inheritance and changed-link evaluation

### Unchanged links

For `unchanged_1to1` lineage, inherit exactly from the immediate parent:

- measured width fields;
- raw samples;
- `wid_adj_routing`;
- QC, model, and correction provenance.

Panelized execution already reuses unchanged parent width records and must be
extended to the new routing/model fields. Regular full-raster execution must
gain equivalent exact inheritance after lineage resolution.

### Changed links

For split, merged, regenerated, or collapsed links:

1. calculate new measured wet/dry/total widths;
2. calculate model predictors;
3. predict correction probability;
4. apply accept/correct/flag logic;
5. calculate and assign `wid_adj_routing`;
6. record parent lineage and complete provenance.

Sequential states inherit from their immediate parent state.

### Acceptance criteria

- Unchanged links retain identical measured and effective widths.
- Changed links do not inherit obsolete parent measurements.
- Every child link is classified exactly once.
- Regular and panelized modes implement the same semantic contract.

## Phase 11 — Dry collapse-corridor handling

When dry width is negligible, measured wet width and effective routing width
describe nearly the same cross-section. A correction may update routing and
relevant no-dry compatibility summaries while preserving originals.

When dry width is meaningful:

- never scale total, wet, and dry measurements together;
- retain the measured decomposition unchanged;
- calculate `wid_adj_routing` independently from upstream continuity;
- do not interpret routing correction as a physical change to island or
  corridor geometry;
- flag extreme routing targets relative to measured wet/total widths.

### Acceptance criteria

- Routing correction does not alter measured total or dry width.
- Maps retain the physical collapse-corridor decomposition.
- Dry-corridor cases have explicit correction provenance.

## Phase 12 — Downstream consumer migration

Use `wid_adj_routing` for:

- routing fractions;
- hierarchy equivalent path widths;
- collapse prioritization where effective flow width is intended;
- equivalent-resistance calculations;
- collapse-footprint buffering derived from effective pre-collapse flow width.

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

Every manifest records the resolved field.

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
