# Broad descriptor audit — 30 October 2026

Reference: **SYN-BROAD-DESCRIPTORS-2026-10-30**.

## Active scope: geometry and slope only

Updated user decision, 30 October 2026: retain the exclusion of `resistance_explicit_depth` and `delta_G_inverse_depth`, but include `delta_G_manning` as a candidate because its inputs are geometry and width/connectivity-derived allocation only. The active scope has **205 representations**. Its interpretation remains conditional on Manning hydraulics; this is not a universal physical law.

Section 8 reads `/Volumes/SyntheticResults20261027/analysis/broad_descriptors_20261030/geometry_with_manning/`. Generate with `investigations/geometry_only_descriptor_audit.py`. The earlier 204-candidate `geometry_only/` export and original 207-candidate export remain historical snapshots. The current scope excludes resistance and inverse-depth G across every active table.

Manning ΔG uses **whole networks**, an exception to the B-domain metrics: unchanged contributions cancel, while changed flow fractions on surviving reaches remain. It leads the descriptive centroid association screen (mean absolute within-type Spearman 0.9814), followed by mean route-length change (0.9564), equivalent-length change (0.9530), area change (0.9496) and spread-related measures (~0.9137). Its peak association is weaker (0.6989); no single universal predictor is established.

See [the paired real-network test](2026-10-30_manning_geometry_candidate.md) and notebook Section 9 for the separate incremental prediction test. No routing or production model was changed.

The original audit record follows.

## Why this was done

The earlier control/predictor matrices evaluated an already chosen set of ten. They did not justify that selection against a wider candidate library. This audit broadens the mechanistic screen using existing definitions, without requiring it to recover the original ten.

## Population and candidate inventory

137,781 completed comparisons: 6,561 geometries × three hydraulic methods × seven collapse treatments. No routing or model fits were run.

207 candidate representations are evaluated. These are not independent concepts: they include existing parent/child/difference/ratio forms of 37 geometric/allocation DAG metrics, historical path-width-weighted equivalents and elongation, signed length–width association, geometric slope/head summaries, current predictors and the spread/entropy differences. Shared variables are counted once. The original ten inputs and two-figure summary remain unchanged.

All metrics use the common closed outer B domain. Historical path-harmonic-width weighting is labelled separately from actual width-based flow allocation; the two should not be treated as equivalent definitions. Current hydraulic-reference candidates remain marked as such. The existing Manning G analytical identity is retained as a reference, excluded from the displayed candidate shortlist.

Deferred inventory: older omega/exposure/length fractions need explicit reconciliation with current B-domain conventions; output-proximal travel-time/celerity/diffusivity metrics are deferred; pixel-width and map-shape metrics lack corresponding synthetic data. This is a broad audit of available compatible definitions, not a claim to have exhausted every metric in the repository.

## Analysis

- Definedness and within-type variability for each method/treatment.
- Ordinary finite-pair Spearman between all eight controls and every candidate (34,776 entries); ordered internal location.
- Exact full-grid main, total and all 28 pair interaction variance contributions. Incomplete or constant candidates receive no normalized decomposition; no imputation.
- Exploratory, full-grid within-type associations with centroid and peak. Ranking averages absolute Spearman equally across the six response-resolved types and requires complete, varying candidates in all six. These are descriptive associations, not held-out R² or unique added information. Signed per-type associations are retained.
- A supplementary variance-pattern distance is exported: half the sum of squared differences between square roots of the disjoint main/pair/higher-order shares. This is a descriptive pattern diagnostic, not a prediction score or selection rule.
- 27 affine aliases detected across the full variable-Manning population by hashing standardized values rounded to eight decimals. This identifies some algebraic overlaps; it is not complete nonlinear redundancy analysis and removes nothing from the library.

## Initial results

For variable Manning, 141 candidates are fully defined and vary within at least one treatment; 40 are partially defined; 26 are constant within every treatment. Constant-within-type topology measures may still distinguish types and must not be rejected as generally uninformative.

Mean absolute within-type Spearman with centroid (six resolved types, complete candidates): inverse-depth G about 0.978; mean route-length change 0.956; equivalent-length change 0.953; channel-area-proxy change 0.950; route-spread/variance descriptors around 0.914; resistance change around 0.885. The Manning analytical reference is about 0.981. Several representations are redundant or monotonically equivalent, so these numbers do not establish separate mechanisms or a final ranking of added value.

Peak ordering differs: area-proxy change about 0.769; equivalent-length change 0.750; mean route-length change 0.744; inverse-depth G 0.729. Their comparative ranking needs held-out testing before any predictive claim. High Spearman can coexist with poor absolute calibration.

The broader set therefore reinforces length/allocation/scale as the central centroid mechanisms, while identifying equivalent-length, area and spread changes as useful simple comparators. It does not demonstrate that the original ten are necessary, optimal, or replaceable by a particular subset.

## Location and reproducibility

Notebook: `synthetic_experiments/notebooks/synthetic_predictor_validation.ipynb`, Section 8 (inventory, exploratory association tables, family-selectable correlation/main/pair matrices). The main two-figure story in Section 7e remains supporting-scope independent.

Outputs: `/Volumes/SyntheticResults20261027/analysis/broad_descriptors_20261030/` on the PhD-drive APFS container. Candidate catalog, coverage, full features, control correlations, factorial signatures, response associations, summary, affine aliases, deferred inventory and provenance hashes are saved there.

Rebuild using `investigations/broad_descriptor_audit.py`, then `investigations/broad_control_correlations.py`, then execute the notebook. Shared-code scaling tests pass for DAG and historical path metrics; existing predictor and difference values are checked for parity during generation.

## What remains before selection

Freeze a physically motivated shortlist after reviewing this screen, then test incremental predictive value with grouped held-out geometries and later real-network validation. Do not tune to the current full-grid descriptive ranking and call it independent validation. Do not force the outcome to contain exactly ten inputs.
