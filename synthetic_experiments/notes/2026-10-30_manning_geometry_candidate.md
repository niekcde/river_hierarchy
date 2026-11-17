# Geometry-only Manning candidate — 30 October 2026

Reference: **GEOMETRY-MANNING-PAIRED-2026-10-30**.

## Why added

The user explicitly requested a geometry-only Manning descriptor after excluding resistance and inverse-depth G. Its inputs qualify: `ΔG_M = Δ Σ L B^(2/5) f^(3/5) S^(-3/10)`, with independently recomputed connectivity/width allocation in each whole network. No fitted n, hydraulic depth, simulated discharge or response enters the feature. The physical exponents assume wide-channel Manning hydraulics, with common constant n and inlet Q for the incremental-storage identity. The descriptor has units m^(7/5), treating slope as dimensionless; it is not itself a time.

The shared implementation is `RAPID/src/rapid_tools/geometric_manning.py`. Tests verify equal-width split invariance, subdivision invariance, the fixed-Manning identity, independence of attached hydraulic values, and changes to allocation on surviving reaches. Recomputed allocation and geometry sums also match 33 saved synthetic states spanning three grid geometries to numerical precision.

## Synthetic candidate screen

205 representations are now included in notebook Section 8. Resistance and inverse-depth G remain excluded. For variable-Manning centroid shift, Manning ΔG leads mean absolute within-type Spearman (0.9814), followed by route-length change (0.9564), equivalent-length change (0.9530), channel area change (0.9496), and spread-related descriptors (~0.9137). For peak shift, Manning ΔG is weaker (0.6989); area, equivalent length and route-length changes lead. These are descriptive full-grid associations, not held-out R², independent concepts, or evidence of unique added value.

## Paired real-network test

Outputs: `/Volumes/SyntheticResults20261027/analysis/real_geometry_manning_20261030/`.
Runner: `RAPID/investigations/geometric_manning_20261030/run.py`.
Notebook: `synthetic_experiments/notebooks/synthetic_predictor_validation.ipynb`, Section 9.

2,185 comparisons, 450 groups, 15 networks; fixed-Manning centroid shift in hours using the existing finite 0.5% event window. Reused exact five repeats × five outer compound-grouped folds, training/network weights and preprocessing. For each fold, reuse the nine-input RF's inner-selected settings and add G_M without new tuning. A refit of the first baseline fold reproduces saved predictions. Also fit a training-only network-weighted affine regression on G_M alone. Twenty-six new RF fits (including the parity check), 25 affine fits; no routing or production model changes.

| Model | Mean R² | Mean MAE, min | Relative-error score, % |
|---|---:|---:|---:|
| Existing nine-input RF | 0.651 | 20.440 | 35.313 |
| Nine + Manning ΔG | 0.645 | 20.438 | 29.983 |
| Manning ΔG alone, affine | 0.620 | 25.969 | 54.689 |

These all-population metrics use equal cohort/network weighting. The relative-error score uses the existing one-minute denominator floor, so is not plain unbounded MAPE. Repeat ranges are exported, not confidence intervals. The objective used to tune the original RF was relative error, not R².

The addition improves relative-error score by 5.33 percentage points (about 15.1% relative reduction), but does not improve pooled R² or materially change MAE. The original-network cohort improves: R² 0.620→0.684 and MAE 0.740→0.609 min. Segmented-cohort R² falls slightly, 0.417→0.406, and MAE 40.140→40.268 min, while its relative-error score improves. Therefore the descriptor is a useful candidate for the shortlist, not a demonstrated universal replacement.

For context, the existing length-only affine baseline has higher all-population R² (0.778), but worse relative-error score (43.580%) than either RF. Do not select by a single score without specifying which errors matter.

## Limits and next decision

This is a paired addition under the current RF settings, not a fully retuned ten-input search. It predicts related states of known networks, not transfer to held-out networks. The inherited 15 full-network singleton exceptions and reuse of historical partitions remain limitations. No peak-response improvement was tested on the real networks here. Synthetic hydraulic assumptions favor this physical descriptor, so synthetic success is not independent physical validation.

Keep Manning ΔG alongside route-length, area/equivalent-length and spread candidates. A later grouped subset comparison must distinguish interchangeable or redundant measures from genuinely complementary information. Do not declare the final ten from descriptive rankings alone.
