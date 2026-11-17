# River-mask cleaning: held-out pilot

**Result: the current models do not reliably reproduce the manual cleanup while preserving river geometry.** The models must be judged on banks, islands and connectivity as well as water overlap.

The highest observed learned-model mean held-out IoU was **0.573** (Image trees: original + binary), versus **0.558** for the original prepared masks. These are exploratory comparisons on 15 rivers, with all training and model selection separated from each test river.

For that model, boundary F1 was **0.396** versus **0.490** originally, and clDice was **0.497** versus **0.563**. Its retained-water loss was **18.19%**. These measures expose costs hidden by water-area overlap alone.

The practical next step is to separate main-river selection from bank/island repair: provide a target-independent river corridor or reliable channel seed, then train and assess local corrections within that context. Add explicit labels for wetland edges and erroneous versus legitimate islands. Retain human review of proposed edits until these geometric checks pass on new rivers.

This is an experimental feasibility benchmark, not a production replacement for reviewed masks. Original/prepared/cleaned inputs were not modified.

## Data and design

- 17 paired prepared/cleaned masks found; 15 used. Run 41 lacks a manual-edit summary; run 56 has no accessible original source at its recorded path.
- 840 native-resolution tiles: 40 input-water-selected tiles plus 8 add-edit and 8 remove-edit locations per river. Each has a 192 × 192 scoring core and 32-pixel context on every side.
- All 840 cached pairs were checked exactly against their georeferenced source rasters. Eligible river bounding boxes do not overlap.
- Four fixed outer folds; every river is tested once. Each fold uses disjoint training, validation and test rivers. Validation chooses U-Net checkpoint and thresholds; test results do not guide training.
- Prepared/cleaned raster zeros are valid land despite their nodata metadata. Cropped target boundaries are aligned using their affine transforms. Training and tile evaluation stay inside the available cleaned extent.
- Original source values are reprojected bilinearly and scaled by the recorded source threshold, clipped to five times that threshold. These values are not assumed to be calibrated water probabilities.
- Image-feature models are Extra Trees with local densities, distance transforms and component sizes. Binary-only versus original-valued features is an input ablation. U-Net uses binary and original-valued channels, boundary/edit-weighted cross-entropy, Dice and soft skeleton loss.
- U-Net is a small residual model with rotations/flips, up to 18 epochs of 192 sampled tiles, early stopping after six unimproved validation epochs. This is a bounded pilot, not an exhaustive architecture or training-budget search.
- The simple baseline removes water components below 16 pixels and fills enclosed holes up to 3 pixels using context tiles. It is local cleanup, not a global main-river selector.

## Primary results

Each river has equal weight. These are sampled source-water-containing tiles inside the cleaned extent, not whole-river or whole-floodplain accuracy.

| Method | IoU ↑ | Boundary F1 ↑ | clDice ↑ | Removal recall ↑ | Addition recall ↑ | Edit precision ↑ | Damage rate ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unchanged mask | 0.558 | 0.490 | 0.563 | 0.000 | 0.000 | — | 0.000% |
| Small-object/hole rules | 0.558 | 0.491 | 0.565 | 0.007 | 0.080 | 0.728 | 0.013% |
| Image trees: binary | 0.540 | 0.370 | 0.515 | 0.672 | 0.406 | 0.563 | 2.796% |
| Image trees: original + binary | 0.573 | 0.396 | 0.497 | 0.689 | 0.448 | 0.583 | 2.469% |
| U-Net | 0.563 | 0.467 | 0.516 | 0.328 | 0.000 | 0.726 | 0.799% |
| U-Net: conservative edits | 0.561 | 0.485 | 0.553 | 0.027 | 0.000 | 0.703 | 0.025% |

Damage rate is the fraction of initially correct pixels changed incorrectly. Edit precision is the fraction of proposed edits agreeing with the cleaned target; undefined values are omitted from macro averages. Boundary matches allow two pixels, whose ground distance varies by river. clDice measures skeleton overlap and is not a guarantee of correct river-network topology.

![Per-river scores](results/river_scores.png)

## Paired uncertainty

95% bootstrap intervals resample whole rivers (5,000 replicates). They describe variation across these rivers, not training-seed uncertainty.

| Method | Mean IoU change | 95% interval | Rivers improved |
|---|---:|---:|---:|
| Small-object/hole rules | +0.001 | [-0.000, +0.001] | 11/15 |
| Image trees: binary | -0.018 | [-0.098, +0.061] | 9/15 |
| Image trees: original + binary | +0.015 | [-0.063, +0.094] | 8/15 |
| U-Net | +0.005 | [-0.049, +0.051] | 12/15 |
| U-Net: conservative edits | +0.003 | [+0.000, +0.007] | 8/15 |

## Saved-edit challenge locations

These are deliberately difficult, label-selected locations and are reported separately. Add/remove actions do not tell us whether an edit was a wetland, island, bank, or scope decision.

| Location | Method | IoU | Addition recall | Removal recall | Edit precision |
|---|---|---:|---:|---:|---:|
| edit_add | Unchanged mask | 0.884 | 0.000 | 0.000 | — |
| edit_remove | Unchanged mask | 0.828 | 0.000 | 0.000 | — |
| edit_add | Small-object/hole rules | 0.884 | 0.004 | 0.010 | 0.700 |
| edit_remove | Small-object/hole rules | 0.829 | 0.001 | 0.008 | 0.680 |
| edit_add | Image trees: binary | 0.745 | 0.418 | 0.553 | 0.198 |
| edit_remove | Image trees: binary | 0.684 | 0.485 | 0.588 | 0.269 |
| edit_add | Image trees: original + binary | 0.765 | 0.474 | 0.630 | 0.237 |
| edit_remove | Image trees: original + binary | 0.725 | 0.531 | 0.632 | 0.325 |
| edit_add | U-Net | 0.832 | 0.000 | 0.194 | 0.156 |
| edit_remove | U-Net | 0.781 | 0.000 | 0.187 | 0.276 |
| edit_add | U-Net: conservative edits | 0.884 | 0.000 | 0.006 | 1.000 |
| edit_remove | U-Net: conservative edits | 0.828 | 0.000 | 0.000 | — |

## Islands and connectivity limitations

Island metrics count enclosed 4-connected land regions of at least 4 pixels; water uses 8-connectivity. Islands intersecting a tile boundary are excluded. Per-tile component counts and island-area IoU are saved in the machine-readable results. These cannot establish whole-network connectivity or identity-preserving island matching.

The target combines manual decisions and automatic component/hole cleanup. A local model may not infer which distant water body belongs to the intended main river. Unusual island shape alone is insufficient evidence of an error. No explicit wetland/island semantic labels or independent expert rereview were available.

## Visual review

The examples include the two largest improvements, two largest regressions, and hardest add/remove edit examples, measured by pixel error. They are selected for diagnosis, not as representative samples.

![Held-out examples](results/held_out_examples.png)

## Reproducibility and artifacts

See [README.md](README.md) for commands. `results/audit.json`, `integrity.json`, `tiles.json`, `splits.json`, and `environment.json` record data, checks and experiment setup. Each fold saves models, training history, validation selection, exact binary predictions and pixel counts. `summary.json`, `per_river.json`, `paired_bootstrap.json`, and `threshold_sensitivity.json` contain numerical results.

Prediction scores are uncalibrated; conservative cutoffs do not establish a confidence guarantee. Threshold-sensitivity calculations use stored float16 scores. Original sources remain external dependencies.


## Bank and island preservation checks

Evaluation-only near-channel strata include retained water and land within 16 pixels. Island-object recall requires at least 50% intersection-over-union with an enclosed reference island of at least four pixels. Cropped islands are excluded. Retained-water loss is the fraction of water already correct in the input that the model deletes. Values below are river means.

| Method | Near-channel IoU ↑ | Island-object recall ↑ | Retained-water loss ↓ |
|---|---:|---:|---:|
| Small-object/hole rules | 0.954 | 0.763 | 0.12% |
| Image trees: binary | 0.736 | 0.062 | 20.53% |
| Image trees: original + binary | 0.762 | 0.109 | 18.19% |
| Unchanged mask | 0.955 | 0.763 | 0.00% |
| U-Net | 0.881 | 0.486 | 7.91% |
| U-Net: conservative edits | 0.952 | 0.763 | 0.24% |

## Repeat training and original-value ablation

| CNN training run | Mean IoU | Boundary F1 | clDice |
|---|---:|---:|---:|
| Primary seed | 0.563 | 0.467 | 0.516 |
| Second seed | 0.563 | 0.454 | 0.522 |

The second run repeats the same architecture, training budget, disjoint river splits and validation selection, changing only the training seed. Two seeds are a sensitivity check, not a complete estimate of training variability.

Adding original-valued features to the tree model changed mean IoU by **+0.033**, with paired river-bootstrap 95% interval **[+0.007, +0.061]**. This tests the value of information retained before binarization; it does not establish that all source encodings will transfer.

## Complete held-out river 03

Both learned models were applied over the **entire prepared raster** without reading cleaned labels. Scoring then used the complete available cleaned extent. River 03 was excluded from both training and validation for these models. A separate global baseline selects the largest 8-connected water component from the full prepared raster and fills tiny holes; this uses no target geometry. This single-river result is a stress test, not a second independent population estimate.

| Method | IoU | Boundary F1 | clDice | Predicted / target water components | Predicted / target islands |
|---|---:|---:|---:|---:|---:|
| Unchanged mask | 0.528 | 0.439 | 0.543 | 1528 / 150 | 93 / 33 |
| Small-object/hole rules | 0.531 | 0.450 | 0.559 | 695 / 150 | 93 / 33 |
| Global largest component + tiny-hole fill | 0.841 | 0.716 | 0.816 | 2 / 150 | 71 / 33 |
| U-Net | 0.564 | 0.449 | 0.590 | 2909 / 150 | 266 / 33 |
| Image trees: original + binary | 0.554 | 0.340 | 0.514 | 2810 / 150 | 360 / 33 |

![Full held-out river](results/whole_river_03/overview.png)

Experimental georeferenced masks: [U-Net](results/whole_river_03/03_unet_experimental.tif), [original-valued image trees](results/whole_river_03/03_trees_raw_experimental.tif), [U-Net scores](results/whole_river_03/03_unet_experimental_probability.tif), and [global component baseline](results/whole_river_03/03_global_component_experimental.tif). These are review artifacts, not replacements for the production masks.

## Verification

- **21 automated tests passed**: grid/crop alignment, invalid offsets/CRS, edge preservation, island handling, empty predictions, connectivity disruption, feature isolation, split integrity, model gradients/reload, GPU-to-CPU checkpoint agreement, streaming georeferencing and non-overwrite behavior.
- **840/840** cached pairs matched the original prepared and cleaned rasters exactly; geographic extents of the eligible rivers did not overlap.
- **5,040** primary held-out prediction artifacts were independently checked against saved confusion counts, harmful-edit counts, score ranges and disjoint split membership.
- Thresholds were checked against their validation selection tables. Saved model SHA-256 hashes are recorded in `results/verification.json`.

## Method references

- [U-Net](https://arxiv.org/abs/1505.04597): encoder-decoder segmentation architecture; this pilot uses a smaller custom residual variant.
- [clDice](https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html): skeleton-overlap supervision and evaluation. The finite-iteration soft-skeleton term used here does not guarantee globally correct topology.


## Per-river primary results

| River | Original IoU | Original-valued trees IoU | U-Net IoU | Trees bank F1 change | U-Net bank F1 change |
|---|---:|---:|---:|---:|---:|
| 02 | 0.838 | 0.806 | 0.857 | -0.217 | -0.079 |
| 03 | 0.600 | 0.766 | 0.633 | +0.029 | -0.022 |
| 05 | 0.544 | 0.767 | 0.770 | +0.179 | +0.142 |
| 10 | 0.496 | 0.516 | 0.526 | -0.026 | -0.024 |
| 11 | 0.678 | 0.596 | 0.686 | -0.235 | +0.003 |
| 13 | 0.750 | 0.747 | 0.759 | -0.182 | -0.023 |
| 15 | 0.926 | 0.815 | 0.929 | -0.284 | -0.001 |
| 21 | 0.257 | 0.342 | 0.277 | -0.049 | +0.003 |
| 22 | 0.965 | 0.918 | 0.880 | -0.187 | -0.138 |
| 37 | 0.710 | 0.441 | 0.432 | -0.453 | -0.222 |
| 45 | 0.369 | 0.439 | 0.372 | -0.030 | -0.005 |
| 46 | 0.364 | 0.665 | 0.409 | +0.102 | -0.006 |
| 48 | 0.570 | 0.274 | 0.550 | -0.140 | +0.034 |
| 54 | 0.268 | 0.421 | 0.292 | +0.059 | -0.016 |
| 55 | 0.031 | 0.082 | 0.067 | +0.033 | +0.015 |
