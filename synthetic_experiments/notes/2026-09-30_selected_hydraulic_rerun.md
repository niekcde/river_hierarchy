# Selected hydraulic rerun of the synthetic analyses

**Reference: SYN-2026-09-30-SELECTED-RERUN**  
**Written: 30 September 2026.**  
**Reason:** after choosing alpha=.65 for synthetic width–depth scaling, the user
requested that the current synthetic notebooks be updated because hydraulic
responses, resistance and predictive relationships can change.

This supersedes the pending-rerun status in **HYD-2026-09-30-SELECTED-ALPHA065**,
not the historical evidence behind that choice. The September 29 checkpoint and
original outputs remain available. No real-case exponent was adopted here:
that decision waits for new runs using updated slopes.

## What is now current

`configs/active_analysis.json` selects `outputs/hydraulic_alpha065_20260930`.
Both the full factorial and full agreed width/length/embedding extension are
routed under the selected settings; no extreme configurations were excluded.
The balanced factorial is still analysed separately from the extension.

| Dataset | Configurations | Comparisons | Unique routing jobs | Maximum absolute fractional volume error |
| --- | ---: | ---: | ---: | ---: |
| full_grid_island | 2,187 | 15,309 | 9,519 | 1.13e-13 |
| embedded_width_length_extension | 140 | 980 | 981 | 2.30e-12 |

All jobs pass the existing initialization, final-settling and conservation
checks. No job activates a celerity cap. The effective-length floor is active
in 1,476 main-grid jobs and zero extension jobs; its influence remains part
of the numerical audit. The reference geometry, original geometry groups, development folds
and exploratory holdout assignments match the earlier analysis exactly.
Thirty-five overlapping comparisons reproduce the focused alpha=.65/native-clock
diagnostic for all three primary responses and resistance.

## Hydraulic and metric contract

BASED provides one reference depth per total-width/reference-slope configuration
at peak reference discharge 1000 m³/s. Parent and child share that reference.
Local depth and kb are

`D_i = D_ref (W_i/W_ref)^0.65`

`kb_i = kb_ref (W_i/W_ref)^0.35`.

Manning n=.035, X=.1, target and minimum effective segment length 200 m, and
celerity bounds 0.05–2.524 m/s are fixed. The 100→1000→100 m³/s pulse retains its
12 h rise and 24 h fall. Width-proportional allocation is unchanged. Each
configuration receives a shared admissible integer timestep and adequate
initialization/recession buffers.

The matched Chezy resistance also uses local kb. Its edge width exponent is
now 3.95 rather than 5: `K_i ∝ L_i / W_i^(2+3 alpha)` at a common reference.
The common reference factor cancels in the matched ratio. Omega remains
`length_fraction * exposure * [-log10(Lambda)]`. This is a geometry predictor
under an assumed depth law, not a Manning-equivalent hydraulic law.

The primary responses remain peak-lag change, 1% plateau-to-plateau event-duration
change, and peak-excess change. Parent and child use the same absolute event
threshold (1% of parent amplitude); 0.5% and 2% threshold checks are retained.
Temporal spread and volume-quantile duration are explicitly secondary.

## What changed in the original balanced grid

| Response | Previous setup: median | Selected setup: median |
| --- | ---: | ---: |
| Peak-lag change | -19.343% | -14.286% |
| Event-duration change | -9.762% | -1.056% |
| Peak-excess change | 9.432% | 1.329% |

These are medians across the same 15,309 comparisons, not an alpha-only effect.
The old setup used kb=20/n=.35; the selected one changes the BASED anchor,
local depth scaling and roughness together. The alpha-only/common-clock
sensitivity remains available in the hydraulic investigation.

The shift is especially large for event duration and peak excess. Earlier
claims about strong widening or separated arrivals therefore need to be read
against the new hydrographs, rather than carried forward unchanged. The main
factor rankings, all 21 pair interactions, signed contrasts and omega residual
analysis are recomputed. Variance shares remain nonnegative; direction is
shown separately with signed contrasts.

## Predictor refresh

Candidate descriptors, estimator specifications and data partitions are held
fixed. Models and development-only subset selection are fitted again to the
new responses and resistance. The selected shared set now has six inputs:
`resistance_log + width_fraction + parent_equivalent_length + parent_dominant_width_fraction + signed_length_width_association + domain_reference_slope`.
It is a near-best set within the bounded search and stated tolerance, not a
proof of globally minimal predictors. The extension does not select features.

Treatment-averaged reused-holdout R² for the fixed boosted estimator:

| Feature group | Lag | Event duration | Peak excess |
| --- | ---: | ---: | ---: |
| Selected six | 0.909 | 0.713 | 0.790 |
| Selected + change + surroundings | 0.941 | 0.789 | 0.839 |

These are exploratory interpolation scores. The full extension separately
checks 63 anchors and evaluates 917 novel comparisons. It is a demanding
extrapolation test with different input ranges; consult the predictor notebook
for its scores, not the original-grid interpolation table alone. Transfer is
poor: the selected shared model averages R²=-2.454 for lag, -0.138 for event
duration and -0.088 for peak excess across extension treatments. Omega alone
averages 0.114 for lag. These results do not support a universal selected
predictor; retain the extension as an explicit limitation, without retuning
the model on its evaluation rows.

## Numerical and hydraulic limits

The seven-geometry nested/full audit was rerun at 200/100/50 m, alongside separate
timestep, X and forcing-duration changes. All 42 pair reruns complete. Between
200 m and 50 m, the largest absolute response differences are 1.31 percentage
points for lag, .54 for duration and .71 for peak excess. This is limited
coverage, not full-grid convergence. Refinement changes the associated length
floor too; peak timing is discrete.

Assigned BASED/local depths and width-allocated reference discharges do not
necessarily satisfy Manning discharge closure. The selected hydraulic notebook
reports both reference coordinate definitions and this mismatch. Its provisional
F²/T⁺ screen uses a 12 h rise-time proxy; it is not the full published wave-regime
diagram or proof of applicability. No historical real-case overlay is presented
as a matched selected-hydraulic validation.

## Where to resume

- [Results synthesis](../notebooks/synthetic_results_synthesis.ipynb): current
  sensitivity, interactions, omega residuals, extension and matched setup changes.
- [Predictor validation](../notebooks/synthetic_predictor_validation.ipynb):
  refreshed fixed benchmarks, compact selection and full-extension transfer.
- [Factor interactions](../notebooks/synthetic_factor_interactions.ipynb):
  detailed primary-response sensitivity and selected focused hydrographs.
- [Hydraulic applicability](../notebooks/synthetic_hydraulic_applicability.ipynb):
  dated diagnostic history followed by the new full selected-run audit.

Geometry examples and routing results are refreshed in their existing notebooks.
Earlier geometry/spread notebooks are marked historical, and notebook snapshots
are stored under the versioned output directory. Reproduction commands are in
[RUNBOOK.md](../RUNBOOK.md). Real-network validation and final framing remain
open until the updated-slope runs are available. Predictive skill for these
deterministic outputs remains conditional on grid ranges, forcing, allocation
and routing assumptions; it is not by itself a general physical law.
