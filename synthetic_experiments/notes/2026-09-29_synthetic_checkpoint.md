# Synthetic checkpoint — 29 September 2026

Reference: **SYN-2026-09-29 — sensitivity/prediction pause; hydraulic applicability resumes**.

## Why this note exists

Written following the decision to pause further synthetic sensitivity and predictor exploration until the additional real-case results are available. The user asked to preserve when and why these conclusions were reached, then resume hydraulic applicability. This is a decision record, not a claim that every validation task is complete.

## What is ready for comparison

- Three current responses: peak-lag change, 1% plateau-to-plateau event-duration change, and peak-excess change. Use the [metric contract](../../RAPID/CURRENT_RESPONSE_METRICS.md); older temporal-spread analyses do not validate the current duration response.
- The balanced synthetic grid supplies controlled main effects, signed contrasts and all 21 pair interactions. Variance contributions remain nonnegative; signed contrasts describe direction. Conclusions are conditional on the tested ranges, forcing, allocation rule and routing model.
- Omega remains a readable lag axis. Shape changes need information beyond omega. Domain scale/proportions, width dominance, length–width association, slope, collapse changes and surrounding lengths are candidate information, not a universal physical formula.
- Original-grid predictor performance is useful interpolation evidence. The extension pilot shows poor transfer of the existing fitted models to the new extremes; this does not establish that the descriptors themselves are useless.
- The width/length and fixed-unit embedding pilot reaches lower omega and smaller lag changes. Small lag changes can still coexist with substantial event-duration changes, and some lag changes are no larger than one timestep.

## Evidence snapshot

| Evidence | Status at this checkpoint |
|---|---|
| Original balanced island-inclusive grid | 2,187 configurations; 15,309 comparisons; seven treatments |
| Width/length and fixed-unit embedding extension | Nine pilot configurations; 63 comparisons completed |
| Full extension | 140 configurations; 980 planned comparisons; not completed |
| Extension predictor diagnostic | 56 novel comparisons scored; seven original-grid anchors excluded; models fitted only to original development rows |
| Numerical audit | One treatment and seven geometries; short-network refinement issue remains |
| Real overlay | Earlier 79-comparison pilot; lag/omega overlap only, not validation of current event duration |

Current entry points:

- [Results synthesis](../notebooks/synthetic_results_synthesis.ipynb): sensitivity, interactions, omega scatter, numerical audit and extension results.
- [Predictor validation](../notebooks/synthetic_predictor_validation.ipynb): compact predictors, original-grid tests and separate extension transfer.
- [Hydraulic applicability](../notebooks/synthetic_hydraulic_applicability.ipynb): next active workstream.

## What the real-case comparison should decide

1. Do actual cases occupy comparable geometry and response ranges?
2. Do the same candidate predictors describe direction and magnitude for all three current responses?
3. Where they fail, can differences be traced to domain geometry, surrounding-network influence, reference hydraulics or numerical resolution?

Real results should determine the emphasis and limits of the final story, not retrospectively change the definitions of a successful test. Freeze the metric definitions and candidate predictor specifications before the additional real-case evaluation. Record later revisions separately. Earlier reused holdouts remain exploratory evidence.

Agreement would support transfer of the synthetic interpretation. Disagreement would delimit it and motivate targeted synthetic follow-ups. Neither a high deterministic R² nor numerical stability establishes a general physical law.

## Parked work and next active work

Sensitivity/prediction is **ready for real-case comparison**, not unconditionally finished. Retain the full extension, targeted high-residual geometries and refinement checks as outstanding tasks where final claims require them. Avoid expanding the grid or searching more predictors without a specific question.

Hydraulic applicability resumes now. First resolve how reference discharge, depth, velocity, slope and event timescale are made mutually consistent. Then distinguish the published wave-regime framework from the repository's provisional threshold, and assess the actual case-study networks rather than treating the gauge-screening cohort as those cases. Keep completed routing outputs intact and label any alternative hydraulic closure as a diagnostic until explicitly adopted in a new experiment.

## Same-day hydraulic restart: first diagnostic completed

The existing hydraulic notebook now compares the saved width-derived depth plus allocated discharge against two local alternatives: width-derived depth plus Manning discharge, and allocated discharge plus normal depth. This adds diagnostics only; routing was not changed or rerun.

For the baseline 500 m centerline, slope 1e-4 and Manning n=0.35, the prescribed 25 m depth and 100 m³/s baseflow imply velocity 0.008 m/s. Manning at that depth instead implies about 3,054 m³/s; retaining 100 m³/s gives normal depth about 3.21 m. All three reference closures remain outside the provisional screen for all original-grid reaches with the 12-hour rise proxy. The immediate next question is therefore both the physical reference state and the validity of the wave-zone/period comparison, not simply a change of discharge to make the plot look acceptable.

The hydraulic notebook executed all eight code cells successfully. Derived tables and the reference-state plot are in `outputs/full_grid_island/applicability/`. The published framework was rechecked against Getirana & Paiva (2013), equations 3–4 and Figure 1; the repository thresholds remain explicitly provisional.

## Same-day follow-up: actual BASED references checked

Reference **HYD-2026-09-29-BASED-MANNING**, added after the user requested the concrete comparison. Examples 03, 21, 55 and 56 were selected to match the earlier real pilot. Actual preparation manifests supply roughness and stored peak-reference conditions; the cached reference summaries match those manifests, and the current BASED model reproduces the stored depths.

Manning-implied discharge / reference discharge is 0.0946, 0.0520, 0.1287 and 0.0777 respectively, with configured n=0.35 in each case. Algebraically implied n is 0.0331, 0.0182, 0.0450 and 0.0272. This diagnoses a substantial discrepancy but does not establish a parameter typo or prescribe replacement roughness.

At the synthetic baseline peak Q=1000 m³/s, width=500 m and slope=1e-4, BASED gives depth=3.465 m and kb=144.30. Its discharge ratio is 0.1133, comparable to the four real references. The saved fixed-kb=20 synthetic reference instead has depth=25 m and ratio=3.0535 at that same peak discharge. Thus procedure alignment and roughness/reference consistency are separate issues. The next targeted task is to trace why n=0.35 is used and whether it is intended as effective routing roughness; no routing or parameter changes were made.

Three roots (03, 21, 56) contain width-refresh-required markers. Agreement of cached and prepared references does not certify current branch-geometry freshness; this is an audit of stored reference conditions, not final updated case validation. The notebook reports the markers explicitly.

Reproducible script: `investigations/based_manning_reference_audit.py`. Results and source hashes: `outputs/full_grid_island/applicability/based_reference_audit/`. Tables and figure are integrated in the existing hydraulic applicability notebook.

## Same-day routing follow-up: roughness sensitivity

Reference **HYD-2026-09-29-MANNING-SENSITIVITY**. User authorized testing n=0.025/0.050 and derived synthetic n; n=0.035 retained as the proposed common baseline. Separate legacy and BASED-depth-only controls distinguish interventions. Seven configurations cover baseline, width/slope extremes and long narrow/dominant branches; all seven collapse treatments are included.

266 comparisons completed (304 unique routes); 28 long-narrow comparisons could not run with a shared all-state stable clock at the retained 200 m segmentation/floor and X=0.1. These are recorded, not silently adjusted. Across 42 fully covered geometry/treatment combinations, response ranges over the four new roughness choices have medians 0.30 pp lag, 4.07 pp duration and 2.91 pp peak excess. The derived n varies from about 0.027 to 0.100: it is a consistency diagnostic, not automatically the preferred physical value.

35 legacy comparisons reproduce the original grid, and completed runs pass volume/plateau checks. No production defaults, original-grid or real-case routing changed. The existing hydraulic applicability notebook contains design, coverage, clipping/timestep summaries, response plots and baseline nested hydrographs. Numerical follow-up remains necessary for the long-narrow case and before interpreting small roughness contrasts as converged results.

## Segmentation follow-up: numerical feasibility and shape versus timing

Reference **HYD-2026-09-29-SEGMENTATION**. User requested an explicit test before adopting adaptive subdivision. Tested baseline, long-narrow and long-dominant configurations at n=0.035 and BASED kb, all seven treatments. Uniform target/floors 200/100/50 m were compared with adaptive maximum lengths at the same levels, the real stability-length helper retaining its 200 m floor, and timestep-only controls. Existing routing solver/forcing were reused through isolated preparation; no production code or policies changed.

147 comparisons completed, 128 unique routes. Uniform refinement and the existing helper do not resolve the long-narrow shared-clock conflict. Adaptive subdivision does, with minimum lengths about 71/36/18 m. Its response percentages vary by at most 0.37 pp lag, 0.22 pp duration and 0.20 pp peak excess across the tested adaptive resolutions.

The larger lesson is that preserved path-mean travel time does not imply preserved peak lag. Across baseline treatments, the lag response moves by up to 4.56 pp on adaptive 200→50 m refinement (nested/main-island), while duration and peak-excess changes are smaller. The long-dominant case instead shows larger duration/peak-excess sensitivity. Timestep-only baseline controls have up to 0.59 pp lag differences but approximately 0.013 pp or less shape-response differences.

At fixed X, subdivision reduces within-path Muskingum spreading; between-path arrival-time variance persists when geometry/allocations are preserved. A finer mesh is therefore not automatically a better physical attenuation model. Moment identities, including the actual external-inflow treatment, matched measured moments; conservation and stock-adapter parity passed. Keep 200 m as the main-analysis candidate and adaptive results as separately labeled numerical sensitivity until a common policy is justified. Full-grid and real-case reruns remain pending.

## Exact edge-controller audit

Reference **HYD-2026-09-29-EDGE-CONTROLLERS**. In the seven focused configurations at n=0.035 and BASED kb, only long-narrow A fails. Its 500 m approach/exit channels impose dt≤330.15 s. The lengthened 12.5 m upper-inner and crossover edges impose dt≥743.14 s; their unlengthened 12.5 m companion/continuation edges already impose dt≥429.05 s. The simple 25 m lengthened A imposes dt≥468.15 s. All segments are 200 m, and the controlling raw celerities lie inside the clipping limits.

Thus the controlling combination is narrow width → very shallow inferred depth under common kb, plus lengthening → reduced slope at fixed head loss. At Wref=500 m the unclipped equal-slope width threshold is 18.52 m; at one-third slope it is 42.21 m, conditional on equal segments and X=0.1. The low-flow branches carry only 2.5% of inlet flow but still constrain the global nonnegative timestep. This localizes the model-assumption question to extrapolating the reference width/depth ratio into narrow internal branches, rather than adjusting the scalar kb merely for feasibility. Full-grid preflight remains outstanding.


## Reference-anchored width–depth scaling test

Reference **HYD-2026-09-29-WIDTH-DEPTH**, requested after the depth-floor discussion. Retain BASED Dref and kbref at the original reference width, with D=Dref(W/Wref)^alpha and local kb=kbref(W/Wref)^(1-alpha). Test alpha=1, 0.8, 0.71, 0.5 at n=0.035, X=0.1, 200 m target/floor, unchanged widths, allocation and forcing. The 0.71 exponent has precedent in Li & Limaye (2025), doi:10.1029/2024JF008196 equation 9; their channel-migration application does not validate our BASED anchoring or routing interpretation. Alpha=0.5 is stronger moderation sensitivity, not an empirically calibrated selection.

Preflight: all 2,187 original factorial configurations pass under each exponent. For all 140 planned embedded-extension configurations, passing counts are 108, 132, 136, 140 respectively. This is a constraint audit, not a new full-grid routing run. In the nominal stored real-base audit, 03 passes throughout; 56 becomes feasible at 0.5; 21 and 55 still fail. The audit uses stored routing widths and slopes, nominal 200m segmentation/floor and n=0.035, not the full real preparation safeguards or parent/child runs. Width-refresh markers remain for 03/21/56.

Seven focused geometries yield 175 primary comparisons plus 175 common-clock controls, 344 unique routes. The three long-narrow variants alpha=1/.8/.71 remain infeasible; alpha=.5 completes. Three stock hydrographs match the diagnostic adapter, all 42 available old-method comparisons reproduce earlier BASED/n=.035 results, and conservation/plateau checks pass (max relative volume error 6.7e-14). Local kb is passed consistently into the matched Chezy resistance solver; no claim is made that Chezy resistance becomes Manning-equivalent.

At baseline simple/full, alpha=1→.5 changes lag shift -27.08→-21.52%, event-duration change -6.20→-0.67%, peak-excess change +5.27→+0.69%. Across baseline treatments maximum absolute changes are 13.33/5.53/4.57 pp. The largest native/common-clock differences across all routed cases are 0.62/0.028/0.032 pp. This is a substantive hydraulic-assumption sensitivity, not simply a harmless stability fix.

The critical 12.5m long-narrow branch deepens .087→.548m, but Manning-implied Q rises only .035→.756m³/s against the steady width-allocated reference Q=25m³/s. Numerical feasibility does not establish hydraulic consistency. Do not select alpha=.5 because it passes; the branch-depth/allocated-discharge interpretation remains unresolved. No production defaults or old grid/real outputs changed.

Reproduce: `investigations/width_depth_scaling.py`. Outputs: `outputs/width_depth_scaling/` including raw hydrographs, local-kb resistance, bounds, real preflight, reference-flow diagnostics and manifest. The existing hydraulic applicability notebook contains the equations, scope, coverage tables, depth/ratio curves, common-clock response panels, resistance table and hydrographs.


## Alpha 0.6 and failure-domain review

Reference **HYD-2026-09-29-ALPHA06-FAILURE-REVIEW**, requested to add the intermediate exponent, count failures accurately, assess possible extreme-case exclusions and bring real geometry into the decision. This extends the prior four-alpha diagnostic; alpha=0.6 is not adopted as a default.

All 2,187 original-grid configurations pass at all five exponents (7,236 unique canonical networks plus reference kb; 15,309 pair slots). In the 140-configuration embedded extension, failures by alpha 1/.8/.71/.6/.5 are respectively 32/8/4/0/0 configurations, 112/24/12/0/0 out of 603 unique networks, and 204/48/24/0/0 out of 980 pair clocks. The corresponding failed state-instance counts are 128/24/12/0/0 out of 1,540 references. Unique network counts exclude clock identity and repeated configuration/treatment references.

At .71 the sole failing geometric combination is outer A=.05 and multiplier=3, repeated at 20/100/400/800km network extents. Failing states are nested/original, crossover/original and nested/main_island_only. All six nested/crossover comparisons fail their pair clock; simple/full remains feasible. Removing those four complete configurations would remove 28 comparison slots (including four feasible simple/full slots) and leave 136/136 passing. At .8 the .05/m2 combination also fails. Under constant kb, all .05 combinations and .10 with multiplier>=1.5 fail the shared configuration clock. Exclusion tables are counterfactual only; nothing was deleted.

Real base checks at .6: 03 and 56 pass; 21 and 55 fail. The remaining slow controllers are example21 link47 (width228m, slope~3.1e-6) and example55 link495 (width175m, slope~6.5e-6). These slopes are ~1.2%/3.2% of their BASED reference slopes and come from raw_wse. Their implied longitudinal head drops are millimetres to about a centimetre; this motivates provenance/uncertainty review, not automatic replacement. Real width minima are ~10.3%,10.8%,3.8%,5.3% of reference width for 03/21/55/56, while slope minima range ~1.2–6.2% of reference. Thus narrow geometry remains relevant, but real slope heterogeneity exceeds the embedded synthetic parents' one-third-reference minimum. Width-refresh markers remain for 03/21/56; these are stored base-only audits, not real response validation or measured depth evidence.

Expanded focused routing completes 224 primary comparisons +224 common-clock controls, 456 unique routes. All volume/plateau and old-method parity checks pass. Baseline simple/full at .6: lag -22.91%, duration -1.10%, peak excess +1.17%. The critical long-narrow depth becomes .379m, but Manning-implied reference Q is .409m³/s versus allocated25m³/s: physical depth/allocation consistency remains unresolved.

The existing hydraulic applicability notebook now includes five-alpha response plots, count definitions, failure maps across width/length, potential exclusions, and real normalized width/slope plots with exact controller IDs. Reproduce `investigations/width_depth_scaling.py` then `investigations/width_depth_failure_review.py`; raw preflight tables, real input hashes and failure-review manifest are in `outputs/width_depth_scaling/`. No production defaults, real routing products or experiment membership changed. Working candidates are .71 with an explicitly delimited stress-test boundary, or .6 with wider synthetic coverage; select using physical evidence, not feasibility alone.


## Fine alpha threshold check · 30 September 2026

Reference **HYD-2026-09-30-ALPHA-THRESHOLD**. User requested the first failure between 0.6 and 0.71. A 0.001-spaced scan plus targeted values covers all 2,187 original-grid and 140 extension configurations; bisection resolves the four limiting configurations. No hydrographs were rerun or exponent adopted.

With the existing integer-timestep scheduler, the first boundary is approximately **alpha=0.669901503**. Alpha=0.6699 passes; 0.66991 fails. On a 0.01 grid, 0.66 passes and 0.67 is the first failure. The limiting combination remains A=.05, multiplier=3, at all four extents. The original grid passes throughout the scanned range.

| Alpha | Limiting continuous dt interval (s) | Integer choices | Outcome |
|---|---|---:|---|
| .60 | 277.879–330.152 | 53 | pass |
| .65 | 314.238–330.152 | 16 | pass |
| .66 | 322.061–330.152 | 8 | pass |
| 2/3 | 327.385–330.152 | 3 | pass |
| .6699 | 329.999–330.152 | 1 | pass |
| .67 | 330.080–330.152 | 0 | fail |
| .71 | 364.201–330.152 | 0 | fail |

At .67 the continuous interval is nonempty but contains no integer second, explaining failure under the actual scheduler. Choosing the last passing exponent would tune to this numerical boundary. Alpha=.65 or .66 retains more margin; 2/3 is feasible here but with only 328/329/330 s available. None of these checks supplies independent physical justification for an exponent. Conditions remain n=.035, X=.1, 200m target/floor, fixed BASED references and existing celerity caps.

Reproduce `investigations/width_depth_alpha_threshold.py`; scan, summary and bisection brackets are in `outputs/width_depth_scaling/alpha_threshold/`.


## Selected synthetic exponent · 30 September 2026

Reference **HYD-2026-09-30-SELECTED-ALPHA065**. The user selected alpha=0.65 following the width–depth sensitivity and refined feasibility scan. This is the selected synthetic depth-scaling assumption, anchored to the existing BASED reference section and retaining the tested n=.035, X=.1 and 200m target/floor context. It is close to the literature-motivated candidate while retaining numerical margin below the tested integer-clock boundary (~.6699015). It is not an independently calibrated or observed optimum.

The user explicitly deferred the real-case decision until **new runs using updated slopes**. Earlier stored-real-base results are historical diagnostics and must not be used as the decision on the refreshed real cases. No real defaults, slopes or hydrographs are changed by this selection.

The machine-readable selection is `configs/selected_hydraulic_reference.json`, consumed by the focused width/depth diagnostic to include alpha=.65. The current hydraulic notebook identifies the selected value and focused results. Existing factorial/extension hydrographs retain their original settings; full-grid rerouting is a separate outstanding step. Branch-depth/allocated-discharge consistency remains an acknowledged limitation, not solved merely by this selection.


Selected-setting verification completed: all seven focused configurations route at alpha=.65, with 49 primary and 49 common-clock comparisons. The complete six-alpha diagnostic contains 273 primary +273 clock-control comparisons and 568 unique routes; old-method parity and conservation/plateau checks pass. Baseline simple/full at .65 has lag shift -23.58%, duration change -1.40%, peak-excess change +1.50%. All 2,187 original-grid and 140 extension preflights pass at .65. These are selected focused reruns plus full-domain preflight, not a completed main-grid rerun.


## Follow-up · 30 September 2026

**SYN-2026-09-30-SELECTED-RERUN:** the selected alpha=.65 / BASED / n=.035 setup now has a full versioned grid and extension rerun, refreshed current notebooks, and matched previous/selected checks. See [the follow-up record](2026-09-30_selected_hydraulic_rerun.md). The earlier numbers above remain dated historical findings. Real-case selection still waits for new updated-slope runs.
