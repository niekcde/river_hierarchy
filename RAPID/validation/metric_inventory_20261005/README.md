# Current metrics and response magnitudes — 5 October 2026

Reference **METRIC-INVENTORY-2026-10-05**. Written to clarify benchmark population exclusions, hydraulic-period uncertainty, updated lag values and whether small shape responses warrant complex prediction models. No routing changed.

## Population

The plotted population has 1,652 deduplicated comparisons across 16 rivers. All have resistance, exposure and length fraction. The common path-geometry population has 1,128 comparisons across the same rivers. It excludes 215 comparisons from example 15 and 309 from example 22 because path enumeration exceeds 10,000. This computational cutoff selects against path-rich domains and changes the population; it is not evidence that those comparisons are invalid. The original feature extraction also rejected many example-37 domains, but these were already outside the current edge-filtered population. Its one plotted comparison remains in both cohorts.

| example_id | plotted | path_geometry | excluded |
| --- | --- | --- | --- |
| 02 | 29 | 29 | 0 |
| 03 | 34 | 34 | 0 |
| 05 | 45 | 45 | 0 |
| 11 | 3 | 3 | 0 |
| 13 | 47 | 47 | 0 |
| 15 | 252 | 37 | 215 |
| 21 | 24 | 24 | 0 |
| 22 | 396 | 87 | 309 |
| 37 | 1 | 1 | 0 |
| 41 | 181 | 181 | 0 |
| 45 | 84 | 84 | 0 |
| 46 | 208 | 208 | 0 |
| 48 | 58 | 58 | 0 |
| 54 | 75 | 75 | 0 |
| 55 | 200 | 200 | 0 |
| 56 | 15 | 15 | 0 |

## Hydraulic period uncertainty

The hydraulic screen uses T⁺=T_0 U S/h. Here T_0 should describe the flood-wave perturbation, not travel time from inlet to outlet. A small wiggle near the peak can make the last local minimum occur minutes before it; a first 1% departure may span several days or multiple rises. Example 05 gives approximately 0.64 h versus 278 h; changing only that choice moves T⁺ by roughly 434-fold. Neither extreme is automatically the correct physical period. This uncertainty does not alter the current parent–child lag denominator or event-duration metric. Select/verify a consistent relevant event rise or duration for the hydraulic diagnostic before interpreting its screen fractions.

## Current gauge/model lags (hours)

Observed values below use first sampled global maxima within each selected gauge window. Model values are recomputed from the saved production hydrograph with guarded interpolation. Positive difference means longer model travel lag. These are not hand-matched shoulders/plateau alternatives. In particular, examples 03, 05 and 48 retain the previously documented peak-matching ambiguity. The current model lag for example 13 is 108.60 h; the earlier review summary cached 111.90 h. This table replaces that cached model-lag column, and does not imply a rerun.

| example | observed_first_peak_lag_h | model_interpolated_lag_h | lag_difference_h | model_downstream_peak_error_h |
| --- | --- | --- | --- | --- |
| 02 | 24.0 | 15.08 | -8.92 | -9.06 |
| 03 | 9.5 | 14.35 | 4.85 | 4.81 |
| 05 | 74.75 | 55.65 | -19.1 | -17.78 |
| 11 | 63.5 | 28.84 | -34.66 | -34.74 |
| 13 | 64.25 | 108.6 | 44.35 | 47.55 |
| 15 | 13.01 | 37.09 | 24.08 | 23.84 |
| 21 | 20.75 | 31.03 | 10.28 | 10.16 |
| 22 | 30.5 | 80.36 | 49.86 | 49.78 |
| 37 | 12.75 | 20.43 | 7.68 | 7.56 |
| 41 | 79.25 | 126.65 | 47.4 | 47.34 |
| 45 | 46.42 | 72.26 | 25.85 | 27.36 |
| 46 | 140.33 | 339.45 | 199.11 | 204.25 |
| 48 | 7.25 | 11.98 | 4.73 | 4.73 |
| 54 | 19.33 | 25.12 | 5.78 | 5.0 |
| 55 | 74.75 | 97.74 | 22.99 | 22.65 |
| 56 | 16.17 | 28.72 | 12.55 | 10.75 |

## Shape and timing magnitude

Absolute magnitudes across all 1,652 plotted comparisons (percent columns already use percent units, not fractions):

| metric | median_absolute | p95_absolute | max_absolute |
| --- | --- | --- | --- |
| interpolated_lag_shift_pct | 0.109351 | 1.156379 | 22.314138 |
| timing_change_minutes | 4.634848 | 52.976756 | 514.187813 |
| event_duration_change_pct | 0.000911 | 0.033721 | 0.867607 |
| peak_excess_change_pct | 0.002908 | 0.068798 | 1.568208 |
| duration_change_minutes | 0.253733 | 6.194249 | 118.02565 |
| peak_excess_change_cms | 0.021538 | 0.765679 | 12.965956 |

98.1% of duration changes and 96.1% of peak-excess changes are below 0.1% in magnitude. All duration changes and 99.88% of excess changes are below 1%. Median absolute duration change is about 15.2 seconds; median absolute excess change is 0.0215 m³/s. Individual duration changes can approach two hours despite a small percentage, because some events are long.

Thus low shape-response R² is not enough to diagnose an important prediction failure: most targets are very close to zero. R² can be poor when absolute errors are tiny relative to discharge or event duration. Report MAE in original units and compare a no-change baseline, then audit the larger-response tail and resolution. Small targets do not automatically explain all errors or establish numerical noise. Neither the production output spacing nor the interpolation alone proves these tiny differences are physically resolved.

The per-network and per-comparison exports retain signed values, distribution tails and counts. Read `response_magnitudes.csv` and `comparison_responses.csv` for the collapse effects; `network_lags.csv` is the distinct gauge/base-model comparison.

## Metric equations used in the current benchmark

Notation: edge e has physical length l_e and width w_e; p denotes the matched parent domain, c its child, and N the full parent network. Path j joins the domain's source to its sink. Shared edges can occur in several paths. Natural logarithms are used for geometry-change ratios; resistance log uses base 10.

### Length, exposure and resistance

- Domain total channel length: L_sum,p = Σ_(e in p) l_e. Full network channel length L_sum,N = Σ_(e in N) l_e. These count each edge once, including parallel branches; neither is a single centerline length.
- Length fraction: f_L = L_sum,p / L_sum,N. Child/parent total-length ratio: r_L = L_sum,c / L_sum,p (available diagnostic, not the main f_L).
- Width-allocation edge proportion: a_e = w_e / Σ_(k outgoing from u) w_k. Propagate inlet share q_inlet=1 and q_v = Σ_(e entering v) q_u a_e. Raw domain exposure E_raw = q_source × Σ_(e leaving source inside p) w_e / Σ_(all e leaving source) w_e.
- Current exposure E: **1 for closed_compound_unit and containing_compound_context**, otherwise E_raw. This is the current domain-aware convention; E must not always be described as the actual fraction of inlet discharge entering a subdomain.
- Anchored depth: h_e = h_ref (w_e/W_ref)^0.65; local k_b,e = w_e/h_e. BASED supplies the retained reference width/depth. Cross-sectional area A_e=w_e h_e and wide-channel hydraulic radius R_h,e=h_e.
- Current Chezy edge resistance: K_e = l_e / (A_e² C_z² g R_h,e), with the repository's dimensionless C_z convention (default 22, g=9.81). Head-flow law ΔH_e=K_e Q_e |Q_e|. This K_e is not RAPID's travel-time K.
- Equivalent resistance: K_eq=ΔH_domain/Q_domain², computed by the nonlinear network solve. Series: Σ K_e; parallel: (Σ K_e^(-1/2))^(-2). General compounds require the solve.
- Resistance ratio λ=K_eq,c/K_eq,p; resistance log R=-log10(λ); main combined axis **ω=f_L E R**. Separate components in the benchmark are R, f_L and E.

### Path geometry (parent and child separately)

For P enumerated paths:

L_j = Σ_(e in j) l_e,
W_j = L_j / Σ_(e in j)(l_e/w_e),
a_j = W_j / Σ_j W_j.

W_j is a length-weighted harmonic width along path j. The path-width shares a_j are not the recursively allocated discharge shares q_v above.

- Arithmetic mean path length: L_bar = (1/P) Σ_j L_j.
- Equivalent length: L_eq = Σ_j a_j L_j = Σ_j W_j L_j / Σ_j W_j.
- Equivalent width: W_eq = Σ_j W_j. This is a path-based aggregate; shared edges are counted on each enumerated path.
- Elongation: e = L_eq/W_eq = (Σ_j W_j L_j)/(Σ_j W_j)².
- Width dominance: D = max_j(W_j)/Σ_j W_j = max_j a_j.
- Path-length coefficient of variation: CV_L = sqrt[(1/P) Σ_j(L_j-L_bar)²]/L_bar. It is unweighted and uses population standard deviation.
- Signed length–width association: A_LW = L_eq/L_bar - 1. The older uncentered L_wa equals A_LW+1. Equal path lengths give A_LW=0 and L_wa=1. Positive association means width weighting favors longer paths; zero does not prove all lengths are equal.
- Domain reference slope: S_ref=(H_source-H_sink)/min_j L_j. This is head drop divided by shortest physical domain path, not mean edge slope.
- Surrounding lengths: L_up = shortest physical path from network inlet to parent-domain source; L_down = shortest physical path from parent-domain sink to network outlet.

### Parent–child changes and combinations

Δlog L_eq=ln(L_eq,c/L_eq,p); Δlog e=ln(e_c/e_p); ΔCV_L=CV_L,c−CV_L,p; ΔD=D_c−D_p.

The selected six are **R, E, L_eq,p, D_p, A_LW,p, S_ref,p**. Size/proportions means **L_eq,p and e_p**. Context means **L_up and L_down**. Change/context adds the four changes above and both surrounding lengths. Product candidates from the earlier search were ωD, ωA_LW and ωS_ref; they are not the current selected-six model.

Resistance and anchored depth are hydraulic-geometry descriptors, not unqualified geometry-only measurements. The current path benchmark does not include entropic braiding index as a fitted feature. Width dominance is not that index. The repository contains other exploratory descriptors; this inventory covers the current benchmark and its retained product candidates, not every historical notebook column.

## Current response equations

Let t_i be the sampled applied-input peak; t_p,t_c the guarded three-point interpolated parent and child outlet peaks. Parent transit lag τ_p=t_p−t_i. Common forcing implies child-minus-parent lag equals t_c−t_p.

- **Percentage lag change:** y_lag=100(t_c−t_p)/(t_p−t_i). This is not normalized by an absolute timestamp or by rise duration.
- **Absolute timing change:** y_time=(t_c−t_p)/3600 in hours (×60 for minutes). Negative means an earlier child peak.
- **Event duration:** initial/final plateau levels B_0,s and B_1,s are taken from each hydrograph. The parent defines shared tolerances ε_0=.01(Q_max,p−B_0,p), ε_1=.01(Q_max,p−B_1,p). For both states, t_start is the first departure beyond ε_0 from the initial plateau, and t_end is the final return within ε_1 of the final plateau, with at least one hour of final settlement. Crossings are linearly interpolated. D_s=t_end,s−t_start,s and y_duration=100(D_c/D_p−1). Later secondary peaks postpone the final return. **This is whole-event duration, not peak width/FWHM or a 1–99% cumulative-volume interval.**
- **Peak excess:** shared parent background B=max(B_0,p,B_1,p); P_p=Q_max,p−B, P_c=Q_max,c−B; y_peak=100(P_c/P_p−1). Negative means a lower child peak above that shared background. It is neither total-flow percentage change nor an attenuation-only metric.

For gauge validation only, τ_obs=t_down,obs−t_up,obs and δτ=τ_p−τ_obs. A direct downstream peak-date error is t_p,UTC−t_down,obs; it can differ from δτ because the applied forcing's peak need not coincide exactly with the sampled upstream-gauge maximum. Both are exported. Observed plateau/secondary-peak alternatives remain sensitivity cases, not silently substituted primary peaks.
