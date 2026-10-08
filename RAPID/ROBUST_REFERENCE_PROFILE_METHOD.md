# Robust reference-profile method

**Decision reference: ROBUST-SLOPES-2026-10-02.** Written after the all-16 review found that ordinary squared-error fitting could flatten sections despite consistently positive cycle slopes. The user requested a bounded implementation, an explicit decision rather than continuing case-by-case experimentation, and allowed small, defensible manual adjustments.

## Decision and scope
Use a common **Huber robust profile fit**, a **0.05 S_ref reference-profile floor**, a documented **SWORD head-difference constraint across disconnected observation gaps**, and the **existing shared-head unit allocation**. Select smoothing using held-out whole cycles. Retain the already documented linear bridge in case 46.

This implementation generates an isolated, reproducible base-network candidate for all 16 examples. It does not overwrite production hydraulic contexts, recompute collapsed states/equivalent resistance, or reroute hydrographs. The numerical and hydraulic audit accompanies the candidate; passing profile checks is not equivalent to passing RAPID stability or validating river hydraulics.

## 1. Inputs and the quantity being fitted
Start with the existing screened RiverSP observations, their cycle/acquisition identifiers, the existing reference-distance grid, and the static SWORD reference profile. Distance increases upstream from the outlet, so positive fitted slope means increasing water-surface elevation upstream.

For acquisition a and location i:

\[
 z_{ai}=H(x_i)+b_a+\epsilon_{ai}.
\]

H is the shared reference water-surface profile; b_a is a constant acquisition stage offset. These offsets accommodate passes at different water levels. They do not permit each pass to have its own slope. Thus H is a common reference shape, not the assertion that every observed flow condition has the same gradient. The mean acquisition offset is fixed to zero to define the vertical level convention.

No new observation screening thresholds were introduced. Existing observations can still include problematic jumps; robust fitting reduces their influence rather than deleting them.

## 2. What “robust fit” means
An ordinary squared-error fit gives residual r a cost r²: a ten-times-larger residual contributes one hundred times the cost. A small number of extreme observations can therefore pull the profile away from the pattern supported by most observations.

Here the observation cost is Huber loss:

\[
 \rho_\delta(r)=
 \begin{cases}
 r^2,& |r|\le\delta,\\
 2\delta|r|-\delta^2,& |r|>\delta.
 \end{cases}
\]

Small residuals retain their normal squared-error treatment. Beyond delta, cost grows linearly, so increasingly extreme observations no longer acquire increasingly large leverage. No observation is automatically classified as false. In particular, a large residual can represent real changing water-surface shape as well as measurement error.

The objective is

\[
 \min_{H,b}\;\sum_{a,i}w_{ai}\rho_\delta(z_{ai}-H_i-b_a)
       +\lambda H^T R H,
\]

subject to the slope, gauge and bridge constraints below. R is the same normalized nonuniform-grid curvature penalty used in the existing implementation. Lambda controls smoothness: larger values discourage changes in gradient more strongly.

## 3. Cycle weights and the outlier threshold
Each cycle has equal total **base** weight. If cycle c contains n_c observations and there are C cycles, each observation receives N/(C n_c), where N is the fixed total-weight scale. N is held at the full observation count during cross-validation, so changing training-set size does not inadvertently change the relative smoothing strength.

Robust influence subsequently reduces effective weight for large residuals. Equal base cycle weight therefore does not mean every inconsistent cycle keeps equal effective influence after robust fitting.

The threshold is determined by one common rule:

\[
 \delta=1.345\times1.4826\times
 \operatorname{weighted\ median}\left|r-
          \operatorname{weighted\ median}(r)\right|.
\]

Residuals come from a squared-loss anchor fit at the previously selected lambda, using the same new gap/bridge constraints. During cross-validation, that anchor uses training cycles only. Delta is then frozen for every candidate lambda in that fold. The final fit estimates delta using all available observations. This avoids choosing a different threshold by hand for each river, or changing the loss scale whenever lambda changes. A tiny numerical safeguard prevents zero scale.

## 4. Smoothing selection: a bounded search
- Use three folds of entire cycles, with fixed seed 20261002. Acquisitions belonging to a cycle remain together.
- Fit each candidate to training cycles, then estimate only a constant offset for each held-out acquisition. Score the remaining held-out shape differences.
- Choose the lambda with the lowest mean held-cycle Huber loss. Also retain ordinary held-cycle MSE/RMSE for transparency. A numerical tie within 1e-6 relative score selects the smoother candidate.
- Search five values at 0.01, 0.1, 1, 10 and 100 times the previous squared-loss lambda. Permit at most three boundary extensions by a factor ten, followed by two half-decade refinements around the best value. Record boundary selections and failed solves explicitly.

This is cross-validation for **method selection using the existing dataset**, not independent physical validation. The search is centred on the previous data-derived selection. Held-acquisition offset removal evaluates profile shape, not independent absolute elevation accuracy. The robust scale is training-only; no held-out water-surface values enter its calculation.

## 5. The reference-profile slope floor
For consecutive profile locations:

\[
 H_{i+1}-H_i\ge0.05S_{ref}(x_{i+1}-x_i).
\]

S_ref is the existing static-reference global slope recorded in the source audit. It is fixed during fitting. The floor prevents the shared profile from becoming arbitrarily flat, while scaling with each river's reference gradient.

**The factor 0.05 is an explicit regularization decision, not a measured physical minimum.** It was retained from the earlier sensitivity comparison. It does not establish that every real river segment must exceed this slope, nor that the routing model is appropriate there. This is a water-surface reference gradient, not a direct river-bed slope measurement.

## 6. Missing observations versus disconnected observations
A gap with no observations can still have an acquisition spanning both sides. That pass constrains their relative height, so it does not automatically require a static-reference anchor.

The fallback applies only where the union of acquisition spatial spans splits into disconnected groups. In the current all-16 inputs, only case 21 meets this condition: approximately **89.21–103.38 km**.

At that gap's endpoints a,b, impose

\[
 H(b)-H(a)=H_{SWORD}(b)-H_{SWORD}(a),
\]

and use a linear bridge inside the gap. For case 21 the reference difference is **2.342809 m**. It is compatible with the selected minimum slope. The constraint is imposed inside every training and final fit, so the surrounding profile and acquisition offsets are fitted consistently with it.

Interpolation alone cannot choose this difference: without a connecting pass, the profile on either side can shift while acquisition offsets compensate. A smooth connection can be drawn for many different endpoint differences. SWORD supplies the otherwise unsupported cross-gap head difference; it does not replace SWOT shapes in the observed sections. The SWORD value is an external reference assumption, not a new SWOT measurement.

If a static gap head difference is negative, missing, or incompatible with the floor, the implementation stops with an explicit diagnostic. It does not silently increase the head loss. That would be a legitimate point for a documented manual reference choice.

## 7. Defensible manual adjustments
Retain the previously documented **case-46 linear bridge at roughly 370–410 km**, resolving endpoints to the saved profile grid. In this implementation it is a constraint inside the optimizer, rather than a post-fit alteration. Its endpoints remain fitted; unlike case 21, it does not receive a new static head-difference anchor.

Further manual adjustments are allowed when linked to identifiable evidence: a documented acquisition discontinuity, a reference-data error, or an observation gap requiring external information. Record the reason, spatial extent, evidence source, chosen constraint and effect on total head loss. Prefer a shared rule or one clearly bounded bridge over hand-selecting individual link slopes.

Do not select an adjustment merely because it makes RAPID stable or produces an expected hydrograph. There are no new per-case observation deletions or hand-picked slope targets in this run.

## 8. From reference profile to network slopes
Interpolate the profile to the existing node reference positions, then retain the existing owned-unit allocation. The unit inlet and outlet heads are fixed by the profile; internal node heads are solved jointly. Each edge slope is

\[
 S_e=(H_u-H_v)/L_e,
\]

using its **actual physical length**. The unit objective remains equal-per-link relative slope error around its domain reference slope, with the existing 0.01-times-domain-reference lower bound. Shared node heads preserve head-loss consistency along alternative paths.

The profile's 0.05 floor is therefore **not a universal final-link floor**. A physically longer branch can have a smaller slope while spanning the same boundary head difference. Neither the width law nor Manning n was changed by this slope implementation.

## 9. Numerical and hydraulic checks
The solver minimizes the convex Huber-plus-curvature objective directly using ADMM with the analytical Huber proximal step. Both constraint residual and stationarity residual must converge. Check positive profile gradients, positive final edge slopes, exact shared-head consistency, unit boundary preservation and the recorded bridge head differences. Focused tests cover gap anchoring, acquisition-level shifts, pass-spanned gaps, incompatible static anchors, robust outlier influence and held-out offset removal.

The base-network hydraulic preflight uses:

- D_e = D_ref (W_e/W_ref)^alpha, with alpha=0.65 as the primary diagnostic and 0.6, 0.71 and 1 as sensitivities.
- Manning celerity c = (5/(3n)) D^(2/3) sqrt(S), with n=0.035 primary, 0.025/0.05 and implied n as sensitivities.
- Stored BASED reference depth as the primary anchor; re-evaluating BASED at the candidate whole-reference slope is a separate diagnostic, not an implicit reference-depth replacement.
- 200 m target segmentation using production rounding and the 200 m effective-length floor; X=0.1.
- A feasible integer timestep in the intersection of all [0.2K,1.8K] intervals, where K=L_effective/c. Audit uncapped, saved/default-capped, and common 0.05–2.524 m/s capped celerity separately. Saved case-specific caps differ, so the common-cap scenario provides a consistent comparison.

This tests a necessary routing coefficient-positivity condition. It does not test all forcing-clock restrictions, adaptive preparation safeguards, all collapsed networks, or physical applicability. The wide-channel Manning-implied n remains a consistency diagnostic; it is not selected simply to obtain stability.

## Reproduction and result locations
- Implementation: `RAPID/src/rapid_tools/robust_reference_profile.py`.
- Fit/select/allocate: `RAPID/investigations/run_bounded_robust_slopes.py`.
- Hydraulic audit: `RAPID/investigations/audit_bounded_robust_hydraulics.py`.
- Outputs and dated results: `RAPID/assets/bounded_robust_slopes_20261002/`.
- Notebook: final bounded-method section of `RAPID/notebooks/slope_smoothing_floor_comparison.ipynb`.

The candidate folder records per-case acquisition offsets, robust weights, cycle folds, training-only scales, all searched CV scores, gap and manual constraints, selected profiles, unit allocations and solver diagnostics. These artifacts define the reproducible candidate; the legacy production workflow remains unchanged until promotion and downstream recomputation are deliberately performed.
