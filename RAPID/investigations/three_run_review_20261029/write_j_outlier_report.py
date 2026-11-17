"""Assemble the review document from the saved, reproducible diagnostics."""
import pandas as pd
from investigate_j_outlier_groups import DEST


def main():
    d=pd.read_csv(DEST/'outlier_diagnostics.csv',dtype={'example':str})
    a=pd.read_csv(DEST/'input_audit.csv',dtype={'example':str})
    t=pd.read_csv(DEST/'dominant_child_reaches.csv',dtype={'example':str})
    controls=pd.read_csv(DEST/'matched_controls.csv',dtype={'outlier_example':str,'control_example':str})
    text=r'''# Fixed-Manning ΔJ outliers: three-group investigation

29 October 2026. Completed real-network simulations only. No new routing, calibration, production predictor changes or outlier exclusions.

## Main finding

The ten opposite-sign cases share a strong model-based mechanism: collapse increases within-reach dispersion very substantially, often through extremely small prescribed head drops. ΔJ describes the mean of the complete fixed routing transfer; it does not describe this change in transfer shape. The saved event centroids can consequently move in the opposite direction. The association is strong and physically interpretable within the implemented equations, but is not a causal intervention or validation of those very low slopes in real rivers.

The eleven aggregate outliers are a different problem. They retain the predicted direction and have reduced transfer spread. Their response-to-ΔJ ratios differ appreciably, and their large shifts amplify departures from one pooled regression line. The four same-sign members are heterogeneous: three resemble weakened versions of the dispersion pattern; one resembles the aggregate group.

## Population and selection

Fixed Manning midpoint run, primary 0.5% common event window. The original pooled OLS fit has slope 0.728806 and intercept −0.162095 minutes. Residual sample SD is 1.41665 minutes; |z| > 3 therefore selects residual magnitudes above approximately 4.25 minutes. This is a descriptive screen, not a statistical significance test or held-out prediction assessment. Network comparisons are dependent and the residual distribution is not assumed normal.

There are 25 flagged comparisons among 2,185 (1.14%), from 13 networks. The artificial single-centerline control remains excluded. The nine all-closed-unit collapses remain included. No simple unit is flagged among 258 simple-unit comparisons.

| Group | Cases | Transfer SD child/parent: median (range) | Interpretation |
|---|---:|---:|---|
| G1: nine all-closed-unit + two full-compound, same sign | 11 | 0.945 (0.481–0.996) | Large advances; reduced spread; different response scaling |
| G2: compound member, same sign | 4 | 1.138 (0.998–1.551) | Three dispersion increases; one aggregate-like advance |
| G3: eight compound member + two island, opposite sign | 10 | 2.673 (1.241–5.473) | Positive ΔJ, negative finite-event centroid; strong dispersion increase |

“Transfer SD” is the analytic standard deviation of the model's transfer, **not the standard deviation measured from the event hydrograph**. A small, very long tail can dominate it.

## What was calculated, and why

The existing ΔJ was not refitted or altered. I calculated the second central moment of each saved fixed-coefficient routing network, alongside its first moment. This is an independent hydraulic diagnostic, computable from saved reference hydraulics and connectivity without response timing as input.

For fixed Muskingum storage, let W be the model storage perturbation, I and O the discharge perturbations, k [s] the cell routing parameter, and X dimensionless:

$$\dot W=I-O,\qquad W=k[X I+(1-X)O].$$

The established storage and continuity equations, their constant-parameter restriction and centered time discretization are given in [Todini (2007), equations 1–8](https://hess.copernicus.org/articles/11/1645/2007/). The following transfer-moment calculations are derived here from those equations and the repository implementation; they are not an empirical resistance law.

Taking the Laplace transform with zero initial perturbation:

$$H(s)=\frac{O(s)}{I(s)}=\frac{1-kXs}{1+k(1-X)s}.$$

Expansion about s=0 gives the cell mean k and second central moment k²(1−2X). The trapezoidal discrete implementation substitutes a frequency variable equal to s+O(s³); therefore the first two low-frequency moments are preserved. This is about the underlying linear transfer. Output binning and finite-event extraction are separate operations.

For N identical cells in reach e:

$$\tau_e=Nk_e=L_e/c_e,\qquad v_e=Nk_e^2(1-2X_e).$$

The saved Cunge implementation has k=Δx/c and X=½[1−βQ/(BS c Δx)], so

$$v_e=\frac{\beta Q_eL_e}{B_eS_ec_e^3}=\frac{2D_eL_e}{c_e^3},\qquad D_e=\frac{\beta Q_e}{2B_eS_e}.$$

D has units m²/s and v has units s². β is the saved diffusion multiplier; it is held fixed in each parent–child comparison. The reach variance is independent of the number of equal cells in this identity. Refinement can nevertheless affect higher moments and event shape.

For each inlet-to-outlet route p, use the product of the actual local split fractions as weight w_p. Rejoining flows are summed. Then

$$J=\sum_p w_p\sum_{e\in p}\tau_e=\sum_e f_e\tau_e,$$

$$\sigma_T^2=\underbrace{\sum_p w_p(\sum_{e\in p}\tau_e-J)^2}_{\text{between-route mean-time spread}}+\underbrace{\sum_e f_ev_e}_{\text{within-route dispersion}}.$$

The implementation evaluates these moments by dynamic programming over the directed acyclic graph, preserving nested connectivity and allocation without enumerating all paths. The route weights are nonnegative; individual numerical impulse coefficients can be signed, so a literal positive-probability interpretation of every cell is not assumed. These are model transfer moments, not a direct measurement of physical channel storage distribution.

For a complete integrable input pulse and stable fixed linear routing, the complete outlet centroid difference equals ΔJ. An observed sign reversal therefore cannot be explained by variance *alone* under those complete-pulse conditions. These saved comparisons instead use a finite common window and baseline, and the forcing can finish at a different level from its initial value. Transfer-shape changes interact with that particular event and measurement. This distinguishes the hydraulic mechanism from a mere artificial translation of a parent curve.

## G3: a consistent dispersion mechanism, with input-representation qualifications

All ten cases increase transfer SD by 24% to 447%. In every case, at least 99.88% of the increase in transfer variance is the within-route term. Separation between fast and slow branch mean travel times is a very small contributor to the added variance. Thus the first mechanism to inspect is local reach hydraulics, not simply unequal path lengths.

Seven of the eight compound-member cases contain reaches at the saved staged slope floor. Those reaches account for approximately 35–96% of the child's within-route dispersion in these seven cases. The eighth, 03/S027_unit_24, is not exactly at its floor but its dominant reach still has slope 2.85×10⁻⁶ and only 0.86 mm of prescribed head drop over 301 m. The two islands were not in the staged correction index; their retained inputs also contain extremely small slopes. They should not be described as having received the same staged correction as the members.

At fixed reference Q, B, L, n and β, Manning normal depth gives h ∝ S⁻³/¹⁰ and c ∝ S³/¹⁰. Consequently τ ∝ S⁻³/¹⁰ while v ∝ S⁻¹⁹/¹⁰. These exponents follow directly from the equations; none is fitted. A small prescribed slope can increase the dispersion much more strongly than it increases the mean delay.

For X<0 the continuous cell transfer can also be written

$$H(s)=\frac{-X}{1-X}+\frac{1}{1-X}\frac{1}{1+k(1-X)s}.$$

It contains a prompt component plus a smaller relaxing component with timescale k(1−X). This is a property of the lumped model, not a claim of instantaneous propagation in a real river. A very small slow component can increase the complete mean and especially the variance while the event's principal response behaves differently. The discrete solver and a cascade require their own interpretation; the single-cell fractions must not be applied to the whole network.

### Worked saved case: 37/island_37_282

ΔJ = +22.38 minutes, measured centroid = −9.59 minutes, pooled-line residual = −25.73 minutes. Transfer SD rises from 2.72 to 14.87 hours. The added transfer variance is 213.71 h²: 213.60 h² local dispersion and only 0.11 h² between-route spread.

Child reach 933 has L=2,010 m, B=354 m, f=0.523, S=8.05×10⁻⁶ and a prescribed head drop of only 16.2 mm. Its reference depth is 9.28 m. This one reach contributes 84.2% of the child's within-route dispersion; reach 938 contributes another 13.0%. Reach 933 has one cell, X=−171.59 and relaxation timescale 175 hours. In the continuous cell representation, roughly 99.42% is prompt and 0.58% is in the slow component. The first moment alone conceals this shape.

Using the actual saved child-minus-parent hydrograph, the portions before and after the parent centroid contribute −6.04 and −3.55 minutes respectively to the −9.59-minute centroid shift. This is exact moment accounting of the actual curves, not a shifted-parent experiment. All ten G3 cases have negative contributions on both sides of the parent centroid.

This island case is also not a clean local-only hydraulic edit: among 750 uniquely matched directed reaches with stable length and width, 650 have slope changes greater than 1%, and 673 have allocation changes greater than 1%. The counts exclude ambiguous parallel endpoint pairs and reversed directions. The effect belongs to the complete saved representation; attributing it solely to two newly collapsed reaches would be premature.

### The most extreme head-drop cases

In 46/S039_unit_18 the dominant child reach has about 0.12 mm of head drop over 539 m; in 46/S093_unit_64 it has about 0.23 mm over 480 m. Their cell relaxation times are approximately 9,541 and 2,728 hours. These are model coefficients, not observed event durations or a prediction that the whole flood lasts that long. They make the sensitivity to prescribed heads/floors particularly consequential. The second case's centroid advance is only 0.37 minutes, so its opposite-sign label should not be overemphasized.

### Counterexamples and limits

Non-outliers also sometimes increase spread: 49 have SD ratios greater than 1.24 and four exceed 2.0; the largest is about 2.18. Greater spread is therefore not a sufficient outlier classifier. Matched non-outliers (same network, edit type and ΔJ sign where available; nearest |ΔJ| on a log scale) have median SD ratio 1.007 for the G3 matches, versus 2.673 for G3. Some controls are reused, and nearest available matches need not be close. This is descriptive evidence, not independent replication.

## G2: retain the label, but distinguish two mechanisms

02/S016_unit_7, 13/S054_unit_53 and 46/S142_unit_37 have positive ΔJ and positive measured delay, but considerably less delay than the pooled line predicts. Their SD ratios are 1.551, 1.152 and 1.123. They each have a floor-limited child reach with substantial dispersion contribution. These fit the same hydraulic tendency as G3 without crossing zero in the finite-event response.

46/S178_unit_44 instead has ΔJ=−90.82 minutes, response=−60.04 minutes and almost unchanged spread (ratio 0.998). It is better investigated alongside the aggregate advances. There is no reason to force all four same-sign members into a single mechanism.

## G1: direction is right; a common proportional response is imperfect

All eleven are advances, and all reduce transfer spread. Their response/ΔJ ratios range from 0.611 to 0.946, whereas the pooled slope is 0.729. With large |ΔJ|, even a moderate proportional difference becomes an outlying absolute residual. Residuals occur on both sides of the fitted line. This is not the sign-reversal mechanism in G3.

For example, network 55's aggregate collapse has ΔJ=−192.97 minutes and response=−170.06 minutes (ratio 0.881); network 54 has ΔJ=−67.97 and response=−41.50 (ratio 0.611). Both move in the predicted direction, but one common response multiplier cannot represent both closely.

Their matched non-outlier controls also tend to reduce spread (median ratio 0.950 versus G1's 0.945). Reduction alone does not distinguish these outliers. Network 15 is exceptional within G1 because its transfer SD nearly halves. These results support keeping G1 as a separate investigation of large edits, event shape and proportional-response heterogeneity. They do not establish one universal physical cause for its regression residuals. No network-specific prediction models were fitted.

## Actual-wave accounting, checks and unresolved issues

For common baseline and interval, q denotes excess flow and C_p the parent centroid. With M_c=∫q_c dt, the exact identity is

$$C_c-C_p=\frac{\int(t-C_p)[q_c(t)-q_p(t)]dt}{M_c}.$$

The script verifies this against every saved outlier centroid and splits the numerator before/after the parent peak and centroid. `hydrograph_segments.csv` contains the contributions. `hydrograph_accounting.csv` records window and post-window volume differences and terminal discharge differences. A finite-window volume deficit is not evidence of mass loss; the window is not a complete equal-volume pulse experiment.

All ten G3 and all four G2 cases remain flagged in the existing 1% and 2% windows. Ten of eleven G1 cases remain flagged. This supports persistence across those modest threshold changes, not equivalence to a complete infinite-duration response.

Saved time/grid refinement checks directly cover eleven of the 25 cases (the nine all-closed-unit collapses and the 02 and 13 same-sign members). All available checks pass; the largest change in centroid contrast is 0.336 seconds. **None of G3 has direct saved refinement coverage.** The analytic first-two-moment identities are grid-independent here, but this does not certify the shape or finite-event centroid in the extreme low-slope cases. Settling criteria based on small discharge differences also need not imply that tail-weighted first or second moments have converged. Do not diagnose a numerical bug from negative X alone, or dismiss this numerical gap.

Source parameter and hydrograph hashes were checked against completion receipts. J was reproduced for all 2,185 comparisons, as was the equivalence between saved K/X dispersion and the hydraulic formula. Parent–child n and diffusion multiplier equality were checked for all outliers. A separate hand-calculated unequal parallel-path example verified mean 21 s, path variance 24 s² and local variance 34 s². No response timing was used to calculate these analytic transfer moments.

## Recommended next investigation

Keep ΔJ and the original outlier flags. Use transfer spread as an explanatory diagnostic rather than immediately adding it to a predictor. The strongest next target is **the low-head-drop inputs in G3**, with the three delayed G2 members as useful comparisons.

Review three representative mechanisms: 37/S082_unit_324 (documented boundary flex and active floors), 37/island_37_282 (retained island inputs with widespread hydraulic changes), and 46/S039_unit_18 (extreme head-drop and relaxation scale). Include 03/S027_unit_24 as a useful low-slope case that is not exactly at its staged floor, and matched non-outliers from the supplied table. Review the original head evidence, boundary contracts and accepted geometry before changing any slopes.

If subsequent simulations are authorized, first test time/grid refinement and tail-moment convergence for those cases. Then a controlled hydraulic-input sensitivity that respects the boundary/head constraints could distinguish slope-floor dependence from a persistent collapse effect. Longer output alone does not turn an event with unequal initial/final plateaus into an isolated pulse. No such new runs were launched here.

For G1, retain the existing pooled analysis and report residuals in both minutes and relative to |ΔJ|. A separate perfect fit for each network would obscure the question and is not required. The evidence currently supports a split between a strong dispersion/input mechanism in G3 and heterogeneous large-edit scaling in G1.

## Files and reproduction

- `outlier_spread.html`: interactive population overview with original z flags.
- `outlier_hydrographs.html`: dropdown for all 25 actual parent/child hydrographs and their differences; no translations. Default view shows the response window plus six hours; the full saved duration remains available by zooming out. To keep the figure compact, each ten-minute bin retains its first sample and both extrema. This display reduction does not affect any computed metric, which uses the original full-resolution records.
- `outlier_diagnostics.csv`: all cases and both moment components.
- `input_audit.csv`, `dominant_child_reaches.csv`, `stable_geometry_pairs.parquet`: staged status, floor contributions and matched hydraulic changes.
- `matched_controls.csv`: auditable control selection, including match distances and fallback type.
- `numerical_check_coverage.csv`, `outlier_window_persistence.csv`: limitations and threshold robustness.
- `provenance.json`, `input_audit_provenance.json`: source receipts and script hashes.

The original `geometry_changes.csv` uses strict rounded geometry signatures: its “new/removed” labels mean unmatched signatures, not necessarily physically added/deleted reaches. Use the tolerance-aware input audit for surviving-geometry statements.

From the repository root, run these scripts with the river-hierarchy-rivgraph Python environment:

```text
RAPID/investigations/three_run_review_20261029/investigate_j_outlier_groups.py
RAPID/investigations/three_run_review_20261029/audit_j_outlier_inputs.py
RAPID/investigations/three_run_review_20261029/plot_j_outlier_groups.py
RAPID/investigations/three_run_review_20261029/write_j_outlier_report.py
```

The notebook adds a saved-diagnostics overview and the same 25-case interactive inspector. The external volume is required to regenerate the analysis or figures; embedded figures and saved tables can be reviewed without rerouting.

## All 25 cases

All timing columns below are minutes; spread ratio is dimensionless. “Checks” counts saved numerical refinement checks, not a new validation run.

'''
    table=d[['group','example','state','delta_J_minutes','centroid_shift_minutes','ols_residual_minutes','sd_ratio','checks']].copy()
    table['group']=table.group.str[:2]
    text+='| '+' | '.join(table.columns)+' |\n| '+' | '.join(['---']*len(table.columns))+' |\n'
    for row in table.itertuples(index=False,name=None):
        text+='| '+' | '.join(f'{v:.3f}' if isinstance(v,float) else str(v) for v in row)+' |\n'
    text+='\n## Case-level input and control audit\n\n'
    for r in d.sort_values(['group','example','state']).itertuples():
        au=a[(a.example==r.example)&(a.state==r.state)].iloc[0]
        top=t[(t.example==r.example)&(t.state==r.state)].iloc[0]
        cc=controls[(controls.outlier_example==r.example)&(controls.outlier_state==r.state)]
        text+=f'### {r.example}/{r.state}\n\n'
        text+=f'{r.group}. ΔJ {r.delta_J_minutes:+.2f} min; centroid {r.centroid_shift_minutes:+.2f} min; residual z={r.residual_z:+.2f}. Transfer SD {r.parent_transfer_sd_hours:.2f} → {r.transfer_sd_hours:.2f} h. Added variance: local {r.delta_diffusion_variance_h2:+.3f} h²; between-route {r.delta_path_variance_h2:+.3f} h².\n\n'
        text+=f'Staged input status: {au.staged_status}. Stable directed-geometry matches: {au.stable_geometry_directed_matches}; >1% slope changes: {au.slope_changed_gt1pct}; >1% allocation changes: {au.allocation_changed_gt1pct}. '
        if pd.notna(au.floor_share_of_child_diffusion):text+=f'Active-floor share of child local dispersion: {au.floor_share_of_child_diffusion:.1%}. '
        text+=f'The largest child local-dispersion contributor is reach {int(top.id_link)} ({top.share_child_diffusion:.1%} of child local dispersion), with slope {top.slope:.3g}, head drop {top.head_drop_m*1000:.3f} mm and cell relaxation {top.cell_relaxation_hours:.2f} h. It may be an inherited reach; this ranking alone does not establish which reach caused the change.\n\n'
        text+='Matched controls: '+('; '.join(f'{v.control_example}/{v.control_state} (ΔJ {v.delta_J_min:+.2f} min, SD ratio {v.sd_ratio:.3f}, match distance {v.match_distance:.3f})' for v in cc.itertuples()) or 'none')+'.\n\n'
    full_path=DEST/'full_saved_record_centroids.csv'
    if full_path.exists():
        full=pd.read_csv(full_path,dtype={'example':str})
        text+='## Follow-up: full saved records and the positive ΔJ contribution\n\n'
        text+='The full common saved record has now been checked for all ten G3 cases, using the original baseline and exact piecewise-linear moments, without clipping or extrapolation. All ten centroid shifts become more positive than in the event window; three become positive (one only +0.020 min), and seven remain negative. Therefore the existing window boundary does not, by itself, account for all sign reversals. The full saved record is still finite and initial and final flows differ, so this is not a complete isolated-pulse centroid test. For 37/island_37_282 the shift changes from −9.587 to −4.005 min, still opposite to ΔJ=+22.377 min.\n\n'
        text+='A descriptive partition of J at slope<10⁻⁵ gives a positive contribution to ΔJ from that low-slope subset in every G3 case, and a negative contribution from the remainder. For 37/island_37_282 these are +37.651 and −15.274 min. Thus the same low-slope hydraulics that increase dispersion also account for the positive total ΔJ in this bookkeeping. This is not a causal slope intervention: membership of the subset can change between parent and child, and allocation and geometry also change. The tail is not an input to J; both the mean and tail behavior follow from the saved hydraulic coefficients.\n\n'
        cols=['example','state','delta_J_min','window_centroid_min','full_saved_excess_centroid_min']
        text+='| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'
        for row in full[cols].itertuples(index=False,name=None):
            text+='| '+' | '.join(f'{v:.3f}' if isinstance(v,float) else str(v) for v in row)+' |\n'
        text+='\nReproduce with `check_j_full_records.py`; results and source hashes are in `full_saved_record_centroids.csv` and `full_saved_record_provenance.json`. No simulations were run.\n'
    (DEST/'REPORT.md').write_text(text)
    print('Wrote',DEST/'REPORT.md')

if __name__=='__main__':main()
