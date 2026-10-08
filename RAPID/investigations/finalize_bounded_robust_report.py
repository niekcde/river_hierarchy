"""Write the dated decision report and append it to the existing investigation notebook."""
from pathlib import Path
import json,base64,numpy as np,pandas as pd,nbformat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'RAPID/assets/bounded_robust_slopes_20261002'
s=pd.read_csv(OUT/'summary.csv',dtype={'example':str});w=pd.read_csv(OUT/'window_summary.csv',dtype={'example':str});v=pd.read_csv(OUT/'validation.csv',dtype={'example':str});hyd=pd.read_csv(OUT/'hydraulic_stability.csv',dtype={'example':str});old=pd.read_csv(ROOT/'RAPID/assets/selected_slope_review_20261002/cases.csv',dtype={'example':str});h=hyd[(hyd.anchor=='stored_BASED')&(hyd.alpha==.65)&(hyd.n_label=='0.035')];common=h[h.caps=='common_0.05_2.524_caps'];table=s.merge(w,on='example').merge(common[['example','integer_feasible']],on='example').merge(old[['example','low_links']],on='example')
rows=['| Case | Low links: squared → selected | Windows outside cycle range: squared → selected | Common-cap timestep feasible |','|---|---:|---:|---|']
for z in table.itertuples():rows.append(f'| {z.example} | {z.low_links} → {z.below_1e5} | {z.old_outside} → {z.new_outside} / {z.n_windows} | {"Yes" if z.integer_feasible else "No"} |')
text='''# Bounded robust-slope implementation — 2026-10-02

**Decision reference: ROBUST-SLOPES-2026-10-02.** This supersedes the earlier exploratory method recommendations in this notebook. The bounded implementation is complete for all 16 base networks; production promotion and downstream collapsed-state/re-routing work have not been performed.

## Decision
Retain the selected Huber robust fit, the 0.05 S_ref profile floor, the SWORD head-difference anchor for the disconnected gap in case 21, the documented bridge in case 46, and the existing shared-head unit allocation. Freeze this as the selected slope candidate. Hydraulic stability remains a separate work item; do not further adjust slopes simply to pass it.

## What the method does
1. Fit one shared reference profile to screened RiverSP observations, allowing a constant stage offset for each acquisition.
2. Give each cycle equal total base weight. Use Huber loss to limit the influence of extreme residuals while preserving ordinary squared-error treatment for small residuals.
3. Determine the Huber threshold from a weighted residual MAD; estimate it on training cycles only during validation. Hold it fixed while searching smoothing values.
4. Select smoothing with three held-out whole-cycle folds. Score held-out shape after removing a constant acquisition offset; retain ordinary RMSE alongside the robust score.
5. Require profile gradients of at least 0.05 times the existing static-reference global slope. This is a regularization assumption, not a measured universal minimum river slope.
6. Where observations are disconnected across a gap, constrain the missing head difference using the static SWORD reference. This applies only to case 21 in this dataset. Interpolate linearly through that gap inside the fit. Retain case 46's existing manual bridge as an internal linear constraint.
7. Transfer profile heads to the network and solve the existing owned units with fixed boundary heads and physical link lengths. Shared node heads preserve head-loss consistency. The profile floor is not an identical floor on every physical branch.

The detailed equations, parameter choices, limitations and policy for defensible manual adjustments are in `RAPID/ROBUST_REFERENCE_PROFILE_METHOD.md`. No new observation deletions or manually chosen local slopes were introduced. Future manual corrections should state the evidence, extent, constraint and resulting head-loss change.

## Numerical and observational result
- All **16 cases / 5,315 links** completed; all reference gradients and final link slopes are positive.
- **120 smoothing candidates / 360 held-cycle-fold fits**, with zero failed CV fits and no selected lambda at a search boundary. All final solvers converged.
- Links below 1e-5 decrease from **47 to 32**; none are below 1e-6. The minimum remains **3.603045e-6** in case 46's compound domain.
- Local 10 km windows outside the cycle P10–P90 slope range decrease from **58/300 to 23/300**. This is an in-sample diagnostic of local cycle agreement, not a confidence interval or independent validation.
- Earlier fixed-lambda robust loss had 31 low links and 31 window flags. The selected candidate has one more low link but fewer window flags; low-link count was not the selection objective.
- Profile constraints are satisfied to micrometre-scale head tolerances; unit boundary and shared-head checks pass. Four focused tests passed for gap identification/anchoring, incompatible references, robust outlier influence and held-out stage-offset removal.

'''+ '\n'.join(rows)+'''

## Case 21: interpolation has an explicit head reference
The disconnected gap is **89.2067–103.3844 km**. The imposed SWORD head difference is **2.342809 m**, corresponding to approximately **1.65e-4 m/m** within the linear bridge. This is an explicit external-reference assumption. No acquisition measures across the gap.

The fitted total network head loss is **43.01697 m**, compared with 50.74912 m for the earlier squared-loss candidate and 44.70925 m for the older production profile. It was not selected to reproduce either previous total. Relative heights in the observed sections are fitted robustly subject to the gap constraint.

## Remaining local uncertainty
Some cycle slopes and fitted gradients still disagree. The short case-46 section around 426–428 km can remain steeper than most cycle slopes; sparse-support sections still depend on the floor. These are retained as documented limitations rather than triggering another round of hand-picked slopes. The 5% profile floor and the static gap head difference are modelling assumptions. A globally good shape fit does not establish physical validity of every link slope.

## Hydraulic preflight: a separate unresolved question
At **alpha=0.65, n=0.035, 200 m target segmentation, X=0.1**, using the stored BASED reference depth:

| Celerity treatment | Base networks with feasible integer timestep |
|---|---:|
| Uncapped | 7/16 |
| Common 0.05–2.524 m/s caps | 9/16 |
| Saved per-case caps (defaults where absent) | 10/16 |

The saved caps differ between cases; their additional pass should not be interpreted as an improvement in the profile method. Common-cap failures are **05, 11, 13, 15, 22, 41 and 46**. With saved caps the failure list excludes 22. Reference-depth recalculation at the candidate global slope is a diagnostic and gives the same primary n=.035 pass counts.

These failures mean the nominal common integer timestep interval is empty. They do not by themselves establish that the new slopes are wrong, that the full preparation safeguards fail, or that a routed hydrograph is unstable. The audit does not include adaptive safeguards, forcing-clock restrictions or all child states. It identifies the remaining hydraulic/timestep problem. Do not choose roughness, depth or slope solely to make the interval nonempty.

'''
fails=common[~common.integer_feasible]
text+='| Case | Controlling slow link | Slope | Width (m) | Estimated depth (m) | Required dt minimum (s) | Allowed dt maximum (s) |\n|---|---:|---:|---:|---:|---:|---:|\n'
for z in fails.itertuples():text+=f'| {z.example} | {z.slow_link} | {z.slow_slope:.3g} | {z.slow_width_m:.1f} | {z.slow_depth_m:.2f} | {z.lower_seconds:.1f} | {z.upper_seconds:.1f} |\n'
text+='''
The audit also stores alpha 0.6/0.71/1.0, n 0.025/0.05/implied, and a diagnostic BASED re-evaluation. Implied n is a wide-channel Manning consistency estimate, not a fitted roughness. All scenario data are in `hydraulic_stability.csv` and `hydraulic_summary.csv`.

## Artifacts and reproduction
- `RAPID/src/rapid_tools/robust_reference_profile.py`: opt-in fitting implementation.
- `RAPID/investigations/run_bounded_robust_slopes.py`: fit, select and allocate all 16; optional case IDs for individual reruns.
- `RAPID/investigations/validate_bounded_robust_slopes.py`: saved-artifact numerical checks and input/code hashes.
- `RAPID/investigations/report_bounded_robust_slopes.py`: local cycle-slope audit.
- `RAPID/investigations/audit_bounded_robust_hydraulics.py`: base-only timestep diagnostics.
- `RAPID/investigations/plot_bounded_robust_slopes.py` and `finalize_bounded_robust_report.py`: plots and notebook report.
- Per-case `floor_0.05/`: profiles, observations/robust weights, acquisition offsets, folds, training thresholds, CV scores, explicit gap/bridge constraints and old-allocation link/node tables.
- `validation.csv`, `input_provenance.json`, `code_provenance.json`: checked results and reproducibility references.

All plots compare with the previous reselected **squared-loss / 0.05 floor** candidate, not the original production profiles. Observation curves are offset-aligned, so their vertical alignment is not independent validation of absolute level. Original notebook results are preserved as historical experiments.
'''
(OUT/'README.md').write_text(text)
fig,axes=plt.subplots(4,4,figsize=(16,12),constrained_layout=True)
for ax,z in zip(axes.flat,s.itertuples()):
 cv=json.loads((OUT/z.example/'floor_0.05/cv.json').read_text());data=pd.DataFrame([r for r in cv.values() if r['status']=='complete']).sort_values('lambda_value');ax.semilogx(data.lambda_value,data.cv_huber,'o-',ms=3);ax.axvline(z.selected_lambda,color='red',ls='--',lw=1);ax.set(title='Case '+z.example,xlabel='Smoothing lambda',ylabel='Held-cycle Huber loss');ax.grid(alpha=.2)
fig.savefig(OUT/'smoothing_selection.png',dpi=125);plt.close(fig)
p=ROOT/'RAPID/notebooks/slope_smoothing_floor_comparison.ipynb';n=nbformat.read(p,as_version=4);tag='bounded-robust-decision-20261002';n.cells=[c for c in n.cells if tag not in c.metadata.get('tags',[])]
banner=nbformat.v4.new_markdown_cell('**Latest decision: ROBUST-SLOPES-2026-10-02.** The final section contains the selected robust method, gap constraints, all-16 results and hydraulic limitations. Earlier sections are retained as the investigation history. See also [the full methodology](../ROBUST_REFERENCE_PROFILE_METHOD.md).');banner.metadata['tags']=[tag];n.cells.insert(0,banner)
c=nbformat.v4.new_markdown_cell(text);c.metadata['tags']=[tag];n.cells.append(c)
for name in ['all_profiles.png','priority_sections.png','all_local_slopes.png','smoothing_selection.png']:
 c=nbformat.v4.new_markdown_cell(f'![Bounded robust method: {name}](attachment:{name})');c['attachments']={name:{'image/png':base64.b64encode((OUT/name).read_bytes()).decode()}};c.metadata['tags']=[tag];n.cells.append(c)
nbformat.validate(n);nbformat.write(n,p)
print('Dated report and existing notebook updated; all figures embedded; notebook valid.')
