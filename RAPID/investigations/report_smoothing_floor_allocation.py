from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'RAPID/assets/smoothing_floor_allocation_20261002'
a=pd.read_csv(P/'allocations.csv');s=pd.read_csv(P/'profiles.csv')
assert len(a)==96 and a.status.eq('complete').all();assert len(s)==32 and not s.selected_grid_edge.any()
checks=[];minima=[]
for r in a.itertuples():
 d=P/f'{r.example:02d}'/f'floor_{r.floor:.2f}'/r.allocation;l=pd.read_parquet(d/'links.parquet');n=pd.read_parquet(d/'nodes.parquet').set_index('id_node');u=pd.read_csv(d/'units.csv');h=n['head'];ph=n.profile_head
 assert len(l)==r.n_links and l.slope.gt(0).all()
 err=(l.id_us_node.map(h)-l.id_ds_node.map(h))-l.slope*l.actual_length_m
 boundary=max(max(abs(h.loc[v.source]-ph.loc[v.source]),abs(h.loc[v.sink]-ph.loc[v.sink])) for v in u.itertuples())
 assert abs(err).max()<1e-10 and boundary<1e-7
 k=l.loc[l.slope.idxmin()]
 minima.append(dict(example=r.example,floor=r.floor,allocation=r.allocation,link=int(k.id_link),slope=k.slope,kind=k.ownership_kind,length_m=k.actual_length_m,drop_mm=k.slope*k.actual_length_m*1000))
 checks.append(dict(example=r.example,floor=r.floor,allocation=r.allocation,head_error_m=float(abs(err).max()),boundary_error_m=boundary))
pd.DataFrame(checks).to_csv(P/'validation.csv',index=False);pd.DataFrame(minima).to_csv(P/'minimum_links.csv',index=False)
pivot=s.pivot(index='example',columns='floor',values='cv_rmse');effect=100*(pivot[.05]/pivot[.01]-1)
summary=a.groupby(['floor','allocation']).agg(cases=('example','size'),below_1e5=('below_1e5','sum'),below_1e6=('below_1e6','sum'),minimum=('min_slope','min'));summary.to_csv(P/'comparison.csv')
cv=[]
for f in P.glob('*/floor_*/cv.json'):
 for v in json.loads(f.read_text()).values():cv.append(dict(example=f.parent.parent.name,floor=float(f.parent.name.split('_')[1]),**{k:z for k,z in v.items() if k!='fold_audits'}))
pd.DataFrame(cv).to_csv(P/'all_cv_candidates.csv',index=False)
text='''# Smoothing × floor × allocation comparison — 2026-10-02

User authorized all proposed tests following the unsuccessful first data-informed section-floor candidate. All tests completed for all 16 base networks. No production files, equivalent resistances, Kb parameters or hydrographs changed.

## Controlled design

- Revert to the simple common longitudinal profile plus acquisition offsets: no section P75 targets, no section-derived interval lower bounds.
- Equal total cycle weight. Within a training split, weights sum to the full-data observation count so smoothing strength has a consistent scale across folds.
- Compare profile floors 0.01 and 0.05 times the existing static reference slope. The higher fraction is a sensitivity/modeling choice, not a universal physical minimum.
- Three complete-cycle folds, deterministic shuffled cycle assignment (seed 20261002), identical folds for both floors. All acquisitions within a cycle remain together.
- Score held-out profile shape after estimating/removing a separate constant offset for each held-out acquisition; average squared error first within cycles, then equally across cycles. This evaluates longitudinal shape, not independent prediction of absolute stage. Scores used to select lambda are model-selection CV, not an untouched final test set.
- Initial logarithmic lambda search from 1e-8 × old lambda through old lambda; adapt the range at boundaries, then refine locally. Choose minimum CV MSE. Numerically indistinguishable scores within 1e-6 relative MSE resolve toward the larger lambda. This only changed case 15, where the low-lambda CV curve is effectively flat; it is not evidence for a uniquely identifiable tiny optimum.
- Preserve the existing case-46 manual bridge in every fold and final fit. No new bridges.
- Compare old domain-wide slope target with 1% relative domain floor, versus local-profile-gradient targets with graph-maxmin allocation. These differ in BOTH target and lower-envelope rule.
- Add a third control: old domain-wide target with graph-maxmin lower envelope. This isolates the lower-envelope change from the local-target change.

All 32 selected full profiles and 96 base-network allocations completed. No final lambda is at the tested search boundary; numerical residuals and CV statuses are saved. All links are positive, share consistent node heads, and preserve the selected profile's unit boundary heads. Worst lower-bound head violation across all allocations is approximately 12 micrometres. No routing stability conclusion follows from these checks.

## Overall slope results

| Profile floor | Allocation | Links below 1e-5 | Links below 1e-6 | Minimum link slope |
|---|---|---:|---:|---:|
'''
labels={'old':'Old domain target + 1% domain floor','maxmin':'Local-gradient target + graph minimum','maxmin_domain':'Old domain target + graph minimum'}
for (floor,allocation),r in summary.iterrows():text+=f'| {floor:.2f} S_ref | {labels[allocation]} | {int(r.below_1e5)} | {int(r.below_1e6)} | {r.minimum:.3g} |\n'
text+='''
The previous production profiles had 57 low links. These new counts are based on re-selected, converged profiles and must not be interpreted as a direct floor-only comparison to production.

The local-gradient target is the major problem in the more elaborate allocation: it transfers flat parts of the fitted longitudinal profile into very small branch targets, sometimes affecting many links in large compound units. The graph-derived rule only prevents slopes that are small *relative to their targets*; it does not make a tiny target physically well supported.

The domain-wide graph control is more restrained, but is not a clear universal improvement. At the higher floor it removes four low links in case 15 and adds four in case 22, leaving the total unchanged. It raises the overall minimum from 3.60e-6 to 4.51e-6. Neither method has been validated against observed internal branch head distributions.

## Per-case comparison: simple profile and old allocation

| Case | Low links at 0.01 | Low links at 0.05 | Held-out RMSE change with higher floor | Head-loss change vs production at 0.05 |
|---|---:|---:|---:|---:|
'''
for ex in sorted(s.example.unique()):
 aa=a[(a.example==ex)&(a.allocation=='old')].set_index('floor');rr=s[(s.example==ex)&(s.floor==.05)].iloc[0]
 text+=f'| {ex:02d} | {int(aa.loc[.01,"below_1e5"])} | {int(aa.loc[.05,"below_1e5"])} | {effect.loc[ex]:+.3f}% | {rr.head_change_pct:+.3f}% |\n'
text+=f'''
Across cases, the largest increase in held-out RMSE from the higher floor is {effect.max():.3f}% (case {effect.idxmax():02d}); some cases improve slightly. This indicates little CV penalty within the tested ranges, not statistically established superiority of the higher floor. Absolute selected-lambda values are often orders of magnitude below the old values. They are specific to the normalized curvature operator and weight scaling; their magnitude is not itself a physical parameter.

Case 21 still changes inlet–outlet head loss by about 13.5% relative to production despite re-selection. It requires visual and observational review. Case 46 retains 28 low links even at the higher floor, and is still the dominant unresolved low-slope case by count. Case 11 retains three low links. These values may reflect profile support, branch lengths or boundary head allocation; they are not all demonstrated errors.

## Recommendation

The simplest next candidate is the re-selected, converged reference profile with 0.05 S_ref and the existing unit allocation. It avoids the previous oversmoothing without introducing spatially varying unit targets; it raises the observed numerical minimum and has little held-out shape-error cost. This is a candidate for review, not a physical lower-bound claim or automatic production adoption.

Keep 0.01 as the sensitivity reference. Do not adopt the local-gradient/maxmin allocation on the basis of this experiment. The domain-wide graph alternative may be retained as a separate allocation sensitivity, but its head redistribution still needs independent assessment.

Before resistance/routing reruns, inspect the profiles (especially 11, 21, 22 and 46) and, if accepted, test Kb/celerity feasibility using these fixed slopes. No smoothing/weighting/geometry changes should be bundled into that next test.

## Files and reproduction

- profiles.csv: 32 lambda selections, held-out errors and head-loss changes.
- allocations.csv / comparison.csv: all 96 completed allocations and aggregates.
- all_cv_candidates.csv and each case's cv.json: every lambda evaluation, fold solver diagnostics and failures (if any).
- minimum_links.csv: precise minimum link, slope, length and drop for every scenario.
- validation.csv: independent shared-head and fixed-boundary checks.
- Each case/floor contains its profile.parquet and selection.json; old, maxmin and maxmin_domain folders contain link/node heads and unit diagnostics.
- Notebook: RAPID/notebooks/slope_smoothing_floor_comparison.ipynb.
- Reproduce with RAPID/investigations/smoothing_floor_allocation_tests.py and RAPID/investigations/report_smoothing_floor_allocation.py using the project Python environment, OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1. CV evaluations are cached; use a new output directory when changing the objective, data or fold rules.
'''
(P/'README.md').write_text(text)
m=json.loads((P/'method.json').read_text());m['script_sha256']=hashlib.sha256((ROOT/'RAPID/investigations/smoothing_floor_allocation_tests.py').read_bytes()).hexdigest();m['n_cv_candidates']=len(cv);m['n_cv_failures']=sum(z['status']!='complete' for z in cv);(P/'method.json').write_text(json.dumps(m,indent=2))
print('Validated 32 profiles and 96 allocations;',len(cv),'CV candidates;',m['n_cv_failures'],'failed candidates')
print('Higher-floor RMSE change %',effect.min(),effect.max())
print(pd.DataFrame(minima).query("floor==.05 and allocation=='old'").nsmallest(1,'slope').to_string(index=False))
