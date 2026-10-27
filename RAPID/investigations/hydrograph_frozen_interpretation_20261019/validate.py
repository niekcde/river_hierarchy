"""Independent artifact checks; does not fit forests or change model outputs."""
import contextlib,io,json
from pathlib import Path
import numpy as np
import pandas as pd
import nbformat
from run import OUT,ROOT,E,C,GROUPS,VIEWS,mask

design=json.loads((OUT/'design.json').read_text())
for f,h in design['sources'].items():assert E.sha(Path(f))==h, f'Source changed: {f}'
p=pd.read_parquet(OUT/'reconstructed_predictions.parquet')
s=pd.read_parquet(OUT/'shap_values.parquet')
assert len(p)==2170 and p.example.nunique()==15
assert len(s)==2170*7 and not s.duplicated(['example','state','feature']).any()
g=s.groupby(['example','state'])
explained=g.shap_minutes.sum()+g.base_minutes.first()
expected=p.set_index(['example','state']).predicted_minutes.loc[explained.index]
np.testing.assert_allclose(explained,expected,atol=1e-6)
assert (g.base_minutes.nunique()==1).all()
perm=pd.read_parquet(OUT/'permutation_repeats.parquet')
assert len(perm[perm.view=='all'])==15*len(GROUPS)*20
assert not perm.duplicated(['view','example','group','repeat']).any()
summary=pd.read_csv(OUT/'permutation_summary.csv')
scores=pd.read_csv(OUT/'scores_summary_exclusions.csv')
for view in VIEWS:
 z=p[mask(p,view)];y=z.centroid_shift_hours.to_numpy()*60;w=E.weights(z)
 mse=np.average((z.predicted_minutes-y)**2,weights=w)
 variance=np.average((y-np.average(y,weights=w))**2,weights=w)
 row=scores[scores.view==view].iloc[0]
 np.testing.assert_allclose(row.r2,1-mse/variance)
 ss=summary[summary.view==view]
 np.testing.assert_allclose(ss.r2_decrease,ss.delta_mse_minutes2/variance)
 sh=s[mask(s,view)].assign(abs_shap=lambda x:abs(x.shap_minutes)).groupby(['example','feature']).abs_shap.mean().groupby('feature').mean()
 ref=pd.read_csv(OUT/'shap_summary.csv').query('view == @view').set_index('feature').mean_abs_shap_minutes
 np.testing.assert_allclose(sh,ref.loc[sh.index])
a=pd.read_csv(OUT/'ale_summary.csv')
for _,g in a.groupby(['view','feature','bins']):
 assert (g.networks>0).all()
 np.testing.assert_allclose(np.average(g.ale_minutes,weights=g.mass),0,atol=1e-10)
 assert g.rows.sum()==scores[scores.view==g.view.iloc[0]].n.iloc[0]
n=nbformat.read(ROOT/'RAPID/notebooks/hydrograph_frozen_forest_interpretation.ipynb',as_version=4);nbformat.validate(n)
namespace={}
with contextlib.redirect_stdout(io.StringIO()):
 for cell in n.cells:
  if cell.cell_type=='code':exec(compile(cell.source,'notebook cell','exec'),namespace)
E.dump(OUT/'validation.json',dict(status='passed',checks=['All source hashes unchanged, including frozen model metadata and saved predictions','2170 predictions and seven SHAP values per comparison','Independent SHAP additivity check from saved values','20 permutation repeats per network/group','R2 loss denominator and equal-network SHAP aggregation','ALE mass centering, row counts and supported common bins','Notebook schema and every code cell executed successfully'],max_saved_shap_additivity_error_minutes=float(np.max(abs(explained-expected)))))
print('All artifact and notebook checks passed.')
