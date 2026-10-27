"""Reconstruct saved held-out Elastic Net contributions without refitting.

Run from repository root. Contributions use each fold's training mean as the
reference and are not causal variable effects. Size perturbation is an algebraic
model diagnostic, not a physically reconstructed network.
"""
import json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
D=pd.read_parquet('RAPID/outputs/hydrograph_residual_review_20261018/diagnostic_rows.parquet');D=D[D.scenario.eq('all')].copy()
rows=[]
for ex,st in [('55','CB001_bubble_38'),('22','CB002_bubble_11')]:
 ck=json.load(open(f'RAPID/outputs/hydrograph_context_timing_20261018/checkpoints/all/{ex}/length_context__sd__log1p__elastic_net.json'))
 cols=ck['audit']['kept'];tr=D[D.example.ne(ex)];test=D[D.example.eq(ex)&D.state.eq(st)]
 w=tr.example.map(1/tr.example.value_counts()).to_numpy();y=tr.observed_hours.to_numpy();ym=np.average(y,weights=w);ys=np.sqrt(np.average((y-ym)**2,weights=w))
 scaler=StandardScaler().fit(tr[cols],sample_weight=w);beta=np.array([ck['audit']['standardized_coefficients'][c] for c in cols]);z=scaler.transform(test[cols])[0]
 contrib=z*beta*ys*60; slope=beta*ys*60/scaler.scale_
 np.testing.assert_allclose(ym*60+sum(contrib),test.predicted_hours.iloc[0]*60,rtol=0,atol=1e-8)
 print(ex,'baseline',ym*60,'pred reconstruct',ym*60+sum(contrib),'saved',test.predicted_hours.iloc[0]*60)
 for col,val,b in zip(cols,contrib,slope):rows.append(dict(example=ex,state=st,feature=col,value=test[col].iloc[0],contribution_minutes=val,slope_minutes_per_input_unit=b))
 print(pd.DataFrame(rows)[lambda f:f.example.eq(ex)].to_string(index=False))
 # Fixed fitted model: change only size f and normalized delta length together,
 # retain same within-domain length ratio, width/resistance etc.
 if ex=='55':
  f=test.B_mean_length_fraction.iloc[0];f2=D[D.example.eq('22')&D.state.eq('CB002_bubble_11')].B_mean_length_fraction.iloc[0]
  delta=test.B_exposed_length_change.iloc[0]
  change=slope[cols.index('B_mean_length_fraction')]*(f2-f)+slope[cols.index('B_exposed_length_change')]*delta*(f2/f-1)
  print('FIXED55_MODEL_SIZE_ONLY_CHANGE_MIN',change)
pd.DataFrame(rows).to_csv('RAPID/outputs/hydrograph_case_checks_20261019/length_predictor_attribution.csv',index=False)
R=pd.read_parquet('RAPID/outputs/hydrograph_response/responses.parquet')
print('ALL responses',len(R),'exact centroid zeros',sum(R.centroid_shift_hours.eq(0)),'near1sec',sum(abs(R.centroid_shift_hours)*3600<1))
print('COHORT',len(D),'zeros',sum(D.centroid_shift_hours.eq(0)),'near1sec',sum(abs(D.centroid_shift_hours)*3600<1))
print(R.loc[(R.centroid_shift_hours.abs()*3600).nsmallest(3).index,['example','state','centroid_shift_hours']].to_string(index=False))
