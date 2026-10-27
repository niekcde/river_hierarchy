"""Compare isolated uncapped fits with capped controls on matching grids/clocks."""
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from joint_calibration import DEST,BUNDLE,SOURCE,PARENTS,replacement_grid,cunge,feature,route_interval_means,sha

def main():
 protocol=json.loads((DEST/'protocol.json').read_text())
 assert json.loads((DEST/'completion.json').read_text())['complete']
 hashes={str(p):sha(p) for p in [BUNDLE/'height_targets.json',BUNDLE/'timing_targets.json']}
 targets=json.loads((DEST/'timing_targets.json').read_text())['records']
 heights=json.loads((DEST/'height_targets.json').read_text())['records']
 table=pd.read_csv(DEST/'comparison.csv',dtype={'example':str})
 rows=[];sections=[]
 for ex in protocol['affected_examples']:
  old=json.loads((BUNDLE/ex/'result.json').read_text());new=json.loads((DEST/ex/'result.json').read_text())
  grid=replacement_grid(ex);grid['celerity_max_mps']=5.
  f=cunge.parameters(grid,old['n_manning'],beta=old['beta'])
  data=pd.read_parquet(SOURCE/ex/'60s.parquet');meta=next(r for r in json.loads((SOURCE/ex/'60s.json').read_text()) if r['method']=='cunge')
  dt=new['native_dt_s']
  control_path=DEST/ex/'capped_control.parquet';audit_path=DEST/ex/'capped_control_audit.json'
  if control_path.exists() and audit_path.exists():
   cached=pd.read_parquet(control_path)
   pd.testing.assert_series_equal(cached.time_utc,data.time_utc)
   q=cached.matched_grid_capped_cms.to_numpy();audit=json.loads(audit_path.read_text())
  else:
   q,audit=route_interval_means(f,data.forcing_cms.to_numpy(),round(60/dt),dt,initial_flow=meta['initial_cms'])
   audit_path.write_text(json.dumps(audit,indent=2))
  valid=data.downstream_cms.notna().to_numpy();t=data.time_utc.astype('int64').to_numpy()/1e9
  target=next(r for r in targets if r['example']==ex);height=next(r for r in heights if r['example']==ex)
  tm,h,kind,_=feature(q[valid],t[valid],ex,target)
  newwave=pd.read_parquet(DEST/ex/'wave.parquet').calibrated_cunge_cms.to_numpy()
  assert .01<=new['n_manning']<=.1 and .5<=new['beta']<=2.
  assert audit['minimum_native_flow_cms']>=-1e-8 and new['minimum_native_flow_cms']>=-1e-8
  grid['celerity_max_mps']=float(max(grid.celerity_times_n/.01)*1.01)
  g=cunge.parameters(grid,new['n_manning'],beta=new['beta'])
  assert (g.rapid_celerity_mps_raw<g.celerity_max_mps).all()
  shifted=data.downstream_cms.to_numpy()+height['downstream_display_offset_cms']
  rec=dict(example=ex,matched_grid_capped_timing_error_minutes=(tm-pd.Timestamp(target['target_utc']).timestamp())/60,
   capped_control_peak_cms=h,old_reported_peak_cms=old['peak_cms'],
   capped_shifted_rmse_cms=float(np.sqrt(np.mean((q[valid]-shifted[valid])**2))),
   uncapped_shifted_rmse_cms=float(np.sqrt(np.mean((newwave[valid]-shifted[valid])**2))),
   max_refinement_feature_shift_minutes=max(abs(c['feature_shift_minutes']) for c in new['numerical_controls']),
   active_upper_capped_links=int((g.rapid_celerity_mps_raw>=g.celerity_max_mps).sum()),
   max_raw_celerity_mps=float(g.rapid_celerity_mps_raw.max()))
  rows.append(rec)
  pd.DataFrame({'time_utc':data.time_utc,'matched_grid_capped_cms':q}).to_parquet(DEST/ex/'capped_control.parquet',index=False)
  fig=go.Figure()
  for label,values in [('Capped: existing n and beta',q),('Uncapped: refitted n and beta',newwave),('Downstream observations, existing display offset',shifted)]:
   fig.add_trace(go.Scatter(x=data.time_utc[valid],y=values[valid],name=label))
  fig.update_layout(template='plotly_white',title=f'{ex}: calibration-window hydrographs',height=450,yaxis_title='Discharge (m³/s)')
  sections.append(fig.to_html(full_html=False,include_plotlyjs=False))
  for p in [SOURCE/ex/'60s.parquet',SOURCE/ex/'60s.json',BUNDLE/ex/'result.json']:
   hashes[str(p)]=sha(p)
 controls=pd.DataFrame(rows);controls.to_csv(DEST/'matched_grid_controls.csv',index=False)
 fig=go.Figure()
 for label,n,b in [('Existing capped','old_n','old_beta'),('Uncapped','new_n','new_beta')]:
  subset=table[table.example.isin(protocol['affected_examples'])]
  fig.add_trace(go.Scatter(x=subset[n],y=subset[b],text=subset.example,mode='markers+text',textposition='top center',name=label))
 fig.update_layout(template='plotly_white',title='Parameter changes in the seven refitted examples',xaxis_title='Manning n',yaxis_title='Beta',height=500)
 html='<html><meta charset="utf-8"><body style="font:15px Arial;margin:24px"><h1>Uncapped joint calibration — separate diagnostic</h1>'
 html+='<p>Existing bounds: n 0.01–0.10, beta 0.5–2. Timing within one minute first; unchanged height references remain soft. Seven affected parents refitted, eight unaffected selections retained without a new search. No production updates or child reruns. Calibration-window agreement is not independent validation. Boundary fits do not establish an unconstrained optimum.</p>'
 html+=table.round(6).to_html(index=False)+fig.to_html(full_html=False,include_plotlyjs=True)
 html+='<h2>Controls and shape comparison</h2><p>The capped control uses the same refined grid and timestep as the uncapped fit. Shape RMSE uses the pre-existing additive downstream display offset; it is diagnostic, not a new tuning objective. Numerical refinement shifts assess discretization sensitivity, not measurement precision.</p>'+controls.round(6).to_html(index=False)
 html+=''.join(sections)+'</body></html>'
 (DEST/'review.html').write_text(html)
 assert all(sha(p)==h for p,h in protocol['source_hashes'].items())
 assert all(sha(p)==h for p,h in hashes.items())
 (DEST/'review_validation.json').write_text(json.dumps(dict(original_sources_unchanged=True,selected_parameters_within_existing_bounds=True,selected_upper_cap_inactive=True,nonnegative_native_flow=True,controls=rows),indent=2))
 print(controls.to_string(index=False))

if __name__=='__main__':main()
