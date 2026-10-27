"""Matched-grid equal-split screen; no production geometry changes.
Uses the exploratory variable-depth kernel, which is not mass conservative.
"""
from pathlib import Path
root=Path(__file__).resolve().parents[2]
# Reuse only the setup and functions, not the original screening experiment.
source=root/'synthetic_experiments/investigations/variable_depth_screen.py'
exec(compile(source.read_text().split('\nfor case in ')[0],str(source),'exec'))
out=root/'synthetic_experiments/outputs/equal_split_width_screen_20261027';out.mkdir(exist_ok=True)
s={**s,'baseflow_cms':500}; t,flow,_=p.forcing(s,192)
refdepth=p.predict_based_depth(discharge_cms=1000,width_m=500,slope=1e-4)
rows=[];pairs=[]
for factor in [1.,1.1,1.2]:
 waves={}
 for state in ['parent','child']:
  edges=[(0,1,5000.,500.),(2,3,5000.,500.)]
  edges+=([(1,2,10000.,250*factor)]*2 if state=='parent' else [(1,2,10000.,500*factor)])
  f=pd.DataFrame(edges,columns=['u','v','link_length_m','width']);f['slope']=1e-4;f['id_link']=np.arange(1,len(f)+1);f['depth_m']=refdepth*(f.width/500)**.65
  g=p.cunge.freeze_grid(f,1000,n_bounds=(.01,.1),alpha=2,min_celerity=.05,max_celerity=20)
  g['count']=np.ceil(g.link_length_m/200).astype(int);g['rapid_effective_length_m']=g.link_length_m/g['count']
  for method in ['fixed_BASED_peak','fixed_Manning_mid','variable_Manning_3point']:
   m=g.copy()
   if method=='fixed_Manning_mid':
    m['reference_discharge_cms']=750*m.reference_flow_fraction
    m['depth_m']=(.025*m.reference_discharge_cms/(m.width*np.sqrt(m.slope)))**.6
    m['celerity_times_n']=(5/3)*m.depth_m**(2/3)*np.sqrt(m.slope)
   m=p.cunge.parameters(m,.025)
   if method.startswith('variable'):w,minimum=route(m,flow,15)
   else:w,_=p.cunge.route(m,flow,4,15)
   record=dict(width_factor=factor,state=state,method=method,**stat(t,w,500,flow))
   if not method.startswith('variable'):record['centroid_hours']=40+float(sum(m.reference_flow_fraction*m.link_length_m/m.rapid_celerity_mps))/3600
   rows.append(record);waves[state,method]=w
   if method=='fixed_BASED_peak':m.to_csv(out/f'links_{factor}_{state}.csv',index=False)
 for method in ['fixed_Manning_mid','variable_Manning_3point']:
  np.testing.assert_allclose(waves['parent',method],waves['child',method],rtol=1e-10,atol=1e-8)
 keep=t<7*86400
 pd.DataFrame({'time_seconds':t[keep],'inflow':flow[keep],**{a+'__'+b:w[keep] for (a,b),w in waves.items()}}).to_parquet(out/f'hydrographs_{factor}.parquet')
r=pd.DataFrame(rows);r.to_csv(out/'state_results.csv',index=False)
for (factor,method),g in r.groupby(['width_factor','method']):
 a=g[g.state=='parent'].iloc[0];b=g[g.state=='child'].iloc[0];pairs.append(dict(width_factor=factor,branch_width_m=250*factor,child_width_m=500*factor,method=method,peak_shift_min=60*(b.peak_hours-a.peak_hours),centroid_shift_min=60*(b.centroid_hours-a.centroid_hours),max_abs_volume_error_pct=g.volume_error_pct.abs().max()))
r=pd.DataFrame(pairs);r.to_csv(out/'pair_results.csv',index=False);print(r.round(4).to_string(index=False))
