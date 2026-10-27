"""Separate paired cap controls; never replace published runs."""
import numpy as np
import pandas as pd
from audit import ROOT,OUT,EXT,read,save,sha,uncapped_grid,cunge,shared_clock,SOURCES
from rapid_tools.common_grid import route_interval_means
from rapid_tools.event_window_batch import equilibrium_check
from rapid_tools.hydrograph_response import distribution

def main():
 dest=OUT/'routing';dest.mkdir(exist_ok=True)
 save(OUT/'protected_extension_before.json',{str(p):sha(p) for p in EXT.rglob('*') if p.is_file()})
 plan=read(EXT/'routing_plan.json');metrics=read(EXT/'state_metrics.parquet');results=[]
 for case in plan['cases']:
  ex=case['example'];folder=ROOT/read(EXT/f'runs/{ex}/completion.json')['final_directory']
  wave=read(folder/'forcing.parquet');nb=int(wave.phase.eq('spinup').sum())
  t=wave.time_utc.astype('int64').to_numpy()/1e9
  r=metrics[(metrics.example==ex)&metrics.window.eq('0.5%')&metrics.state.eq('S000_base')].iloc[0]
  mask=(t>=r.rise_seconds)&(t<=r.fall_seconds)
  calibration=read(OUT/f'conditional_recalibration/{ex}.json')
  jobs=[j for j in case['jobs'] if j['state_id'] in ['S000_base','all_closed_units_collapsed']]
  grids=[];configs=[]
  for job in jobs:
   f=read(job['parameters']);grid=uncapped_grid(f)
   for scenario,n,cap in [('capped_control',float(f.n_manning.iloc[0]),5.),('uncapped',float(f.n_manning.iloc[0]),float(grid.celerity_max_mps.iloc[0])),('uncapped_refit',calibration['new_n'],float(grid.celerity_max_mps.iloc[0]))]:
    g=grid.copy();g['celerity_max_mps']=cap
    g=cunge.parameters(g,n,beta=float(f.diffusion_multiplier.iloc[0]))
    np.testing.assert_allclose(g.link_length_m,f.link_length_m,rtol=0,atol=0)
    grids.append(g);configs.append((job['state_id'],scenario,n))
  dt=shared_clock(grids,[30,60],min(j['native_dt_seconds'] for j in jobs))
  rows=[];curves=[];cache={}
  for g,(state,scenario,n) in zip(grids,configs):
   equilibrium_check(g,dt)
   key=(state,n,tuple(g.rapid_celerity_mps))
   if key in cache:q,a=cache[key]
   else:
    q,a=route_interval_means(g,wave.forcing_cms.to_numpy()[nb:],round(60/dt),dt,initial_flow=case['initial_cms'])
    q=np.r_[np.full(nb,case['initial_cms']),q];cache[key]=(q,a)
   assert a['minimum_native_flow_cms']>=0
   published=read(folder/state/'outlet.parquet').q_outlet_cms.to_numpy()
   rows.append(dict(example=ex,state=state,scenario=scenario,n=n,dt_seconds=dt,cells=int(g['count'].sum()),minimum_native_flow_cms=a['minimum_native_flow_cms'],peak_seconds=float(t[mask][np.argmax(q[mask])]),max_absolute_difference_from_published_cms=float(abs(q-published).max())))
   curves.append(q)
  baseline=min(float(r.baseline_cms),min(float(q[mask].min()) for q in curves))
  for row,q in zip(rows,curves):
   row.update(centroid_seconds=distribution(t[mask],q[mask]-baseline)['centroid_seconds'],baseline_cms=baseline,published_baseline_cms=float(r.baseline_cms))
  for scenario in ['capped_control','uncapped','uncapped_refit']:
   parent=next(row for row in rows if row['scenario']==scenario and row['state']=='S000_base')
   for row in rows:
    if row['scenario']==scenario:
     row['centroid_shift_hours']=(row['centroid_seconds']-parent['centroid_seconds'])/3600
     row['peak_shift_hours']=(row['peak_seconds']-parent['peak_seconds'])/3600
  pd.DataFrame({'time_utc':wave.time_utc,**{state+'__'+scenario:q for (state,scenario,n),q in zip(configs,curves)}}).to_parquet(dest/f'{ex}.parquet',index=False)
  results.extend(rows);pd.DataFrame(results).to_csv(dest/'metrics.csv',index=False)
  print(ex,'paired routes complete',flush=True)
 save(dest/'sources.json',SOURCES)

if __name__=='__main__':main()
