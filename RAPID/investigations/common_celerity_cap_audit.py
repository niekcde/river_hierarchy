"""Compare shared cap pairs across all 16 selected base networks."""
from pathlib import Path
import json
import numpy as np,pandas as pd,pyogrio
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'RAPID/assets/bounded_robust_slopes_20261002';OUT=BASE/'common_caps';OUT.mkdir(exist_ok=True)

def interval_conflicts(lower,upper):
 start=np.maximum(1,np.ceil(lower-1e-9).astype(int));end=np.floor(upper+1e-9).astype(int);valid=end>=start
 positions=np.r_[start[valid],end[valid]+1];changes=np.r_[np.ones(valid.sum(),int),-np.ones(valid.sum(),int)];order=np.argsort(positions);pos,idx=np.unique(positions[order],return_index=True);delta=np.add.reduceat(changes[order],idx);active=np.cumsum(delta);best=int(np.argmax(active));dt=int(pos[best]);slow=lower>dt+1e-9;fast=upper<dt-1e-9
 return dt,slow,fast

def main():
 refs=pd.read_csv(BASE/'hydraulic_references.csv',dtype={'example':str}).set_index('example');frames=[]
 for ex,a in refs.iterrows():
  root=ROOT/'network_variants/output'/f'{ex}_indep{"_direction130" if ex=="15" else ""}__anchored';m=json.loads((root/'states/S000_base/hydraulic_context/hydraulic_context_manifest.json').read_text());widths=pyogrio.read_dataframe(m['input_paths']['directed_links'],columns=['id_link','wid_adj_routing'],read_geometry=False);g=pd.read_parquet(BASE/ex/'floor_0.05/old/links.parquet').merge(widths,on='id_link',validate='one_to_one');g['example']=ex;g['depth_m']=a.stored_depth_m*(g.wid_adj_routing/a.reference_width_m)**.65;g['celerity_raw_mps']=5/(3*.035)*g.depth_m**(2/3)*np.sqrt(g.slope);g['effective_length_m']=np.maximum(200,g.actual_length_m/np.maximum(1,np.rint(g.actual_length_m/200)));frames.append(g)
 data=pd.concat(frames,ignore_index=True);data.to_csv(OUT/'all_base_links.csv',index=False);rows=[]
 lower=[0.,.05,.075,.1,.125,.15,.175,.2,.25,.3,.35,.4,.45,.5,.6,.75,1.]
 upper=[.5,.65,1.,1.5,2.,2.5,2.524,3.,3.5,4.,4.5,5.,6.,8.,10.,np.inf]
 for lo in lower:
  for hi in upper:
   if hi<=lo:continue
   for ex,g in data.groupby('example'):
    raw=g.celerity_raw_mps.to_numpy();c=np.clip(raw,lo,hi);K=g.effective_length_m.to_numpy()/c;L=g.actual_length_m.to_numpy();lb=.2*K;ub=1.8*K;dt,slow,fast=interval_conflicts(lb,ub);clipped=~np.isclose(raw,c,rtol=1e-12,atol=1e-12);exceptions=slow|fast
    rows.append(dict(example=ex,cap_min=lo,cap_max=hi,n_links=len(g),channel_length_m=L.sum(),clipped_count=int(clipped.sum()),clipped_length_m=float(L[clipped].sum()),lower_clipped_count=int((raw<lo).sum()),upper_clipped_count=int((raw>hi).sum()),feasible=bool(not exceptions.any()),minimum_exception_count=int(exceptions.sum()),slow_exceptions=int(slow.sum()),fast_exceptions=int(fast.sum()),exception_length_m=float(L[exceptions].sum()),diagnostic_dt_s=dt,lower_dt_s=float(lb.max()),upper_dt_s=float(ub.min()),affected_union_count=int((clipped|exceptions).sum()),affected_union_length_m=float(L[clipped|exceptions].sum()),max_speed_change_factor=float(np.maximum(raw/c,c/raw).max())))
 d=pd.DataFrame(rows);d.to_csv(OUT/'scenarios.csv',index=False);s=d.groupby(['cap_min','cap_max']).agg(passing=('feasible','sum'),clipped_links=('clipped_count','sum'),lower_clipped=('lower_clipped_count','sum'),upper_clipped=('upper_clipped_count','sum'),clipped_length_m=('clipped_length_m','sum'),exceptions=('minimum_exception_count','sum'),max_exceptions_one_case=('minimum_exception_count','max'),affected_union=('affected_union_count','sum'),max_speed_change_factor=('max_speed_change_factor','max')).reset_index();s['clipped_length_pct']=100*s.clipped_length_m/data.actual_length_m.sum();s.to_csv(OUT/'summary.csv',index=False);print('DONE',len(data),'links',len(s),'common cap pairs',flush=True)
 print(s[((s.cap_min.isin([0,.05,.1,.2,.3,.5]))&(s.cap_max.isin([2.5,3.,4.,5.,np.inf])))].to_string(index=False))
 print('FEWEST ALL PASS',s[s.passing==16].sort_values('clipped_links').head(8).to_string(index=False));print('LOWEST UNION',s.sort_values(['affected_union','clipped_links']).head(10).to_string(index=False))
if __name__=='__main__':main()
