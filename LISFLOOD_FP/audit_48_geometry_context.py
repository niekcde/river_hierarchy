"""Read-only cross-network geometry/steady-hydraulics audit; no rerouting."""
from pathlib import Path
import json
import numpy as np,pandas as pd
import broad_study as b
r=b.OUT/'investigation';out=r/'geometry_context48';out.mkdir(exist_ok=True)
def wquant(x,w,p):
 order=np.argsort(x);x=np.asarray(x)[order];w=np.asarray(w)[order];return float(np.interp(p,np.cumsum(w)/sum(w),x))
rows=[];reachrows=[];sources={}
for ex in sorted(b.CASES):
 path=r/'current_mc/states'/ex/'S000_base/parameters.parquet';f=pd.read_parquet(path);L=f.link_length_m.to_numpy();W=f.width.to_numpy();S=((f.head_us-f.head_ds)/f.link_length_m).to_numpy();cal=json.loads((b.OUT/ex/'calibration.json').read_text());n=cal['selected_n'];cells=b.v.grid(b.frame(ex,'S000_base'),'max100');m=b.build(cells,n);ini=dict(np.load(Path(cal['final']['folder'])/'initial.npz'));h=ini['h'];q=ini['q'];N=len(h);eta=m['z'][:N]+h;js=np.where(m['b']<N)[0];hf=np.maximum(np.maximum(eta[m['a'][js]],eta[m['b'][js]])-np.maximum(m['z'][m['a'][js]],m['z'][m['b'][js]]),1e-12);fr=abs(q[js])/(hf*np.sqrt(9.80665*hf));cellfr=np.zeros(N);np.maximum.at(cellfr,m['a'][js],fr)
 row=dict(example=ex,reaches=len(f),total_km=sum(L)/1000,width_min_m=min(W),width_median_m=float(np.median(W)),width_length_median_m=wquant(W,L,.5),width_max_m=max(W),slope_median_m_per_km=float(np.median(S))*1000,slope_length_median_m_per_km=wquant(S,L,.5)*1000,slope_length_p90_m_per_km=wquant(S,L,.9)*1000,slope_max_m_per_km=max(S)*1000,length_share_slope_gt_2mkm=float(sum(L[S>.002])/sum(L)),length_share_slope_gt_3mkm=float(sum(L[S>.003])/sum(L)),longest_reach_km=max(L)/1000,mc_n=float(f.n_manning.iloc[0]),local_strict_n=n,calibration_status=cal['status'],initial_depth_length_median_m=wquant(h,m['dx'],.5),initial_face_fr_max=float(max(fr)),initial_length_share_outgoing_fr_gt06=float(sum(m['dx'][cellfr>.6])/sum(m['dx'])),event_max_fr=cal['final'].get('maximum_face_froude'),zero_or_negative_widths=int(sum(W<=0)),zero_or_negative_slopes=int(sum(S<=0)),max_slope_column_difference=float(max(abs(S-f.slope.to_numpy()))))
 # Endpoint head mismatch and adjacent total widths (whole junction, not one branch).
 nodeheads={};jrat=[]
 for z in f.itertuples():nodeheads.setdefault(z.u,[]).append(z.head_us);nodeheads.setdefault(z.v,[]).append(z.head_ds)
 row['max_node_head_spread_m']=max(max(v)-min(v) for v in nodeheads.values())
 for node in set(f.u)&set(f.v):
  incoming=f[f.v==node];outgoing=f[f.u==node];ratio=outgoing.width.sum()/incoming.width.sum();jrat.append(max(ratio,1/ratio))
 row['junction_total_width_ratio_max']=max(jrat);row['junction_total_width_ratio_p90']=float(np.quantile(jrat,.9));rows.append(row)
 Q=float(b.forcing(ex).forcing_cms.iloc[0]);ncommon=.020799605931143154
 for i,z in enumerate(f.itertuples()):
  Qref=z.reference_discharge_cms;hu=(Qref*ncommon/(z.width*np.sqrt(S[i])))**.6;fn=Qref/z.width/hu/np.sqrt(9.80665*hu)
  reachrows.append(dict(example=ex,id_link=z.id_link,length_m=L[i],width_m=W[i],slope_m_per_km=S[i]*1000,head_us=z.head_us,head_ds=z.head_ds,mc_reference_q_cms=Qref,mc_reference_depth_m=z.depth_m,common_n_normal_depth_m=hu,common_n_normal_fr=fn))
 sources[str(path)]=b.sha(path)
pd.DataFrame(rows).to_parquet(out/'network_comparison.parquet',index=False);pd.DataFrame(reachrows).to_parquet(out/'reach_comparison.parquet',index=False);b.save(out/'network_comparison.json',rows);b.save(out/'source_hashes.json',sources)
print(pd.DataFrame(rows)[['example','width_length_median_m','slope_length_median_m_per_km','slope_length_p90_m_per_km','longest_reach_km','local_strict_n','initial_depth_length_median_m','initial_length_share_outgoing_fr_gt06','event_max_fr']].round(4).to_string(index=False))
checks=json.loads((b.OUT/'calibrated_grid_checks.json').read_text());print('CALIBRATED GRID CHECKS',[(x['example'],x['valid']) for x in checks]);print('48',json.dumps(next(x for x in rows if x['example']=='48'),indent=2))
