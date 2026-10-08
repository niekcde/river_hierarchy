"""Targeted reference-flow sensitivity, isolated from live hierarchy runs.

Compare each affected child with (1) its actual width-based allocation,
(2) parent Q retained only in shared-link X coefficients (diagnostic only),
and (3) conserved parent shared-link flow where uniquely feasible.
The third changes routing shares and steady flows consistently; infeasible
cases are explicitly retained as infeasible, never patched by dropping mass.
"""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src')]
from rapid_tools import cunge
from rapid_tools.k_values import compute_dt_bounds
OUT=ROOT/'RAPID/outputs/cunge_reference_flow_sensitivity_20261006'
AUDIT=ROOT/'RAPID/outputs/cunge_readiness_audit_20261006'
ASSETS=ROOT/'RAPID/assets/bounded_robust_slopes_20261002/child_head_reconstruction_20261003'
KEYS=['u','v','link_length_m','width','slope','depth_m']

def conserved(parent,child):
 a=parent.copy();a['_pi']=np.arange(len(a));b=child.copy();b['_ci']=np.arange(len(b));shared=a.merge(b,on=KEYS,suffixes=('_p','_c'))
 assert shared._ci.is_unique and shared._pi.is_unique
 _,_,root,end=cunge.topology(child)
 nodes=list(dict.fromkeys(child.u.tolist()+child.v.tolist()));index={u:i for i,u in enumerate(nodes)}
 A=np.zeros((len(nodes),len(child)))
 for i,r in enumerate(child.itertuples()):A[index[r.u],i]=1;A[index[r.v],i]=-1
 fixed=shared._ci.to_numpy(int);q=np.zeros(len(child));q[fixed]=shared.reference_discharge_cms_p.to_numpy();unknown=np.setdiff1d(np.arange(len(child)),fixed)
 total=parent.loc[parent.u.eq(root),'reference_discharge_cms'].sum();rhs=np.zeros(len(nodes));rhs[index[root]]=total;rhs[index[end]]=-total;rhs-=A[:,fixed]@q[fixed]
 mat=A[:,unknown]
 if len(unknown):sol,_,rank,_=np.linalg.lstsq(mat,rhs,rcond=None);q[unknown]=sol
 else:rank=0
 residual=float(np.max(abs(A@q-(rhs+A[:,fixed]@q[fixed]))))
 info=dict(unknown_edges=len(unknown),rank=int(rank),max_balance_residual_cms=residual,min_solved_q_cms=float(q.min()),unique=bool(rank==len(unknown)))
 feasible=rank==len(unknown) and residual<1e-8*max(1,total) and q.min()>0
 info['feasible']=bool(feasible)
 info['positive_balanced_solution_found']=bool(residual<1e-8*max(1,total) and q.min()>0)
 if not feasible:return None,shared,info
 result=child.copy();result['reference_discharge_cms']=q;result['reference_flow_fraction']=q/total
 for u,rows in result.groupby('u'):result.loc[rows.index,'split_fraction']=rows.reference_discharge_cms/rows.reference_discharge_cms.sum()
 return result,shared,info

def held_splits(parent,child,shared):
 """Retain parent shares of surviving edges at surviving bifurcations.

 Changed/new outgoing edges receive the remainder in their child width ratio.
 Removed bifurcations cannot retain a former fractional share: the only
 remaining branch must carry all incoming flow. Recompute Q throughout DAG.
 """
 result=child.copy();changes=[]
 # Include tiny width/depth re-evaluation differences when the directed
 # endpoints, length and slope are exactly preserved. This captures a
 # bifurcation where BOTH branch widths changed, as in example 22.
 a=parent.copy();a['_pi']=np.arange(len(a));b=child.copy();b['_ci']=np.arange(len(b))
 matches=a.merge(b,on=['u','v','link_length_m','slope'],suffixes=('_p','_c'))
 matches=matches[np.isclose(matches.width_p,matches.width_c,rtol=1e-5,atol=0)&np.isclose(matches.depth_m_p,matches.depth_m_c,rtol=1e-5,atol=0)]
 assert matches._ci.is_unique and matches._pi.is_unique
 for u,rows in child.groupby('u'):
  old=parent[parent.u.eq(u)]
  if len(rows)<2 or len(old)<2:continue
  match=matches[matches.u.eq(u)]
  if match.empty:continue
  fixed=match._ci.to_numpy(int);other=rows.index.difference(fixed)
  values=match.split_fraction_p.to_numpy();remaining=1-float(values.sum())
  if len(other)==0:
   if not np.isclose(remaining,0,atol=1e-12):continue
  elif remaining<=0:continue
  before=result.loc[rows.index,'split_fraction'].to_numpy().copy()
  result.loc[fixed,'split_fraction']=values
  if len(other):result.loc[other,'split_fraction']=remaining*child.loc[other,'width']/child.loc[other,'width'].sum()
  after=result.loc[rows.index,'split_fraction'].to_numpy()
  if np.max(abs(before-after))>1e-14:changes.append(dict(node=int(u),before=before.tolist(),after=after.tolist()))
 order,edges,root,end=cunge.topology(result);flow={root:1.};allocation=np.zeros(len(result))
 for u in order:
  ids=edges.get(u,[])
  if ids:assert np.isclose(result.iloc[ids].split_fraction.sum(),1,atol=1e-12,rtol=0)
  for i in ids:
   allocation[i]=flow[u]*result.iloc[i].split_fraction
   v=result.iloc[i].v;flow[v]=flow.get(v,0.)+allocation[i]
 assert np.isclose(flow[end],1,atol=1e-12,rtol=0)
 total=parent.loc[parent.u.eq(root),'reference_discharge_cms'].sum()
 result['reference_flow_fraction']=allocation;result['reference_discharge_cms']=allocation*total
 return result,changes

def settled(flow,q,event_last,step):
 t=np.arange(len(q))*step;amp=max(float(q.max()-max(flow[0],flow[-1])),1e-12)
 last=np.flatnonzero((t>=event_last)&(abs(q-flow[-1])>.001*amp));j=int(last[-1]+1) if len(last) else int(round(event_last/step))
 return j<len(t) and t[j]+172800<=t[-1]

def main():
 OUT.mkdir(parents=True,exist_ok=True);cal=pd.read_csv(ROOT/'RAPID/outputs/cunge_calibration_alpha2/comparison.csv',dtype={'example':str});cal=cal[cal.method.eq('cunge')].set_index('example');results=[];metadata=[];aliases=[]
 for ex in ['13','15','22','54','56','41','46']:
  inv=pd.read_csv(ASSETS/ex/'inventory.csv');audit=pd.read_csv(AUDIT/ex/'exterior_invariance.csv');bad=audit[~audit.unchanged_parameters];source=Path(cal.loc[ex,'output_dir']);m=json.loads((source/'case.json').read_text());old=pd.read_parquet(source/'cunge_best_wave.parquet');step=m['forcing_step_s'];n=float(cal.loc[ex,'n_manning']);done=set()
  for row in bad.itertuples():
   childrow=inv[inv.family.eq(row.family)&inv.state.eq(row.state)].iloc[0];p=inv[inv.family.eq(row.family)&inv.state.eq(row.parent_state)]
   if p.empty:p=inv[inv.state.eq(row.parent_state)].drop_duplicates('fingerprint')
   assert len(p)==1
   parentrow=p.iloc[0];key=parentrow.fingerprint+'_'+childrow.fingerprint
   aliases.append(dict(example=ex,family=row.family,state=row.state,parent_state=row.parent_state,key=key))
   if key in done:continue
   done.add(key);dest=OUT/ex/key;dest.mkdir(parents=True,exist_ok=True)
   parent=cunge.freeze_grid(pd.read_parquet(parentrow.frame),m['reference_discharge_cms']);child=cunge.freeze_grid(pd.read_parquet(childrow.frame),m['reference_discharge_cms'])
   held,shared,info=conserved(parent,child)
   reference_only=child.copy();reference_only.loc[shared._ci,'reference_discharge_cms']=shared.reference_discharge_cms_p.to_numpy()
   # Same grid for all child variants, sufficient at the fastest permitted n.
   variants={'actual':child,'X_only':reference_only}
   split_held,split_changes=held_splits(parent,child,shared)
   info['held_bifurcation_splits']=split_changes
   if split_changes:variants['parent_bifurcation_splits']=split_held
   if held is not None:variants['conserved_parent_flow']=held
   count=np.maximum.reduce([np.maximum(1,np.ceil(g.link_length_m/(2*g.reference_discharge_cms/(g.width*g.slope*np.clip(g.celerity_times_n/.01,.05,5))))).astype(int).to_numpy() for g in variants.values()])
   frames={}
   for label,g in variants.items():
    g=g.copy();g['count']=count;g['rapid_effective_length_m']=g.link_length_m/count;frames[label]=cunge.parameters(g,n)
   frames['parent']=cunge.parameters(parent,n)
   upper=min(compute_dt_bounds(f.rapid_k,x_value=f.rapid_x,coefficient_policy='signed_inflow')[1] for f in frames.values());nsub=int(np.ceil(step/min(15,.99*upper)));dt=step/nsub
   flow=old.q_inflow_cms.to_numpy();event_last=(pd.Timestamp(m['last_sample'])-pd.Timestamp(m['first_sample'])).total_seconds();tail_start=int(round(event_last/step))+1
   for attempt in range(7):
    waves={};audits={}
    for label,f in frames.items():waves[label],audits[label]=cunge.route(f,flow,nsub,dt)
    if all(settled(flow,q,event_last,step) for q in waves.values()):break
    flow=np.r_[flow,np.full(max(len(flow)-tail_start,int(np.ceil(3*86400/step))),flow[-1])]
   else:raise ValueError('No common settled horizon: '+key)
   t=np.arange(len(flow))*step;actual=waves['actual'];amplitude=max(np.ptp(flow),1e-12);basepeak=int(flow.argmax());records=[]
   for label,q in waves.items():
    rec=dict(example=ex,state=row.state,family=row.family,key=key,variant=label,n_manning=n,flow_constraints_feasible=info['feasible'],constraint_balance_residual_cms=info['max_balance_residual_cms'],dt_s=dt,simulation_days=t[-1]/86400,
      lag_h=(q.argmax()-basepeak)*step/3600,peak_cms=float(q.max()),minimum_native_flow_cms=audits[label]['minimum_native_flow_cms'],tail_settled=True,
      max_abs_outlet_difference_cms=float(np.max(abs(q-actual))),max_wave_difference_pct_inlet_amplitude=float(np.max(abs(q-actual))/amplitude*100),
      peak_difference_pct=100*(float(q.max()/actual.max())-1),peak_time_difference_s=float((q.argmax()-actual.argmax())*step),
      outlet_rmse_difference_cms=float(np.sqrt(np.mean((q-actual)**2))),grid_changed_for_comparison=bool(np.any(count!=child['count'].to_numpy())))
    rec['screening_pass']=bool(rec['minimum_native_flow_cms']>=0 and rec['max_wave_difference_pct_inlet_amplitude']<=.01 and abs(rec['peak_difference_pct'])<=.01 and rec['peak_time_difference_s']==0) if label not in ['actual','parent'] else None
    records.append(rec)
   df=pd.DataFrame(dict(time_seconds=t,time_utc=pd.Timestamp(m['first_sample'])+pd.to_timedelta(t,unit='s'),inflow_cms=flow,**waves));df.to_parquet(dest/'waveforms.parquet',index=False)
   (dest/'constraint_check.json').write_text(json.dumps(info,indent=2));results.extend(records)
   metadata.append(dict(example=ex,key=key,parent_frame=str(parentrow.frame),child_frame=str(childrow.frame),parent_sha256=hashlib.sha256(Path(parentrow.frame).read_bytes()).hexdigest(),child_sha256=hashlib.sha256(Path(childrow.frame).read_bytes()).hexdigest(),source_calibration=str(source),shared_links=len(shared),constraint_check=info))
   pd.DataFrame(results).to_csv(OUT/'sensitivity_results.csv',index=False);pd.DataFrame(aliases).to_csv(OUT/'state_aliases.csv',index=False)
   print(json.dumps(dict(example=ex,state=row.state,conserved_feasible=info['feasible'],variants=[{k:r[k] for k in ['variant','max_wave_difference_pct_inlet_amplitude','peak_time_difference_s','screening_pass']} for r in records if r['variant'] not in ['actual','parent']])),flush=True)
 (OUT/'provenance.json').write_text(json.dumps(dict(sources=metadata,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),criteria='Wave difference <=0.01% inlet amplitude; peak magnitude difference <=0.01%; zero saved-resolution peak-time change; positive native flow. Diagnostic screen, not measured uncertainty.',scope='No active-run inputs or routing policies changed. X-only is diagnostic, not a proposed production model.'),indent=2))
 print('DONE',flush=True)

if __name__=='__main__':main()
