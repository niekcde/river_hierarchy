"""Read-only frozen compound floor audit; never writes production assets."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/investigations/member_boundary_review_20261027')]
import numpy as np,pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from feasibility import head_series,path_constraints
from network_variants.reviewed_geometry import ReviewedGeometry
OUT=ROOT/'RAPID/outputs/frozen_floor_review_20261027'
PRIOR=ROOT/'RAPID/outputs/member_boundary_review_20261027_complete'
BASE=ROOT/'RAPID/assets/bounded_robust_slopes_20261002'
release=ReviewedGeometry();hashes={}
def track(p):
 p=Path(p);hashes.setdefault(str(p),hashlib.sha256(p.read_bytes()).hexdigest());return p
def frame(p):return pd.read_parquet(track(p))
def js(p):return json.loads(track(p).read_text())
def outside_paths(record,parent,child,selected):
 path=Path(record.get('source_bundle') or '')/'exterior_paths.json'
 if record.get('source_bundle') and path.exists():return js(path)
 groups={(u,v):g for (u,v),g in child.groupby(['u','v'])};paths=[]
 for p in parent[~parent.id_link.isin(selected)].itertuples():
  g=groups.get((p.u,p.v),child.iloc[:0]);g=g[np.isclose(g.link_length_m,p.link_length_m,rtol=1e-8,atol=1e-5)]
  if len(g)>1 and p.id_link in set(g.id_link):g=g[g.id_link.eq(p.id_link)]
  paths.append(dict(parent_link=int(p.id_link),child_links=g.id_link.astype(int).tolist(),path_unique=len(g)==1))
 return paths

def lp_check(edges,heads,fixed,drop):
 free=sorted((set(edges.u)|set(edges.v))-fixed);idx={v:i for i,v in enumerate(free)}
 A=lil_matrix((len(edges),len(free)));b=-np.array(drop)
 for j,r in enumerate(edges.itertuples()):
  for v,s in [(r.u,1),(r.v,-1)]:
   if v in idx:A[j,idx[v]]=-s
   else:b[j]+=s*heads.loc[v]
 if not free:return bool((b>=-1e-9).all())
 res=linprog(np.zeros(len(free)),A_ub=A.tocsr(),b_ub=b,bounds=[(None,None)]*len(free),method='highs',options={'primal_feasibility_tolerance':1e-9})
 if res.success:assert np.max(A@res.x-b)<1e-8
 return bool(res.success)

def main():
 OUT.mkdir(exist_ok=False);rows=[];witness=[];cache={};baseline=[];lps=[]
 states=js(PRIOR/'state_results.json');exceptions=[]
 for i,z in enumerate(states):
  ex=str(z['example']).zfill(2);state=z['state'];r=release.lookup(ex,'indep',state)
  if z['feasibility_status'].startswith('not_tested'):exceptions.append(z);continue
  if ex not in cache:
   p=frame(release.lookup(ex,'indep','S000_base')['original_frame']);old=frame(BASE/ex/'floor_0.05/old/links.parquet').set_index('id_link');h=head_series(p)
   pos=frame(BASE/ex/'floor_0.05/old/nodes.parquet').set_index('id_node').reference_section_dist_out_m
   units=pd.read_csv(track(BASE/ex/'floor_0.05/old/units.csv'))
   for q in p.itertuples():
    low=old.loc[q.id_link,'lower_bound']
    if np.isfinite(low) and q.slope<low:baseline.append(dict(example=ex,id_link=int(q.id_link),floor=float(low),reference_slope=float(q.slope),drop_deficit_m=float((low-q.slope)*q.link_length_m)))
   cache[ex]=p,old,h,pos,units
  p,old,h,pos,units=cache[ex];c=js(r['contract_path']);f=frame(r['original_frame']);selected=c['selected_parent_links'];o=old.loc[selected]
  floors=o.lower_bound.unique();domains=o[['ownership_kind','ownership_domain_id']].drop_duplicates()
  assert len(floors)==1 and np.isfinite(floors[0]) and len(domains)==1,(ex,state)
  floor=float(floors[0]);domain=domains.iloc[0];u=units[(units.kind==domain.ownership_kind)&(units.domain==domain.ownership_domain_id)]
  assert len(u)==1,(ex,state,domains.to_dict(),units.columns)
  u=u.iloc[0];ref=(h.loc[u.source]-h.loc[u.sink])/(pos.loc[u.source]-pos.loc[u.sink]);assert np.isclose(floor,.01*ref,rtol=1e-7,atol=1e-12),(ex,state,floor,ref)
  paths=outside_paths(r,p,f,selected);outside={lid:v['parent_link'] for v in paths for lid in v['child_links']};patch=f[~f.id_link.isin(outside)];ext=p[~p.id_link.isin(selected)];fixed=set(ext.u)|set(ext.v)|set(c['boundary_nodes'])
  weights=np.maximum(patch.link_length_m.to_numpy()*floor,1e-10)
  checks=path_constraints(patch,h.loc[sorted(fixed&set(h.index))],weights)
  bad=[v for v in checks if v['available_drop_m']<v['path_weight']-1e-9]
  pi=p.set_index('id_link');exbad=[]
  for q in f[f.id_link.isin(outside)].itertuples():
   lid=outside[q.id_link];lo=old.loc[lid,'lower_bound'];lo=min(lo,pi.loc[lid,'slope']) if np.isfinite(lo) else 0.
   req=max(lo*q.link_length_m,1e-10);available=h.loc[q.u]-h.loc[q.v]
   if available<req-1e-9:exbad.append(dict(child_link=int(q.id_link),parent_link=int(lid),required_drop_m=float(req),available_drop_m=float(available)))
  status='feasible' if not bad else 'nonpositive_fixed_head_path' if any(v['available_drop_m']<=1e-10 for v in bad) else 'frozen_floor_infeasible'
  row=dict(example=ex,state=state,previous_status=z['feasibility_status'],frozen_status=status,compound=int(domain.ownership_domain_id),floor=floor,reference_slope=float(ref),replacement_links=len(patch),fixed_fixed_links=int((patch.u.isin(fixed)&patch.v.isin(fixed)).sum()),exterior_floor_conflicts=len(exbad))
  rows.append(row)
  if bad or exbad:witness.append(dict(**row,conflicts=bad,exterior_conflicts=exbad))
  if bad or z['feasibility_status']!='feasible' or i%40==0:
   result=lp_check(patch,h,fixed,weights);assert result==(not bad),(ex,state);lps.append(dict(example=ex,state=state,lp_feasible=result))
  if i%150==0:print(i,'cases inspected',flush=True)
 pd.DataFrame(rows).to_csv(OUT/'floor_checks.csv',index=False)
 for name,data in [('conflicts',witness),('exceptions',exceptions),('reference_floor_residuals',baseline),('independent_lp_checks',lps)]: (OUT/(name+'.json')).write_text(json.dumps(data,indent=2))
 transitions=pd.DataFrame(rows).groupby(['previous_status','frozen_status']).size().reset_index(name='cases');transitions.to_csv(OUT/'transitions.csv',index=False)
 for path,digest in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
 summary=dict(tested=len(rows),exceptions=len(exceptions),transitions=transitions.to_dict('records'),exterior_conflict_cases=sum(v['exterior_floor_conflicts']>0 for v in rows),lp_checks=len(lps),baseline_residuals=len(baseline),max_baseline_drop_deficit_m=max(v['drop_deficit_m'] for v in baseline),source_hashes=hashes)
 (OUT/'manifest.json').write_text(json.dumps(summary,indent=2));print({k:v for k,v in summary.items() if k!='source_hashes'})
if __name__=='__main__':main()
