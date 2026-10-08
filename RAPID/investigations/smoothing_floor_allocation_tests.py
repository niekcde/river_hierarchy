"""All-16 controlled CV: smoothing x profile floor x unit allocation; isolated outputs."""
from pathlib import Path
import sys,json,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd,networkx as nx
from scipy import sparse
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'RAPID/investigations'),str(ROOT/'RAPID/src')]
from run_data_informed_base_slopes import qp,sources
from rapid_tools.reference_profile import _curvature_penalty
OUT=ROOT/'RAPID/assets/smoothing_floor_allocation_20261002';OUT.mkdir(parents=True,exist_ok=True)
FLOORS=[.01,.05]

def fit_profile(f,o,penalty,lam,eta,ref,weight_total,adjustments):
 x=f.reference_section_dist_out_m.to_numpy();n=len(x);acqs=pd.Index(sorted(o.acquisition_id.unique()));ai=acqs.get_indexer(o.acquisition_id);ni=np.searchsorted(x,o.reference_section_dist_out_m);N=len(o);nv=n+len(acqs)
 M=sparse.coo_matrix((np.ones(2*N),(np.r_[np.arange(N),np.arange(N)],np.r_[ni,n+ai])),shape=(N,nv)).tocsr()
 center=float(f.fitted_wse_m.median());y=o.wse_m.to_numpy()-center;w=weight_total/(o.cycle.nunique()*o.groupby('cycle').wse_m.transform('size').to_numpy())
 P=2*M.T@sparse.diags(w)@M+sparse.block_diag((2*lam*penalty,sparse.csr_matrix((len(acqs),len(acqs)))))
 q=-2*M.T@(w*y);D=sparse.hstack([sparse.diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n)),sparse.csr_matrix((n-1,len(acqs)))])
 gauge=sparse.csr_matrix((np.ones(len(acqs))/len(acqs),(np.zeros(len(acqs)),n+np.arange(len(acqs)))),shape=(1,nv))
 A=sparse.vstack([D,gauge]);lower=np.r_[eta*ref*np.diff(x),0.];upper=np.r_[np.full(n-1,np.inf),0.]
 offsets=o.groupby('acquisition_id').acquisition_stage_offset_m.first().reindex(acqs).to_numpy();m=offsets.mean();initial=np.r_[f.fitted_wse_m-center+m,offsets-m]
 sol,audit=qp(P,q,A,lower,upper,initial,tol=1e-6,maxiter=150000)
 h=sol[:n]+center
 for bridge in adjustments:
  lo=float(bridge.get('resolved_start_distance_m',bridge['requested_start_distance_m']));hi=float(bridge.get('resolved_end_distance_m',bridge['requested_end_distance_m']))
  il=np.argmin(abs(x-lo));ih=np.argmin(abs(x-hi));h[il:ih+1]=np.interp(x[il:ih+1],[x[il],x[ih]],[h[il],h[ih]])
 assert np.all(np.diff(h)>0),'nonpositive fitted profile'
 return h,audit

def score(o,x,h):
 d=o[['cycle','acquisition_id','wse_m']].copy();d['res']=d.wse_m.to_numpy()-np.interp(o.reference_section_dist_out_m,x,h)
 d['res']-=d.groupby('acquisition_id').res.transform('mean');return d.assign(squared=d.res**2).groupby('cycle').squared.mean()

def units(ex,eta,f,h,links,nodes,method):
 x=f.reference_section_dist_out_m.to_numpy();gradient=np.diff(h)/np.diff(x);nh=pd.Series(np.interp(nodes.reference_section_dist_out_m,x,h),index=nodes.id_node.astype(int));base=nh.copy();pos=nodes.set_index('id_node').reference_section_dist_out_m;audits=[];bounds={}
 for (kind,did),g in links[links.ownership_kind.isin(['simple_unit','compound_bubble'])].groupby(['ownership_kind','ownership_domain_id']):
  G=nx.MultiDiGraph()
  for r in g.itertuples():G.add_edge(int(r.id_us_node),int(r.id_ds_node),key=int(r.id_link),length=r.actual_length_m)
  src=[v for v in G if G.in_degree(v)==0];snk=[v for v in G if G.out_degree(v)==0];assert len(src)==len(snk)==1 and nx.is_directed_acyclic_graph(G)
  src,snk=src[0],snk[0];drop=base[src]-base[snk];ref=drop/(pos[src]-pos[snk]);assert drop>0 and ref>0
  Ts=[]
  for r in g.itertuples():
   if method in ['old','maxmin_domain']:T=ref
   else:
    lo,hi=sorted([pos[r.id_us_node],pos[r.id_ds_node]]);w=np.maximum(0,np.minimum(x[1:],hi)-np.maximum(x[:-1],lo));T=float(np.average(gradient,weights=w)) if w.sum()>0 else float(gradient[min(len(gradient)-1,max(0,np.searchsorted(x,lo)-1))])
   Ts.append(T);G[r.id_us_node][r.id_ds_node][r.id_link]['weight']=T*r.actual_length_m
  dist={v:-np.inf for v in G};dist[src]=0
  for u in nx.topological_sort(G):
   for _,v,z in G.out_edges(u,data=True):dist[v]=max(dist[v],dist[u]+z['weight'])
  rho=drop/dist[snk] if method!='old' else .01
  vs=list(G);index={v:i for i,v in enumerate(vs)};initial=np.array([1-dist[v]/dist[snk] for v in vs])
  fixed=[index[src],index[snk]]
  if method!='old':
   tail={v:-np.inf for v in G};tail[snk]=0
   for u in reversed(list(nx.topological_sort(G))):
    for _,v,z in G.out_edges(u,data=True):tail[u]=max(tail[u],z['weight']+tail[v])
   fixed=[index[v] for v in vs if abs(dist[v]+tail[v]-dist[snk])<=1e-10*max(1,dist[snk])]
  free=[i for i in range(len(vs)) if i not in fixed];C=sparse.lil_matrix((len(g),len(vs)));scales=np.array(Ts)*g.actual_length_m.to_numpy()/drop
  for i,r in enumerate(g.itertuples()):C[i,index[r.id_us_node]]=1/scales[i];C[i,index[r.id_ds_node]]=-1/scales[i]
  C=C.tocsr();hh=initial.copy()
  if free:
   Cf=C[:,free];constant=C[:,fixed]@initial[fixed];A=sparse.diags(scales)@Cf;lower=(rho-constant)*scales
   values,qa=qp(2*Cf.T@Cf,2*Cf.T@(constant-1),A,lower,np.full(len(g),np.inf),initial[free],tol=1e-7,maxiter=150000);hh[free]=values
  else:qa=dict(iterations=0,primal_residual=0,dual_residual=0)
  hh[index[src]]=1;hh[index[snk]]=0
  for v in vs:
   if v not in [src,snk]:nh[v]=base[snk]+drop*hh[index[v]]
  for i,r in enumerate(g.itertuples()):bounds[r.id_link]=rho*Ts[i]
  audits.append(dict(kind=kind,domain=int(did),n_links=len(g),source=src,sink=snk,drop_m=drop,lower_fraction=rho,**qa))
 result=links[['id_link','id_us_node','id_ds_node','actual_length_m','ownership_kind','ownership_domain_id']].copy();result['old_slope']=links.slope_used;result['head_us']=result.id_us_node.map(nh);result['head_ds']=result.id_ds_node.map(nh);result['slope']=(result.head_us-result.head_ds)/result.actual_length_m;result['lower_bound']=result.id_link.map(bounds)
 assert (result.slope>0).all() and np.isfinite(result.slope).all()
 for a in audits:assert abs(nh[a['source']]-base[a['source']])<1e-7 and abs(nh[a['sink']]-base[a['sink']])<1e-7
 folder=OUT/ex/f'floor_{eta:.2f}'/method;folder.mkdir(parents=True,exist_ok=True);result.to_parquet(folder/'links.parquet',index=False)
 nd=nodes[['id_node','reference_section_dist_out_m']].copy();nd['profile_head']=nd.id_node.map(base);nd['head']=nd.id_node.map(nh);nd.to_parquet(folder/'nodes.parquet',index=False);pd.DataFrame(audits).to_csv(folder/'units.csv',index=False)
 return dict(example=ex,floor=eta,allocation=method,n_links=len(result),n_units=len(audits),below_1e5=int((result.slope<1e-5).sum()),below_1e6=int((result.slope<1e-6).sum()),min_slope=float(result.slope.min()),max_floor_violation_m=float(max(0,((result.lower_bound-result.slope)*result.actual_length_m).max())),head_residual_m=float(abs(result.slope*result.actual_length_m-(result.head_us-result.head_ds)).max()),status='complete')

def run(ex):
 p=Path(sources[ex]);f=pd.read_parquet(p/'reference_profile.parquet');o=pd.read_parquet(p/'reference_profile_observations.parquet');meta=json.loads((p/'reference_profile_audit.json').read_text())['summary'];selection=json.loads((p/'reference_profile_selection.json').read_text());bridges=selection.get('manual_adjustments',[])
 suffix='_direction130' if ex=='15' else '';hc=ROOT/'network_variants/output'/f'{ex}_indep{suffix}__anchored/states/S000_base/hydraulic_context';nodes=pd.read_csv(hc/'node_hydraulic_context.csv');links=pd.read_csv(hc/'link_hydraulic_context.csv');x=f.reference_section_dist_out_m.to_numpy();penalty=_curvature_penalty(x)
 cycles=np.array(sorted(o.cycle.unique()));rng=np.random.default_rng(20261002);rng.shuffle(cycles);foldmap={v:i%3 for i,v in enumerate(cycles)};fold=o.cycle.map(foldmap).to_numpy();out=OUT/ex;out.mkdir(exist_ok=True);(out/'folds.json').write_text(json.dumps({str(k):v for k,v in foldmap.items()},indent=2))
 rows=[];results=[];lam0=meta['lambda_value']
 for eta in FLOORS:
  directory=out/f'floor_{eta:.2f}';directory.mkdir(exist_ok=True)
  cache=directory/'cv.json';records=json.loads(cache.read_text()) if cache.exists() else {}
  def evaluate(lam):
   key=f'{lam:.12g}'
   if key in records:return records[key]
   scores=[];foldaud=[]
   try:
    for k in range(3):
     h,qa=fit_profile(f,o[fold!=k].copy(),penalty,lam,eta,meta['reference_slope_m_per_m'],len(o),bridges);cs=score(o[fold==k],x,h);scores.extend(cs.tolist());foldaud.append(qa)
    rec=dict(lambda_value=lam,cv_mse=float(np.mean(scores)),cv_se=float(np.std(scores,ddof=1)/np.sqrt(len(scores))),n_cycles=len(scores),fold_audits=foldaud,status='complete')
   except Exception as err:rec=dict(lambda_value=lam,cv_mse=None,status='failed',error=repr(err))
   records[key]=rec;cache.write_text(json.dumps(records,indent=2));print(ex,eta,'CV',key,rec['status'],rec.get('cv_mse'),flush=True);return rec
  grid=lam0*10.**np.array([-8,-6,-4,-2,0],float)
  for lam in grid:evaluate(float(lam))
  for _ in range(3):
   good=[v for v in records.values() if v['status']=='complete'];assert good,'no successful CV candidates';best=min(good,key=lambda v:v['cv_mse']);lo=min(v['lambda_value'] for v in good);hi=max(v['lambda_value'] for v in good)
   if best['lambda_value']==lo:evaluate(lo/100)
   elif best['lambda_value']==hi:evaluate(hi*100)
   else:break
  best=min([v for v in records.values() if v['status']=='complete'],key=lambda v:v['cv_mse'])
  for factor in [.1, .316227766,3.16227766,10.]:evaluate(best['lambda_value']*factor)
  good=[v for v in records.values() if v['status']=='complete'];best=min(good,key=lambda v:v['cv_mse'])
  # Resolve numerically flat CV minima in favor of the largest indistinguishable lambda.
  best=max([v for v in good if v['cv_mse']<=best['cv_mse']+1e-6*max(best['cv_mse'],1e-12)],key=lambda v:v['lambda_value']);lam=best['lambda_value']
  h,qa=fit_profile(f,o,penalty,lam,eta,meta['reference_slope_m_per_m'],len(o),bridges)
  profile=f[['reference_section_dist_out_m','static_wse_m']].copy();profile['old_profile']=f.fitted_wse_m;profile['fitted_wse_m']=h;profile['slope']=np.r_[np.nan,np.diff(h)/np.diff(x)];profile.to_parquet(directory/'profile.parquet',index=False)
  ends=[nodes.reference_section_dist_out_m.min(),nodes.reference_section_dist_out_m.max()];drop=np.diff(np.interp(ends,x,h))[0];oldrop=np.diff(np.interp(ends,x,f.fitted_wse_m))[0]
  chosen=dict(example=ex,floor=eta,selected_lambda=lam,old_lambda=lam0,lambda_ratio=lam/lam0,cv_mse=best['cv_mse'],cv_rmse=float(np.sqrt(best['cv_mse'])),cv_se=best['cv_se'],head_change_pct=100*(drop/oldrop-1),in_sample_cycle_rmse=float(np.sqrt(score(o,x,h).mean())),min_profile_slope=float(np.min(np.diff(h)/np.diff(x))),selected_grid_edge=lam in [min(v['lambda_value'] for v in good),max(v['lambda_value'] for v in good)],n_failed_cv=sum(v['status']=='failed' for v in records.values()),manual_bridges_retained=len(bridges),solver=qa)
  (directory/'selection.json').write_text(json.dumps(chosen,indent=2));rows.append(chosen)
  for method in ['old','maxmin','maxmin_domain']:
   try:results.append(units(ex,eta,f,h,links,nodes,method))
   except Exception as err:results.append(dict(example=ex,floor=eta,allocation=method,status='failed',error=repr(err)));print('UNIT FAIL',ex,eta,method,repr(err),flush=True)
  print('FINISHED',ex,eta,flush=True)
 pd.DataFrame([{k:v for k,v in z.items() if k!='solver'} for z in rows]).to_csv(out/'profiles.csv',index=False);pd.DataFrame(results).to_csv(out/'allocations.csv',index=False)
 return rows,results

def main():
 answers=[]
 with ProcessPoolExecutor(max_workers=2) as pool:
  jobs={ex:pool.submit(run,ex) for ex in sources}
  for ex,job in jobs.items():
   try:answers.append(job.result())
   except Exception as err:print('CASE FAILED',ex,repr(err),flush=True)
 pd.DataFrame([{k:v for k,v in r.items() if k!='solver'} for rows,_ in answers for r in rows]).to_csv(OUT/'profiles.csv',index=False);pd.DataFrame([r for _,rows in answers for r in rows]).to_csv(OUT/'allocations.csv',index=False)
 (OUT/'method.json').write_text(json.dumps(dict(floors=FLOORS,sources=sources,cv='3 whole-cycle folds, equal-cycle score after held-acquisition offset removal; fixed seed20261002; lambda chosen by minimum CV MSE; adaptive logarithmic search',weights='Equal total cycle weight; overall observation weight kept constant across folds',manual_bridges='Existing bridge applied to every fold and final fit',allocation='Old domain-wide target and 1% floor versus local-gradient target and graph maxmin lower envelope; both fixed to same profile boundary heads',production_changed=False),indent=2))
 print('ALL DONE',flush=True)
if __name__=='__main__':main()
