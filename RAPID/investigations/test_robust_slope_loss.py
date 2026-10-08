"""Controlled Huber-loss test: existing selected lambda, floor and allocation fixed."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from scipy import sparse
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parent))
import smoothing_floor_allocation_tests as base
ROOT=base.ROOT;PREV=base.OUT;OUT=ROOT/'RAPID/assets/robust_slope_loss_20261002';OUT.mkdir(exist_ok=True,parents=True)
def median(v,w):
 i=np.argsort(v);return np.interp(.5*np.sum(w),np.cumsum(w[i]),v[i])
def fit(f,o,lam,ref,bridges):
 x=f.reference_section_dist_out_m.to_numpy();n=len(x);acqs=pd.Index(sorted(o.acquisition_id.unique()));ai=acqs.get_indexer(o.acquisition_id);ni=np.searchsorted(x,o.reference_section_dist_out_m);N=len(o)
 M=sparse.coo_matrix((np.ones(2*N),(np.r_[np.arange(N),np.arange(N)],np.r_[ni,n+ai])),shape=(N,n+len(acqs))).tocsr();center=float(f.fitted_wse_m.median());y=o.wse_m.to_numpy()-center;w=N/(o.cycle.nunique()*o.groupby('cycle').wse_m.transform('size').to_numpy())
 penalty=sparse.block_diag((2*lam*base._curvature_penalty(x),sparse.csr_matrix((len(acqs),len(acqs)))))
 D=sparse.hstack([sparse.diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n)),sparse.csr_matrix((n-1,len(acqs)))])
 gauge=sparse.csr_matrix((np.ones(len(acqs))/len(acqs),(np.zeros(len(acqs)),n+np.arange(len(acqs)))),shape=(1,n+len(acqs)));A=sparse.vstack([D,gauge]);lower=np.r_[.05*ref*np.diff(x),0];upper=np.r_[np.full(n-1,np.inf),0]
 residual=o.wse_m.to_numpy()-np.interp(o.reference_section_dist_out_m,x,f.fitted_wse_m);offset=pd.Series(residual).groupby(ai).mean().reindex(range(len(acqs))).to_numpy();m=offset.mean();sol=np.r_[f.fitted_wse_m-center+m,offset-m];r=y-M@sol;scale=max(1e-6,1.4826*median(abs(r-median(r,w)),w));delta=1.345*scale
 history=[]
 for iteration in range(400):
  r=y-M@sol;rw=np.minimum(1,delta/np.maximum(abs(r),1e-15));ww=w*rw
  new,qa=base.qp(2*M.T@sparse.diags(ww)@M+penalty,-2*M.T@(ww*y),A,lower,upper,sol,tol=1e-6,maxiter=150000)
  step=float(np.max(abs(new-sol)));sol=new;r=y-M@sol;objective=float(np.sum(w*np.where(abs(r)<=delta,r*r,2*delta*abs(r)-delta*delta))+.5*sol@(penalty@sol));history.append(dict(iteration=iteration+1,step_m=step,objective=objective,**qa))
  if step<1e-4:break
 h=sol[:n]+center
 for b in bridges:
  lo=float(b.get('resolved_start_distance_m',b['requested_start_distance_m']));hi=float(b.get('resolved_end_distance_m',b['requested_end_distance_m']));il=np.argmin(abs(x-lo));ih=np.argmin(abs(x-hi));h[il:ih+1]=np.interp(x[il:ih+1],[x[il],x[ih]],[h[il],h[ih]])
 return h,dict(scale_m=scale,huber_delta_m=delta,iterations=len(history),converged=step<1e-4,final_step_m=step,downweighted_fraction=float(np.mean(abs(r)>delta)),history=history)
def run(ex):
 source=Path(base.sources[ex]);f=pd.read_parquet(PREV/ex/'floor_0.05/profile.parquet');o=pd.read_parquet(source/'reference_profile_observations.parquet');meta=json.loads((source/'reference_profile_audit.json').read_text())['summary'];bridges=json.loads((source/'reference_profile_selection.json').read_text()).get('manual_adjustments',[]);selection=json.loads((PREV/ex/'floor_0.05/selection.json').read_text());lam=selection['selected_lambda'];h,qa=fit(f,o,lam,meta['reference_slope_m_per_m'],bridges)
 folder=OUT/ex/'floor_0.05';folder.mkdir(parents=True,exist_ok=True);p=f.copy();p['previous_candidate']=f.fitted_wse_m;p['fitted_wse_m']=h;p['slope']=np.r_[np.nan,np.diff(h)/np.diff(f.reference_section_dist_out_m)];p.to_parquet(folder/'profile.parquet',index=False);(folder/'solver.json').write_text(json.dumps(qa,indent=2))
 suffix='_direction130' if ex=='15' else '';hc=ROOT/'network_variants/output'/f'{ex}_indep{suffix}__anchored/states/S000_base/hydraulic_context';nodes=pd.read_csv(hc/'node_hydraulic_context.csv');links=pd.read_csv(hc/'link_hydraulic_context.csv');base.OUT=OUT;allocation=base.units(ex,.05,f,h,links,nodes,'old');x=f.reference_section_dist_out_m.to_numpy();ends=[nodes.reference_section_dist_out_m.min(),nodes.reference_section_dist_out_m.max()]
 row=dict(example=ex,lambda_value=lam,**{k:v for k,v in qa.items() if k!='history'},**{k:v for k,v in allocation.items() if k!='example'},old_drop=float(np.diff(np.interp(ends,x,f.fitted_wse_m))[0]),new_drop=float(np.diff(np.interp(ends,x,h))[0]),old_rmse=float(np.sqrt(base.score(o,x,f.fitted_wse_m).mean())),new_rmse=float(np.sqrt(base.score(o,x,h).mean())))
 (folder/'summary.json').write_text(json.dumps(row,indent=2));print(ex,'DONE',qa['iterations'],qa['converged'],allocation['below_1e5'],flush=True);return row
if __name__=='__main__':
 with ProcessPoolExecutor(max_workers=2) as pool:rows=list(pool.map(run,base.sources))
 pd.DataFrame(rows).to_csv(OUT/'summary.csv',index=False);(OUT/'method.json').write_text(json.dumps(dict(sources=base.sources,method='Huber IRLS; delta=1.345 times equal-cycle weighted MAD scale of initial candidate residuals; scale fixed during fit; no per-case manual tuning',controls='Selected squared-loss lambda, 0.05 S_ref floor, equal-cycle base weights, existing bridges and old unit allocation held fixed',validation='Controlled loss-only experiment; no new CV selection or independent validation; no production modifications'),indent=2));print('ALL DONE')
