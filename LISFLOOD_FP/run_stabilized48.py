"""Controlled network adaptation of q-centred smoothing: diagnostic only."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,subprocess,time,hashlib
import numpy as np,pandas as pd
import broad_study as b
from run_positivity48 import AUDIT
root=b.OUT/'investigation';out=root/'stabilized48';out.mkdir(exist_ok=True);kernel=Path(__file__).with_name('inertial_stabilized_diagnostic_kernel.py')

def stencil(cells,m):
    N=len(cells);M=len(m['a']);incoming=[[] for _ in range(N)];outgoing=[[] for _ in range(N)];left=np.full(M,-1,dtype=np.int64);right=left.copy();reach=cells.id_link.to_numpy()
    for j,(a,k) in enumerate(zip(m['a'],m['b'])):
        outgoing[a].append(j)
        if k<N:incoming[k].append(j)
    for j,(a,k) in enumerate(zip(m['a'],m['b'])):
        if k==N or reach[a]!=reach[k] or len(incoming[a])!=1 or len(outgoing[k])!=1:continue
        l=incoming[a][0];rr=outgoing[k][0];u=m['a'][l];v=m['b'][rr]
        if v==N or reach[u]!=reach[a] or reach[v]!=reach[a]:continue
        if not np.allclose(m['wf'][[l,j,rr]],m['wf'][j],rtol=0,atol=1e-12):continue
        if not np.allclose(m['dxf'][[l,j,rr]],m['dxf'][j],rtol=0,atol=1e-9):continue
        left[j]=l;right[j]=rr
    return left,right

def solve(d,dest,label):
    inp=dest/(label+'_input.npz');op=dest/(label+'_output.npz');np.savez_compressed(inp,**d)
    p=subprocess.run(['/opt/anaconda3/bin/python',str(kernel),str(inp),str(op)],capture_output=True,text=True)
    if p.returncode:raise RuntimeError(p.stdout+p.stderr)
    return dict(np.load(op))

def run(length,theta,cfl=.35):
    tag=f'dx{length}_theta{theta:g}_cfl{cfl:g}';dest=out/tag;dest.mkdir(exist_ok=True)
    if (dest/'result.json').exists():return json.loads((dest/'result.json').read_text())
    start=time.monotonic();print('START',tag,flush=True)
    template=root/('limiter_retained_48_n0.0207996059'+('' if length==100 else '_dx50'));src=next(template.glob('_input*.npz'));d=dict(np.load(src));f=pd.read_parquet(root/'current_mc/states/48/S000_base/parameters.parquet');cells=b.v.grid(f,f'max{length}');m=b.build(cells,float(d['n']));left,right=stencil(cells,m);m.update(theta=theta,smooth_left=left,smooth_right=right)
    if length in [50,100]:assert all(np.array_equal(m[k],d[k]) for k in ['z','w','dx','a','b','wf','dxf'])
    else:d['h'],d['q']=b.initial_guess(f,cells,m,float(d['forcing'][0]))
    d.update(m);d['cfl']=cfl
    spin=[];Q=float(d['forcing'][0]);init=dict(d);init.update(times=np.array([0.,86400.]),forcing=np.array([Q,Q]),sample_indices=np.array([0,1]))
    # Re-equilibrate with the changed scheme, including for existing grids.
    for day in range(1,61):
        x=solve(init,dest,'spin');audit=dict(zip(AUDIT,map(float,x['audit'])));spin.append(audit);init.update(h=x['h'],q=x['q'])
        if audit['final_node_imbalance_cms']/Q<1e-8:break
    else:raise RuntimeError('No converged steady state '+tag)
    initial_drift=None if length not in [50,100] else float(max(abs(x['h']-d['h'])))
    d.update(h=x['h'],q=x['q']);x=solve(d,dest,'event');audit=dict(zip(AUDIT,map(float,x['audit'])));force=b.forcing('48');means=np.diff(x['wave'][:,4])/np.diff(d['times']);w=force[['time_utc','phase']].copy();w['q_outlet_cms']=means;w.to_parquet(dest/'wave.parquet',index=False)
    obs=pd.read_parquet(b.OUT/'inputs/48/observed_downstream.parquet');origin=force.interval_start_utc.iloc[0];mask=(w.time_utc>=obs.time.min())&(w.time_utc<=obs.time.max())&(w.time_utc>=force.loc[force.phase.eq('input_event'),'interval_start_utc'].iloc[0]);tt=(w.time_utc-origin).dt.total_seconds().to_numpy();tm,height,kind,_=b.feature(means[mask],tt[mask],'48',b.TARGETS['48']);target=(pd.Timestamp(b.TARGETS['48']['target_utc'])-origin).total_seconds()
    res=dict(tag=tag,length=length,theta=theta,cfl=cfl,n=float(d['n']),seconds=time.monotonic()-start,smoothed_faces=int(sum(left>=0)),total_faces=len(left),initial_max_depth_drift_m=initial_drift,spinup_audits=spin,audit=audit,timing_error_minutes=(tm-target)/60,feature_kind=kind,feature_height_cms=height,relative_mass_residual=audit['maximum_mass_residual_m3']/x['wave'][-1,3],kernel_sha256=hashlib.sha256(kernel.read_bytes()).hexdigest(),folder=str(dest))
    (dest/'result.json').write_text(json.dumps(res,indent=2));print('DONE',json.dumps(res),flush=True);return res
if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(lambda args:run(*args),[(length,theta) for length in [100,50] for theta in [.9,.8,.7]]))
    (out/'initial_results.json').write_text(json.dumps(results,indent=2))
