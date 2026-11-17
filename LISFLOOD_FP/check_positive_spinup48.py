"""Check that the event result is not inherited from clipped warm-start history."""
from pathlib import Path
import numpy as np,pandas as pd,json
import broad_study as b
from run_positivity48 import AUDIT,root,out,kernel
b.v.KERNEL=kernel;b.v.AUDIT=AUDIT
f=pd.read_parquet(root/'current_mc/states/48/S000_base/parameters.parquet');results=[]
for length,folder in [(100,'limiter_retained_48_n0.0207996059'),(50,'limiter_retained_48_n0.0207996059_dx50')]:
    dest=out/f'clean_spinup_{length}m';dest.mkdir(exist_ok=True);b.v.OUT=dest;d=dict(np.load(next((root/folder).glob('_input*.npz'))));cells=b.v.grid(f,f'max{length}');m=b.build(cells,float(d['n']));Q=float(d['forcing'][0]);initial=b.initial_guess(f,cells,m,Q);days=[]
    for day in range(1,61):
        x=b.v.solve(m,[0,86400],[Q,Q],initial,cfl=.35,max_dt=150);initial=(x['h'],x['q']);days.append(x['audit'])
        if x['audit']['final_node_imbalance_cms']/Q<1e-8:break
    np.savez_compressed(dest/'initial.npz',h=initial[0],q=initial[1]);res=dict(length=length,days=day,converged=days[-1]['final_node_imbalance_cms']/Q<1e-8,max_h_difference_m=float(max(abs(initial[0]-d['h']))),max_q_difference=float(max(abs(initial[1]-d['q']))),audit_by_day=days);results.append(res);print(json.dumps(res),flush=True)
(out/'clean_spinup_results.json').write_text(json.dumps(results,indent=2))
