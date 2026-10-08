"""Repeat targeted cap checks using the corrected head allocations."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from reconstruct_child_heads import OUT,BASE
from route_full_signed_validation import *
from reconstruct_child_heads import OUT


def run(ex,states):
    inv=pd.read_csv(OUT/ex/'inventory.csv');selected=inv[inv.state.isin(['S000_base']+states)&inv.status.eq('ready')].drop_duplicates('fingerprint')
    selected=pd.concat([selected[selected.state=='S000_base'],selected[selected.state!='S000_base']])
    frames={r.state:pd.read_parquet(r.frame) for r in selected.itertuples()}
    t,inp,sp,nfft,q0,qe,fdt=grid_input(ex,selected.max_travel_seconds.max(),120.)
    upper=min(1.8*(f.rapid_effective_length_m/f.rapid_celerity_mps_raw.clip(lower=.05)).min() for f in frames.values())
    dt=max(x for x in range(1,fdt+1) if fdt%x==0 and x<.99*upper);rows=[]
    for policy in ['cap5','no_upper_cap']:
        for state,f0 in frames.items():
            f=f0.copy()
            if policy=='no_upper_cap':f['rapid_k']=f.rapid_effective_length_m/f.rapid_celerity_mps_raw.clip(lower=.05)
            q=q0+irfft(sp*response(f,dt,nfft,120.),nfft)[:len(t)];m=summarize(t,q,inp,q0,qe)
            if state=='S000_base':base_peak=m['peak_time_s']
            rows.append(dict(example=ex,state=state,policy=policy,dt_s=dt,peak_time_s=m['peak_time_s'],timing_change_minutes=(m['peak_time_s']-base_peak)/60))
    return rows

if __name__=='__main__':
    rows=[]
    for ex,states in [('22',['CB001_bubble_9','S037_unit_220']),('46',['S010_unit_105']),('37',['S156_unit_18','island_37_161'])]:
        rows+=run(ex,states)
    pd.DataFrame(rows).to_csv(OUT/'targeted_cap_sensitivity.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))
