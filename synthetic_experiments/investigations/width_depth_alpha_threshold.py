"""Resolve synthetic alpha feasibility boundary; no hydrograph reruns/default edits."""
import json,hashlib
import numpy as np
import pandas as pd
from width_depth_scaling import ROOT,OUT,routing,make_states,canonical_network,_compute_subreach_count
from embedded_width_length_extension import embed

def main():
    out=OUT/'alpha_threshold';out.mkdir(exist_ok=True)
    refs=pd.read_csv(OUT/'state_preflight.csv',usecols=['dataset','configuration_id','reference_kb']).drop_duplicates().set_index(['dataset','configuration_id']).reference_kb
    original=json.loads((ROOT/'synthetic_experiments/outputs/full_grid_island/configurations.json').read_text())
    cfg=json.loads((ROOT/'synthetic_experiments/configs/embedded_extension.json').read_text())
    base={k:cfg[k] for k in ['total_width_m','reference_slope','inner_fraction','internal_location']}
    ext=[dict(configuration_id=r['configuration_id'],parameters=dict(base,total_length_m=20000,outer_a_fraction=r['outer_a_fraction'],length_multiplier=r['length_multiplier']),extent=r['network_extent_m']) for r in pd.read_csv(ROOT/'synthetic_experiments/outputs/embedded_width_length_extension/configurations.csv').to_dict('records')]
    alphas=np.unique(np.r_[np.arange(600,711)/1000, .65,.66,.665,2/3,.669,.6699,.66991,.66992,.67,.675,.68,.69,.70,.71])
    rows=[];limits=[]
    for dataset,configs in [('full_grid_island',original),('embedded_width_length_extension',ext)]:
        for idx,config in enumerate(configs):
            params=config['parameters'];wref=params['total_width_m'];kb=refs.loc[(dataset,config['configuration_id'])]
            states,_=make_states(params)
            if 'extent' in config:states={k:embed(g,config['extent'],params['reference_slope']) for k,g in states.items()}
            e=np.array([e for g in states.values() for e in canonical_network(g)['edges']]);L,w,slope=e[:,2],e[:,3],e[:,4]
            count=np.array([_compute_subreach_count(float(l),200) for l in L]);effective=np.maximum(L/count,200)
            def bounds(a):
                a=np.atleast_1d(a);depth=wref/kb*(w[:,None]/wref)**a[None,:]
                c=np.clip(5/(3*.035)*np.sqrt(slope[:,None])*depth**(2/3),.05,2.524)
                K=effective[:,None]/c;lo=.2*K.max(axis=0);hi=1.8*K.min(axis=0)
                return lo,hi,np.ceil(lo+1e-8)<=np.floor(hi)
            lo,hi,ok=bounds(alphas)
            for a,l,h,f in zip(alphas,lo,hi,ok):rows.append(dict(dataset=dataset,configuration_id=config['configuration_id'],alpha=a,dt_lower_seconds=l,dt_upper_seconds=h,feasible=bool(f),integer_timesteps=max(0,int(np.floor(h)-np.ceil(l+1e-8)+1))))
            if not ok[-1]:
                assert ok[0] and not np.any(np.diff(ok.astype(int))>0)
                left,right=.6,.71
                for _ in range(45):
                    mid=(left+right)/2
                    if bounds(mid)[2][0]:left=mid
                    else:right=mid
                l,h,_=bounds(left)
                limits.append(dict(dataset=dataset,configuration_id=config['configuration_id'],max_feasible_alpha=left,min_infeasible_alpha=right,dt_lower_seconds=l[0],dt_upper_seconds=h[0],outer_a_fraction=params['outer_a_fraction'],length_multiplier=params['length_multiplier']))
            if idx%1000==0:print(dataset,idx,flush=True)
    d=pd.DataFrame(rows);d.to_csv(out/'scan.csv',index=False)
    summary=d.groupby(['dataset','alpha']).agg(configurations=('feasible','size'),passing=('feasible','sum'),min_integer_timesteps=('integer_timesteps','min')).reset_index();summary['failing']=summary.configurations-summary.passing;summary.to_csv(out/'summary.csv',index=False)
    limits=pd.DataFrame(limits);limits.to_csv(out/'boundaries.csv',index=False)
    first=limits.sort_values('max_feasible_alpha').iloc[0]
    controlling=d[(d.dataset==first.dataset)&(d.configuration_id==first.configuration_id)]
    controlling.to_csv(out/'limiting_configuration.csv',index=False)
    (out/'manifest.json').write_text(json.dumps(dict(reference='HYD-2026-09-30-ALPHA-THRESHOLD',settings='n=.035,X=.1,target/floor=200m,BASE D references unchanged,caps .05/2.524; all-state shared integer clock',alphas=alphas.tolist(),minimum_boundary=float(first.max_feasible_alpha),script_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),source_preflight_sha256=hashlib.sha256((OUT/'state_preflight.csv').read_bytes()).hexdigest(),scope='2,187 original +140 extension configurations; preflight only, no exponent adoption'),indent=2))
    print(limits.to_string(index=False));print(controlling[controlling.alpha.isin([.6,.65,.66,2/3,.669,.6699,.66991,.66992,.67,.68,.70,.71])].to_string(index=False))
if __name__=='__main__':main()
