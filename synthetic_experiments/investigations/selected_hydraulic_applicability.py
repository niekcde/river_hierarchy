"""Recompute reference-flow hydraulic diagnostics for the selected synthetic grid.

Not a real-case validation or a hydrodynamic solution. Q allocations are steady
reference values, not local transient peaks; the 12h timescale is a rise proxy.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from width_depth_scaling import width_shares
from analysis_paths import RUN


def main():
    out=RUN.parent/'hydraulic_applicability';out.mkdir(exist_ok=True)
    manifest=json.loads((RUN/'manifest.json').read_text());rows=[]
    for index,rid in enumerate(manifest['run_ids']):
        job=json.loads((RUN/'jobs'/f'{rid}.json').read_text());s=job['settings'];ref=job['hydraulic_reference'];net=job['network'];shares=width_shares(net)
        for edge,(u,v,L,w,slope) in enumerate(net['edges']):
            depth=ref['width_m']/s['kb']*(w/ref['width_m'])**ref['depth_alpha'];um=depth**(2/3)*np.sqrt(slope)/s['n_manning'];qm=w*depth*um
            for label,q0 in [('base',s['baseflow_cms']),('peak',s['peak_cms'])]:
                q=shares[edge]*q0;uq=q/(w*depth);raw=5*um/3
                row=dict(run_id=rid,edge=edge,reference_flow=label,width_m=w,length_m=L,slope=slope,depth_m=depth,kb_local=w/depth,flow_share=shares[edge],allocated_q_cms=q,manning_q_cms=qm,q_manning_over_allocated=qm/q,velocity_q_mps=uq,velocity_manning_mps=um,celerity_raw_mps=raw,celerity_used_mps=np.clip(raw,s['min_celerity_mps'],s['max_celerity_mps']))
                for mode,velocity in [('allocated',uq),('manning',um)]:
                    row[mode+'_f2']=velocity**2/(9.80665*depth);row[mode+'_tplus']=s['rise_hours']*3600*velocity*slope/depth
                    row[mode+'_screen_pass']=bool(row[mode+'_f2']<=.9 and row[mode+'_tplus']>=80)
                rows.append(row)
        if index%2000==0:print('Hydraulic reference audit',index,flush=True)
    d=pd.DataFrame(rows);d.to_parquet(out/'reaches.parquet',index=False)
    summaries=[]
    for flow,g in d.groupby('reference_flow'):
        for mode in ['allocated','manning']:
            passed=g[mode+'_screen_pass']
            summaries.append(dict(reference_flow=flow,velocity_definition=mode,branch_run_rows=len(g),screen_pass_rows=int(passed.sum()),screen_pass_length_fraction=float(g.loc[passed,'length_m'].sum()/g.length_m.sum()),median_f2=g[mode+'_f2'].median(),median_tplus=g[mode+'_tplus'].median(),q_ratio_p05=g.q_manning_over_allocated.quantile(.05),q_ratio_median=g.q_manning_over_allocated.median(),q_ratio_p95=g.q_manning_over_allocated.quantile(.95)))
    pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False)
    (out/'provenance.json').write_text(json.dumps(dict(plan_id=manifest['plan_id'],selected_reference=manifest['config']['selected_hydraulic_reference'],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Selected full-grid reference-flow audit only; no real-case overlay',screen='Repository provisional F2<=.9 and Tplus>=80; 12h rise-time proxy, not full wave-regime boundary or proof of RAPID validity',weighting='Canonical branches across unique routed jobs; shared geometries/clock variants are not independent river observations',velocity_definitions='Allocated: Qwidth/(WD); Manning: D^(2/3)*sqrt(S)/n, unclipped; depth fixed from peak-reference BASED/local-alpha law'),indent=2))
    print(pd.DataFrame(summaries).to_string(index=False))
if __name__=='__main__':main()
