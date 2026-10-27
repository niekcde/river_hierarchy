"""Additional shared-line lowering required by single-reach references.
Fixed accepted times; minimize integral lowering relative to accepted background.
Diagnostic only: do not replace accepted definitions.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from review_revised_backgrounds_20261013 import ROOT, pairs, read_wave, iso


def main():
    out=ROOT/'RAPID/outputs/final_straight_background_review_20261013'
    source=out/'candidate_definitions.json'
    definitions=json.loads(source.read_text())['candidates']
    rows=[];provenance={}
    for c in definitions:
        ex=c['example'];a=pd.Timestamp(c['fixed_start_utc']).timestamp();b=pd.Timestamp(c['fixed_end_utc']).timestamp()
        q0=c['candidate_start_cms'];qf=c['candidate_end_cms']
        ref=ROOT/f'network_variants/output/{ex}_indep__anchored/reference_section/rapid/cunge/run/outlet.parquet'
        parent=Path(pairs[pairs.example_id.eq(ex)].iloc[0].parent_outlet_parquet)
        rt,rq=read_wave(ref);bt,bq=read_wave(parent)
        assert rt[0]<=a<b<=rt[-1] and bt[0]<=a<b<=bt[-1]
        t=np.unique(np.r_[a,b,rt[(rt>a)&(rt<b)],bt[(bt>a)&(bt<b)]])
        w=(t-a)/(b-a);bg=q0*(1-w)+qf*w
        single=np.interp(t,rt,rq);base=np.interp(t,bt,bq)
        deficit=bg-single
        fit=linprog([1.,1.],A_ub=-np.column_stack([1-w,w]),b_ub=-np.maximum(deficit,0),bounds=[(0,None),(0,None)],method='highs')
        assert fit.success,fit.message
        d=fit.x
        margin=max(0.,float(np.max(bg-d[0]*(1-w)-d[1]*w-single)))+1e-7
        d=d+margin;new=bg-d[0]*(1-w)-d[1]*w
        assert np.min(single-new)>=-1e-8 and np.min(base-new)>=-1e-8
        assert np.max(new-bg)<=1e-8 # Earlier passing children remain valid since line only moves down.
        v=float(np.trapz(base-bg,t));addition=float(np.trapz(bg-new,t))
        rows.append(dict(example=ex,start_utc=iso(a),end_utc=iso(b),additional_start_lowering_cms=float(d[0]),additional_end_lowering_cms=float(d[1]),
            accepted_start_cms=q0,accepted_end_cms=qf,new_start_cms=float(q0-d[0]),new_end_cms=float(qf-d[1]),
            start_lowering_pct_of_level=100*d[0]/q0,end_lowering_pct_of_level=100*d[1]/qf,
            parent_excess_volume_increase_pct=100*addition/v,maximum_old_deficit_cms=float(deficit.max()),
            maximum_old_deficit_utc=iso(t[deficit.argmax()]),minimum_new_single_excess_cms=float(np.min(single-new)),
            minimum_new_base_excess_cms=float(np.min(base-new)),outlet=str(ref.resolve())))
        provenance[ex]={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ref,parent]}
    df=pd.DataFrame(rows);df.to_csv(out/'single_reach_additional_adjustments.csv',index=False)
    (out/'single_reach_additional_adjustments.json').write_text(json.dumps(dict(status='Diagnostic candidates, not adopted',
        method='Fixed accepted times; minimize additional start+end lowering. Downward-only change preserves validity of previously checked children.',
        limitation='Base excess volume sensitivity is not a change in parent-child response metrics. Endpoints remain reference levels, not curve crossing times.',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),provenance=provenance,cases=rows),indent=2)+'\n')
    print(df[['example','additional_start_lowering_cms','additional_end_lowering_cms','end_lowering_pct_of_level','parent_excess_volume_increase_pct']].to_string(index=False,float_format=lambda x:f'{x:.4f}'))

if __name__=='__main__':main()
