"""Diagnostic cost of replacing reviewed backgrounds by straight lines.

No replacement is adopted. Parallel lowering preserves the window but changes
both endpoint reference levels; its effect on event volume is reported.
"""
import json
import numpy as np
import pandas as pd

from review_revised_backgrounds_20261013 import OUT, pairs, read_wave, check, DEFAULTS


def main():
    definitions=json.loads((OUT/'candidate_definitions.json').read_text())['cases']
    rows=[]
    for ex,group in pairs.groupby('example_id'):
        choice=next(c for c in definitions[ex] if c['key']==DEFAULTS.get(ex,'proposed'))
        knots=choice['knots'];a,q0=knots[0];b,qf=knots[-1]
        straight=[(a,q0),(b,qf)]
        horizontal=[(a,min(q0,qf)),(b,min(q0,qf))]
        t,q=read_wave(group.iloc[0].parent_outlet_parquet)
        parent,curve=check(t,q,straight)
        constant,_=check(t,q,horizontal)
        old,(ot,oq,ob)=check(t,q,knots)
        old_signed=float(np.trapz(oq-ob,ot))
        base_offset=max(0.,-parent['minimum_excess_cms'])
        all_offset=base_offset
        horizontal_offset=max(0.,-constant['minimum_excess_cms'])
        worst='S000_base'
        for r in group.itertuples():
            ct,cq=read_wave(r.outlet_parquet)
            ix=(ct>a)&(ct<b)
            times=np.r_[a,ct[ix],b]
            values=np.interp(times,ct,cq)
            deficit=max(0.,float(np.max(np.interp(times,[a,b],[q0,qf])-values)))
            if deficit>all_offset:
                all_offset,worst=deficit,r.state_id
            horizontal_offset=max(horizontal_offset,float(min(q0,qf)-values.min()))
        tt,qq,bg=curve
        peak_excess=float(np.max(qq-bg))
        straight_signed=float(np.trapz(qq-bg,tt))
        def volume_change(volume):return 100*(volume/old_signed-1) if old_signed>0 else np.nan
        row=dict(example=ex,current_choice=choice['key'],current_base_pass=old['base_pass'],
            straight_base_pass=parent['base_pass'],base_only_straight_lowering_cms=base_offset,
            all_states_straight_lowering_cms=all_offset,worst_state=worst,
            all_states_lowering_pct_of_parent_peak_excess=100*all_offset/peak_excess,
            base_only_lowered_straight_volume_change_pct=volume_change(straight_signed+base_offset*(b-a)),
            all_states_lowered_straight_volume_change_pct=volume_change(straight_signed+all_offset*(b-a)),
            horizontal_reference_cms=min(q0,qf),horizontal_base_pass=constant['base_pass'],
            horizontal_extra_lowering_for_all_states_cms=horizontal_offset,
            horizontal_volume_change_pct=volume_change(float(np.trapz(qq-min(q0,qf),tt))),
            start_utc=choice['start_utc'],end_utc=choice['end_utc'])
        assert base_offset<=all_offset+1e-9
        rows.append(row)
        print(ex,'straight lowering: base',round(base_offset,4),'all states',round(all_offset,4),flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/'simple_background_comparison.csv',index=False)
    document='''<!doctype html><meta charset="utf-8"><title>Straight background sensitivity</title><style>body{font:14px Arial;margin:24px}p{max-width:1200px;line-height:1.5}table{border-collapse:collapse}th,td{padding:8px;border:1px solid #ddd}</style><h1>Simple background alternatives — diagnostics, not adopted definitions</h1><p>Each comparison retains the current absolute-time event window. A diagonal joins its two endpoint levels. The required parallel lowering is max(B−Q,0), checked first on the base and then on all children. Such lowering guarantees nonnegative excess but moves both reference levels below their selected endpoints, leaving positive excess at the window boundaries. This is not evidence that the lowered line is physical baseflow.</p><p>The horizontal alternative uses min(start level, end level). For unequal endpoint levels it includes some of the changing background as event excess. Volume changes compare signed integrals on the same window against the current piecewise background; they are sensitivity diagnostics, not parent–child response values. Selecting a line using the most extreme child also makes the reference depend on the set of experiments.</p>TABLE'''
    columns=['example','base_only_straight_lowering_cms','all_states_straight_lowering_cms',
        'base_only_lowered_straight_volume_change_pct','all_states_lowered_straight_volume_change_pct',
        'horizontal_volume_change_pct','worst_state']
    (OUT/'simple_background_comparison.html').write_text(document.replace('TABLE',frame[columns].to_html(index=False,float_format=lambda v:f'{v:.5g}')))
    print(frame[columns].to_string(index=False))


if __name__=='__main__':main()
