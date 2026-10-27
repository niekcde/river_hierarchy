"""Expose all-state background failures and a common two-endpoint adjustment.

The adjustment minimizes total endpoint lowering at fixed window times. It is
a shared reference convention, not physical baseflow identification.
"""
from pathlib import Path
import base64
import gzip
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog

from review_revised_backgrounds_20261013 import ROOT,pairs,read_wave,iso,rising_crossing

OUT=ROOT/'RAPID/outputs/final_straight_background_review_20261013'
OUT.mkdir(exist_ok=True)
EXAMPLES=['02','03','05','13','15','21','22','37','41','45','46','48','54','55','56']
CHOICES={'05':'later_start','13':'later_start','22':'earlier_3h','37':'later_start','54':'q337'}


def main():
    definitions=json.loads((ROOT/'RAPID/outputs/straight_anchor_review_20261013/candidate_definitions.json').read_text())['cases']
    panels,summary,checks,provenance=[],[],[],{}
    fig,axes=plt.subplots(15,2,figsize=(14,45))
    overview,overview_axes=plt.subplots(5,3,figsize=(16,18))
    for row,ex in enumerate(EXAMPLES):
        reference=next(c for c in definitions[ex] if c['key']==CHOICES.get(ex,'unchanged'))
        (a,q0),(b,qf)=reference['knots']
        g=pairs[pairs.example_id.eq(ex)]
        bt,bq=read_wave(g.iloc[0].parent_outlet_parquet)
        if ex=='02':
            a=rising_crossing(bt,bq,1056.,pd.Timestamp('2025-07-31T17:25:30Z').timestamp());q0=1056.
        t=np.r_[a,bt[(bt>a)&(bt<b)],b]
        base=np.interp(t,bt,bq)
        waves=[base];names=['S000_base'];aliases=['Base'];paths=[g.iloc[0].parent_outlet_parquet]
        for r in g.itertuples():
            ct,cq=read_wave(r.outlet_parquet)
            assert ct[0]<=a<b<=ct[-1]
            # All interior sample clocks match, so endpoint interpolation preserves
            # every breakpoint required for an exact linear-reconstruction check.
            np.testing.assert_array_equal(ct[(ct>a)&(ct<b)],t[1:-1])
            waves.append(np.interp(t,ct,cq));names.append(r.state_id);aliases.append(r.state_aliases);paths.append(r.outlet_parquet)
        q=np.asarray(waves);w=(t-a)/(b-a);background=q0*(1-w)+qf*w
        low=q.min(axis=0);high=q.max(axis=0);residual=q-background
        minimum=residual.min(axis=1);argmin=residual.argmin(axis=1)
        envelope_deficit=np.maximum(background-low,0)
        # min integral lowering = (duration/2) * (d_start+d_end).
        solution=linprog([1.,1.],A_ub=-np.column_stack([1-w,w]),b_ub=-envelope_deficit,
                         bounds=[(0,None),(0,None)],method='highs')
        if not solution.success:raise RuntimeError(solution.message)
        lowering=solution.x
        raw_adjusted=background-lowering[0]*(1-w)-lowering[1]*w
        margin=max(0.,float(np.max(raw_adjusted-low)))+1e-7
        lowering=lowering+margin
        adjusted=background-lowering[0]*(1-w)-lowering[1]*w
        adjusted_residual=q-adjusted
        assert adjusted_residual.min()>=-1e-8
        passed=minimum>=-1e-8
        peak_excess=float((base-background).max())
        signed_volume=float(np.trapz(base-background,t))
        volume_addition=float(np.trapz(background-adjusted,t))
        worst=int(np.argmin(minimum[1:]))+1
        highlighted=sorted(set([worst,int(np.argmin(residual[1:,0]))+1,int(np.argmin(residual[1:,-1]))+1]))
        badpoint=int(np.argmin(low-background))
        failing_interior_only=(minimum[1:]<-1e-8)&(residual[1:,0]>=-1e-8)&(residual[1:,-1]>=-1e-8)
        info=dict(example=ex,children=len(g),passing_before=int(passed[1:].sum()),passing_after=len(g),base_pass_before=bool(passed[0]),reference_choice='q1056' if ex=='02' else CHOICES.get(ex,'unchanged'),
            failing_at_start=int((residual[1:,0]<-1e-8).sum()),failing_at_end=int((residual[1:,-1]<-1e-8).sum()),
            failing_interior_only=int(failing_interior_only.sum()),current_start_cms=q0,current_end_cms=qf,
            lower_start_by_cms=float(lowering[0]),lower_end_by_cms=float(lowering[1]),
            candidate_start_cms=float(q0-lowering[0]),candidate_end_cms=float(qf-lowering[1]),
            worst_current_deficit_cms=float(-minimum[worst]),worst_state=names[worst],worst_aliases=aliases[worst],
            worst_time_utc=iso(t[argmin[worst]]),parent_excess_volume_change_pct=100*volume_addition/signed_volume,
            start_lowering_pct_of_peak_excess=100*lowering[0]/peak_excess,
            end_lowering_pct_of_peak_excess=100*lowering[1]/peak_excess,
            fixed_start_utc=iso(a),fixed_end_utc=iso(b),numerical_margin_cms=margin)
        info['selection_status']='review larger adjustment' if ex=='37' else 'accepted shared-line method'
        summary.append(info)
        for i in range(1,len(names)):
            checks.append(dict(example=ex,state_id=names[i],state_aliases=aliases[i],
                minimum_current_excess_cms=float(minimum[i]),minimum_utc=iso(t[argmin[i]]),
                start_excess_cms=float(residual[i,0]),end_excess_cms=float(residual[i,-1]),
                minimum_adjusted_excess_cms=float(adjusted_residual[i].min()),outlet=paths[i]))
        h=(t-a)/3600
        p=dict(example=ex,info=info,x=h.tolist(),utc=[iso(v) for v in t],
               base=base.tolist(),low=low.tolist(),high=high.tolist(),background=background.tolist(),adjusted=adjusted.tolist(),
               highlighted=[dict(state=names[i],aliases=aliases[i],q=q[i].tolist()) for i in highlighted],
               default_highlight=names[worst],ranges=dict(event=[0,float(h[-1])],start=[0,min(12,float(h[-1]))],
                   end=[max(0,float(h[-1])-12),float(h[-1])],worst=[max(0,float(h[badpoint])-6),min(float(h[-1]),float(h[badpoint])+6)]))
        panels.append(p)
        ax=overview_axes.flat[row]
        ax.fill_between(h,low,high,color='#b8d6ea',alpha=.6,label='All-state range')
        ax.plot(h,base,color='#ce3d45',lw=1,label='Base')
        ax.plot(h,background,color='#9b59b6',ls=':',lw=1,label='Before adjustment')
        ax.plot(h,adjusted,color='#168a65',ls='--',lw=1.3,label='Shared straight background')
        ax.set(title=f'{ex} · all {len(g)} children pass',xlabel='Hours from selected start',ylabel='Q (m³/s)');ax.grid(alpha=.2)
        for col,window in enumerate(['start','end']):
            ax=axes[row,col];lo,hi=p['ranges'][window];use=(h>=lo)&(h<=hi)
            ax.fill_between(h[use],(low-background)[use],(high-background)[use],color='#b8d6ea',alpha=.6,label='All states: min–max')
            ax.plot(h[use],(base-background)[use],color='#ce3d45',lw=1.3,label='Base')
            ax.plot(h[use],residual[worst,use],color='#222',lw=1.1,label='Worst state overall')
            ax.plot(h[use],(adjusted-background)[use],color='#168a65',ls='--',lw=1.3,label='Proposed common background')
            ax.axhline(0,color='#9b59b6',ls='--',lw=1,label='Current background')
            ax.set(title=f'{ex} · {window} · {info["passing_before"]}/{len(g)} children currently pass',
                   xlabel='Hours from reviewed start',ylabel='Q − current background (m³/s)')
            ax.grid(alpha=.2)
        provenance[ex]=dict(outlets={path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths})
        print(ex,info,flush=True)
    overview_axes.flat[0].legend(fontsize=7)
    overview.suptitle('15 networks: shared straight backgrounds checked against every state',fontsize=15)
    overview.tight_layout(rect=(0,0,1,.97));overview.savefig(OUT/'15_network_shared_backgrounds.png',dpi=140);plt.close(overview)
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Shared straight backgrounds: all-state endpoint checks\nBlue band: all-state range; green: one shared lower line, proposed for review',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(OUT/'all_state_start_end_review.png',dpi=140);plt.close(fig)
    pd.DataFrame(summary).to_csv(OUT/'common_adjustment_candidates.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'state_checks.csv',index=False)
    (OUT/'candidate_definitions.json').write_text(json.dumps(dict(status='Shared-line method applied to all networks; 37 flagged for larger adjustment',
        objective='Minimize total start+end lowering at fixed reviewed times, subject to background <= every saved linearly reconstructed state; one shared line per network',
        limitation='New endpoint levels generally lie below the base curve. They are reference levels at fixed analysis boundaries, not new model crossing times. The fit depends on the evaluated state set and does not establish physical baseflow.',
        candidates=summary),indent=2))
    payload=base64.b64encode(gzip.compress(json.dumps(panels,separators=(',',':'),allow_nan=False).encode())).decode()
    old=(ROOT/'RAPID/outputs/straight_anchor_review_20261013/15_network_straight_anchor_trials.html').read_text()
    library=old.split('<script>',1)[1].split('</script>',1)[0]
    script=Path(__file__).with_name('review_all_state_backgrounds_20261013.js').read_text()
    doc='''<!doctype html><meta charset="utf-8"><title>All-state background review</title><style>body{font:14px Arial;margin:24px;color:#202a34}p{max-width:1300px;line-height:1.5}.controls{position:sticky;top:0;background:#f4f6f9;padding:12px;z-index:20;display:flex;gap:10px;flex-wrap:wrap}select{padding:7px}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}</style><h1>15 networks: shared straight backgrounds</h1><p><a href="15_network_shared_backgrounds.png">15-panel event overview</a> · <a href="all_state_start_end_review.png">All start/end close-ups</a></p><p>The blue band is the exact pointwise minimum–maximum range of the base and every unique child. Its edges can come from different states; they are not individual hydrographs. Red is the base. Black highlights an actual child. Negative excess means the selected background is above that hydrograph.</p><p>The proposed correction is <b>one shared straight line per network</b>, with the existing times fixed. It minimizes the sum of start- and end-level lowering while staying below every state. Adjustments for 03, 15, 41, 46 and 55 were accepted; the same minimal-lowering rule is extended here to the remaining networks. This is an event reference convention, not physical baseflow identification. Its endpoint levels usually no longer touch the base curve; positive residual flow remains at those window boundaries. Nothing is clipped or rerouted.</p><div class="controls">Network <select id="network"></select>View <select id="view"><option value="end">End: last 12 h</option><option value="start">Start: first 12 h</option><option value="event">Full reviewed event</option><option value="worst">Largest deficit</option></select>Units <select id="units"><option value="excess">Q − selected background</option><option value="discharge">Discharge Q</option></select>Background <select id="background"><option value="adjusted">Shared adjusted line</option><option value="background">Before adjustment</option></select>Highlight <select id="highlight"></select></div><p id="notes"></p><div id="plot"></div><h2>Fixed-time adjustment candidates</h2><p>Start/end failure counts may overlap. The volume change is for the base event under the changed reference, not a parent–child response. Download <a href="common_adjustment_candidates.csv">candidate values</a> or <a href="state_checks.csv">every state check</a>.</p>TABLE<script>LIBRARY</script><script id="payload" type="application/octet-stream">PAYLOAD</script><script>SCRIPT</script>'''
    cols=['example','children','passing_before','failing_at_start','failing_at_end','failing_interior_only','lower_start_by_cms','lower_end_by_cms','candidate_start_cms','candidate_end_cms','parent_excess_volume_change_pct']
    table=pd.DataFrame(summary)[cols].to_html(index=False,float_format=lambda v:f'{v:.6g}')
    html=OUT/'all_state_background_review.html';html.write_text(doc.replace('TABLE',table).replace('LIBRARY',library).replace('PAYLOAD',payload).replace('SCRIPT',script))
    (OUT/'manifest.json').write_text(json.dumps(dict(provenance=provenance,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        html_sha256=hashlib.sha256(html.read_bytes()).hexdigest(),networks=len(panels),children=len(checks)),indent=2))


if __name__=='__main__':main()
