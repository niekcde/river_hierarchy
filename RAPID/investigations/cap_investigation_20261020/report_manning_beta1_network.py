"""Compare the beta=1 full-network sensitivity run with beta=.5 and BASED."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from manning_beta1_network import DEST, OUT, ROOT, digest, table


def main():
    manifest=json.loads((DEST/'manifest.json').read_text())
    assert manifest['complete'] and manifest['states']==411
    previous=OUT/'manning_calibrated_network22_20261027'
    joint=json.loads((OUT/'manning_joint_reference22_20261027/result.json').read_text())
    audit=pd.read_csv(DEST/'routing_audit.csv')
    assert len(audit)==411 and audit.state.nunique()==411 and audit.settled.all()
    assert audit.n.nunique()==audit.beta.nunique()==1
    np.testing.assert_allclose(audit.n,joint['beta1']['n_manning'])
    np.testing.assert_allclose(audit.beta,1)
    assert audit.minimum_native_flow_cms.min()>=0
    keys=['example','state','window']
    a=pd.read_csv(DEST/'comparison.csv',dtype={'example':str})
    b=pd.read_csv(previous/'comparison.csv',dtype={'example':str})
    fields=['centroid_shift_hours','peak_shift_hours','peak_excess_change_pct',
            'duration80_change_pct','asymmetry_change','volume_change_pct']
    a=a.rename(columns={c:c.replace('_manning','_beta1') for c in a if c.endswith('_manning')})
    b=b[keys+[f+'_manning' for f in fields]].rename(columns={f+'_manning':f+'_beta05' for f in fields})
    data=a.merge(b,on=keys,validate='one_to_one')
    assert len(data)==1230 and not data[ [f+'_'+s for f in fields for s in ['based','beta05','beta1']] ].isna().any().any()
    for s in ['based','beta05','beta1']:
        v=data['centroid_minutes_'+s]=60*data['centroid_shift_hours_'+s]
        data['response_class_'+s]=np.select([v>1,v< -1],['delay','advance'],default='near_zero')
    data.to_csv(DEST/'three_way_comparison.csv',index=False)
    main=data[data.window.eq('0.5%')].copy()
    accepted=main[main.edit_type.ne('Single-centerline reference control')]
    regular=accepted[accepted.edit_type.ne('Segmented full network')]
    assert len(main)==410 and len(accepted)==409 and len(regular)==408
    counts=[]
    for window,g0 in data.groupby('window'):
        g=g0[~g0.edit_type.isin(['Single-centerline reference control','Segmented full network'])]
        for s in ['based','beta05','beta1']:
            v=g['centroid_minutes_'+s]
            counts.append(dict(window=window,model=s,cases=len(g),advance_gt1min=int((v< -1).sum()),
                within_1min=int((abs(v)<=1).sum()),delay_gt1min=int((v>1).sum()),
                positive_subminute=int(((v>0)&(v<1)).sum())))
    counts=pd.DataFrame(counts);counts.to_csv(DEST/'population_summary.csv',index=False)
    transitions=regular.groupby(['response_class_beta05','response_class_beta1']).size().rename('count').reset_index()
    transitions.to_csv(DEST/'beta_class_transitions.csv',index=False)
    membership=pd.DataFrame({'state':regular.state,'edit_type':regular.edit_type})
    for s in ['based','beta05','beta1']:membership['delay_'+s]=regular['centroid_minutes_'+s]>1
    membership.to_csv(DEST/'delay_membership.csv',index=False)
    robust=int(membership[['delay_'+s for s in ['based','beta05','beta1']]].all(axis=1).sum())
    both=int((membership.delay_beta05 & membership.delay_beta1).sum())
    lost=int((membership.delay_beta05 & ~membership.delay_beta1).sum())
    gained=int((~membership.delay_beta05 & membership.delay_beta1).sum())
    changes=int((regular.response_class_beta05!=regular.response_class_beta1).sum())
    cases=['S057_unit_5','island_22_018','S296_unit_30','CB001_bubble_9','all_closed_units_collapsed','fully_collapsed']
    selected=main.set_index('state').loc[cases].reset_index()
    selected.to_csv(DEST/'selected_comparisons.csv',index=False)
    bytype=[]
    for kind,g in regular.groupby('edit_type'):
        bytype.append(dict(type=kind,cases=len(g),**{s+'_delays_gt1min':int((g['centroid_minutes_'+s]>1).sum()) for s in ['based','beta05','beta1']}))
    bytype=pd.DataFrame(bytype);bytype.to_csv(DEST/'three_way_type_summary.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for kind,g in accepted.groupby('edit_type'):
        axes[0].scatter(g.centroid_minutes_beta05,g.centroid_minutes_beta1,s=20,alpha=.65,label=kind)
    extent=max(abs(accepted.centroid_minutes_beta05).max(),abs(accepted.centroid_minutes_beta1).max())*1.1
    axes[0].plot([-extent,extent],[-extent,extent],color='grey',ls='--')
    axes[0].set(xscale='symlog',yscale='symlog',xlabel='Manning beta=0.5 centroid shift (min)',
        ylabel='Manning beta=1 centroid shift (min)',title='409 accepted comparisons; symmetric-log axes')
    axes[0].axhline(0,color='grey',lw=.5);axes[0].axvline(0,color='grey',lw=.5);axes[0].legend(fontsize=7)
    x=np.arange(3)
    for s,offset,label in [('based',-.26,'BASED'),('beta05',0,'Manning beta=0.5'),('beta1',.26,'Manning beta=1')]:
        c=counts[(counts.window=='0.5%') & (counts.model==s)].iloc[0]
        bars=axes[1].bar(x+offset,[c.advance_gt1min,c.within_1min,c.delay_gt1min],width=.25,label=label)
        axes[1].bar_label(bars,fontsize=8)
    axes[1].set(xticks=x,xticklabels=['Advance >1 min','Within ±1 min','Delay >1 min'],ylabel='Regular collapse cases',title='408 regular collapse cases')
    axes[1].legend(fontsize=8);axes[0].grid(alpha=.2);axes[1].grid(axis='y',alpha=.2)
    fig.savefig(DEST/'overview.png',dpi=160);plt.close(fig)
    report=f'''# Network 22: beta=1 full-network sensitivity check

All 411 states rerouted with Manning-derived reference depths, n={joint['beta1']['n_manning']:.9f} and beta=1. The calibration is reused from the completed reference-only beta profile. Both parameters remain frozen across collapses. Existing BASED and Manning beta=0.5 outputs are unchanged; no RF fitting or production replacement.

## Design and calibration

The beta=0.5 Manning alternative uses n={joint['selected']['n_manning']:.9f}. Both Manning calibrations reproduce the reference crest timing within +0.5 min. Their respective peak discharges are {joint['selected']['peak_cms']:.3f} and {joint['beta1']['peak_cms']:.3f} m³/s; both remain below the soft 20,300–20,400 m³/s target. Beta=0.5 remains the better height fit in the tested calibration range. This check evaluates robustness to a calibrated alternative, not which formulation is physically validated.

Each state uses h=(n Q_ref/(B sqrt(S)))^(3/5), c=(5/(3n))h^(2/3)sqrt(S), K=dx/c, and X=0.5−beta Q_ref/(2 B S c dx). Depth is fixed through time. Geometry, reference flow allocation, branch fractions and grid are inherited for each state. Both n and beta differ between the Manning alternatives: this is not an isolated diffusion-only perturbation.

The same accepted RAPID/Cunge routing engine, forcing, initial state, shared baseline, original 0.5%, 1%, and 2% windows and response routines are used. Every response is child minus its own model's uncollapsed parent. No collapse response is used for calibration.

## Population and centroid response

There are 408 regular collapses, one segmented full-network collapse, and one separate single-centerline reference control, plus the uncollapsed parent. Counts below use only the 408 regular collapses. Positive means delayed; negative means advanced. Sub-minute positive shifts are included in the within-±1-min category, not added to it.

{table(counts[counts.window.eq('0.5%')])}

Between the two calibrated Manning models, **{both}** cases are delayed by >1 min in both; **{lost}** leave that delay category and **{gained}** enter it at beta=1. In total **{changes}** cases change the advance/near-zero/delay class. **{robust}** cases are delayed by >1 min under all three models. Threshold counts do not establish physical accuracy; a small crossing of the 1-minute threshold need not be a substantial waveform change.

{table(bytype)}

## Selected cases

Centroid shifts in minutes, using the original 0.5% window:

{table(selected[['state','centroid_minutes_based','centroid_minutes_beta05','centroid_minutes_beta1']])}

![Three-way results](overview.png)

Full six-response results for all three windows are in three_way_comparison.csv: centroid, peak timing, peak excess, central 80% duration, asymmetry and excess volume. The two full-collapse cases are distinct representations and remain separately labelled.

## Verification and limits

All 411 outputs have nonnegative routed flows and settled tails; {int(audit.tail_extensions.sum())} tail extensions were needed. Every state uses n and beta above. Completion receipts retain hashes of parameters, outlets and metrics. Source hashes were rechecked at completion. Representative and extreme states were rerouted at half the native timestep: maximum change in centroid contrast was {manifest['maximum_centroid_clock_difference_s']:.6f} seconds. Peak-bin differences are recorded in clock_checks.csv. The completed joint calibration also checked half timestep and halved cell length for both parameter pairs; this full-state check does not establish spatial convergence for every collapse.

The observation-clock ambiguity and unmet soft height target remain. Neither reduced delays nor closer agreement between alternatives establishes the true river response. These are fixed-reference-depth representation-sensitivity results for network 22 only.

## Reproduction

Run manning_beta1_network.py, then report_manning_beta1_network.py in RAPID/investigations/cap_investigation_20261020 with the river-hierarchy-rivgraph Python environment. State resume checks n, beta, original parameter hash and output hashes. Inputs and outputs are isolated under this directory.
'''
    (DEST/'REPORT.md').write_text(report)
    manifest['report_script_sha256']=digest(Path(__file__))
    manifest['comparison_sources']={str(previous/'comparison.csv'):digest(previous/'comparison.csv')}
    manifest['summary_output_hashes']={str(p):digest(p) for p in DEST.iterdir() if p.is_file() and p.name!='manifest.json'}
    (DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(counts.to_string(index=False))
    print(selected[['state','centroid_minutes_based','centroid_minutes_beta05','centroid_minutes_beta1']].to_string(index=False))
    print(dict(shared_Manning_delays=both,lost=lost,gained=gained,class_changes=changes,all_three_delays=robust))


if __name__=='__main__':main()
