"""Trace the actual event at matched domain boundaries and review response definitions."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from scipy.signal import lfilter
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src')]
from rapid_tools.cunge import topology
OUT=ROOT/'RAPID/outputs/cb006_hydrograph_audit_20261012'
meta=json.loads((OUT/'provenance.json').read_text());r=meta['record'];domain=meta['domain']
frames=[pd.read_parquet(r[k]) for k in ['parent_parameters_parquet','parameters_parquet']]
a,b=[f.set_index('id_link') for f in frames];common=a.index.intersection(b.index)
assert set(a.index)-set(b.index)==set(domain['parent_links'])
assert set(b.index)-set(a.index)==set(domain['child_links'])
rows=[]
for column in a:
 equal=a.loc[common,column].eq(b.loc[common,column]) | (a.loc[common,column].isna() & b.loc[common,column].isna())
 rows.append(dict(column=column,exterior_links=len(common),changed_links=int((~equal).sum()),role='State identity metadata' if column=='geometry_id' else 'Unchanged parameter or metadata'))
 if column!='geometry_id':assert equal.all(),column
pd.DataFrame(rows).to_csv(OUT/'exterior_parameter_audit.csv',index=False)
forcing=pd.read_parquet(r['forcing_parquet']);flow=forcing.forcing_cms.to_numpy();dt=5.;nsub=12

def trace(f):
 f=f.reset_index(drop=True);order,edges,root,end=topology(f)
 k=f.rapid_k.to_numpy();x=f.rapid_x.to_numpy();den=dt+2*k*(1-x)
 aa=(dt-2*k*x)/den;bb=(dt+2*k*x)/den;dd=(2*k*(1-x)-dt)/den
 signals={root:np.repeat(flow,nsub)};saved={}
 for u in order:
  sig=signals.pop(u)
  if u in [domain['source'],domain['sink'],end]:
   bins=sig.reshape(-1,nsub);initial=np.r_[sig[0],bins[:-1,-1]]
   saved[u]=bins.mean(axis=1)+(initial-bins[:,-1])/(2*nsub)
  for i in edges.get(u,[]):
   e=f.iloc[i];val=sig*e.split_fraction;eq=flow[0]*e.reference_flow_fraction
   for _ in range(int(e['count'])):val,_=lfilter([aa[i],bb[i]],[1.,-dd[i]],val,zi=[(bb[i]+dd[i])*eq])
   signals[e.v]=signals.get(e.v,0.)+val
 return saved,end

series=pd.DataFrame({'time_utc':forcing.time_utc,'network_inlet':flow});summary=[]
for label,f in zip(['parent','child'],frames):
 saved,end=trace(f)
 for place,node in [('domain_inlet',domain['source']),('domain_outlet',domain['sink']),('network_outlet',end)]:
  series[label+'_'+place]=saved[node];i=int(saved[node].argmax())
  summary.append(dict(state=label,location=place,peak_utc=str(forcing.time_utc.iloc[i]),peak_cms=float(saved[node][i])))
 published=pd.read_parquet(r['parent_outlet_parquet' if label=='parent' else 'outlet_parquet'])
 np.testing.assert_allclose(series[label+'_network_outlet'],published.q_outlet_cms.iloc[:len(flow)],rtol=0,atol=1e-8)
 print(label,'boundary signals checked',flush=True)
np.testing.assert_array_equal(series.parent_domain_inlet,series.child_domain_inlet)
series.to_parquet(OUT/'boundary_hydrographs.parquet',index=False)
summary=pd.DataFrame(summary);summary.to_csv(OUT/'boundary_peak_summary.csv',index=False)
print(summary.to_string(index=False),flush=True)
# Nearest crossings around each selected peak; linear interpolation of saved means.
t=np.arange(len(series))*60.
def crest(q,level):
 q=np.asarray(q);i=int(q.argmax());left=np.flatnonzero(q[:i]<level);right=np.flatnonzero(q[i+1:]<level)
 if not len(left) or not len(right):return [np.nan]*5
 j=left[-1];k=i+1+right[0]
 tl=t[j]+60*(level-q[j])/(q[j+1]-q[j]);tr=t[k-1]+60*(level-q[k-1])/(q[k]-q[k-1])
 return [tl,tr,tr-tl,t[i]-tl,tr-t[i]]
responses=[]
for place in ['domain_outlet','network_outlet']:
 for frac in [.9,.95,.99]:
  for label in ['parent','child']:
   q=series[label+'_'+place].to_numpy();values=crest(q,frac*q.max())
   responses.append(dict(location=place,state=label,threshold_type='fraction_of_own_total_peak',threshold=frac,threshold_cms=frac*q.max(),**dict(zip(['rise_crossing_s','fall_crossing_s','width_s','rise_to_peak_s','peak_to_fall_s'],values))))
 # A shared level checks whether own-peak normalization changes the conclusion.
 for label in ['parent','child']:
  level=.95*series['parent_'+place].max();values=crest(series[label+'_'+place],level)
  responses.append(dict(location=place,state=label,threshold_type='common_absolute_level',threshold=.95,threshold_cms=level,**dict(zip(['rise_crossing_s','fall_crossing_s','width_s','rise_to_peak_s','peak_to_fall_s'],values))))
pd.DataFrame(responses).to_csv(OUT/'crest_width_diagnostics.csv',index=False)

# Separate input-to-output attenuation from parent-to-child peak change.
attenuation=[]
for scope in ['domain','network']:
 inlet=series.parent_domain_inlet if scope=='domain' else series.network_inlet
 for label in ['parent','child']:
  q=series[label+'_'+scope+'_outlet']
  attenuation.append(dict(scope=scope,state=label,inlet_peak_cms=inlet.max(),outlet_peak_cms=q.max(),peak_lag_minutes=int(q.argmax()-inlet.argmax()),total_peak_attenuation_pct=100*(1-q.max()/inlet.max())))
attenuation=pd.DataFrame(attenuation);attenuation.to_csv(OUT/'peak_attenuation.csv',index=False)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(12,7.5))
for row,place in enumerate(['domain_outlet','network_outlet']):
 time=(series.time_utc-series.time_utc.iloc[int(series['parent_'+place].argmax())]).dt.total_seconds()/3600
 sel=time.abs()<=40
 for label,color in [('parent','#2563eb'),('child','#dc2626')]:
  q=series[label+'_'+place]
  axes[row,0].plot(time[sel],q[sel],color=color,label=label)
  axes[row,1].plot(time[sel],q[sel]/q.max(),color=color,label=label)
 axes[row,0].set(title=place.replace('_',' ').capitalize()+': actual discharge',ylabel='Q (m³/s)')
 axes[row,1].axhline(.95,color='gray',ls='--',lw=1,label='95% of own total peak')
 axes[row,1].set(title='Own-peak normalization: slight crest narrowing',ylabel='Q / own total peak',ylim=(.94,1.001),xlim=(-28,28))
 for ax in axes[row]:ax.set_xlabel('Hours from parent peak at this location');ax.legend();ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(OUT/'domain_and_network_crests.png',dpi=160);plt.close(fig)

import nbformat as nb
book=ROOT/'RAPID/notebooks/hydrograph_response_worked_examples_20261012.ipynb';notebook=nb.read(book,as_version=4)
tag='cb006_response_walkthrough'
notebook.cells=[cell for cell in notebook.cells if cell.metadata.get('audit_section')!=tag]
def md(s):return nb.v4.new_markdown_cell(s,metadata={'audit_section':tag})
def code(s):return nb.v4.new_code_cell(s,metadata={'audit_section':tag})
intro=[md('''## Start here: follow one real event through the target region and the network

Use **46 / CB006_bubble_101** throughout this review. The response variables **describe different aspects of the change**; the hydraulic model and controlled comparisons explain its cause. A peak-height response alone cannot explain timing or widening.

My earlier wording “faster target region” needs qualification: **134.08 → 73.50 min is the model mean-delay diagnostic through the target domain (nodes 25→23), not the actual event's peak lag.** The actual peak lag across this domain is **57 → 62 min**. Its crest is therefore five minutes later locally as well as at the network outlet. The whole-network model mean delay also decreases by 60.58 min; there is no evidence here of a local mean-delay improvement being reversed by slower exterior reaches.

| Clock or response | Parent | Child | Child minus parent |
|---|---:|---:|---:|
| Target-domain model mean delay | 134.085 min | 73.504 min | −60.581 min |
| Actual peak lag: domain inlet → domain outlet | 57 min | 62 min | +5 min |
| Actual peak lag: network inlet → network outlet | 8420 min | 8425 min | +5 min |
| Whole-network model mean delay | 10284.989 min | 10224.408 min | −60.581 min |
| Network outlet peak | 1883.352 m³/s | 1888.382 m³/s | +5.030 m³/s (+0.267%) |

**Same event, different clocks.** A shape-changing filter need not move its maximum in the same direction as its mean delay. Model mean delay is neither a measured water-particle time nor an event-excess centroid. If “slower” means later crest, it is slightly slower by that definition **both locally and at the outlet**. If “slower” means larger model mean delay, it is faster both locally and over the whole network.

The following cells use independently reconstructed boundary hydrographs. They reproduce the published network outlets to within 10⁻⁸ m³/s. Their window is the supplied forcing period, which contains these peaks and threshold crossings, not the additional published settling tail.'''),code('''from pathlib import Path
import pandas as pd
from IPython.display import display, Image
CASE_ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'RAPID').is_dir())
CASE=CASE_ROOT/'RAPID/outputs/cb006_hydrograph_audit_20261012'
display(pd.read_csv(CASE/'boundary_peak_summary.csv'))
display(Image(filename=str(CASE/'domain_and_network_crests.png')))
'''),md('''### 1. Verify that the perturbation is confined to the target domain

Exactly **11 domain reaches disappear and one replacement reach appears**. Every saved column was compared on the **530 exterior links**, with NaNs treated as equal. Connectivity, length, width, depth, heads, slope, roughness, celerity, subdivision, K, X, diffusion, split fractions and reference discharges are identical. The only differing exterior column is `geometry_id`, the identifier of the entire network state; it is metadata, not a routing coefficient. The domain-inlet hydrograph is identical sample for sample. The independent checks use the same forcing, 5-second routing clock and reporting convention.

**Unchanged downstream parameters do not mean unchanged downstream discharge.** They receive the changed domain-outlet signal. Here the domain-outlet peak shifts 06:04:30→06:09:30 UTC and the network-outlet peak shifts 12:15:30→12:20:30 UTC. The peak-to-peak gap between those two locations happens to remain 371 minutes; this is a feature lag, not tracked water-particle travel.'''),code("display(pd.read_csv(CASE/'exterior_parameter_audit.csv'))"),md('''### 2. Height and attenuation: almost equal peaks, slightly less attenuation

The existing notebook response is **parent-to-child total peak change**, `100 (Qpeak_child/Qpeak_parent − 1)`: **+0.267%** at the network outlet. It is small, but not zero.

A distinct diagnostic is **input-to-output total peak attenuation**, `100 (1 − Qpeak_out/Qpeak_in)`. At the target domain it decreases from **0.4399% to 0.1626%**; over the network it decreases from **15.6830% to 15.4578%**. The whole-network difference is **−0.2252 percentage points**, not −0.2252% parent-to-child peak change. Compare maxima of the same event at each location, not simultaneous discharges.

These numbers say that the child preserves slightly more peak height. **They do not say that the peak must arrive earlier.** They use total Q, including background discharge; they are not event-excess attenuation.'''),code("display(pd.read_csv(CASE/'peak_attenuation.csv'))"),md('''### 3. Widening and limb duration: specify the threshold first

As a baseline-free **additional diagnostic**, measure the connected crest above 95% of each curve's **own total peak**. At the network outlet the width is **44.4127 → 44.1453 h**, a **16.04-minute narrowing (about 0.60%)**. Rise-to-peak duration decreases by **7.55 min**, and peak-to-falling-crossing duration by **8.49 min**. The target-domain crest also narrows, by **14.99 min**. The same narrowing direction holds at 90% and 99% of each total peak.

This is **not** the catalogue's width above a fraction of peak **excess**. No event background has been chosen, so it must retain its distinct label.

A shared absolute threshold asks a different question: above **1789.1847 m³/s** (95% of the parent network peak), the child stays above the threshold **57.07 minutes longer**. Its slightly greater height moves the crossings outwards even though the own-peak-normalized crest narrows. This is why a generic “duration” or “attenuation” label is insufficient.

The table below reports crossing and limb-duration changes in minutes. Negative width change means narrower under that particular threshold definition.'''),code('''widths=pd.read_csv(CASE/'crest_width_diagnostics.csv')
width_comparison=[]
for (location,definition,threshold),g in widths.groupby(['location','threshold_type','threshold']):
    g=g.set_index('state');p=g.loc['parent'];c=g.loc['child']
    width_comparison.append(dict(location=location,definition=definition,threshold=threshold,
        parent_width_hours=p.width_s/3600,child_width_hours=c.width_s/3600,
        width_change_minutes=(c.width_s-p.width_s)/60,
        rise_duration_change_minutes=(c.rise_to_peak_s-p.rise_to_peak_s)/60,
        fall_duration_change_minutes=(c.peak_to_fall_s-p.peak_to_fall_s)/60))
display(pd.DataFrame(width_comparison))
'''),md('''### 4. Bulk timing and event-volume attenuation remain distinct questions

The catalogue's event-excess centroid, median volume time, temporal SD, volume quantile durations, quantile asymmetry, volume change and standardized shape require a defensible **event window and B(t)**. Do not insert the model mean-delay or model transfer spread into those response columns. The real record has unequal initial/final discharge, so subtracting an arbitrary constant or minimum would change the question.

For this case the demonstrated description is: **slightly later crests at both scales, slightly less total-peak attenuation, slightly narrower own-total-peak-normalized crests, and shorter model mean delay.** The event-excess bulk timing direction is not yet established. Each metric card below now states what this real case does—and does not—supply; the synthetic examples remain to explain the exact definitions.''')]
notebook.cells[1:1]=intro
notes={
'sampled_peak_shift':'Network outlet: **+300 s**. Domain outlet: **+300 s**. This is a later largest interval mean, not an inferred mean delay.',
'reviewed_feature_shift':'Network outlet: **+300 s**, agreeing with the sampled shift for this single crest.',
'54_left_right_center':'**Not applicable**: this case uses a single reviewed crest. Keep the network-54 example for the multiple-lobe definition.',
'total_peak_change_pct':'Network outlet: **+0.267082%** (+5.030096 m³/s). This small height increase corresponds to slightly less attenuation of the common input peak, but says nothing by itself about timing.',
'relative_timing':'Using the common network-inlet sampled maximum: **100×5/8420 = +0.059382%**. At the matched domain: **100×5/57 = +8.771930%**. Same five-minute shift, different denominator and spatial scope; never mix them.',
'peak_excess_change_pct':'**Pending an explicit background convention.** The +0.267082% figure is total-peak change, not excess-peak change.',
'centroid_shift':'**Event-excess centroid pending B(t)/window.** The −60.581 min model mean-delay change is a separate hydraulic diagnostic, not this response.',
'volume_median_shift':'**Pending B(t)/window.** A later crest cannot determine whether half the excess volume passes earlier or later.',
'temporal_sd_change':'**Event-excess SD pending B(t)/window.** Model transfer spread decreases, but must not be substituted for this event response.',
'volume_duration80_change':'**Pending B(t)/window.** No cumulative event-excess 10–90% interval is inferred from a crest-width diagnostic.',
'volume_duration50_change':'**Pending B(t)/window.** This is a cumulative-volume width, not time above 50% peak height.',
'connected_crest_width':'**Catalogue excess-height width pending B(t).** Separate diagnostic: network width above 95% of own **total** peak narrows by **16.04 min**. Above the common 1789.1847 m³/s threshold it lengthens by **57.07 min**; these are different definitions.',
'quantile_asymmetry_change':'**Pending B(t)/window.** The two crest limbs changing by different amounts is not a calculation of volume-quantile asymmetry.',
'rise_fall_duration_change':'**Excess-based version pending B(t).** At 95% of own total peak, network rise duration changes **−7.55 min** and fall duration **−8.49 min**. These explicitly labelled total-Q diagnostics accompany the +5 min crest shift.',
'volume_change_pct':'**Event-excess volume pending B(t)/window.** Unit model DC gain is a conservation property, not a measured event-window volume-change percentage.',
'standardized_shape_distance':'**Not computed.** It needs the chosen event decomposition and normalization; nearly overlapping curves do not imply exactly zero standardized distance.'}
result=[]
for cell in notebook.cells:
 result.append(cell)
 if cell.cell_type=='markdown':
  name=cell.source.split('\n',1)[0].removeprefix('### ')
  if name in notes:result.append(md('**46 / CB006 real-case interpretation:** '+notes[name]))
notebook.cells=result;nb.validate(notebook);nb.write(notebook,book)
print('Updated real-case walkthrough and',len(notes),'metric interpretations',flush=True)
