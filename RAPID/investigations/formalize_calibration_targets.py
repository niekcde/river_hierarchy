"""Observation-only target specification. Does not alter calibration or routing."""
from pathlib import Path
import json, hashlib, html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'RAPID/reviews/calibration_targets_20261006'
OUT.mkdir(exist_ok=True)
comparison=ROOT/'RAPID/outputs/cunge_calibration_alpha2/comparison.csv'
c=pd.read_csv(comparison,dtype={'example':str});c=c[c.method.eq('cunge')]
notes={
'02':('retain','Retain the selected observed peak.'),
'03':('plateau_midpoint','Midpoint of the contiguous exact-maximum plateau.'),
'05':('retain','Retain the selected observed peak despite adjacent small fluctuations.'),
'11':('crest_span_midpoint','User-specified 16.063–16.823 day span, resolved to exact observation timestamps. Diagnostic only: Cunge remains unresolved.'),
'13':('crest_span_midpoint','User-specified 5.583–5.885 day plateau, resolved to exact observation timestamps; short intervening dips are retained.'),
'15':('plateau_midpoint','Midpoint of the contiguous exact-maximum plateau.'),
'21':('retain','Retain the selected observed peak.'),
'22':('crest_span_midpoint','Midpoint of the full span covering the two equal-maximum crest blocks; the intervening dip belongs to this crest.'),
'37':('crest_span_midpoint','User accepted: span all THREE equal highest samples; use midpoint from first to last, retaining all three.'),
'41':('retain','Retain the first selected maximum as requested; a second equal maximum 30 minutes later is documented, not silently substituted.'),
'45':('three_crest_span_midpoint','User corrected: three nearly equal observed crests on 2024-06-20 at 20:00, 20:20 and 20:40 UTC. Narrow span 20:00–20:40; nominal target 20:20.'),
'46':('retain','Retain the selected observed peak.'),
'48':('middle_equal_maximum','User selected the June 2 version: middle chronological sample among the three equal global maxima, 2025-06-02 23:45 UTC. The earlier broad-crest alternative is not adopted.'),
'54':('two_crest_midpoint','Midpoint between maxima of the two broad crest lobes, selected in observation-only day windows 4.2–4.75 and 4.8–5.3.'),
'55':('retain','Retain the selected observed peak.'),
'56':('retain','Retain current peak provisionally. Local sharpness alone is insufficient evidence to reject it.')}
records=[];hashes={};charts=[]
def stamp(t):return pd.Timestamp(t).isoformat()
def span_max(g,origin,lo=None,hi=None):
 t=pd.to_datetime(g.time,utc=True);q=g.discharge_cms.to_numpy();x=(t-origin).dt.total_seconds()/86400
 mask=np.ones(len(q),bool) if lo is None else (x.ge(lo)&x.le(hi)).to_numpy()
 ids=np.flatnonzero(mask&(q==q[mask].max()));return t.iloc[ids[0]],t.iloc[ids[-1]]
for row in c.itertuples():
 ex=row.example;folder=Path(row.output_dir);series={}
 for side in ['upstream','downstream']:
  path=folder/f'observed_{side}.parquet';series[side]=pd.read_parquet(path);hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
 path=folder/'cunge_best_wave.parquet';w=pd.read_parquet(path);hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
 down=series['downstream'];tt=pd.to_datetime(down.time,utc=True);origin=tt.min();a,b=span_max(down,origin);old=a
 upa,upb=span_max(series['upstream'],origin);inlet_t=pd.to_datetime(w.time_utc,utc=True);inlet_q=w.q_inflow_cms.to_numpy();inlet_peak=inlet_t.iloc[inlet_q.argmax()]
 rule,note=notes[ex];status='specified';alternatives=[]
 if rule=='retain':a=b=old
 if ex=='37':status='user_accepted'
 if ex=='45':
  a=pd.Timestamp('2024-06-20T20:00:00Z');b=pd.Timestamp('2024-06-20T20:40:00Z');status='user_corrected'
  assert a in set(tt) and b in set(tt)
 if ex=='54':
  aa,ab=span_max(down,origin,4.2,4.75);ba,bb=span_max(down,origin,4.8,5.3);a=aa+(ab-aa)/2;b=ba+(bb-ba)/2
 if ex=='48':
  ids=np.flatnonzero(down.discharge_cms.to_numpy()==down.discharge_cms.max())
  assert len(ids)==3
  tiny=tt.iloc[ids[len(ids)//2]]
  assert tiny==pd.Timestamp('2025-06-02T23:45:00Z')
  status='user_selected'
 target=a+(b-a)/2 if ex!='48' else tiny
 full_a,full_b=span_max(down,origin);upstep=float(np.median(np.diff(pd.to_datetime(series['upstream'].time,utc=True).astype('int64')))/6e10);downstep=float(np.median(np.diff(tt.astype('int64')))/6e10)
 rec=dict(example=ex,status=status,selection_rule=rule,rationale=note,plot_origin_utc=stamp(origin),old_downstream_peak_utc=stamp(old),observed_maximum_span_start_utc=stamp(full_a),observed_maximum_span_end_utc=stamp(full_b),crest_start_utc=stamp(a) if target is not None else None,crest_end_utc=stamp(b) if target is not None else None,target_utc=stamp(target) if target is not None else None,target_plot_day=(target-origin).total_seconds()/86400 if target is not None else None,shift_from_old_target_minutes=(target-old).total_seconds()/60 if target is not None else None,observed_upstream_first_max_utc=stamp(upa),observed_upstream_last_max_utc=stamp(upb),upstream_sampling_minutes=upstep,downstream_sampling_minutes=downstep,prepared_inlet_first_max_utc=stamp(inlet_peak),prepared_minus_observed_upstream_peak_minutes=(inlet_peak-upa).total_seconds()/60,diagnostic_target_lag_from_observed_upstream_h=(target-upa).total_seconds()/3600 if target is not None else None,target_offset_from_prepared_inlet_h=(target-inlet_peak).total_seconds()/3600 if target is not None else None,model_peak_policy='First sampled event maximum; record ties and report interpolation sensitivity separately',calibration_scope='diagnostic_only_unresolved_cunge' if ex=='11' else 'base_only_same_target_for_both_methods',alternatives=alternatives)
 records.append(rec)
 fig=go.Figure()
 for side,color in [('upstream','grey'),('downstream','black')]:
  g=series[side];q=g.discharge_cms.to_numpy();norm=(q-q.min())/np.ptp(q)
  fig.add_trace(go.Scatter(x=pd.to_datetime(g.time,utc=True),y=norm,name='Observed '+side,line=dict(color=color,dash='dash'),customdata=q,hovertemplate='%{x|%Y-%m-%d %H:%M:%S} UTC<br>Normalized Q %{y:.4f}<br>Q %{customdata:.3f} m³/s<extra>%{fullData.name}</extra>'))
 fig.add_trace(go.Scatter(x=[old],y=[1],mode='markers',marker=dict(color='royalblue',symbol='x',size=11),name='Previous downstream target'))
 if target is not None:
  fig.add_vrect(x0=stamp(a),x1=stamp(b),fillcolor='orange',opacity=.18,line_width=0)
  fig.add_trace(go.Scatter(x=[target],y=[1.035],mode='markers',marker=dict(color='darkorange',symbol='diamond',size=11),name='Specified/proposed target',hovertemplate='%{x|%Y-%m-%d %H:%M:%S} UTC<extra>Target time; marker height is schematic</extra>'))
 else:
  for alt in alternatives:fig.add_trace(go.Scatter(x=[alt['target_utc']],y=[1.035],mode='markers',name=alt['label'],marker=dict(size=11,symbol='diamond')))
 start=origin;end=tt.max();left=old-pd.Timedelta(days=1);right=old+pd.Timedelta(days=1)
 if ex=='48':left=origin+pd.Timedelta(days=8);right=origin+pd.Timedelta(days=12)
 fig.update_layout(template='plotly_white',height=470,title=dict(text=f'{ex} · {rule} · {status}',font=dict(size=17)),margin=dict(t=95,b=90,l=65,r=25),legend=dict(orientation='h',y=-.23),xaxis=dict(title='Observed UTC — no time shifts',range=[stamp(left),stamp(right)]),yaxis=dict(title='Independently normalized discharge',range=[0,1.07]),updatemenus=[dict(type='buttons',direction='right',x=0,y=1.13,buttons=[dict(label='Crest detail',method='relayout',args=[{'xaxis.range':[stamp(left),stamp(right)]}]),dict(label='Full event',method='relayout',args=[{'xaxis.range':[stamp(start),stamp(end)]}])])])
 charts.append('<section><p>'+html.escape(note)+'</p>'+fig.to_html(full_html=False,include_plotlyjs=True if not charts else False,config={'responsive':True,'displaylogo':False})+'</section>')
policy='''FORMAL CALIBRATION TARGET SPECIFICATION — 6 October 2026
Status: observation-based review specification; not applied to calibration or production.

PRIMARY OBJECTIVE
Calibrate the base-state model downstream event-crest time against the specified downstream UTC target. Use the identical target for fixed-X and Cunge. Retain existing forcing timestamps, Q reference, geometry, parameter bounds and routing policy for this target-only comparison. All child states inherit the resulting base n; children are not calibrated.

CREST INTERVALS
Intervals describe which observed crest times are being represented; they are not statistical confidence intervals. A midpoint is a deterministic representative time, not a claim of sub-sample observational precision. Retained individual peaks have a point interval, which does not imply zero measurement uncertainty. Record downstream observation cadence and model report cadence separately. Report signed error to midpoint and distance outside the crest interval; use the existing sampling-tolerance diagnostic separately and state its definition. Do not widen intervals to rescue a fit. Exact data timestamps replace rounded plotted-day endpoints for 11 and 13.

MODEL TIMING AND UPSTREAM REFERENCES
Primary residual = model downstream peak UTC minus target downstream UTC. The present calibration implementation uses model lag relative to PREPARED inlet maximum minus observed gauge-to-gauge lag relative to OBSERVED upstream maximum. These are not necessarily identical clocks: the prepared peak can differ from the observed peak. A future implementation must use the UTC residual, or exactly equivalent target offset from the prepared inlet. This table records both upstream references explicitly. Upstream equal-max spans are diagnostic; this specification does not silently redefine upstream peaks. Future gauge-to-gauge midpoint lags must use a reviewed upstream crest definition. Never resample or translate the forcing merely to align these timestamps.

MODEL PEAK DEFINITION
For the ordinary unimodal/merged-crest cases, retain the existing sampled event maximum as model observable during the target-only experiment, with first-tie handling explicitly reported. Inspect plateau/model interpolation sensitivity separately; do not silently change both definitions. Event 48 uses the user-selected middle of the three equal observed global maxima on June 2. Its nominal target is an actual observation timestamp, not the midpoint of the first-to-last crest interval. The earlier broad-crest alternative is not adopted; the model observable remains its event maximum.

INTERPRETATION CHOICES
37 has three equal highest samples; the user accepted the full first-to-last span midpoint. 45 was corrected by the user to the narrow 20 June 20:00–20:40 UTC span with nominal target 20:20 UTC; all other targets were accepted. The completed reviewed calibration snapshot predates this correction, so 45 requires a target-specific refit before production. 48 is resolved by the user: 2025-06-02 23:45 UTC, the middle of three equal maxima. Its crest span remains 2025-06-02 22:15 to 2025-06-03 01:30 UTC. Case 56 retains its current peak as requested, without declaring it erroneous. Case 11 stays a diagnostic case with unresolved Cunge calibration; target revision alone is not evidence of resolution.

RECALIBRATION HANDOFF
Freeze the resolved table and include its hash in calibration provenance. Recalibrate base states only; report old/new n, UTC residual, crest-interval residual, shape diagnostics and numerical checks. The inlet-reference correction may matter even for retained observed targets: reuse completed hierarchy outputs only when the final chosen n and all routing inputs remain identical. Do not treat preliminary targets or ambiguous feature matches as production-ready. Existing outputs and workbooks are unchanged.

'''
lines=[]
for r in records:lines.append(f"{r['example']} | {r['status']} | target {r['target_utc']} | shift {r['shift_from_old_target_minutes']} min | crest {r['crest_start_utc']} to {r['crest_end_utc']}\n  {r['rationale']}")
(OUT/'SPECIFICATION.txt').write_text(policy+'\nTARGET TABLE\n'+'\n'.join(lines)+'\n')
(OUT/'targets.json').write_text(json.dumps(dict(schema_version=2,status='user_reviewed_case45_correction_pending_refit',records=records),indent=2))
(OUT/'provenance.json').write_text(json.dumps(dict(source_hashes=hashes,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),comparison_sha256=hashlib.sha256(comparison.read_bytes()).hexdigest(),scope='Observation-only specification; model outlet hydrographs not used to choose targets. Prepared inlet read only to record reference clock.'),indent=2))
small=pd.DataFrame(records)[['example','status','selection_rule','target_utc','shift_from_old_target_minutes','crest_start_utc','crest_end_utc']]
(OUT/'review.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>Calibration target review</title><style>body{font:15px system-ui;margin:24px}table{border-collapse:collapse;font-size:12px}td,th{padding:7px;border:1px solid #ddd}section{margin-top:35px}p{max-width:1150px}</style></head><body><h1>Calibration target specification · 16 events</h1><p>Observation-only review. No calibration or routing changed. Orange shading = selected crest interval; diamond = nominal target time (schematic marker height). Blue cross = previous selected maximum. Normalize y only; UTC unchanged. See SPECIFICATION.txt for objective and unresolved choices.</p>'+small.to_html(index=False,border=0)+''.join(charts)+'</body></html>')
print(small.to_string(index=False));print('INLET OFFSETS');print(pd.DataFrame(records)[['example','prepared_minus_observed_upstream_peak_minutes']].to_string(index=False))
