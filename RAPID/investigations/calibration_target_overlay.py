"""Add explicit original/reviewed timing targets to both calibration figures.
Only figure and display metadata outputs are changed; calibration is untouched.
"""
from pathlib import Path
import sys,json,html
import pandas as pd,numpy as np
import plotly.graph_objects as go
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
from cunge_hydrograph_figure import build_figure
TARGETS=ROOT/'RAPID/reviews/calibration_targets_20261006/targets.json'
records={r['example']:r for r in json.loads(TARGETS.read_text())['records']}
OLD=ROOT/'RAPID/outputs/cunge_calibration_alpha2';NEW=ROOT/'RAPID/outputs/cunge_calibration_reviewed_20261006'

def q_at(g,stamp):
 t=pd.to_datetime(g.time,utc=True);return float(np.interp(pd.Timestamp(stamp).value,t.astype('int64').to_numpy(),g.discharge_cms.to_numpy()))

def build(root):
 results=pd.read_csv(root/'comparison.csv',dtype={'example':str});paths={r.example:Path(r.output_dir) for r in results.itertuples()};fig=build_figure(results,paths);table=[]
 for j,ex in enumerate(sorted(paths)):
  row,col=divmod(j,4);row+=1;col+=1;r=records[ex];g=pd.read_parquet(paths[ex]/'observed_downstream.parquet');q=g.discharge_cms.to_numpy();qmin=float(q.min());span=float(np.ptp(q));origin=pd.Timestamp(r['plot_origin_utc']);target=pd.Timestamp(r['target_utc']);a=pd.Timestamp(r['crest_start_utc']);b=pd.Timestamp(r['crest_end_utc'])
  targetq=q_at(g,target) if r['selection_rule'] in ['retain','middle_equal_maximum'] else (q_at(g,a)+q_at(g,b))/2
  x=lambda stamp:(pd.Timestamp(stamp)-origin).total_seconds()/86400
  fig.add_vrect(x0=x(a),x1=x(b),fillcolor='#E69F00',opacity=.14,line_width=0,row=row,col=col)
  fig.add_shape(type='line',x0=x(target),x1=x(target),y0=0,y1=1.015,line=dict(color='#B87300',width=1,dash='dot'),row=row,col=col)
  fig.add_trace(go.Scatter(x=[x(r['old_downstream_peak_utc'])],y=[1.0],mode='markers',name='Previous observed target',legendgroup='previous_target',showlegend=j==0,marker=dict(color='#0072B2',symbol='x',size=11,line=dict(width=2)),hovertemplate='Previous observed target<br>'+r['old_downstream_peak_utc']+'<extra>'+ex+'</extra>'),row=row,col=col)
  fig.add_trace(go.Scatter(x=[x(target)],y=[(targetq-qmin)/span],mode='markers',name='Reviewed calibration target',legendgroup='reviewed_target',showlegend=j==0,marker=dict(color='#E69F00',symbol='diamond-open',size=15,line=dict(width=2.5)),hovertemplate='Reviewed target<br>'+target.strftime('%Y-%m-%d %H:%M:%S UTC')+'<br>Display Q: '+f'{targetq:.3f}'+' m³/s<br>Height is illustrative; calibration uses TIME only.<extra>'+ex+'</extra>'),row=row,col=col)
  table.append(dict(example=ex,target_utc=target.isoformat(),crest_start_utc=a.isoformat(),crest_end_utc=b.isoformat(),selection_rule=r['selection_rule'],display_target_discharge_cms=targetq,display_height_definition='observed sample' if r['selection_rule'] in ['retain','middle_equal_maximum'] else 'mean of observed crest endpoint discharges',timing_only=True))
 label='Previous calibration model curves' if root==OLD else 'Recalibrated model curves'
 fig.update_layout(height=1640,margin=dict(t=195,b=90,l=85,r=35),title=dict(text='<b>'+label+' · all 16 reviewed targets</b><br><sup>Diamond = reviewed timing target; blue × = previous observed target; shading = crest span. UTC unchanged.</sup>',font=dict(size=21)),legend=dict(y=1.065,font=dict(size=11),orientation='h',x=.5,xanchor='center'))
 assert len(fig.data)==96
 # A separate UTC detail makes the two gauges and the crest midpoints unambiguous.
 p=paths['54'];detail=go.Figure();gdata={}
 for side,color in [('upstream','grey'),('downstream','black')]:
  g=pd.read_parquet(p/f'observed_{side}.parquet');gdata[side]=g
  detail.add_trace(go.Scatter(x=pd.to_datetime(g.time,utc=True),y=g.discharge_cms,name=side.title()+' gauge',line=dict(color=color,dash='dash')))
 for method,color in [('fixed_x','#00BFD8'),('cunge','#E31A1C')]:
  w=pd.read_parquet(p/f'{method}_best_wave.parquet');detail.add_trace(go.Scatter(x=pd.to_datetime(w.time_utc,utc=True),y=w.q_outlet_cms,name=method+' model',line=dict(color=color)))
 for side,times,color in [('upstream',['2023-06-21T17:20Z','2023-06-22T06:20Z'],'grey'),('downstream',['2023-06-22T12:40Z','2023-06-22T23:50Z'],'black')]:
  tt=[pd.Timestamp(t) for t in times];qq=[q_at(gdata[side],t) for t in tt]
  detail.add_trace(go.Scatter(x=tt,y=qq,mode='markers',name=side.title()+' crest endpoints',marker=dict(color=color,size=10,symbol='circle-open'),hovertemplate='%{x|%d June %H:%M} UTC<br>%{y:.3f} m³/s<extra>'+side+'</extra>'))
  if side=='downstream':
   target=pd.Timestamp(records['54']['target_utc']);qm=sum(qq)/2
   detail.add_trace(go.Scatter(x=[tt[0],target,tt[1]],y=[qq[0],qm,qq[1]],mode='lines',line=dict(color='#E69F00',dash='dot'),showlegend=False,hoverinfo='skip'))
   detail.add_trace(go.Scatter(x=[target],y=[qm],mode='markers',name='Downstream midpoint target (time only)',marker=dict(color='#E69F00',symbol='diamond',size=14),hovertemplate='%{x|%d June %H:%M} UTC<br>Display midpoint Q %{y:.3f} m³/s<br>Only time is calibrated<extra></extra>'))
 detail.update_layout(template='plotly_white',height=640,margin=dict(t=85,b=125,l=85,r=35),title='54 · upstream versus downstream crests · '+label,xaxis=dict(title='UTC (24-hour clock)',range=['2023-06-21T00:00Z','2023-06-24T06:00Z'],tickformat='%d Jun<br>%H:%M'),yaxis_title='Discharge (m³/s) — no normalization',legend=dict(orientation='h',y=-.22,font=dict(size=11)))
 other=NEW if root==OLD else OLD
 revision_note='<p><b>45 target corrected:</b> 20 June 2024, 20:20 UTC, spanning 20:00–20:40. The reviewed-calibration snapshot includes the corrected 45 refit; the historical snapshot retains its old curves. All other target times are unchanged.</p>'
 doc='<!doctype html><html><head><meta charset="utf-8"><title>Calibration targets — '+label+'</title><style>body{font:15px Arial;margin:20px}.figure{overflow-x:auto}p{max-width:1200px;line-height:1.5}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:7px}</style></head><body><p><b>'+label+'.</b> The model curves belong to this directory’s calibration. Reviewed targets are shown for comparison; adding a marker does not recalibrate the curves. <a href="../'+other.name+'/calibration_hydrographs.html">Open the other calibration snapshot</a>.</p><p>Orange diamonds mark every reviewed calibration target. Their heights are the mean discharge of the selected crest endpoints, or the observed height for a selected individual peak. This makes the midpoint visible even above a valley. <b>Calibration fits timing only, not the marker’s height.</b> Orange shading denotes the selected crest span, not a confidence interval. Coincident old/new markers are overlaid. Case 11 remains unresolved; case 54 needs a timing-observable review.</p><div class="figure">'+fig.to_html(full_html=False,include_plotlyjs=True,config={'displaylogo':False})+'</div><h2>54: distinguish the gauges</h2><p>21 June 17:20 UTC is the upstream first crest. The next major upstream crest is 22 June 06:20 UTC. The downstream crest maxima are 22 June 12:40 and 23:50 UTC; their target is 18:15 UTC. The height of the orange midpoint is illustrative and is not constrained by the calibration.</p>'+detail.to_html(full_html=False,include_plotlyjs=False,config={'displaylogo':False})+'<h2>All target timestamps</h2>'+pd.DataFrame(table).to_html(index=False,border=0)+'</body></html>'
 doc=doc.replace('<body>','<body>'+revision_note,1)
 (root/'calibration_hydrographs.html').write_text(doc);fig.write_json(root/'calibration_hydrographs.plotly.json');detail.write_json(root/'case54_gauge_target_detail.plotly.json');(root/'calibration_target_display.json').write_text(json.dumps(table,indent=2))
 print(root/'calibration_hydrographs.html')

if __name__=='__main__':
 build(OLD);build(NEW)
