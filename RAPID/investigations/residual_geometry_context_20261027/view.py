from pathlib import Path
import sys,json,html
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import pandas as pd,geopandas as gpd,pyogrio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs
from network_variants.reviewed_geometry import ReviewedGeometry
r=ReviewedGeometry();BASE=ROOT/'RAPID/outputs/accepted_boundary_trial_20261027_v4';OUT=ROOT/'RAPID/outputs/residual_geometry_context_20261027';bc=json.loads((BASE/'37/S289_unit_143/boundary_contract.json').read_text());selected=set(bc['original_selected']);boundary=set(bc['boundary_nodes']);frames=[]
for state in ['S000_base','S289_unit_143']:
 rec=r.lookup('37','indep',state);f=pd.read_parquet(rec['original_frame']);g=pyogrio.read_dataframe(rec['links']).set_index('id_link');f=f.merge(g[['geometry']],left_on='id_link',right_index=True);frames.append(f)
frames.append(gpd.read_parquet(BASE/'37/S289_unit_143/geometry_candidate.parquet'))
parent=frames[0].set_index('id_link');origin=parent.loc[420].geometry.coords[0];ox,oy=origin
# Canonical node positions from directed geometry endpoints. Reference line orientation is checked against frame raw nodes.
positions=[]
for col,f in enumerate(frames):
 if col<2:
  rec=r.lookup('37','indep',['S000_base','S289_unit_143'][col]);ns=pyogrio.read_dataframe(rec['nodes']).set_index('id_node');mp=dict(zip(f.u,f.id_us_node))|dict(zip(f.v,f.id_ds_node));positions.append({int(v):ns.loc[raw].geometry for v,raw in mp.items()})
 else:
  from shapely.geometry import Point
  positions.append({int(q.u):Point(q.geometry.coords[0]) for q in f.itertuples()}|{int(q.v):Point(q.geometry.coords[-1]) for q in f.itertuples()})
refpos=positions[0]
def window(nodes,pad=100):
 points=[refpos[n] for n in nodes];x=[v.x for v in points];y=[v.y for v in points];pad=max(pad,.12*max(max(x)-min(x),max(y)-min(y)));return [(min(x)-pad-ox)/1000,(max(x)+pad-ox)/1000],[(min(y)-pad-oy)/1000,(max(y)+pad-oy)/1000]
unit_nodes=set(frames[0][frames[0].id_link.isin(selected)].u)|set(frames[0][frames[0].id_link.isin(selected)].v)
context_nodes=unit_nodes|{287,293,299,275};views={'whole_unit':window(unit_nodes,150),'surrounding_network':window(context_nodes,500),'reversed_junction':window([281,286,287,293],100),'connector_close_up':window([286,287],70),'approved_296_restoration':window([294,295,296,299],100)}
fig=make_subplots(rows=1,cols=3,subplot_titles=['Reference: selected unit','Published collapse','Isolated geometry after approved repairs'],horizontal_spacing=.045);tables=[]
for col,(f,points) in enumerate(zip(frames,positions),1):
 for q in f.itertuples():
  line=q.geometry;hot=q.id_link==434 if col==1 else q.id_link==979;chosen=q.id_link in selected if col==1 else (getattr(q,'role',None)=='selected_replacement' if col==3 else q.id_link in [975,976,977,978,980,981,982,983,984,985,987,988,989,990,991,992])
  # Do not infer published unit membership from unmatched exterior paths.
  color='#d34b38' if hot else '#087f8c' if col==1 and chosen or col==3 and chosen else '#7c8fa0';width=4 if hot else 2 if color=='#087f8c' else 1.3
  xr,yr=views['surrounding_network'];bounds=line.bounds
  if bounds[2]<ox+xr[0]*1000 or bounds[0]>ox+xr[1]*1000 or bounds[3]<oy+yr[0]*1000 or bounds[1]>oy+yr[1]*1000:continue
  parts=[line] if line.geom_type=='LineString' else list(line.geoms);xx=[];yy=[]
  for part in parts:
   xy=list(part.coords);xx.extend([(v[0]-ox)/1000 for v in xy]+[None]);yy.extend([(v[1]-oy)/1000 for v in xy]+[None])
  name='Restored parent '+str(-q.id_link) if q.id_link<0 else 'Link '+str(q.id_link)
  fig.add_trace(go.Scatter(x=xx,y=yy,mode='lines',line=dict(color=color,width=width),text=f'{name}: {int(q.u)} → {int(q.v)}; {q.link_length_m:.3f} m',hovertemplate='%{text}<extra></extra>',showlegend=False),row=1,col=col)
  if q.u in context_nodes or q.v in context_nodes or hot:
   part=max(parts,key=lambda x:x.length);up=points[int(q.u)];rev=up.distance(part.interpolate(0))>up.distance(part.interpolate(1,normalized=True));a=part.interpolate(.55 if rev else .45,normalized=True);b=part.interpolate(.45 if rev else .55,normalized=True);sx='x' if col==1 else f'x{col}';sy='y' if col==1 else f'y{col}';fig.add_annotation(x=(b.x-ox)/1000,y=(b.y-oy)/1000,ax=(a.x-ox)/1000,ay=(a.y-oy)/1000,xref=sx,yref=sy,axref=sx,ayref=sy,showarrow=True,arrowhead=2,arrowwidth=1.5,arrowcolor=color,text='')
  if q.u in [286,287] or q.v in [286,287]:tables.append(dict(panel=col,link=int(q.id_link),u=int(q.u),v=int(q.v),length_m=float(q.link_length_m)))
 for node,point in points.items():
  xr,yr=views['surrounding_network'];x=(point.x-ox)/1000;y=(point.y-oy)/1000
  if not xr[0]<=x<=xr[1] or not yr[0]<=y<=yr[1]:continue
  fig.add_trace(go.Scatter(x=[x],y=[y],mode='markers+text',text=[str(node)],textposition='top left' if node in [286,296] else 'bottom right',textfont=dict(size=10),marker=dict(size=9 if node in boundary else 5,color='#cc951d' if node in boundary else '#263744'),showlegend=False,hovertemplate=f'Node {node}'+(' — selected boundary' if node in boundary else '')+'<extra></extra>'),row=1,col=col)
 xr,yr=views['whole_unit'];fig.update_xaxes(range=xr,title_text='Local easting (km)',row=1,col=col);fig.update_yaxes(range=yr,scaleanchor='x' if col==1 else f'x{col}',scaleratio=1,row=1,col=col)
fig.update_layout(height=750,template='plotly_white',margin=dict(l=35,r=15,t=50,b=50))
residual=json.loads((OUT/'endpoint_audit.json').read_text());solved=[v for v in residual if v['solved']]
page='''<!doctype html><meta charset="utf-8"><title>Unit 143 context and remaining exterior geometry</title><style>body{font:16px system-ui;margin:28px;color:#263744;line-height:1.5}p{max-width:1250px}button{font:inherit;padding:9px;margin:5px}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border:1px solid #ddd}.note{background:#f1f6f8;padding:15px}#plot{min-width:1050px}</style><h1>37 / unit 143 — whole-unit context</h1><p>Red is the additional exterior connector 434 / 979. Gold nodes are selected boundaries. Reference teal is the original selected unit; candidate teal is its declared replacement. The middle published panel uses neutral colours because unmatched exterior reaches must not be labelled as selected. Flow arrows show directed connectivity; hover reaches for IDs and lengths.</p><p>The right panel includes the approved restoration at node 296. It does not yet contain a fitted head solution or any additional geometry change for connector 979. The reversal is already present in the published collapse.</p><div id="views"></div><div style="overflow-x:auto"><div id="plot"></div></div><div class="note"><b>Decision under consideration:</b> keep the existing reversal and explicitly permit boundary flexibility, using the same relative-exterior-drop objective as the other accepted reversal cases. Boundary 286 remains a boundary; exterior node 287 does not become selected. Any release of 287 must be declared explicitly. The original reference heads give −0.039021 m drop along 286 → 287, so both cannot stay fixed. Your “looks reasonable” comment is recorded as tentative, not a final approval.</div><h2>Connections around 286 and 287</h2>__CONNECTIONS__<h1>What to do with the remaining 21 solved cases</h1><p>These contain <b>34 exterior reach instances</b> with residual geometry differences. The endpoint audit identifies which shared boundary coordinates differ from reference. The two additional residual reaches in blocked unit 143 are listed separately below.</p><p><b>Recommended next trial:</b> keep the current declared unit selection and approved exceptions; restore each residual exterior reach to its original polyline and length; restore its moved shared boundary coordinate to the reference position; reconnect the incident selected-side centreline to that same coordinate. Then rerun the local frozen-floor/head solve. Merely changing a length number or freezing a slope would leave geometry and shared heads inconsistent.</p><p>This restoration is a proposal, not an applied change. For every moved boundary, check all incident reaches together, not reach by reach. If restoring the coordinate produces a crossing, overlap, zero-length connector or infeasible local head constraints, retain that case as an explicit exception for review; do not expand the region automatically. Accepted direction reversals and the unit-287 scope decision remain explicit.</p><p>Keeping current exterior geometry is an alternative, but it explicitly relaxes the original fixed-exterior experiment. With fixed endpoint heads and changed length, slope changes follow directly from S = ΔH/L.</p>__RESIDUAL__<script>__PLOTLY__</script><script>const figure=__FIG__;const views=__VIEWS__;Plotly.newPlot('plot',figure.data,figure.layout,{responsive:true,scrollZoom:true});Object.keys(views).forEach(k=>{const b=document.createElement('button');b.textContent=k.replaceAll('_',' ');b.onclick=()=>{const r=views[k];let u={};for(let i=1;i<=3;i++){const s=i===1?'':i;u['xaxis'+s+'.range']=r[0];u['yaxis'+s+'.range']=r[1]}Plotly.relayout('plot',u)};document.getElementById('views').appendChild(b)});</script>'''
def table(rows,cols):return '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v.get(k,'')))+'</td>' for k in cols)+'</tr>' for v in rows)+'</table>'
page=page.replace('__CONNECTIONS__',table(tables,['panel','link','u','v','length_m'])).replace('__RESIDUAL__',table(residual,['key','parent_link','child_link','length_delta_m','moved_endpoints','solved'])).replace('__PLOTLY__',get_plotlyjs()).replace('__FIG__',fig.to_json()).replace('__VIEWS__',json.dumps(views))
(OUT/'unit143_and_residual_geometry.html').write_text(page);(OUT/'unit143_connections.json').write_text(json.dumps(tables,indent=2));print('Context HTML ready')
