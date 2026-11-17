"""Standalone offline review of the 27 unresolved boundary interpretations."""
from check import *
import html
import pyogrio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs

def main():
 cases=[z for z in js(PRIOR/'state_results.json') if z['feasibility_status'].startswith('not_tested')];payload=[];cache={}
 for z in cases:
  ex=str(z['example']).zfill(2);state=z['state'];r=release.lookup(ex,'indep',state);c=js(r['contract_path']);child=frame(r['original_frame']);cg=pyogrio.read_dataframe(r['links'],columns=['id_link']);cn=pyogrio.read_dataframe(r['nodes'],columns=['id_node']).set_index('id_node')
  if ex not in cache:
   br=release.lookup(ex,'indep','S000_base');cache[ex]=(frame(br['original_frame']),pyogrio.read_dataframe(br['links'],columns=['id_link']),pyogrio.read_dataframe(br['nodes'],columns=['id_node']).set_index('id_node'))
  parent,pg,pn=cache[ex];heads=head_series(parent);selected=set(c['selected_parent_links']);paths=outside_paths(r,parent,child,selected);outside={lid for p in paths for lid in p['child_links']};active=set(child.id_link)-outside
  bad={int(v) for v in (z.get('exterior_path_errors') or []) if str(v).isdigit()};sub=set(z.get('exterior_subdivision_nodes') or []);manual={k:v for k,v in c.items() if k.startswith('reviewed_')}
  suspect=[p for p in paths if p['parent_link'] in bad or (set(child[child.id_link.isin(p['child_links'])].u)|set(child[child.id_link.isin(p['child_links'])].v))&sub]
  highlight={p['parent_link'] for p in suspect};chigh={lid for p in suspect for lid in p['child_links']}
  # Include incident candidates if a saved path is absent/invalid; mark as candidates, never accept automatically.
  ends=set(parent[parent.id_link.isin(highlight)].u)|set(parent[parent.id_link.isin(highlight)].v)
  candidates=set(child[child.u.isin(ends)|child.v.isin(ends)].id_link)-chigh
  focus=pg[pg.id_link.isin(selected|highlight)];bounds=focus.total_bounds;ox,oy=bounds[:2];dx=max(bounds[2]-ox,100);dy=max(bounds[3]-oy,100)
  fig=make_subplots(rows=1,cols=2,subplot_titles=['Reference','Saved collapse'],horizontal_spacing=.07)
  tables=[]
  for col,geo,f,ns,chosen,hot in [(1,pg,parent,pn,selected,highlight),(2,cg,child,cn,active,chigh)]:
   pairs=pd.concat([f[['id_us_node','u']].rename(columns={'id_us_node':'raw','u':'node'}),f[['id_ds_node','v']].rename(columns={'id_ds_node':'raw','v':'node'})]).drop_duplicates();inverse=dict(zip(pairs.node,pairs.raw));fi=f.set_index('id_link');near=geo.cx[ox-.25*dx:bounds[2]+.25*dx,oy-.25*dy:bounds[3]+.25*dy]
   for row in near.itertuples():
    lid=int(row.id_link);q=fi.loc[lid];color='#d44937' if lid in hot else '#087f8c' if lid in chosen else '#9b59b6' if col==2 and lid in candidates else '#c4c9cd';width=3 if lid in hot|chosen else 1
    geom=row.geometry;parts=[geom] if geom.geom_type=='LineString' else list(geom.geoms);xx=[];yy=[]
    for line in parts:
     xy=list(line.coords);xx += [(p[0]-ox)/1000 for p in xy]+[None];yy += [(p[1]-oy)/1000 for p in xy]+[None]
    label=f'{lid}: {int(q.u)} → {int(q.v)}; L={q.link_length_m:.3f} m'
    fig.add_trace(go.Scatter(x=xx,y=yy,mode='lines',line=dict(color=color,width=width),text=label,hovertemplate='%{text}<extra></extra>',showlegend=False),row=1,col=col)
    if lid in hot|chosen or col==2 and lid in candidates:
     line=max(parts,key=lambda g:g.length);us=ns.loc[inverse[q.u]].geometry;reverse=us.distance(line.interpolate(0))>us.distance(line.interpolate(1,normalized=True));a=line.interpolate(.55 if reverse else .45,normalized=True);b=line.interpolate(.45 if reverse else .55,normalized=True)
     fig.add_annotation(x=(b.x-ox)/1000,y=(b.y-oy)/1000,ax=(a.x-ox)/1000,ay=(a.y-oy)/1000,xref='x' if col==1 else 'x2',yref='y' if col==1 else 'y2',axref='x' if col==1 else 'x2',ayref='y' if col==1 else 'y2',showarrow=True,arrowhead=2,arrowsize=1,arrowwidth=1.5,arrowcolor=color,text='')
     tables.append(dict(side='reference' if col==1 else 'collapsed',link=lid,upstream=int(q.u),downstream=int(q.v),length_m=float(q.link_length_m),reference_head_us=float(heads.loc[q.u]) if q.u in heads.index else None,reference_head_ds=float(heads.loc[q.v]) if q.v in heads.index else None))
   nodes=set(c['boundary_nodes'])|sub|ends
   for k in ('reviewed_internal_parent_nodes','reviewed_provisional_parent_nodes','reviewed_required_boundary_nodes'):nodes.update(c.get(k,[]))
   for v in sorted(nodes):
    if v not in inverse:continue
    p=ns.loc[inverse[v]].geometry
    fig.add_trace(go.Scatter(x=[(p.x-ox)/1000],y=[(p.y-oy)/1000],mode='markers+text',text=[str(v)],textposition='top center',marker=dict(size=8,color='#d49b16' if v in c['boundary_nodes'] else '#8e44ad'),hovertext=f'Node {v}; reference head '+(f'{heads.loc[v]:.8f} m' if v in heads.index else 'not a reference node'),hoverinfo='text',showlegend=False),row=1,col=col)
   fig.update_xaxes(range=[-.25*dx/1000,1.25*dx/1000],title_text='Local easting (km)',row=1,col=col);fig.update_yaxes(range=[-.25*dy/1000,1.25*dy/1000],scaleanchor='x' if col==1 else 'x2',row=1,col=col)
  fig.update_layout(height=670,template='plotly_white',margin=dict(l=50,r=25,t=40,b=50))
  group=f'{ex}: manual interpretation' if manual else f'{ex}: subdivision {sorted(sub)}' if sub else f'{ex}: exterior {sorted(bad)}'
  payload.append(dict(key=ex+'/'+state,group=group,figure=json.loads(fig.to_json()),table=tables,details=dict(previous_decision=r.get('decision'),review_folder=r.get('review_folder'),contract_path=r.get('contract_path'),saved_boundary_nodes=c['boundary_nodes'],manual_interpretation=manual,path_records=suspect,exception=z.get('exception'),subdivision_nodes=sorted(sub)),question='Confirm the saved manual interpretation and which nodes are fixed.' if manual else 'Does this chain preserve the exterior reach direction and geometry, with only an added intermediate node?' if sub else 'Identify the actual directed child correspondence for the highlighted parent reach. Purple reaches are inspection candidates only.'))
 payload.sort(key=lambda x:(x['group'],x['key']))
 shell='''<!doctype html><meta charset="utf-8"><title>27 boundary exceptions</title><style>body{font:15px system-ui;margin:24px;color:#263744}select,button,textarea{font:inherit;padding:8px}table{border-collapse:collapse;font-size:13px}td,th{padding:6px;border:1px solid #ddd}pre{white-space:pre-wrap;background:#f2f4f6;padding:15px}textarea{width:90%;height:75px}.key{font-weight:bold}p{max-width:1100px;line-height:1.5}</style><h1>Review the 27 boundary exceptions</h1><p>Read-only evidence from the published geometry. These cases remain outside the frozen-floor feasibility result. No hydraulic inputs were changed. Review grouped cases individually; matching issue labels do not imply identical geometry.</p><p>Teal: selected/replacement reaches. Red: problematic exterior reach/path. Purple: possible correspondence or additional node. Gold: saved boundary node. Arrows follow frame direction; hover lines for IDs, endpoints and lengths. Some selected reaches are very short: zoom to inspect.</p><p><b>Review order:</b> verify identity and direction → check retained geometry/length → confirm existing boundary interpretation → record outcome. A subdivision needs an intermediate head consistent with the original reach; it is not automatically routing-equivalent.</p><button onclick="move(-1)">Previous</button> <select id="pick" onchange="show()"></select> <button onclick="move(1)">Next</button><h2 id="title"></h2><p id="question"></p><div id="plot"></div><details><summary>Saved decisions, contracts and exterior paths</summary><pre id="details"></pre></details><h3>Directed reach evidence</h3><div id="table"></div><h3>Your review</h3><select id="outcome" onchange="save()"><option>Unreviewed</option><option>Verified correspondence</option><option>Verified subdivision</option><option>Apply existing manual interpretation</option><option>Unresolved geometry/direction issue</option></select><br><textarea id="notes" placeholder="Record parent ↔ child link IDs, boundary interpretation, and reasoning" oninput="save()"></textarea><p>Notes persist locally in this browser when permitted. Export them before closing or switching browser.</p><button onclick="download()">Download review decisions (JSON)</button><script>__PLOTLY__</script><script>const cases=__DATA__;const pick=document.getElementById('pick');let reviews={};try{reviews=JSON.parse(localStorage.getItem('river-boundary-review-20261027')||'{}')}catch(e){};cases.forEach((c,i)=>pick.add(new Option(c.group+' — '+c.key,i)));function esc(v){return String(v??'—').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;')}function show(){const c=cases[+pick.value];document.getElementById('title').textContent=c.key;document.getElementById('question').textContent=c.question;Plotly.react('plot',c.figure.data,c.figure.layout,{responsive:true,scrollZoom:true});document.getElementById('details').textContent=JSON.stringify(c.details,null,2);const cols=Object.keys(c.table[0]||{});document.getElementById('table').innerHTML='<table><tr>'+cols.map(k=>'<th>'+esc(k)+'</th>').join('')+'</tr>'+c.table.map(r=>'<tr>'+cols.map(k=>'<td>'+esc(r[k])+'</td>').join('')+'</tr>').join('')+'</table>';document.getElementById('outcome').value=reviews[c.key]?.outcome||'Unreviewed';document.getElementById('notes').value=reviews[c.key]?.notes||''}function move(n){pick.value=Math.max(0,Math.min(cases.length-1,+pick.value+n));show()}function save(){reviews[cases[+pick.value].key]={outcome:document.getElementById('outcome').value,notes:document.getElementById('notes').value};try{localStorage.setItem('river-boundary-review-20261027',JSON.stringify(reviews))}catch(e){}}function download(){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(reviews,null,2)],{type:'application/json'}));a.download='boundary-review-decisions.json';a.click();URL.revokeObjectURL(a.href)}show();</script>'''
 (OUT/'exceptions_review.html').write_text(shell.replace('__PLOTLY__',get_plotlyjs()).replace('__DATA__',json.dumps(payload).replace('</','<\\/')))
 (OUT/'exception_review_evidence.json').write_text(json.dumps([{k:v for k,v in c.items() if k!='figure'} for c in payload],indent=2));print('HTML cases',len(payload))
if __name__=='__main__':main()
