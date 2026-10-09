"""Read-only graph-stage visualization and explicit human review records."""
from pathlib import Path
from functools import lru_cache
import json
from datetime import datetime,timezone
import pandas as pd,geopandas as gpd
from shapely.geometry import box
from shapely.ops import polygonize,unary_union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
ROOT=Path(__file__).resolve().parents[3]
AUDIT=ROOT/'RAPID/outputs/anchoring_audit_20261008'
OUT=ROOT/'RAPID/outputs/collapse_creation_review_20261008'

def read_csv(path):return pd.read_csv(path,dtype={'example':str})
def int_ids(value):return [] if pd.isna(value) or str(value)=='' else [int(v) for v in str(value).split(',')]

def load_review_data():
 return read_csv(AUDIT/'review_registry.csv'),read_csv(AUDIT/'boundary_geometry_details.csv'),read_csv(AUDIT/'unit_cut_classification.csv')

def cut_queue(kind='priority',example=None):
 _,b,q=load_review_data()
 choices={'priority':['projection_cut_unresolved_over_100m'],'other_candidates':['projection_cut_unresolved_up_to_100m'],'all_candidates':['projection_cut_unresolved_over_100m','projection_cut_unresolved_up_to_100m'],'displaced_controls':['displaced_projection_nodes_present'],'exact_controls':['exact_boundary_nodes_present'],'all':list(q.review_class.unique())}
 if kind not in choices:raise ValueError(f'Choose one of {list(choices)}')
 q=q[q.review_class.isin(choices[kind])]
 if example is not None:q=q[q.example.eq(str(example).zfill(2))]
 representatives=[]
 for r in q.itertuples():
  names=r.states.split(';');rows=b[b.example.eq(r.example)&b.state.isin(names)].copy()
  if kind=='priority':rows=rows[rows.nearest_child_node_distance_m.gt(100)]
  elif kind in ['other_candidates','all_candidates']:rows=rows[(rows.nearest_child_node_distance_m-rows.nearest_child_line_distance_m).abs().gt(1e-5)]
  elif kind=='displaced_controls':rows=rows[rows.missing_canonical_node]
  representatives.append(rows.iloc[0].state if len(rows) else names[0])
 q=q.copy();q['representative_state']=representatives
 return q.sort_values(['example','states']).reset_index(drop=True)

def case_row(example,state,family=None):
 a,_,_=load_review_data();f=a[a.example.eq(str(example).zfill(2))&a.state.eq(state)]
 if family:f=f[f.family.eq(family)]
 if len(f)!=1:raise ValueError(f'Expected exactly one parent/state reference, found {len(f)}')
 return f.iloc[0]

@lru_cache(maxsize=20)
def vectors(path,kind):
 cols=['id_link','id_us_node','id_ds_node','geometry'] if kind=='links' else ['id_node','geometry']
 return gpd.read_file(path,columns=cols)

def stage_paths(row):
 vp=Path(row.variant_manifest);m=json.loads(vp.read_text()) if vp.exists() else {};real=m.get('local_patch',{}).get('realized_assembly',{}).get('output_paths',{});p=real.get('realized_links');n=real.get('realized_nodes')
 if p and Path(p).exists():stage=('Realized assembly (before anchoring)',p,n)
 else:
  p=m.get('output_paths',{}).get('rivgraph_links');n=m.get('output_paths',{}).get('rivgraph_nodes')
  stage=('Raw RivGraph (pre-anchor proxy)',p,n) if p and Path(p).exists() else ('Pre-anchor geometry unavailable',None,None)
 return [('Parent: original base IDs',row.base_links_path,row.base_nodes_path),stage,('Final anchored: child IDs',row.child_links_path,row.child_nodes_path)]

def catalog(example):
 suffix='_direction130' if str(example)=='15' else ''
 path=ROOT/'network_variants/output'/f'{str(example).zfill(2)}_island{suffix}__anchored/setup/island_candidate_geometries.gpkg'
 return gpd.read_file(path,columns=['island_id','geometry']) if path.exists() else None

def _lines(fig,frame,col,color,name,dash='solid',width=1.5):
 first=True
 for idx,r in frame.iterrows():
  parts=list(r.geometry.geoms) if r.geometry.geom_type=='MultiLineString' else [r.geometry]
  lid=r.get('id_link',idx);uv=f"{r.get('id_us_node','?')} → {r.get('id_ds_node','?')}"
  for part in parts:
   if part.geom_type!='LineString':continue
   coords=list(part.coords)
   fig.add_trace(go.Scatter(x=[p[0] for p in coords],y=[p[1] for p in coords],mode='lines',name=name,legendgroup=name,showlegend=first and col==1,line=dict(color=color,width=width,dash=dash),hovertemplate=f'{name}<br>link {lid}<br>nodes {uv}<extra></extra>'),1,col);first=False

def _polygons(fig,geoms,col,color,name):
 first=True
 for geom in geoms:
  parts=list(geom.geoms) if geom.geom_type=='MultiPolygon' else [geom]
  for p in parts:
   if p.geom_type!='Polygon':continue
   xy=list(p.exterior.coords);fig.add_trace(go.Scatter(x=[v[0] for v in xy],y=[v[1] for v in xy],mode='lines',line=dict(color=color,width=1),fill='toself',fillcolor=color,name=name,legendgroup=name,showlegend=first and col==1,hovertemplate=name+'<extra></extra>'),1,col);first=False

def boundary_table(example,state):
 _,b,_=load_review_data();return b[b.example.eq(str(example).zfill(2))&b.state.eq(state)].copy()

def case_metadata(example,state):
 r=case_row(example,state);meta={k:r[k] for k in ['example','family','state','selected_unit_ids','required_boundary_nodes','missing_boundary_nodes','selected_parent_link_ids','execution_backend','island_target_domain_mismatch','outside_links_not_retained_count','outside_property_changed_count','source_manifest']}
 meta['parent']='S000_base';meta['unit_ids_refer_to']='original parent hierarchy, never the renumbered child hierarchy'
 if (OUT/'target_inventory.csv').exists():
  t=read_csv(OUT/'target_inventory.csv');t=t[t.example.eq(r.example)&t.state.eq(r.state)&t.family.eq(r.family)]
  if len(t):meta.update(t.iloc[0][['declared_target_islands','actual_full_target_islands','partial_target_islands','target_error']].to_dict())
 return pd.DataFrame({'field':list(meta),'value':[str(v) for v in meta.values()]})

def plot_case(example,state,anchor=None,radius_m=80,context_padding_m=None,expected_anchors=None):
 r=case_row(example,state);base=vectors(r.base_links_path,'links');bn=vectors(r.base_nodes_path,'nodes');selected=base[base.id_link.isin(int_ids(r.selected_parent_link_ids))];required=bn[bn.id_node.isin(int_ids(r.required_boundary_nodes))];islands=catalog(r.example)
 expected=bn[bn.id_node.isin(expected_anchors or [])]
 if expected_anchors and len(expected)!=len(set(expected_anchors)):raise ValueError('Expected anchor IDs must exist in the original parent')
 target_ids=[state] if r.family=='island' else []
 if (OUT/'target_inventory.csv').exists():
  t=read_csv(OUT/'target_inventory.csv');t=t[t.example.eq(r.example)&t.state.eq(state)&t.family.eq(r.family)]
  if len(t) and pd.notna(t.iloc[0].actual_full_target_islands):target_ids+=str(t.iloc[0].actual_full_target_islands).split(';')
 targets=islands[islands.island_id.isin(target_ids)] if islands is not None else None
 if anchor is not None:
  point=bn[bn.id_node.eq(int(anchor))]
  if len(point)!=1 or int(anchor) not in set(required.id_node):raise ValueError('Choose an original required parent boundary from the boundary table')
  p=point.iloc[0].geometry;bounds=(p.x-radius_m,p.y-radius_m,p.x+radius_m,p.y+radius_m)
 else:
  geom=selected.geometry.union_all()
  if len(expected):geom=geom.union(expected.geometry.union_all())
  if targets is not None and len(targets):geom=geom.union(targets.geometry.union_all())
  xmin,ymin,xmax,ymax=geom.bounds;pad=context_padding_m or max(40,max(xmax-xmin,ymax-ymin)*.18);bounds=(xmin-pad,ymin-pad,xmax+pad,ymax+pad)
 region=box(*bounds);stages=stage_paths(r);fig=make_subplots(rows=1,cols=3,subplot_titles=[z[0] for z in stages],horizontal_spacing=.065)
 faces=list(polygonize(unary_union(selected.geometry)))
 for col,(title,lp,npth) in enumerate(stages,1):
  _polygons(fig,[g.intersection(region) for g in faces],col,'rgba(255,180,0,0.12)','Recorded replacement domain')
  if targets is not None:_polygons(fig,[g.intersection(region) for g in targets.geometry if g.intersects(region)],col,'rgba(70,160,60,0.28)','Target island(s)')
  nearby=base[base.intersects(region)].copy();nearby.geometry=nearby.geometry.intersection(region)
  _lines(fig,nearby,col,'#777','Parent links','dash',1.5)
  if lp:
   f=vectors(lp,'links');assert f.crs==base.crs;f=f[f.intersects(region)].copy();f.geometry=f.geometry.intersection(region);_lines(fig,f,col,'#0673bd' if col==2 else '#d52c2c' if col==3 else '#333',title,width=2)
   if npth and Path(npth).exists():
    n=vectors(npth,'nodes');n=n[n.intersects(region)];texts=[f"{'P' if col==1 else 'C'}{v}" for v in n.id_node] if 'id_node' in n else list(map(str,n.index));fig.add_trace(go.Scatter(x=n.geometry.x,y=n.geometry.y,mode='markers',marker=dict(size=7,color='#0673bd' if col==2 else '#d52c2c' if col==3 else '#333'),text=texts,name='Graph nodes',showlegend=False,hovertemplate='%{text}<extra></extra>'),1,col)
  ports=required[required.intersects(region)];fig.add_trace(go.Scatter(x=ports.geometry.x,y=ports.geometry.y,mode='markers+text',text=['P'+str(v) for v in ports.id_node],textposition='top right',marker=dict(size=13,symbol='x',color='#9b00b5'),name='Recorded parent boundary',legendgroup='ports',showlegend=col==1,hovertemplate='%{text}<extra></extra>'),1,col)
  ep=expected[expected.intersects(region)]
  if len(ep):fig.add_trace(go.Scatter(x=ep.geometry.x,y=ep.geometry.y,mode='markers+text',text=['Expected P'+str(v) for v in ep.id_node],textposition='bottom right',marker=dict(size=12,symbol='diamond-open',color='#087c46'),name='Expected boundary (review specification)',legendgroup='expected',showlegend=col==1,hovertemplate='%{text}<extra></extra>'),1,col)
  xaxis='x' if col==1 else f'x{col}';fig.update_xaxes(range=[bounds[0],bounds[2]],title_text='Easting (m)',row=1,col=col);fig.update_yaxes(range=[bounds[1],bounds[3]],scaleanchor=xaxis,scaleratio=1,title_text='Northing (m)' if col==1 else None,row=1,col=col)
 fig.update_layout(title=dict(text=f'Network {r.example} · {state} · parent unit(s) {r.selected_unit_ids}'+(f' · boundary P{anchor}' if anchor is not None else ''),x=.02),template='plotly_white',height=650,margin=dict(t=130,b=100,l=65,r=30),legend=dict(orientation='h',x=0,y=-.14),hovermode='closest')
 return fig

DECISIONS={'cut':['correct_projected_cut','missing_cut','wrong_connection','needs_investigation'],'island':['retain_independent_canonical','approved_manual_composite_retain','reuse_validated_unit_output','incorrect_target_set','incorrect_creation','valid_distinct_intervention','needs_investigation']}
def save_review(example,state,kind,decision,notes,reviewer,canonical_state=None):
 if kind not in DECISIONS or decision not in DECISIONS[kind]:raise ValueError('Unknown review decision')
 if not notes.strip() or not reviewer.strip():raise ValueError('Reviewer and evidence notes are required')
 r=case_row(example,state)
 if decision in ['reuse_validated_unit_output','retain_independent_canonical']:
  if not canonical_state:raise ValueError('Name the canonical independent state')
  pairs=read_csv(OUT/'island_independent_target_pairs.csv');match=pairs[pairs.example.eq(r.example)&pairs.island_state.eq(state)&pairs.independent_state.eq(canonical_state)]
  if match.empty or not match.target_island.eq(state).all() or (match.island_partial_count.gt(0)|match.independent_partial_count.gt(0)).any():raise ValueError('No clean singleton target-set match; investigate before recording reuse')
 entry=dict(example=r.example,state=state,fingerprint=r.fingerprint,kind=kind,decision=decision,notes=notes,reviewer=reviewer,canonical_state=canonical_state,time_utc=datetime.now(timezone.utc).isoformat(),effect='review record only; no production deletion/exclusion')
 OUT.mkdir(parents=True,exist_ok=True)
 with (OUT/'human_reviews.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
 return entry

def saved_reviews():
 path=OUT/'human_reviews.jsonl';return pd.read_json(path,lines=True,dtype={'example':str}) if path.exists() else pd.DataFrame()
