"""Small-buffer raster-backed review of network 41, unit 57."""
import sys,json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import from_bounds
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import plotly.graph_objects as go
from shapely.geometry import box
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'RAPID/src'))
from rapid_tools import collapse_creation_review as review

def run():
 row=review.case_row('41','S061_unit_57');m=json.loads(Path(row.variant_manifest).read_text())
 islands=review.catalog('41');collapsed=islands[islands.island_id.isin(['island_41_037','island_41_038'])]
 parent=review.vectors(row.base_links_path,'links');pn=review.vectors(row.base_nodes_path,'nodes');child=review.vectors(row.child_links_path,'links')
 selected=parent[parent.id_link.isin(review.int_ids(row.selected_parent_link_ids))]
 xmin,ymin,xmax,ymax=collapsed.geometry.union_all().union(selected.geometry.union_all()).bounds
 bounds=(xmin-150,ymin-150,xmax+150,ymax+150);region=box(*bounds)
 with rasterio.open(m['source_paths']['cleaned_mask']) as base:
  w=from_bounds(*bounds,transform=base.transform).round_offsets().round_lengths();a=base.read(1,window=w);tr=base.window_transform(w)
  with rasterio.open(m['output_paths']['collapsed_mask']) as after:
   assert after.transform==base.transform and after.crs==base.crs
   b=after.read(1,window=w)
 z=np.where(a>0,0,np.where(b>0,1,2));assert z.shape==a.shape
 xs=tr.c+(np.arange(a.shape[1])+.5)*tr.a;ys=tr.f+(np.arange(a.shape[0])+.5)*tr.e
 extent=(tr.c,tr.c+a.shape[1]*tr.a,tr.f+a.shape[0]*tr.e,tr.f)
 out=ROOT/'RAPID/outputs/boundary_contract_repair_20261008/full/41/S061_unit_57';out.mkdir(exist_ok=True,parents=True)
 fig,ax=plt.subplots(figsize=(12,10));ax.imshow(z,cmap=ListedColormap(['white','#a6a6a6','black']),vmin=0,vmax=2,extent=extent,origin='upper',interpolation='nearest')
 interactive=go.Figure(go.Heatmap(x=xs,y=ys,z=z,zmin=0,zmax=2,colorscale=[[0,'white'],[.2499,'white'],[.25,'#a6a6a6'],[.7499,'#a6a6a6'],[.75,'black'],[1,'black']],showscale=False,hovertemplate='E %{x:.0f} m<br>N %{y:.0f} m<extra></extra>',name='Raster context'))
 for value,color,label in [(0,'white','Water'),(1,'#a6a6a6','Collapsed island footprint'),(2,'black','Remaining islands / banks')]:
  interactive.add_trace(go.Scatter(x=[None],y=[None],mode='markers',marker=dict(size=12,color=color,line=dict(color='#555',width=1)),name=label))
 for frame,color,label,dash in [(parent,'#c25a13','Parent links','dot'),(child,'#0789bf','Current child links','solid')]:
  x=[];y=[]
  for v in frame[frame.intersects(region)].itertuples():
   geom=v.geometry.intersection(region)
   parts=list(geom.geoms) if hasattr(geom,'geoms') else [geom]
   for part in parts:
    if part.geom_type!='LineString':continue
    xx,yy=part.xy;x+=list(xx)+[None];y+=list(yy)+[None]
    ax.plot(xx,yy,color=color,lw=1.1,ls=':' if dash=='dot' else '-',alpha=.8)
  interactive.add_trace(go.Scatter(x=x,y=y,mode='lines',line=dict(color=color,width=1.7,dash=dash),name=label,visible=True))
 for v in islands[islands.intersects(region)].itertuples():
  pt=v.geometry.intersection(region).representative_point();label='Island '+v.island_id.rsplit('_',1)[1].lstrip('0');color='black' if v.island_id in set(collapsed.island_id) else 'white'
  ax.text(pt.x,pt.y,label,ha='center',va='center',color=color,fontsize=10,bbox=dict(facecolor='#555' if color=='white' else '#c9c9c9',alpha=.85,edgecolor='none',pad=2))
  interactive.add_annotation(x=pt.x,y=pt.y,text=label,showarrow=False,font=dict(color=color,size=13),bgcolor='#555' if color=='white' else '#c9c9c9')
 ports=pn[pn.id_node.isin([110,112,114,115,127,128])]
 positions={110:(10,10),112:(10,-20),114:(10,10),115:(10,10),127:(-52,-17),128:(10,10)}
 for v in ports.itertuples():
  ax.scatter(v.geometry.x,v.geometry.y,c='#a0009a',s=25,zorder=5);ax.annotate('P'+str(v.id_node),(v.geometry.x,v.geometry.y),xytext=positions[v.id_node],textcoords='offset points',color='#850080',fontsize=10,bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=1))
  dx,dy=positions[v.id_node];interactive.add_annotation(x=v.geometry.x,y=v.geometry.y,text='P'+str(v.id_node),showarrow=True,ax=dx,ay=-dy,font=dict(color='#850080'),bgcolor='white',arrowcolor='#850080')
 ax.set(xlim=bounds[::2],ylim=bounds[1::2],xlabel='Easting (m)',ylabel='Northing (m)',title='Network 41 · S061_unit_57\nIslands 37 and 38 collapsed · 150 m context buffer');ax.set_aspect('equal')
 ax.legend(handles=[Patch(facecolor=c,edgecolor='#555',label=l) for c,l in [('white','Water'),('#a6a6a6','Collapsed island footprint'),('black','Remaining islands / banks')]],loc='upper center',bbox_to_anchor=(.5,-.08),ncol=3)
 fig.tight_layout();fig.savefig(out/'island_context.png',dpi=170,bbox_inches='tight');plt.close(fig)
 interactive.update_layout(template='plotly_white',title='Network 41 · S061_unit_57 · Islands 37 and 38 collapsed<br><sup>150 m buffer; orange dotted: parent; blue: current child. Toggle overlays using legend.</sup>',height=950,margin=dict(t=100,b=100),legend=dict(orientation='h',y=-.1),xaxis=dict(range=[bounds[0],bounds[2]],title='Easting (m)'),yaxis=dict(range=[bounds[1],bounds[3]],title='Northing (m)',scaleanchor='x',scaleratio=1))
 interactive.write_html(out/'island_context.html',include_plotlyjs=True)
 evidence=dict(actual_collapsed_islands=['island_41_037','island_41_038'],selection_metadata=m['collapsed_selection'],removed_land_components=m['bank_label_summary'],correction='Prior declared_target_islands field was inferred from selected-parent polygon faces at 99.9% containment; it was not an explicit island declaration.',buffer_m=150,production_changed=False)
 (out/'island_context_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n');print(out/'island_context.html')
if __name__=='__main__':run()
