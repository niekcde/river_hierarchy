"""Four concrete missing-boundary examples, in original projected coordinates."""
import pandas as pd,geopandas as gpd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shapely.geometry import box
from audit_all_anchoring import OUT,ids

def run():
 a=pd.read_csv(OUT/'transition_audit.csv',dtype={'example':str});cases=[('41','S016_unit_4'),('13','S039_unit_33'),('15','S135_unit_144'),('02','S026_unit_16')]
 fig=make_subplots(rows=2,cols=2,subplot_titles=[f'{ex}: {state}' for ex,state in cases],horizontal_spacing=.09,vertical_spacing=.12)
 for ix,(ex,state) in enumerate(cases):
  r=a[(a.example==ex)&a.state.eq(state)].iloc[0];bl=gpd.read_file(r.base_links_path,columns=['id_link','geometry']);cl=gpd.read_file(r.child_links_path,columns=['id_link','geometry']);bn=gpd.read_file(r.base_nodes_path,columns=['id_node','geometry']).set_index('id_node');cn=gpd.read_file(r.child_nodes_path,columns=['id_node','geometry']);sel=bl[bl.id_link.isin(ids(r.selected_parent_link_ids))];xmin,ymin,xmax,ymax=sel.total_bounds;pad=max(xmax-xmin,ymax-ymin)*.2;window=box(xmin-pad,ymin-pad,xmax+pad,ymax+pad);row=ix//2+1;col=ix%2+1
  for name,frame,color,dash,width in [('Parent',bl,'#555','dash',3),('Child',cl,'#00a7b5','solid',2)]:
   first=True
   for z in frame[frame.intersects(window)].itertuples():
    line=z.geometry.intersection(window);parts=list(line.geoms) if line.geom_type=='MultiLineString' else [line]
    for part in parts:
     if part.geom_type!='LineString':continue
     xy=list(part.coords);fig.add_trace(go.Scatter(x=[x[0] for x in xy],y=[x[1] for x in xy],mode='lines',line=dict(color=color,dash=dash,width=width),name=name,legendgroup=name,showlegend=ix==0 and first,hovertemplate=f'{name} link {z.id_link}<extra></extra>'),row,col);first=False
  missing=bn.loc[ids(r.missing_boundary_nodes)];fig.add_trace(go.Scatter(x=missing.geometry.x,y=missing.geometry.y,mode='markers+text',marker=dict(color='red',symbol='x',size=12),text=[f'node {v}' for v in missing.index],textposition='top right',name='Missing original boundary',legendgroup='missing',showlegend=ix==0,hovertemplate='%{text}<br>E %{x:.1f} m<br>N %{y:.1f} m<extra></extra>'),row,col)
  nearby=cn[cn.intersects(window)];fig.add_trace(go.Scatter(x=nearby.geometry.x,y=nearby.geometry.y,mode='markers',marker=dict(color='#00a7b5',size=5),name='Child nodes',legendgroup='nodes',showlegend=ix==0,text=nearby.id_node,hovertemplate='Child node %{text}<extra></extra>'),row,col)
  axis=str(ix+1) if ix else '';fig.update_yaxes(scaleanchor='x'+axis,scaleratio=1,title_text='Northing (m)',row=row,col=col);fig.update_xaxes(title_text='Easting (m)',row=row,col=col)
 fig.update_layout(title=dict(text='Missing collapse boundaries: parent and final child geometry',x=.5),height=1050,width=1300,margin=dict(t=125,b=70,l=90,r=60),legend=dict(orientation='h',y=1.08,x=.5,xanchor='center'))
 fig.write_html(OUT/'missing_anchor_examples.html',include_plotlyjs=True)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 preview,axs=plt.subplots(2,2,figsize=(12,10),layout='constrained')
 for trace in fig.data:
  axis=trace.xaxis or 'x';j=int(axis[1:])-1 if axis[1:] else 0;ax=axs.flat[j]
  if trace.mode=='lines':ax.plot(trace.x,trace.y,color=trace.line.color,linestyle='--' if trace.line.dash=='dash' else '-',linewidth=1)
  elif trace.name=='Child nodes':ax.scatter(trace.x,trace.y,c='#00a7b5',s=12)
  else:
   ax.scatter(trace.x,trace.y,c='red',marker='x',s=50)
   for x,y,label in zip(trace.x,trace.y,trace.text):ax.annotate(label,(x,y),xytext=(5,5),textcoords='offset points',color='red',fontsize=8)
 for j,(ex,state) in enumerate(cases):
  ax=axs.flat[j];ax.set_title(f'{ex}: {state}');ax.set_aspect('equal');ax.set_xlabel('Easting (m)');ax.set_ylabel('Northing (m)');ax.ticklabel_format(useOffset=False,style='plain');ax.tick_params(labelsize=7)
 preview.suptitle('Parent: dashed grey | Child: cyan | Missing original boundary: red cross')
 preview.savefig(OUT/'missing_anchor_examples.png',dpi=140);plt.close(preview)
 print(OUT/'missing_anchor_examples.html')
if __name__=='__main__':run()
