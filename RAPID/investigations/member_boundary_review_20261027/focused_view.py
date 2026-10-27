"""Visual inspection of saved boundaries and three slope-feasibility examples."""
from pathlib import Path
import sys,json,html
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT)]
import pandas as pd
import pyogrio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from network_variants.reviewed_geometry import ReviewedGeometry
from feasibility import check_slopes,head_series
OUT=ROOT/'RAPID/outputs/member_boundary_review_20261027'


def main():
    release=ReviewedGeometry();ex='22';br=release.lookup(ex,'indep','S000_base')
    parent=pd.read_parquet(br['original_frame']);heads=head_series(parent)
    pg=pyogrio.read_dataframe(br['links'],columns=['id_link']);pn=pyogrio.read_dataframe(br['nodes'],columns=['id_node']).set_index('id_node')
    pos=pd.read_parquet(ROOT/f'RAPID/assets/bounded_robust_slopes_20261002/{ex}/floor_0.05/old/nodes.parquet').set_index('id_node').reference_section_dist_out_m
    body=['<!doctype html><meta charset="utf-8"><title>Existing member boundaries — slope feasibility</title><style>body{font:16px system-ui;margin:30px;color:#263744;max-width:1300px}p{max-width:1000px;line-height:1.55}section{border-top:1px solid #ccc;margin-top:35px}code{background:#eee}</style><h1>Existing member boundaries: slope feasibility only</h1><p>Saved collapse selections and reviewed node identities are reused. No geometry, head, slope, hydraulic parameter or routing input has been changed. Labels show reference boundary heads; hover also shows the currently saved collapsed-case head.</p>']
    cases=[]
    for state in ['S296_unit_30','S023_unit_136','S166_unit_309']:
        r=release.lookup(ex,'indep',state);contract=json.loads(Path(r['contract_path']).read_text());child=pd.read_parquet(r['original_frame']);cg=pyogrio.read_dataframe(r['links'],columns=['id_link']);cn=pyogrio.read_dataframe(r['nodes'],columns=['id_node']).set_index('id_node')
        selected=contract['selected_parent_links'];boundary=contract['boundary_nodes'];paths=json.loads((Path(r['source_bundle'])/'exterior_paths.json').read_text());exterior={lid for p in paths for lid in p['child_links']};patch=child[~child.id_link.isin(exterior)];ext=parent[~parent.id_link.isin(selected)]
        qa=check_slopes(patch,heads,set(ext.u)|set(ext.v)|set(boundary),pos);cases.append(dict(state=state,**qa))
        critical=qa['critical_floor_path'] if qa['status']=='relative_floor_infeasible' else qa['critical_positive_path'];witness=set(critical['path_links'])
        fg=make_subplots(rows=1,cols=2,subplot_titles=['Reference member and existing boundary','Saved collapsed geometry; same boundary identities'],horizontal_spacing=.06)
        bounds=pg[pg.id_link.isin(selected)].total_bounds;ox,oy=bounds[0],bounds[1]
        for col,geo,frame,ns,active in [(1,pg,parent,pn,set(selected)),(2,cg,child,cn,set(patch.id_link))]:
            for name,ids,color,width in [('Exterior context',set(geo.id_link)-active,'#b9bec3',1),('Selected / replacement',active,'#087f8c',2.5),('Critical directed path',witness if col==2 else set(),'#c54437',4)]:
                if not ids:continue
                xx=[];yy=[]
                for line in geo[geo.id_link.isin(ids)].geometry:
                    for part in ([line] if line.geom_type=='LineString' else line.geoms):
                        xy=list(part.coords);xx.extend([(p[0]-ox)/1000 for p in xy]+[None]);yy.extend([(p[1]-oy)/1000 for p in xy]+[None])
                fg.add_trace(go.Scatter(x=xx,y=yy,mode='lines',line=dict(color=color,width=width),name=name,showlegend=col==2,hoverinfo='skip'),row=1,col=col)
            pairs=pd.concat([frame[['id_us_node','u']].rename(columns={'id_us_node':'raw','u':'node'}),frame[['id_ds_node','v']].rename(columns={'id_ds_node':'raw','v':'node'})]).drop_duplicates();inverse=dict(zip(pairs.node,pairs.raw));current=head_series(frame)
            pts=[ns.loc[inverse[b]].geometry for b in boundary]
            fg.add_trace(go.Scatter(x=[(p.x-ox)/1000 for p in pts],y=[(p.y-oy)/1000 for p in pts],mode='markers+text',marker=dict(color='#da941f',size=10,line=dict(color='white',width=1)),text=[f'{b}: {heads.loc[b]:.5f} m' for b in boundary],textposition='top center',textfont=dict(size=10),customdata=[[b,heads.loc[b],current.loc[b]] for b in boundary],hovertemplate='Boundary %{customdata[0]}<br>Reference head %{customdata[1]:.7f} m<br>Saved case head %{customdata[2]:.7f} m<extra></extra>',name='Fixed reference boundary',showlegend=col==2),row=1,col=col)
            mx=max((bounds[2]-ox)/1000,.1);my=max((bounds[3]-oy)/1000,.1)
            fg.update_xaxes(range=[-.15*mx,1.15*mx],title_text='Local easting (km)',row=1,col=col)
            fg.update_yaxes(range=[-.15*my,1.15*my],title_text='Local northing (km)',scaleanchor='x' if col==1 else 'x2',scaleratio=1,row=1,col=col)
        fg.update_layout(height=650,template='plotly_white',legend=dict(orientation='h',y=-.16),margin=dict(t=65,b=95))
        desc=f"Critical path {critical['upstream']} → {critical['downstream']}: reference drop {critical['available_drop_m']:.9f} m."
        if 'required_drop_m' in critical:desc+=f" Current 1% floor requires {critical['required_drop_m']:.9f} m."
        body.append(f'<section><h2>22 / {state}: {qa["status"]}</h2><p>{html.escape(desc)}</p>'+fg.to_html(full_html=False,include_plotlyjs=True if len(cases)==1 else False)+'</section>')
    (OUT/'focused_boundary_review.html').write_text('\n'.join(body));(OUT/'focused_cases.json').write_text(json.dumps(cases,indent=2))
    print('Saved',OUT/'focused_boundary_review.html')

if __name__=='__main__':main()
