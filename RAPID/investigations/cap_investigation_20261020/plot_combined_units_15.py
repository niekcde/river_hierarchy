"""Actual mapped before/after geometry for two combined cases in example 15."""
import hashlib
import json
from pathlib import Path
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
from shapely.ops import unary_union

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'RAPID/outputs/uncapped_event_window_20261020'
EXP=BASE/'combination_expansion_20261026'
DEST=EXP/'mapped_examples_15'
COLORS=['#177eab','#d66b28']
SOURCES={}


def track(path):
    path=Path(path);SOURCES[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();return path


def main():
    DEST.mkdir(exist_ok=True)
    plan=json.loads(track(ROOT/'RAPID/outputs/event_window_plan_20261013/plan.json').read_text())
    original=next(c for c in plan['cases'] if c['example']=='15')
    job=next(j for j in original['jobs'] if j['state_id']=='S000_base')
    geo_path=Path(job['parameters']).parents[3]/'variant/widths/links_with_width_families.gpkg'
    geo=gpd.read_file(track(geo_path)).set_index('id_link')
    assert geo.crs.is_projected
    current=next(c for c in json.loads(track(EXP/'routing_plan.json').read_text())['cases'] if c['example']=='15')
    jobs={j['state_id']:j for j in current['jobs']}
    parent=pd.read_parquet(track(jobs['S000_base']['parameters']))
    segments=gpd.read_file(track(ROOT/'RAPID/outputs/hydrograph_closed_unit_extension_20261019/variants/15/all_closed_units_collapsed/links.gpkg'))
    assert segments.crs==geo.crs
    blocks={b['state']:b for b in json.loads(track(ROOT/'RAPID/outputs/hydrograph_closed_unit_pairing_20261019/eligible_blocks.json').read_text())['15']}
    variants=json.loads(track(EXP/'variants.json').read_text())
    graph=nx.MultiDiGraph()
    for r in parent.itertuples():graph.add_edge(int(r.id_us_node),int(r.id_ds_node),key=int(r.id_link))
    records=[]
    for state,names,labels,stem in [
        ('expanded_a3d1891a10d0',['S097_unit_11','S059_unit_10'],['Unit 11','Unit 10'],'units_11_and_10'),
        ('expanded_9b1ae31de20e',['CB007_bubble_9','CB006_bubble_10'],['Compound 9','Compound 10'],'compounds_9_and_10')]:
        v=next(r for r in variants if r['example']=='15' and r['state']==state)
        assert v['components']==names
        selected=[blocks[n] for n in names]
        connectors=[];node=selected[0]['sink']
        while node!=selected[1]['source']:
            edges=list(graph.out_edges(node,keys=True));assert len(edges)==1
            _,node,key=edges[0];connectors.append(key)
        pids=set(connectors)|set(selected[0]['parent_ids'])|set(selected[1]['parent_ids'])
        line=pd.read_csv(track(EXP/f'parameters/15/{state}/lineage.csv'))
        child=pd.read_parquet(track(jobs[state]['parameters']))
        pparts=[geo.loc[b['parent_ids']].geometry.tolist() for b in selected]
        cparts=[];childids=[]
        for b,n in zip(selected,names):
            part=segments[segments.block_state.eq(n)]
            assert len(part)==1
            ids=line.loc[line.role.eq('collapsed_block') & line.parent_link_id.isin(b['parent_ids']),'child_link_id'].unique()
            assert len(ids)==1
            c=child[child.id_link.eq(ids[0])];assert len(c)==1
            replacement=part.geometry.iloc[0]
            np.testing.assert_allclose(replacement.length,c.link_length_m.iloc[0],rtol=1e-9,atol=1e-6)
            cparts.append([replacement]);childids.append(int(ids[0]))
        for key in connectors:
            p=parent[parent.id_link.eq(key)].iloc[0];c=child[child.id_link.eq(key)].iloc[0]
            np.testing.assert_allclose(p[['link_length_m','width']].to_numpy(float),c[['link_length_m','width']].to_numpy(float))
        background=[geo.loc[i].geometry for i in connectors]
        allgeom=unary_union(sum(pparts+cparts,[])+background)
        xmin,ymin,xmax,ymax=allgeom.bounds
        origin=np.array([xmin,ymin]);dx=(xmax-xmin)/1000;dy=(ymax-ymin)/1000
        pad=max(dx,dy)*.09
        fig,axes=plt.subplots(1,2,figsize=(11,7),layout='constrained',sharex=True,sharey=True)
        def draw(ax,lines,color,lw):
            for geometry in lines:
                xy=(np.array(geometry.coords)-origin)/1000
                ax.plot(xy[:,0],xy[:,1],color=color,lw=lw,solid_capstyle='round')
        for ax,parts,title in zip(axes,[pparts,cparts],['Before: both original units','After: both units collapsed together']):
            draw(ax,background,'#727e86',2)
            for parts_i,color,label in zip(parts,COLORS,labels):
                draw(ax,parts_i,color,2.6)
                center=unary_union(parts_i).centroid
                xy=(np.array([center.x,center.y])-origin)/1000
                ax.annotate(label,xy,xytext=(10,9),textcoords='offset points',color=color,fontweight='bold',fontsize=11,
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=2))
            ax.set_title(title,fontsize=12)
            ax.set_aspect('equal',adjustable='box')
            ax.set_xlim(-pad,dx+pad);ax.set_ylim(-pad,dy+pad)
            ax.set_xlabel('Local easting (km)')
            ax.spines[['top','right']].set_visible(False)
            ax.grid(alpha=.15)
        axes[0].set_ylabel('Local northing (km)')
        handles=[Line2D([0],[0],color=c,lw=3,label=f'{label}: {len(b["parent_ids"])} original links → 1 channel') for c,label,b in zip(COLORS,labels,selected)]
        handles.append(Line2D([0],[0],color='#727e86',lw=2,label=f'Unchanged connector ({len(connectors)} link)'))
        fig.legend(handles=handles,loc='outside lower center',ncol=1,frameon=False)
        fig.suptitle(f'Example 15: {labels[0]} + {labels[1]}\nOne combined case; each selected unit gets its own equivalent channel',fontsize=14)
        fig.savefig(DEST/f'{stem}.png',dpi=200)
        fig.savefig(DEST/f'{stem}.pdf')
        plt.close(fig)
        record=dict(example='15',state=state,components=names,labels=labels,parent_links_by_unit=[len(b['parent_ids']) for b in selected],
                    collapsed_child_links=childids,connector_links=connectors,parent_region_links=len(pids),child_region_links=2+len(connectors),
                    crs=str(geo.crs),geometry='Actual source and equivalent-channel geometries, same bounds and scale in both panels.')
        records.append(record)
        print(json.dumps(record,indent=2))
    (DEST/'manifest.json').write_text(json.dumps(dict(examples=records,sources=SOURCES,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')


if __name__=='__main__':main()
