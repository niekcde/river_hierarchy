"""Closed outer-unit composition pilot. Writes only separate workspace outputs."""
from pathlib import Path
import sys,json,hashlib,itertools
import numpy as np
import pandas as pd
import networkx as nx
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src'),str(ROOT/'RAPID/investigations/hydrograph_predictors_20261014')]
from build import descriptors,pair_features
from rapid_tools.current_domains import validate_domain,allocation_tables
from rapid_tools import cunge,common_grid
from rapid_tools.event_window_batch import save,sha,settling_hours,equilibrium_check
from rapid_tools.common_grid_production import shared_clock
SCOPES=ROOT/'RAPID/outputs/hydrograph_predictor_scopes_20261014'
SOURCE=ROOT/'RAPID/outputs/hydrograph_relative_sweep_20261019/input.parquet'
PLAN=ROOT/'RAPID/outputs/event_window_plan_20261013/plan.json'
OUT=ROOT/'RAPID/outputs/hydrograph_closed_unit_pairing_20261019'
SOURCES={}
def read(path):
    path=Path(path);SOURCES[str(path)]=sha(path)
    if path.suffix=='.json':return json.loads(path.read_text())
    if path.suffix=='.parquet':return pd.read_parquet(path)
    return pd.read_csv(path,dtype={'example':str})
def graph(f):
    g=nx.MultiDiGraph()
    for r in f.itertuples():g.add_edge(int(r.u),int(r.v),key=int(r.id_link))
    return g

def adjacent_pairs(parent,blocks):
    """Only consecutive closed blocks joined by a directed unbranched connector."""
    g=graph(parent);starts={b['source']:b for b in blocks};pairs=[]
    if len(starts)!=len(blocks):raise ValueError('Duplicate block inlets')
    for a in blocks:
        node=a['sink'];connector=[];seen=set()
        while node not in starts:
            if node in seen:raise ValueError('Cycle')
            seen.add(node)
            if (node!=a['sink'] and g.in_degree(node)!=1) or g.out_degree(node)!=1:break
            edge=next(iter(g.out_edges(node,keys=True)));connector.append(edge[2]);node=edge[1]
        if node in starts and starts[node]['state']!=a['state']:
            b=starts[node]
            if set(a['parent_ids'])&set(b['parent_ids']):raise ValueError('Overlapping pair')
            union=set(a['parent_ids'])|set(b['parent_ids'])|set(connector)
            domain=parent[parent.id_link.isin(union)]
            validate_domain(parent,domain)
            m=descriptors(domain);alloc=allocation_tables(domain)[1]
            con=float(alloc.loc[alloc.id_link.isin(connector),'mean_length_contribution_m'].sum())
            pairs.append(dict(example=a['example'],state_a=a['state'],state_b=b['state'],pair_type='+'.join(sorted([a['kind'],b['kind']])),connector_ids=connector,
                parent_ids=sorted(union),source=a['source'],sink=b['sink'],parent_mean_length_m=m['route_length_mean'],connector_mean_length_m=con,connector_fraction=con/m['route_length_mean']))
    return pairs

def inventory():
    plan=read(PLAN);d=read(SOURCE);domains=read(SCOPES/'domain_graphs.csv')
    records=d[d.edit_type.isin(['Simple','Full compound'])][['example','state','edit_type','B_scope']].merge(domains,left_on=['example','state','B_scope'],right_on=['example','state','scope'],validate='one_to_one')
    all_blocks={};parents={};all_pairs=[];audit=[]
    cases={c['example']:c for c in plan['cases']}
    for ex,c in cases.items():
        jobs={j['state_id']:j for j in c['jobs']};base=jobs['S000_base'];p=read(base['parameters'])
        assert SOURCES[base['parameters']]==c['source_sha256'][base['parameters']]
        assert np.array_equal(p.u,p.id_us_node) and np.array_equal(p.v,p.id_ds_node)
        blocks=[]
        for r in records[records.example.eq(ex)].itertuples():
            pg=read(SCOPES/'graphs'/f'{r.parent_graph}.parquet');cg=read(SCOPES/'graphs'/f'{r.child_graph}.parquet')
            part,bounds=validate_domain(p,pg)
            np.testing.assert_allclose(part.set_index('id_link').loc[pg.id_link,['link_length_m','width']],pg[['link_length_m','width']],rtol=1e-12)
            if len(cg)!=1:raise ValueError('Eligible full block not a single-link replacement')
            block=dict(example=ex,state=r.state,kind='simple' if r.edit_type=='Simple' else 'compound',source=int(bounds[0]),sink=int(bounds[1]),parent_ids=sorted(pg.id_link.astype(int)),child_id=int(cg.id_link.iloc[0]),child_source=int(r.child_source),child_sink=int(r.child_sink),child_graph=r.child_graph)
            blocks.append(block)
        # Remove true contained regions, fail on partial overlap. Labels alone do not establish outermost.
        outer=[]
        for b in blocks:
            ids=set(b['parent_ids']);containers=[a['state'] for a in blocks if ids<set(a['parent_ids'])]
            if containers:audit.append(dict(example=ex,state=b['state'],status='excluded_contained',reason=','.join(containers)));continue
            if any(ids&set(a['parent_ids']) for a in outer):raise ValueError('Overlapping outer blocks')
            outer.append(b);audit.append(dict(example=ex,state=b['state'],status='eligible',reason='closed and non-overlapping outer block'))
        # The contracted topology must be a chain for the segmented whole-network variant.
        remove=set().union(*(set(b['parent_ids']) for b in outer));q=p[~p.id_link.isin(remove)].copy()
        edges=list(zip(q.u,q.v))+[(b['source'],b['sink']) for b in outer];g=nx.MultiDiGraph();g.add_edges_from(edges)
        chain=nx.is_weakly_connected(g) and nx.is_directed_acyclic_graph(g) and all(g.in_degree(n)<=1 and g.out_degree(n)<=1 for n in g)
        all_blocks[ex]=outer;parents[ex]=p
        pairs=adjacent_pairs(p,outer);all_pairs.extend(pairs)
        print(ex,len(outer),'blocks',len(pairs),'pairs','single_path',chain,flush=True)
        if not chain:audit.append(dict(example=ex,state='all_closed_units_collapsed',status='unavailable',reason='outer blocks do not remove all branching'))
    OUT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(audit).to_csv(OUT/'block_audit.csv',index=False)
    save(OUT/'eligible_blocks.json',all_blocks);save(OUT/'candidate_pairs.json',all_pairs)
    pd.DataFrame(all_pairs).drop(columns=['parent_ids','connector_ids']).to_csv(OUT/'candidate_pairs.csv',index=False)
    save(OUT/'inventory_sources.json',SOURCES)
    return cases,parents,all_blocks,all_pairs



def select_pilot(pairs):
    # Geometry-only purposeful coverage: small, intermediate and large domains;
    # two of each pair type, two pairs in each of three networks. No response access.
    d=pd.DataFrame(pairs);chosen=[]
    specifications=[('03','simple+simple','parent_mean_length_m',.25),('03','compound+compound','parent_mean_length_m',.5),
        ('55','simple+simple','parent_mean_length_m',1.),('55','compound+simple','connector_fraction',1.),
        ('22','compound+simple','parent_mean_length_m',1.),('22','compound+compound','parent_mean_length_m',1.)]
    for ex,kind,sort,quantile in specifications:
        subset=d[d.example.eq(ex)&d.pair_type.eq(kind)].sort_values([sort,'state_a','state_b'])
        if subset.empty:raise ValueError('Planned pilot type unavailable')
        row=subset.iloc[round(quantile*(len(subset)-1))].to_dict()
        row['state']='pair_'+row['state_a']+'__'+row['state_b'];row['selection_rule']=f'{sort} quantile {quantile}, geometry only'
        chosen.append(row)
    save(OUT/'selection.json',chosen)
    pd.DataFrame(chosen).drop(columns=['parent_ids','connector_ids']).to_csv(OUT/'selected_pairs.csv',index=False)
    return chosen


def head_map(frame):
    values={}
    for r in frame.itertuples():
        for node,h in [(int(r.u),r.head_us),(int(r.v),r.head_ds)]:
            if node in values and not np.isclose(h,values[node],atol=1e-8,rtol=0):raise ValueError('Inconsistent node head')
            values[node]=float(h)
    return values


def compose(parent,blocks,sections):
    """Splice validated whole-block sections, preserving all outside physical inputs."""
    selected=set();nodes=set();heads=head_map(parent);parts=[];lineage=[]
    for i,b in enumerate(sorted(blocks,key=lambda b:b['state'])):
        ids=set(b['parent_ids']);part,bounds=validate_domain(parent,parent[parent.id_link.isin(ids)])
        if selected&ids:raise ValueError('Selected regions overlap')
        internal=set(part.u)|set(part.v);internal-={b['source'],b['sink']}
        if nodes&internal:raise ValueError('Selected interiors overlap')
        nodes|=internal;selected|=ids
        row=sections[b['state']].copy()
        if len(row)!=1:raise ValueError('Replacement must have one section')
        # Explicit saved-domain boundary correspondence, never raw numeric ID coincidence.
        if int(row.id_us_node.iloc[0])!=b['child_source'] or int(row.id_ds_node.iloc[0])!=b['child_sink']:raise ValueError('Child boundary mismatch')
        np.testing.assert_allclose(row[['head_us','head_ds']].iloc[0],[heads[b['source']],heads[b['sink']]],rtol=0,atol=1e-8)
        new_id=int(parent.id_link.max())+1+i
        row['id_link']=new_id;row['u']=row['id_us_node']=b['source'];row['v']=row['id_ds_node']=b['sink']
        row['section_role']='collapsed_block';row['block_state']=b['state'];parts.append(row)
        lineage.extend(dict(parent_link_id=int(j),child_link_id=new_id,role='collapsed_block',block_state=b['state']) for j in sorted(ids))
    unchanged=parent[~parent.id_link.isin(selected)].copy();unchanged['section_role']='unchanged';unchanged['block_state']=''
    lineage.extend(dict(parent_link_id=int(j),child_link_id=int(j),role='unchanged',block_state='') for j in unchanged.id_link)
    child=pd.concat([unchanged,*parts],ignore_index=True).sort_values('id_link').reset_index(drop=True)
    if child.id_link.duplicated().any():raise ValueError('ID collision')
    g=graph(child)
    if not nx.is_weakly_connected(g) or not nx.is_directed_acyclic_graph(g):raise ValueError('Invalid composed graph')
    cunge.topology(child);head_map(child)
    # Closed serial edits must preserve outside allocation and hydraulic parameters.
    rebuilt=cunge.freeze_grid(child,float(parent.loc[parent.u.eq(cunge.topology(parent)[2]),'reference_discharge_cms'].sum()),
        n_bounds=(float(parent.grid_n_min.iloc[0]),float(parent.grid_n_max.iloc[0])),alpha=float(parent.cunge_alpha.iloc[0]),
        min_celerity=float(parent.celerity_min_mps.iloc[0]),max_celerity=float(parent.celerity_max_mps.iloc[0]))
    for c in ['split_fraction','reference_flow_fraction','reference_discharge_cms']:
        np.testing.assert_allclose(child[c],rebuilt[c],rtol=1e-10,atol=1e-10)
    pd.testing.assert_frame_equal(child.loc[child.section_role.eq('unchanged'),parent.columns].reset_index(drop=True),parent.loc[~parent.id_link.isin(selected)].sort_values('id_link').reset_index(drop=True),check_dtype=False)
    assert len(lineage)==len(parent) and len(set(r['parent_link_id'] for r in lineage))==len(parent)
    return child,pd.DataFrame(lineage)


def load_sections(ex,parent,blocks,case):
    jobs={j['state_id']:j for j in case['jobs']};sections={};audits=[]
    for b in blocks:
        job=jobs[b['state']];f=read(job['parameters']);assert SOURCES[job['parameters']]==case['source_sha256'][job['parameters']]
        cg=read(SCOPES/'graphs'/f'{b["child_graph"]}.parquet');part,_=validate_domain(f,cg)
        np.testing.assert_allclose(part[['link_length_m','width']].to_numpy(),cg[['link_length_m','width']].to_numpy(),rtol=1e-12)
        sections[b['state']]=part
        # Assemble each singly to exercise boundaries/head consistency before combinations.
        one,_=compose(parent,[b],sections)
        audits.append(dict(example=ex,state=b['state'],status='validated',length_m=float(part.link_length_m.iloc[0]),width_m=float(part.width.iloc[0]),depth_m=float(part.depth_m.iloc[0]),slope=float(part.slope.iloc[0])))
    return sections,audits


def geometry_frame(path):
    import geopandas as gpd
    path=Path(path);SOURCES[str(path)]=sha(path)
    frame=gpd.read_file(path)
    if not frame.crs or not frame.crs.is_projected:raise ValueError('Metric geometry required')
    return frame


def prepare(cases,parents,all_blocks,pairs):
    import geopandas as gpd
    from shapely.geometry import Point
    from shapely.ops import unary_union
    chosen=select_pilot(pairs);audit=[];jobs_out=[];features=[];single_checks=[]
    for ex in sorted({r['example'] for r in chosen}):
        case=cases[ex];parent=parents[ex];blocks=all_blocks[ex];lookup={b['state']:b for b in blocks}
        sections,checks=load_sections(ex,parent,blocks,case);audit.extend(checks)
        oldjobs={j['state_id']:j for j in case['jobs']}
        basepath=Path(oldjobs['S000_base']['parameters']).parents[3]/'variant/widths/links_with_width_families.gpkg'
        pg=geometry_frame(basepath);crs=pg.crs;pg=pg.set_index('id_link')
        geometries={int(i):pg.loc[i].geometry for i in parent.id_link}
        # Verify mapped section geometries terminate at parent boundary coordinates.
        points={}
        for r in parent.itertuples():
            line=geometries[int(r.id_link)]
            for node,xy in [(int(r.u),line.coords[0]),(int(r.v),line.coords[-1])]:
                pt=Point(xy)
                if node in points and points[node].distance(pt)>1e-5:raise ValueError('Parent line orientations/node coordinates disagree')
                points[node]=pt
        replacements={}
        for b in blocks:
            path=Path(oldjobs[b['state']]['parameters']).parents[3]/'variant/widths/links_with_width_families.gpkg'
            cg=geometry_frame(path);line=cg.set_index('id_link').loc[b['child_id']].geometry
            if Point(line.coords[0]).distance(points[b['source']])>1e-5 or Point(line.coords[-1]).distance(points[b['sink']])>1e-5:raise ValueError('Replacement geometric anchors not preserved: '+ex+'/'+b['state'])
            np.testing.assert_allclose(line.length,sections[b['state']].link_length_m.iloc[0],rtol=1e-9,atol=1e-6)
            replacements[b['state']]=line
        selected=[r for r in chosen if r['example']==ex]
        variants=[dict(state=r['state'],blocks=[lookup[r['state_a']],lookup[r['state_b']]],parent_ids=r['parent_ids'],connector_ids=r['connector_ids'],kind='pair') for r in selected]
        variants.append(dict(state='all_closed_units_collapsed',blocks=blocks,parent_ids=parent.id_link.astype(int).tolist(),connector_ids=[],kind='segmented_whole_network'))
        # Reuse singly collapsed sections, but route controlled single edits as well:
        # their geometry outside the selected block is explicitly unchanged.
        for state in sorted({r[k] for r in selected for k in ['state_a','state_b']}):
            b=lookup[state];variants.append(dict(state='control_'+state,blocks=[b],parent_ids=b['parent_ids'],connector_ids=[],kind='single_control'))
        for variant in variants:
            state=variant['state'];bs=variant['blocks'];child,lineage=compose(parent,bs,sections)
            reverse,_=compose(parent,list(reversed(bs)),sections);pd.testing.assert_frame_equal(child,reverse)
            if variant['kind']=='segmented_whole_network':
                g=graph(child);assert all(g.in_degree(n)<=1 and g.out_degree(n)<=1 for n in g)
            dest=OUT/'variants'/ex/state;dest.mkdir(parents=True,exist_ok=True)
            child.to_parquet(dest/'parameters.parquet',index=False);lineage.to_csv(dest/'lineage.csv',index=False)
            lines=[replacements[r.block_state] if r.section_role=='collapsed_block' else geometries[int(r.id_link)] for r in child.itertuples()]
            geo=gpd.GeoDataFrame(child[['id_link','u','v','section_role','block_state','link_length_m','width','depth_m','slope']].copy(),geometry=lines,crs=crs)
            geo.to_file(dest/'links.gpkg',driver='GPKG',index=False)
            parent_domain=parent[parent.id_link.isin(variant['parent_ids'])];edited=set().union(*(set(b['parent_ids']) for b in bs))
            cd_ids=set(lineage[lineage.parent_link_id.isin(variant['parent_ids'])].child_link_id);child_domain=child[child.id_link.isin(cd_ids)]
            validate_domain(parent,parent_domain);validate_domain(child,child_domain)
            m=pair_features(descriptors(parent_domain),descriptors(child_domain),'B_');m.update(example=ex,state=state,kind=variant['kind'],selected_blocks=len(bs))
            pa=allocation_tables(parent_domain)[1];ca=allocation_tables(child_domain)[1]
            m['edited_parent_mean_length_m']=float(pa[pa.id_link.isin(edited)].mean_length_contribution_m.sum())
            changed_child=set(lineage[lineage.role.eq('collapsed_block')].child_link_id)
            m['edited_child_mean_length_m']=float(ca[ca.id_link.isin(changed_child)].mean_length_contribution_m.sum())
            m['edited_mean_length_ratio']=m['edited_child_mean_length_m']/m['edited_parent_mean_length_m']
            m['unchanged_mean_length_m']=m['B_parent_route_length_mean']-m['edited_parent_mean_length_m']
            m['unchanged_length_fraction']=m['unchanged_mean_length_m']/m['B_parent_route_length_mean']
            m['domain_size']=m['B_parent_route_length_mean']/1000
            for c in ['route_length_mean','route_length_sd','width_mean','width_cv','route_entropy']:m['log1p__B_ratio_'+c]=float(np.log1p(m['B_ratio_'+c]))
            features.append(m)
            maxdt=min([oldjobs['S000_base']['native_dt_seconds']]+[oldjobs[b['state']]['native_dt_seconds'] for b in bs]);dt=shared_clock([child],[30,60],maxdt);equilibrium_check(child,dt)
            jobs_out.append(dict(example=ex,state=state,kind=variant['kind'],parameters=str(dest/'parameters.parquet'),native_dt_seconds=dt,selected_states=[b['state'] for b in bs],unchanged_reaches=int(child.section_role.eq('unchanged').sum()),collapsed_sections=len(bs),parent_domain_ids=variant['parent_ids']))
            print('PREPARED',ex,state,flush=True)
        # Base geometry for maps; do not modify source GPKG.
        gpd.GeoDataFrame(parent[['id_link','u','v','link_length_m','width']].copy(),geometry=[geometries[int(i)] for i in parent.id_link],crs=crs).to_file(OUT/'variants'/ex/'parent.gpkg',driver='GPKG',index=False)
    pd.DataFrame(audit).to_csv(OUT/'section_hydraulic_audit.csv',index=False);pd.DataFrame(features).to_parquet(OUT/'predictors.parquet',index=False)
    save(OUT/'prepared_jobs.json',jobs_out)
    for p,h in SOURCES.items():assert sha(p)==h, 'Source changed: '+p
    save(OUT/'preparation_manifest.json',dict(status='prepared_not_routed',sources=SOURCES,code_sha256={str(Path(__file__)):sha(__file__)},construction='Compose previously validated whole-block sections; preserve parent connectors and outer reaches exactly, including frozen routing parameters. No mask/raster regeneration or new width fitting.',calibration='Existing Manning n, beta, reference depth and width retained; legacy reference untouched.'))
    return jobs_out

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    data=inventory()
    if args.prepare:prepare(*data)
