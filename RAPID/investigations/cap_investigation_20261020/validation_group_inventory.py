"""Inventory dependence groups for validation; no split selection or model fits."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from slope_ratio_audit import ROOT, OUT, EXT, SC, read, SOURCES

DEST=OUT/'validation_group_inventory_20261023'


def integers(value):
    if pd.isna(value) or str(value).strip()=='':return []
    return [int(float(x.strip())) for x in str(value).strip('[]').split(',') if x.strip()]


def grouped(records, use_links=False, use_context=False):
    """Transitive components; tokens are network-local, never node-touch adjacency."""
    g=nx.Graph();g.add_nodes_from(r['state'] for r in records)
    owners={};edges=[]
    for r in records:
        tokens=[('family',t) for t in r['family_tokens']]
        if use_links:tokens += [('reach',i) for i in r['parent_footprint_links']]
        if use_context:tokens += [('context_reach',i) for i in r['descriptor_links']]
        for kind,key in tokens:
            token=kind,key
            if token in owners and owners[token]!=r['state']:
                first=owners[token];g.add_edge(first,r['state'])
                edges.append(dict(state_a=first,state_b=r['state'],reason=kind,token=str(key)))
            else:owners[token]=r['state']
    assignments={}
    for members in nx.connected_components(g):
        name=hashlib.sha256('|'.join(sorted(members)).encode()).hexdigest()[:12]
        assignments.update({s:name for s in members})
    return assignments,edges


def response_bin(hours):
    m=hours*60
    if abs(m)<=1:return 'near_zero_<=1min'
    sign='advance' if m<0 else 'delay'
    scale='1to5min' if abs(m)<=5 else '5to60min' if abs(m)<=60 else 'over60min'
    return sign+'_'+scale


def main():
    DEST.mkdir(exist_ok=True)
    d=read(OUT/'resistance_inverse_depth_panels_20261023/input.parquet')
    assert len(d)==2185 and not d.duplicated(['example','state']).any()
    meta=read(SC/'geometry_responses.parquet').set_index(['example','state'])
    graphs=read(SC/'domain_graphs.csv').set_index(['example','state','scope'])
    oldplan=read(ROOT/'RAPID/outputs/event_window_plan_20261013/plan.json')
    active={};jobs={}
    for plan in [oldplan,read(EXT/'routing_plan.json'),read(OUT/'routing_plan.json')]:
        for c in plan['cases']:
            for j in c['jobs']:
                for alias in {j['state_id'],*j.get('aliases',[])}:jobs[c['example'],alias]=j
                if j['state_id']=='S000_base':active[c['example']]=j
    records=[];assignments=[];edge_rows=[];summary=[];memberships=[];join_audit=[]
    for ex,frame in d.groupby('example'):
        case=next(c for c in oldplan['cases'] if c['example']==ex)
        base=Path(next(j for j in case['jobs'] if j['state_id']=='S000_base')['parameters']).parents[3]
        units=read(base/'hierarchy/unit_summary.csv').set_index('unit_id')
        membership=read(base/'variant/summary/unit_link_membership.csv')
        full=read(active[ex]['parameters']);parent_ids=set(full.id_link.astype(int))
        assert set(membership.id_link.astype(int))<=parent_ids
        assert set(membership.unit_id.astype(int))<=set(units.index)
        family=nx.Graph();family.add_nodes_from(int(i) for i in units.index)
        for uid,u in units.iterrows():
            for other in set(integers(u.parents)+integers(u.children)):
                assert other in units.index
                family.add_edge(int(uid),other)
        compound=units[units.in_compound_bubble.eq(True)]
        assert compound.compound_bubble_id.notna().all()
        for bid,group in compound.groupby('compound_bubble_id'):
            ids=sorted(map(int,group.index))
            family.add_edges_from((ids[0],i) for i in ids[1:])
        family_of={uid:min(c) for c in nx.connected_components(family) for uid in c}
        unit_links=membership.groupby('unit_id').id_link.apply(lambda v:set(map(int,v))).to_dict()
        for uid,u in units.iterrows():
            memberships.append(dict(example=ex,unit_id=int(uid),family_id=family_of[int(uid)],
                in_compound=bool(u.in_compound_bubble),compound_id=int(u.compound_bubble_id) if u.in_compound_bubble else None,
                parents=u.parents,children=u.children,parent_links=','.join(map(str,sorted(unit_links.get(uid,set()))))))
        network_records=[];cache={}
        for row in frame.itertuples():
            common=dict(example=ex,state=row.state,cohort=row.cohort,
                centroid_shift_hours=row.centroid_shift_hours,response_bin=response_bin(row.centroid_shift_hours))
            if row.cohort=='segmented':
                line=read(EXT/f'variants/{ex}/{row.state}/lineage.csv')
                ids=set(line.loc[line.role.eq('collapsed_block'),'parent_link_id'].astype(int))
                assert ids and ids<=parent_ids
                z=dict(**common,edit_type='Segmented full network',selected_unit_ids=[],family_tokens=[],
                    compound_ids=[],parent_footprint_links=sorted(ids),descriptor_links=sorted(parent_ids),
                    footprint_convention='collapsed-block lineage, unchanged connectors excluded',
                    descriptor_convention='full network for resistance; deliberately conservative overlap sensitivity',aliases=[],nested=False)
            else:
                m=meta.loc[(ex,row.state)];selected=integers(m.selected_unit_ids)
                assert selected and set(selected)<=set(units.index)
                j=jobs[ex,row.state];aliases=j.get('aliases',[])
                families={a.split('/')[0] for a in aliases};primary=aliases[0].split('/')[0] if aliases else ''
                chosen=units.loc[selected];is_member=chosen.in_compound_bubble.eq(True).any()
                assert bool(m.is_compound_member)==bool(is_member)
                nested=bool(((chosen.n_children>0)|(chosen.depth_from_root>0)).any())
                if 'compound_bubbles' in families:kind='Full compound'
                elif primary=='island':kind='Island'
                elif is_member:kind='Compound member'
                elif nested:kind='Nested'
                elif chosen['class'].eq('simple_bifurcation_confluence_pair').all():kind='Simple'
                else:kind='Unclassified'
                footprint=set().union(*(unit_links[i] for i in selected))
                gr=graphs.loc[(ex,row.state,m.B_scope)];key=gr.parent_graph
                if key not in cache:
                    geom=read(SC/'graphs'/f'{key}.parquet').set_index('id_link')
                    assert set(geom.index)<=parent_ids
                    active_part=full.set_index('id_link').loc[geom.index]
                    np.testing.assert_array_equal(active_part[['id_us_node','id_ds_node']],geom[['id_us_node','id_ds_node']])
                    np.testing.assert_allclose(active_part[['link_length_m','width']],geom[['link_length_m','width']],rtol=1e-10,atol=1e-7)
                    cache[key]=set(map(int,geom.index))
                context=cache[key]
                assert footprint and footprint<=context,(ex,row.state,'membership outside B')
                z=dict(**common,edit_type=kind,selected_unit_ids=selected,
                    family_tokens=sorted({family_of[i] for i in selected}),
                    compound_ids=sorted(set(chosen.loc[chosen.in_compound_bubble.eq(True),'compound_bubble_id'].astype(int))),
                    parent_footprint_links=sorted(footprint),descriptor_links=sorted(context),
                    footprint_convention='saved source/anchor unit membership (conservative for islands)',
                    descriptor_convention='saved '+str(m.B_scope)+' parent domain',aliases=aliases,nested=nested)
                join_audit.append(dict(example=ex,state=row.state,metadata='matched',job='matched',scope=m.B_scope,selected_units=len(selected)))
            network_records.append(z);records.append(z)
        regular=[r for r in network_records if r['cohort']=='original']
        scenarios=[('family_only',regular,False,False),('family_and_reach',regular,True,False),
            ('family_reach_and_context',regular,True,True),('including_segmented_edited',network_records,True,False),
            ('including_segmented_context',network_records,True,True)]
        for scenario,rr,links,context in scenarios:
            groups,edges=grouped(rr,links,context)
            counts=pd.Series(list(groups.values())).value_counts()
            summary.append(dict(example=ex,scenario=scenario,records=len(rr),groups=len(counts),
                singleton_groups=int(counts.eq(1).sum()),largest_group=int(counts.max()),
                largest_fraction=float(counts.max()/len(rr)),net_effective_groups=float(len(rr)**2/(counts**2).sum()),
                at_least_five_groups=len(counts)>=5,only_one_group=len(counts)==1))
            assignments.extend(dict(example=ex,state=r['state'],scenario=scenario,group_id=ex+'_'+groups[r['state']]) for r in rr)
            edge_rows.extend(dict(example=ex,scenario=scenario,**edge) for edge in edges)
        print(ex,'inventoried',len(regular),'regular cases',flush=True)
    p=pd.DataFrame(records);a=pd.DataFrame(assignments);n=pd.DataFrame(summary)
    assert len(p)==2185 and len(join_audit)==2170 and not p.duplicated(['example','state']).any()
    assert not p.edit_type.eq('Unclassified').any(),p[p.edit_type.eq('Unclassified')][['example','state']]
    p.to_parquet(DEST/'records.parquet',index=False)
    a.to_csv(DEST/'group_assignments.csv',index=False)
    n.to_csv(DEST/'network_summary.csv',index=False)
    pd.DataFrame(memberships).to_csv(DEST/'unit_families.csv',index=False)
    pd.DataFrame(edge_rows).drop_duplicates().to_parquet(DEST/'relation_edges.parquet',index=False)
    pd.DataFrame(join_audit).to_csv(DEST/'join_audit.csv',index=False)
    merged=a.merge(p[['example','state','cohort','edit_type','response_bin','centroid_shift_hours']],on=['example','state'],validate='many_to_one')
    # Every family and every shared-reach link must remain in one conservative group.
    for ex,g in p[p.cohort.eq('original')].groupby('example'):
        assigned=a[(a.example==ex)&(a.scenario=='family_and_reach')].set_index('state').group_id
        owners=defaultdict(set)
        for r in g.itertuples():
            for token in [('f',x) for x in r.family_tokens]+[('e',x) for x in r.parent_footprint_links]:owners[token].add(assigned[r.state])
        assert all(len(v)==1 for v in owners.values())
    group_summary=merged.groupby(['scenario','example','group_id']).agg(records=('state','size'),
        shift_min_hours=('centroid_shift_hours','min'),shift_max_hours=('centroid_shift_hours','max'),
        response_bins=('response_bin',lambda x:' | '.join(sorted(set(x)))),edit_types=('edit_type',lambda x:' | '.join(sorted(set(x)))))
    group_summary.to_csv(DEST/'group_summary.csv')
    merged.groupby(['scenario','edit_type','response_bin']).agg(records=('state','size'),groups=('group_id','nunique'),networks=('example','nunique')).to_csv(DEST/'stratum_support.csv')
    merged.groupby(['scenario','edit_type']).agg(records=('state','size'),groups=('group_id','nunique'),networks=('example','nunique')).to_csv(DEST/'type_support.csv')
    merged.groupby(['scenario','response_bin']).agg(records=('state','size'),groups=('group_id','nunique'),networks=('example','nunique')).to_csv(DEST/'response_support.csv')
    merged.groupby(['scenario','example','edit_type','response_bin']).size().rename('records').to_csv(DEST/'network_mix.csv')
    totals=[]
    for scenario,g in merged.groupby('scenario'):
        sizes=g.groupby('group_id').size()
        totals.append(dict(scenario=scenario,records=len(g),groups=len(sizes),largest_group=int(sizes.max()),
            nominal_fivefold_size=len(g)/5,largest_over_nominal=float(sizes.max()/(len(g)/5)),
            at_least_five_groups=len(sizes)>=5,networks_only_one_group=int(n[(n.scenario==scenario)].only_one_group.sum())))
    pd.DataFrame(totals).to_csv(DEST/'feasibility_summary.csv',index=False)
    for path,h in SOURCES.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
    manifest=dict(sources=SOURCES,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        population='Current 2185 model comparisons; response bins use current input centroid shifts, never historical scope responses.',
        grouping='Network-qualified transitive components. Family links use explicit true compound membership plus all saved parent/child nesting; singleton bubble IDs do not imply compound membership. Reach-overlap uses selected-unit memberships; islands use conservative anchor units. Context sensitivity adds overlap of parent B descriptor domains.',
        segmented='Two sensitivities: edited collapsed-block lineage overlap; additionally full-network resistance context. Regular-only grouping is the candidate for within-network validation.',
        bins='Descriptive predefined thresholds in minutes: abs<=1 near zero; otherwise signed (1,5], (5,60], >60. Not optimized for balance or scores.',
        limits='These are non-overlapping dependence groups, not statistically independent observations. Shared network forcing/calibration remains. >=5 groups is necessary, not sufficient for balanced folds; sparse response/type strata may not cover all five folds. No folds selected or fits performed.',
        matched=2185,unmatched=0,fits=0,production_modified=False)
    (DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(pd.DataFrame(totals).to_string(index=False),flush=True)
    print(n[n.scenario.eq('family_and_reach')].to_string(index=False),flush=True)


if __name__=='__main__':main()
