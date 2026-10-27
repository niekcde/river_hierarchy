"""Reuse reviewed unit/compound domains and parent identities on current geometry."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import networkx as nx
from build import ROOT, descriptors, pair_features, alias_lookup, digest
sys.path.insert(0,str(ROOT))
from rapid_tools.hydrograph_response import load
from rapid_tools.current_domains import allocation_tables,validate_domain
from network_variants.equivalent_transition_domain import (
    extract_complete_directed_domain, audit_hydraulic_domain_closure,
    select_smallest_closed_parent_domain, load_parent_to_child_anchor_map,
    complete_anchor_map_from_unchanged_lineage,
)
OUT=ROOT/'RAPID/outputs/hydrograph_predictor_scopes_20261014'
HASHES={}
def read(path):
    path=Path(path);HASHES[str(path)]=digest(path)
    if path.suffix=='.json':return json.loads(path.read_text())
    if path.suffix=='.parquet':return pd.read_parquet(path)
    return pd.read_csv(path)

def ids(x):
    if x is None or (isinstance(x,float) and np.isnan(x)):return []
    if isinstance(x,(list,tuple)):return [int(v) for v in x]
    return [int(float(v)) for v in str(x).split(',') if v.strip()]

def terminals(f):
    g=nx.MultiDiGraph();g.add_edges_from(zip(f.id_us_node.astype(int),f.id_ds_node.astype(int)))
    source=[n for n in g if g.in_degree(n)==0];sink=[n for n in g if g.out_degree(n)==0]
    if len(source)!=1 or len(sink)!=1 or not nx.is_directed_acyclic_graph(g) or not nx.is_weakly_connected(g):
        raise ValueError('Not a connected single-source/sink DAG')
    return source[0],sink[0]

def save_graph(f):
    source,sink=terminals(f)
    frame=f[['id_link','id_us_node','id_ds_node','link_length_m','width']].sort_values('id_link').reset_index(drop=True)
    key=hashlib.sha256(frame.to_json(orient='split',double_precision=15).encode()).hexdigest()[:24]
    path=OUT/'graphs'/f'{key}.parquet'
    if not path.exists():frame.to_parquet(path,index=False)
    return key,source,sink

def reviewed_mapping(child,context,parent_nodes):
    """Use saved reviewed identities, verified against the routed frame's u/v."""
    nodes=read(context/'node_hydraulic_context.csv')
    if 'parent_identity' not in nodes:raise ValueError('No reviewed parent_identity in current context')
    mapping=nodes[['id_node','parent_identity']].dropna().astype(int)
    if mapping.id_node.duplicated().any() or mapping.parent_identity.duplicated().any():
        raise ValueError('Reviewed node correspondence is not injective')
    raw_to_identity=mapping.set_index('id_node').parent_identity
    for raw,routed in [('id_us_node','u'),('id_ds_node','v')]:
        got=child[raw].map(raw_to_identity)
        if got.isna().any() or not np.array_equal(got.to_numpy(),child[routed].to_numpy()):
            raise ValueError('Node context does not agree with current routed endpoint identities')
    return {int(r.parent_identity):int(r.id_node) for r in mapping.itertuples() if r.parent_identity in parent_nodes}

def graph_pair(parent,child,p,mapping):
    source,sink=terminals(p)
    if source not in mapping or sink not in mapping:raise ValueError(f'Unit/compound anchors removed or unmapped: {source}->{sink}')
    c=extract_complete_directed_domain(child,mapping[source],mapping[sink]);cs,ct=terminals(c)
    if (cs,ct)!=(mapping[source],mapping[sink]):raise ValueError('Child domain terminals disagree')
    pc=audit_hydraulic_domain_closure(p,parent,source,sink)['is_closed']
    cc=audit_hydraulic_domain_closure(c,child,cs,ct)['is_closed']
    return p,c,bool(pc),bool(cc)

def main():
    (OUT/'graphs').mkdir(parents=True,exist_ok=True)
    _,response=load();lookup=alias_lookup(response)
    plan_path=ROOT/'RAPID/outputs/event_window_plan_20261013/plan.json';plan=read(plan_path)
    features=[];audit=[];graphrows=[]
    for case in plan['cases']:
        ex=case['example'];bj=next(j for j in case['jobs'] if j['state_id']=='S000_base')
        parent=read(bj['parameters']);base=Path(bj['parameters']).parents[3]
        assert HASHES[bj['parameters']]==case['source_sha256'][bj['parameters']]
        units=read(base/'hierarchy/unit_summary.csv');membership=read(base/'variant/summary/unit_link_membership.csv')
        bubbles=read(base/'hierarchy/bubble_edge_resistance.csv');unitsidx=units.set_index('unit_id')
        parent_nodes=set(parent.id_us_node)|set(parent.id_ds_node);pm=descriptors(parent)
        visits=allocation_tables(parent)[0].set_index('node').visit_probability
        island_manifest=read(ROOT/f'network_variants/output/{ex}_island__anchored/island_analysis_manifest.csv').set_index('island_id')
        registries={}
        for job in case['jobs']:
            if job['state_id']=='S000_base':continue
            state='fully_collapsed' if job['state_id']=='single_centerline' else job['state_id']
            assert lookup[ex,state]==state
            child=read(job['parameters']);assert HASHES[job['parameters']]==case['source_sha256'][job['parameters']]
            if 'id_us_node' not in child:child=child.rename(columns={'u':'id_us_node','v':'id_ds_node'})
            row=dict(example=ex,state=state,**pair_features(pm,descriptors(child),'N_'))
            scopes={};selection=[];member_compound=None;source_paths=[];metadata_error=''
            if state=='fully_collapsed':
                scopes['N']=(parent,child,True,True)
            else:
                family,sid=job['aliases'][0].split('/')
                canonical=Path(job['parameters']).parents[3]
                try:
                    if family=='island':
                        meta=island_manifest.loc[sid]
                        selection=ids(meta.source_unit_id)
                        analysis=Path(meta.analysis_state_dir);variant=analysis/'variant'
                        context=Path(meta.hydraulic_frame_path).parent
                        # Existing manifests can retain an obsolete variant path; use actual analysis products.
                        if not (variant/'summary/variant_manifest.json').exists():variant=Path(meta.analysis_variant_dir)
                    else:
                        variant=canonical/'variant';context=canonical/'hydraulic_context'
                        registry=ROOT/f'network_variants/output/{ex}_{family}__anchored/state_registry.csv'
                        if str(registry) not in registries:registries[str(registry)]=read(registry).set_index('state_id')
                        selection=ids(registries[str(registry)].loc[sid,'selected_unit_ids'])
                    selected=unitsidx.loc[selection]
                    # Explicitly test actual membership, not a non-null bubble ID (standalone units also have IDs).
                    if selected.in_compound_bubble.astype(bool).any():
                        bids=selected.compound_bubble_id.dropna().astype(int).unique()
                        if len(bids)!=1:raise ValueError('Selection spans more than one containing compound')
                        member_compound=int(bids[0])
                    row['is_compound_member']=member_compound is not None
                    row['selected_unit_ids']=','.join(map(str,selection))
                    try:
                        mapping=reviewed_mapping(child,context,parent_nodes)
                        source_paths.append(str(context/'node_hydraulic_context.csv'));mapping_method='reviewed_parent_identity_verified_against_routing'
                    except (FileNotFoundError,ValueError) as err:
                        # Existing stable/lineage mapper is a fallback, not numeric-ID matching.
                        nm=variant/'matching/node_match.csv';ll=variant/'matching/link_lineage.csv'
                        if not nm.exists():raise ValueError('No verified reviewed identity or existing node-match source: '+str(err))
                        HASHES[str(nm)]=digest(nm)
                        if ll.exists():HASHES[str(ll)]=digest(ll)
                        mapping,_=load_parent_to_child_anchor_map(node_match_path=nm,child_links=child,link_lineage_path=ll)
                        mapping_method='existing_stable_node_and_lineage_map'
                    row['mapping_method']=mapping_method
                    uids=set(membership[membership.unit_id.isin(selection)].id_link.astype(int))
                    up=parent[parent.id_link.isin(uids)].copy()
                    if len(up)!=len(uids):raise ValueError('Selected parent unit link IDs absent')
                    try:
                        us,ut=terminals(up)
                        # Avoid silently completing truncated path membership.
                        complete=extract_complete_directed_domain(parent,us,ut)
                        if set(complete.id_link)!=uids:raise ValueError('Selected U membership is not complete between its terminals')
                        scopes['U']=graph_pair(parent,child,up,mapping)
                    except ValueError as error:
                        audit.append(dict(example=ex,state=state,scope='U',status='unavailable',reason=str(error)))
                    if member_compound is not None:
                        # Reuse established selector with all members to identify the full containing compound.
                        compound_units=units[units.compound_bubble_id.eq(member_compound)].unit_id.astype(int).tolist()
                        domain=select_smallest_closed_parent_domain(full_links=parent,unit_summary=units,
                            unit_link_membership=membership,bubble_edge_resistance=bubbles,
                            selected_unit_ids=compound_units,selected_compound_bubble_id=member_compound)
                        try:
                            cp,cc,closedp,closedc=graph_pair(parent,child,domain.links,mapping)
                            if not(closedp and closedc):raise ValueError('Containing compound is not closed in both states')
                            scopes['C']=(cp,cc,closedp,closedc)
                        except ValueError as error:
                            audit.append(dict(example=ex,state=state,scope='C',status='unavailable',reason=str(error)))
                except (ValueError,KeyError,FileNotFoundError) as error:
                    metadata_error=str(error)
                    audit.append(dict(example=ex,state=state,scope='metadata',status='unavailable',reason=metadata_error))
            for scope,(p,c,closedp,closedc) in scopes.items():
                a,b=descriptors(p),descriptors(c)
                row.update(pair_features(a,b,scope+'_'))
                pg,source,sink=save_graph(p);cg,cs,ct=save_graph(c)
                exposure=float(visits.loc[source]);row.update({scope+'_exposure':exposure,scope+'_length_fraction':a['total_length']/pm['total_length'],
                    scope+'_mean_length_fraction':a['route_length_mean']/pm['route_length_mean'],
                    scope+'_exposed_length_change':exposure*(b['route_length_mean']-a['route_length_mean'])/pm['route_length_mean'],
                    scope+'_exposed_sd_change':exposure*(b['route_length_sd']-a['route_length_sd'])/pm['route_length_mean'],
                    scope+'_closed':closedp and closedc})
                graphrows.append(dict(example=ex,state=state,scope=scope,parent_graph=pg,child_graph=cg,parent_source=source,parent_sink=sink,
                    child_source=cs,child_sink=ct,parent_closed=closedp,child_closed=closedc,parent_links=len(p),child_links=len(c),
                    route_convention='local width normalization conditional on U entry; external exchanges excluded' if not(closedp and closedc) else 'closed domain width allocation'))
                audit.append(dict(example=ex,state=state,scope=scope,status='available',reason='',parent_closed=closedp,child_closed=closedc,
                    source_paths=' | '.join(source_paths)))
            # The resistance benchmark B uses C for compound members, closed U otherwise, N for centerlines.
            bench='N' if state=='fully_collapsed' else 'C' if member_compound is not None else 'U'
            row['B_scope']=bench
            row['B_available']=bench in scopes and all(scopes[bench][2:])
            if row['B_available']:
                for k,v in list(row.items()):
                    if k.startswith(bench+'_'):row['B_'+k[len(bench)+1:]]=v
            features.append(row)
        print(ex,'scope extraction',len(features),flush=True)
    f=pd.DataFrame(features);assert len(f)==len(response) and not f.duplicated(['example','state']).any()
    merged=response.merge(f,on=['example','state'],how='outer',validate='one_to_one',indicator=True)
    assert merged._merge.eq('both').all();merged=merged.drop(columns='_merge')
    merged.to_parquet(OUT/'geometry_responses.parquet',index=False)
    pd.DataFrame(audit).to_csv(OUT/'domain_audit.csv',index=False)
    pd.DataFrame(graphrows).to_csv(OUT/'domain_graphs.csv',index=False)
    for path in [Path(__file__),ROOT/'network_variants/equivalent_transition_domain.py',ROOT/'RAPID/outputs/hydrograph_response/manifest.json']:
        HASHES[str(path)]=digest(path)
    assert all(digest(p)==h for p,h in HASHES.items())
    report=dict(source_hashes=HASHES,comparisons=len(merged),response_unmatched=0,
        scope_convention='U=selected unit(s), C=full containing compound before/after SAME edit, N=whole-network centerline comparison. B=C if a compound member, closed U otherwise, N for centerline.',
        U_open_convention='Route allocation conditional on entry into induced selected unit, normalized over in-domain outgoing widths. Excludes external exchange; geometric descriptor only; no U resistance if open.',
        benchmark_available=int(merged.B_available.sum()),outputs='Fresh current geometry; no routed outputs changed')
    (OUT/'geometry_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print('B coverage',merged.B_available.value_counts().to_dict(),flush=True)

if __name__=='__main__':main()
