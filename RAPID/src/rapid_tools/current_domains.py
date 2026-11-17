"""Validate saved domain membership against current routed physical graphs.

Only topology/selection metadata is reused. Lengths, widths and other attributes
always come from the active parameters; old resistance values are not imported.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx

AUDITED_SCOPE_ROOT = Path(__file__).resolve().parents[3]/'RAPID/outputs/hydrograph_predictor_scopes_20261014'


def load_audited_domain_pair(record):
    """Reuse current scope audit when a state has no legacy membership CSV.

    Resolve aliases explicitly and revalidate topology, closure and geometric
    attributes against the active parameters. Cached resistance is never used.
    """
    root = AUDITED_SCOPE_ROOT
    table_path = root/'geometry_responses.parquet'
    if not table_path.exists() or record.get('example_id') is None:
        raise ValueError(f'No saved local domain membership or current scope audit for {record.state_id}')
    table = pd.read_parquet(table_path, columns=['example','state','aliases','B_scope','B_available','selected_unit_ids'])
    example = str(record.example_id).zfill(2)
    table = table[table.example.astype(str).str.zfill(2).eq(example)]
    matches = table[table.apply(lambda r: record.state_id in {r.state, *str(r.aliases).split(' | ')}, axis=1)]
    if len(matches) != 1 or not bool(matches.iloc[0].B_available):
        raise ValueError(f'No unique validated outer domain for {example}/{record.state_id}')
    selected = matches.iloc[0]
    graphs = pd.read_csv(root/'domain_graphs.csv', dtype={'example':str})
    graphs = graphs[graphs.example.eq(example)&graphs.state.eq(selected.state)&graphs.scope.eq(selected.B_scope)]
    if len(graphs) != 1:
        raise ValueError('Ambiguous current scope graph mapping')
    graph = graphs.iloc[0]
    parts = []; bounds = []
    for side, key in [('parent','parent_parameters_parquet'),('child','parameters_parquet')]:
        saved = pd.read_parquet(root/'graphs'/f'{graph[side+"_graph"]}.parquet')
        full = pd.read_parquet(record[key])
        part, terminals = validate_domain(full, saved)
        expected = (int(graph[side+'_source']),int(graph[side+'_sink']))
        if terminals != expected:
            raise ValueError('Audited domain boundaries disagree with current topology')
        columns = ['link_length_m','width']
        if not np.allclose(part.set_index('id_link').sort_index()[columns],
                           saved.set_index('id_link').sort_index()[columns], rtol=1e-12, atol=1e-10):
            raise ValueError('Audited domain geometry differs from current parameters; refresh the scope audit')
        parts.append(part); bounds.append(terminals)
    return *parts, dict(source=bounds[0][0],sink=bounds[0][1],child_source=bounds[1][0],child_sink=bounds[1][1],
        domain_type='validated outer '+{'C':'compound','U':'unit','N':'network'}[selected.B_scope],
        selected_unit_ids=str(selected.selected_unit_ids),membership_path=str(root/'domain_graphs.csv'),
        parent_links=sorted(parts[0].id_link.astype(int)),child_links=sorted(parts[1].id_link.astype(int)),
        canonical_state=str(selected.state),attribute_source='Active parameters; audited topology, closure, length and width revalidated')


def validate_domain(full, membership):
    ids=set(membership.id_link.astype(int))
    if not ids or not ids.issubset(set(full.id_link)):
        raise ValueError('Domain link IDs are absent from the current physical graph')
    part=full[full.id_link.isin(ids)].copy()
    topology=part.set_index('id_link')[['id_us_node','id_ds_node']].sort_index()
    saved=membership.set_index('id_link')[['id_us_node','id_ds_node']].sort_index()
    if saved.index.duplicated().any() or not np.array_equal(topology.to_numpy(),saved.to_numpy()):
        raise ValueError('Saved domain endpoints differ from current reach endpoints')
    g=nx.MultiDiGraph()
    for r in part.itertuples():g.add_edge(int(r.id_us_node),int(r.id_ds_node),key=int(r.id_link))
    sources=[u for u in g if g.in_degree(u)==0];sinks=[u for u in g if g.out_degree(u)==0]
    if len(sources)!=1 or len(sinks)!=1 or not nx.is_directed_acyclic_graph(g) or not nx.is_weakly_connected(g):
        raise ValueError('Selected domain is not a closed single-source/sink DAG')
    source,sink=sources[0],sinks[0]
    external=full[~full.id_link.isin(ids)]
    internal=set(g)-{source,sink}
    if external.id_us_node.isin(internal).any() or external.id_ds_node.isin(internal).any():
        raise ValueError('External reach crosses an internal domain node')
    if external.id_us_node.eq(source).any() or external.id_ds_node.eq(sink).any():
        raise ValueError('Domain omits an outgoing source or incoming sink branch')
    return part,(source,sink)


def load_domain_pair(record):
    if record.get('mode') == 'centerline' and record.state_id == 'fully_collapsed':
        parent = pd.read_parquet(record.parent_parameters_parquet)
        child = pd.read_parquet(record.parameters_parquet)
        if not {'id_us_node', 'id_ds_node'}.issubset(child.columns):
            child = child.rename(columns={'u': 'id_us_node', 'v': 'id_ds_node'})
        parent, pbounds = validate_domain(parent, parent)
        child, cbounds = validate_domain(child, child)
        return parent, child, dict(
            source=pbounds[0], sink=pbounds[1], child_source=cbounds[0], child_sink=cbounds[1],
            domain_type='Whole-network centerline collapse',
            parent_links=sorted(parent.id_link.astype(int)), child_links=sorted(child.id_link.astype(int)),
            attribute_source='Current physical parameters')
    registry_root=Path(record.registry_path).parent
    canonical=Path(record.parameters_parquet).parents[3]
    candidate=registry_root/'states'/record.state_id/'equivalent_resistance'
    if not (candidate/'selected_hydraulic_domain_links.csv').exists():
        candidate=canonical/'equivalent_resistance'
    path=candidate/'selected_hydraulic_domain_links.csv'
    if not path.exists(): return load_audited_domain_pair(record)
    domain=pd.read_csv(path)
    parent_full=pd.read_parquet(record.parent_parameters_parquet)
    child_full=pd.read_parquet(record.parameters_parquet)
    parent,pbounds=validate_domain(parent_full,domain[domain.domain_role.eq('parent')])
    child,cbounds=validate_domain(child_full,domain[domain.domain_role.eq('child')])
    comparison=pd.read_csv(candidate/'bubble_equivalent_state_comparison.csv')
    if len(comparison)!=1: raise ValueError('Ambiguous selected-domain comparison metadata')
    cmp=comparison.iloc[0]
    expected=(int(cmp.parent_upstream_boundary_node),int(cmp.parent_downstream_boundary_node))
    if pbounds!=expected: raise ValueError('Domain anchors disagree with comparison metadata')
    child_expected=(int(cmp.child_upstream_boundary_node),int(cmp.child_downstream_boundary_node))
    if cbounds!=child_expected: raise ValueError('Child anchors disagree with comparison metadata')
    if pbounds!=cbounds:
        methods=[cmp.get('child_upstream_anchor_match_method'),cmp.get('child_downstream_anchor_match_method')]
        if any(pd.isna(m) or not str(m).strip() for m in methods):
            raise ValueError('Renumbered anchors lack recorded correspondence methods')
    return parent,child,dict(source=pbounds[0],sink=pbounds[1],child_source=cbounds[0],child_sink=cbounds[1],
        domain_type=str(cmp.hydraulic_domain_type),selected_unit_ids=str(cmp.selected_unit_ids),
        membership_path=str(path),parent_links=sorted(parent.id_link.astype(int)),
        child_links=sorted(child.id_link.astype(int)),
        attribute_source='Current published physical parameters; membership topology validated')


def allocation_tables(frame):
    """Accumulate incoming probability at every confluence before redistributing."""
    g=nx.MultiDiGraph();out={}
    for r in frame.itertuples():
        u,v=int(r.id_us_node),int(r.id_ds_node)
        g.add_edge(u,v,key=int(r.id_link));out.setdefault(u,[]).append(r)
    order=list(nx.topological_sort(g));sources=[u for u in order if g.in_degree(u)==0]
    if len(sources)!=1: raise ValueError('Allocation needs one domain inlet')
    visit={sources[0]:1.};edges=[];nodes=[]
    for u in order:
        outgoing=out.get(u,[]);r_u=visit.get(u,0.)
        nodes.append(dict(node=u,in_degree=g.in_degree(u),out_degree=g.out_degree(u),visit_probability=r_u))
        total=sum(r.width for r in outgoing)
        for r in outgoing:
            p=float(r.width/total);r_e=r_u*p
            visit[int(r.id_ds_node)]=visit.get(int(r.id_ds_node),0.)+r_e
            edges.append(dict(id_link=int(r.id_link),u=u,v=int(r.id_ds_node),L_e_m=r.link_length_m,
                W_e_m=r.width,p_outgoing=p,r_edge=r_e,mean_length_contribution_m=r_e*r.link_length_m))
    return pd.DataFrame(nodes),pd.DataFrame(edges)
