"""Isolated geometry/identity repair. Does not generate routing parameters.

Only explicit reviewed correspondences and prescribed boundary cuts are accepted.
Raw child IDs and parent hydraulic IDs remain separate namespaces.
"""
from __future__ import annotations

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import substring


def node_mapping(frame):
    pairs = pd.concat([
        frame[['id_us_node', 'u']].rename(columns={'id_us_node': 'raw', 'u': 'parent'}),
        frame[['id_ds_node', 'v']].rename(columns={'id_ds_node': 'raw', 'v': 'parent'}),
    ]).drop_duplicates()
    if pairs.raw.duplicated().any() or pairs.parent.duplicated().any():
        raise ValueError('Node mapping must be consistent and injective')
    return dict(zip(pairs.raw.astype(int), pairs.parent.astype(int)))


def repair(links, nodes, parent_nodes, mapping, required, *, cuts=(), correspondences=None, tolerance_m=1e-5, cut_link_ids=None):
    """Split unique exact-hit reaches and preserve explicitly reviewed identities.

    Displaced reviewed nodes keep their current position. No hidden snapping or
    nearest-node heuristic is applied. Link output contains geometry/lineage only;
    inherited whole-link hydraulics would be invalid for repaired fragments.
    """
    correspondences = correspondences or {}
    if not set(cuts).union(correspondences.values()) <= set(required):
        raise ValueError('Repair must reference prescribed boundaries')
    if len(set(mapping.values())) != len(mapping):
        raise ValueError('Mapping collision')
    mapping = dict(mapping)
    nodes = nodes[['id_node', 'geometry']].copy()
    parent = parent_nodes.set_index('id_node').geometry
    positions = nodes.set_index('id_node').geometry
    links = links[['id_link', 'id_us_node', 'id_ds_node', 'geometry']].copy()
    records = []
    for r in links.itertuples():
        geom = r.geometry
        if geom.geom_type != 'LineString':
            raise ValueError('Expected LineString')
        a, b = positions.loc[r.id_us_node], positions.loc[r.id_ds_node]
        if Point(geom.coords[0]).distance(a) > tolerance_m:
            geom = LineString(list(geom.coords)[::-1])
        if Point(geom.coords[0]).distance(a) > tolerance_m or Point(geom.coords[-1]).distance(b) > tolerance_m:
            raise ValueError(f'Link {r.id_link}: endpoints do not match graph')
        records.append(dict(id_link=int(r.id_link), id_us_node=int(r.id_us_node), id_ds_node=int(r.id_ds_node),
                            source_link=int(r.id_link), geometry=geom))
    actions = []
    for raw, original in correspondences.items():
        if raw not in mapping or original not in parent.index:
            raise ValueError('Unknown reviewed node')
        if original in mapping.values() and mapping[raw] != original:
            raise ValueError('Reviewed correspondence would cause mapping collision')
        previous = mapping[raw]
        mapping[raw] = original
        actions.append(dict(action='identity_restored' if previous != original else 'already_correct',
                            parent_node=original, raw_node=raw, previous_identity=previous,
                            displacement_m=float(positions.loc[raw].distance(parent.loc[original]))))
    next_node = int(nodes.id_node.max()) + 1
    next_link = int(links.id_link.max()) + 1
    new_nodes = []
    for original in cuts:
        if original in mapping.values():
            raise ValueError(f'Parent {original} already represented; do not add another cut')
        point = parent.loc[original]
        candidates = []
        for i, record in enumerate(records):
            if cut_link_ids is not None and record['source_link'] != cut_link_ids[original]:
                continue
            geom = record['geometry']; d = geom.project(point)
            if geom.distance(point) <= tolerance_m and tolerance_m < d < geom.length - tolerance_m:
                candidates.append((i, d))
        if len(candidates) != 1:
            raise ValueError(f'Parent {original}: expected unique interior cut, found {len(candidates)}')
        i, d = candidates[0]; old = records.pop(i); geom = old['geometry']
        new_nodes.append(dict(id_node=next_node, geometry=geom.interpolate(d)))
        mapping[next_node] = original
        left = dict(old, id_ds_node=next_node, geometry=substring(geom, 0, d))
        right = dict(old, id_link=next_link, id_us_node=next_node, geometry=substring(geom, d, geom.length))
        records.extend([left, right])
        actions.append(dict(action='cut_inserted', parent_node=original, raw_node=next_node,
                            source_link=old['source_link'], displacement_m=float(geom.distance(point))))
        next_node += 1; next_link += 1
    result = gpd.GeoDataFrame(records, crs=links.crs)
    if new_nodes:
        nodes = gpd.GeoDataFrame(pd.concat([nodes, gpd.GeoDataFrame(new_nodes, crs=nodes.crs)], ignore_index=True), crs=nodes.crs)
    result['u'] = result.id_us_node.map(mapping)
    result['v'] = result.id_ds_node.map(mapping)
    nodes['parent_identity'] = nodes.id_node.map(mapping)
    result['geometry_length_m'] = result.length
    return result, nodes, actions


def topology(links):
    graph = nx.MultiDiGraph()
    graph.add_edges_from(zip(links.id_us_node, links.id_ds_node))
    return dict(nodes=graph.number_of_nodes(), links=graph.number_of_edges(),
                components=nx.number_weakly_connected_components(graph),
                cycle_rank=graph.number_of_edges()-graph.number_of_nodes()+nx.number_weakly_connected_components(graph),
                dag=nx.is_directed_acyclic_graph(graph),
                inlets=sorted(n for n in graph if graph.in_degree(n)==0),
                outlets=sorted(n for n in graph if graph.out_degree(n)==0))
