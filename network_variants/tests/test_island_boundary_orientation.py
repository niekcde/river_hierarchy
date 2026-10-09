import geopandas as gpd
import networkx as nx
import pytest
from shapely.geometry import Point,LineString
from network_variants.island_boundary_orientation import orient_with_exterior_constraints


def graph_data():
    points={0:(-1,0),4:(0,0),5:(1,1),7:(2,1),6:(3,0),9:(4,0),274:(1,0),275:(2,0)}
    edges=[(3,0,4),(6,5,7),(9,6,9),(409,7,275),(410,275,274),(412,275,6),(413,274,5),(414,4,274)]
    return gpd.GeoDataFrame([dict(id_link=k,id_us_node=u,id_ds_node=v,geometry=LineString([points[u],points[v]])) for k,u,v in edges],crs=32631)


def test_surviving_island_branch_constrains_replacement_direction():
    links=graph_data();inserted={409,410,412,413,414}
    out,qa=orient_with_exterior_constraints(links,inserted_link_ids=inserted,source_nodes={4,7},sink_nodes={5,6},parent_node_order={0:0,4:1,5:2,7:3,6:4,9:5})
    g=nx.MultiDiGraph();g.add_edges_from(zip(out.id_us_node,out.id_ds_node))
    assert nx.is_directed_acyclic_graph(g)
    assert [(n,g.in_degree(n)) for n in g if g.in_degree(n)==0]==[(0,0)]
    assert [n for n in g if g.out_degree(n)==0]==[9]
    assert out.loc[out.id_link.eq(410),['id_us_node','id_ds_node']].values.tolist()==[[274,275]]
    for lid in [3,6,9]:
        a=links[links.id_link.eq(lid)].iloc[0];b=out[out.id_link.eq(lid)].iloc[0]
        assert (a.id_us_node,a.id_ds_node)==(b.id_us_node,b.id_ds_node)
        assert a.geometry.equals_exact(b.geometry,0)
    assert g.number_of_edges()-g.number_of_nodes()+1==1


def test_refuses_conflicting_protected_interface_constraints():
    links=graph_data()
    with pytest.raises(ValueError,match='Shared-role'):
        orient_with_exterior_constraints(links,inserted_link_ids={409,410,412,413,414},source_nodes={4,7,5},sink_nodes={5,6},parent_node_order={4:1,5:2,7:3,6:4})
