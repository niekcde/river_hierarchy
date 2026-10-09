from pathlib import Path
import sys
import geopandas as gpd
from shapely.geometry import Point,LineString
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rapid_tools.boundary_contract_review import plot_comparison,identity_label,correspondence_table


def fixture():
    pn=gpd.GeoDataFrame(dict(id_node=[0,1,2,3]),geometry=[Point(0,-500),Point(1000,-500),Point(0,500),Point(1000,500)],crs=32631)
    parent=gpd.GeoDataFrame(dict(id_link=[1,2],id_us_node=[0,2],id_ds_node=[1,3]),geometry=[LineString([(0,-500),(1000,-500)]),LineString([(0,500),(1000,500)])],crs=32631)
    child=gpd.GeoDataFrame(dict(id_link=[80],id_us_node=[10],id_ds_node=[11],u=[4],v=[5]),geometry=[LineString([(200,0),(800,0)])],crs=32631)
    cn=gpd.GeoDataFrame(dict(id_node=[10,11],parent_identity=[4,5]),geometry=[Point(200,0),Point(800,0)],crs=32631)
    return parent,pn,child,cn


def test_link_between_parent_branches_is_not_dropped_from_viewport():
    p,pn,c,cn=fixture()
    fig=plot_comparison(c,c,p,cn,cn,pn,[0,1,2,3],[1,2],'Comparison')
    assert fig.layout.meta['visible_link_ids']['current']==[80]
    assert fig.layout.meta['visible_link_ids']['trial']==[80]
    assert len(fig.layout.annotations)>=3
    assert fig.layout.xaxis2.matches=='x' and fig.layout.yaxis3.matches=='y'
    assert any(t.name=='Selected parent links' for t in fig.data)


def test_context_nodes_are_not_shown_as_unit_boundaries_or_original_parent_ids():
    p,pn,c,cn=fixture()
    fig=plot_comparison(c,c,p,cn,cn,pn,[0,1,2,3],[1,2],'Comparison')
    assert identity_label(4,set(pn.id_node))=='H4'
    context=[t for t in fig.data if t.name=='Child context nodes']
    assert context and all('not a prescribed boundary' in v for t in context for v in t.customdata)
    assert any('H4' in v for t in context for v in t.customdata)


def test_identity_change_is_distinct_from_new_node_or_coordinate_shift():
    p,pn,c,cn=fixture();after=cn.copy();after.loc[0,'parent_identity']=0
    table=correspondence_table(cn,after,pn,[0],extra_child_nodes=[11])
    r=table[table.child_node.eq('C10')].iloc[0]
    assert r.before_identity=='H4' and r.trial_identity=='P0'
    assert r.identity_changed and not r.graph_node_added and r.coordinate_shift_m==0
    assert not table[table.child_node.eq('C11')].iloc[0].prescribed_boundary
