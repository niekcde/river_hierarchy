import sys
from pathlib import Path
import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rapid_tools.anchor_repair_trial import repair, topology


def fixture():
    links=gpd.GeoDataFrame(dict(id_link=[1],id_us_node=[10],id_ds_node=[11],geometry=[LineString([(0,0),(10,0)])]),crs=32631)
    nodes=gpd.GeoDataFrame(dict(id_node=[10,11],geometry=[Point(0,0),Point(10,0)]),crs=32631)
    parent=gpd.GeoDataFrame(dict(id_node=[1,2,3,4],geometry=[Point(0,0),Point(10,0),Point(3,0),Point(7,0)]),crs=32631)
    return links,nodes,parent,{10:1,11:2}


def test_two_cuts_preserve_line_and_create_real_nodes():
    l,n,p,m=fixture()
    a,b,events=repair(l,n,p,m,[3,4],cuts=[3,4])
    assert list(a.sort_values('u').u)==[1,3,4]
    assert set(zip(a.u,a.v))=={(1,3),(3,4),(4,2)}
    assert a.length.sum()==10
    assert topology(a)['cycle_rank']==topology(l)['cycle_rank']
    assert len(b)==4 and len(events)==2
    assert a.geometry.union_all().equals(l.geometry.iloc[0])


def test_ambiguous_cut_fails():
    l,n,p,m=fixture()
    l.loc[1]=[2,10,11,l.geometry.iloc[0]]
    with pytest.raises(ValueError,match='unique interior cut, found 2'):
        repair(l,n,p,m,[3],cuts=[3])


def test_identity_collision_fails():
    l,n,p,m=fixture()
    with pytest.raises(ValueError,match='collision'):
        repair(l,n,p,m,[2],correspondences={10:2})


def test_displaced_reviewed_identity_does_not_snap_geometry():
    l,n,p,m=fixture()
    a,b,events=repair(l,n,p,m,[3],correspondences={10:3})
    assert b.geometry.equals(n.geometry)
    assert a.u.iloc[0]==3
    assert events[0]['displacement_m']==3


def test_existing_identity_is_noop_and_duplicate_cut_rejected():
    l,n,p,m=fixture()
    a,b,events=repair(l,n,p,m,[1],correspondences={10:1})
    assert events[0]['action']=='already_correct'
    assert a.geometry.equals(l.geometry)
    with pytest.raises(ValueError,match='already represented'):
        repair(l,n,p,m,[1],cuts=[1])


def test_unknown_or_nonboundary_repairs_rejected():
    l,n,p,m=fixture()
    with pytest.raises(ValueError,match='prescribed'):
        repair(l,n,p,m,[3],cuts=[4])
    with pytest.raises(ValueError,match='Unknown'):
        repair(l,n,p,m,[3],correspondences={42:3})


def test_same_number_in_child_and_parent_namespaces_is_not_same_node():
    l,n,p,m=fixture()
    l['id_us_node']=64
    n.loc[n.id_node==10,'id_node']=64
    m={64:100,11:2}
    p.loc[p.id_node==1,'id_node']=66
    p.loc[p.id_node==3,'id_node']=64
    a,b,events=repair(l,n,p,m,[64,66],cuts=[64],correspondences={64:66})
    assert set(zip(a.u,a.v))=={(66,64),(64,2)}
    assert b.loc[b.id_node==64,'parent_identity'].iloc[0]==66
    assert b.loc[b.parent_identity==64,'id_node'].iloc[0]!=64
