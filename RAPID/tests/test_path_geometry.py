import itertools
import networkx as nx
import numpy as np
import pandas as pd
from rapid_tools.path_geometry import path_geometry


def check(edges):
    f=pd.DataFrame([dict(id_link=i,id_us_node=u,id_ds_node=v,link_length_m=l,width=w,head_us=100-u,head_ds=100-v) for i,(u,v,l,w) in enumerate(edges)])
    result,source,sink=path_geometry(f,f.id_link.tolist());g=nx.MultiDiGraph()
    for r in f.itertuples():g.add_edge(r.id_us_node,r.id_ds_node,key=r.id_link,l=r.link_length_m,w=r.width)
    lengths=[];widths=[]
    for path in nx.all_simple_edge_paths(g,source,sink):
        length=sum(g.edges[e]['l'] for e in path);lengths.append(length);widths.append(length/sum(g.edges[e]['l']/g.edges[e]['w'] for e in path))
    l=np.array(lengths);w=np.array(widths);leq=np.average(l,weights=w)
    expected=dict(equivalent_length=leq,elongation=leq/w.sum(),dominant_width_fraction=w.max()/w.sum(),path_length_cv=l.std()/l.mean(),association=leq/l.mean()-1,slope=(sink-source)/l.min(),paths=len(l))
    for key,value in expected.items():np.testing.assert_allclose(result[key],value,rtol=1e-8,atol=1e-10)


def test_parallel_and_shared_paths():
    check([(0,1,100,5),(0,1,600,50),(1,2,30,3),(1,2,700,10),(2,3,20,20)])


def test_random_dags_match_enumeration():
    rng=np.random.default_rng(7)
    for _ in range(8):
        edges=[(i,i+1,float(rng.uniform(10,1000)),float(rng.uniform(1,200))) for i in range(7)]
        edges += [(i,j,float(rng.uniform(10,1000)),float(rng.uniform(1,200))) for i,j in itertools.combinations(range(8),2) if j>i+1 and rng.random()<.25]
        check(edges)


def test_billion_paths_without_enumeration():
    f=pd.DataFrame([dict(id_link=2*i+j,id_us_node=i,id_ds_node=i+1,link_length_m=100.,width=10.,head_us=40-i,head_ds=39-i) for i in range(30) for j in range(2)])
    result,_,_=path_geometry(f,f.id_link.tolist())
    assert result['paths']==2**30
    np.testing.assert_allclose(result['equivalent_length'],3000)
    np.testing.assert_allclose(result['dominant_width_fraction'],1/2**30)
    assert result['path_length_cv']==0


def test_single_path_invariants_are_exact():
    f=pd.DataFrame([dict(id_link=i,id_us_node=i,id_ds_node=i+1,
        link_length_m=l,width=w,head_us=4-i,head_ds=3-i)
        for i,(l,w) in enumerate([(23.7,3.2),(100.3,11.1),(89.1,2.9)])])
    result,_,_=path_geometry(f,f.id_link.tolist())
    assert result['dominant_width_fraction']==1.
    assert result['association']==0.
    assert result['path_length_cv']==0.
    assert result['quadrature_relative_error']==0.
