import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linprog
from feasibility import check_slopes,path_constraints


def chain():
    return pd.DataFrame(dict(id_link=[1,2,3],u=[0,1,2],v=[1,2,3],link_length_m=[100.,100.,100.]))


def test_surviving_internal_heads_are_free():
    f=chain();h=pd.Series({0:10.,1:4.,2:6.,3:0.});pos=pd.Series({0:300.,3:0.})
    result=check_slopes(f,h,{0,3},pos)
    assert result['status']=='feasible' and result['free_nodes']==2
    assert result['maximum_uniform_positive_slope']==pytest.approx(10/300)


def test_actual_boundary_reversal_has_explicit_path_witness():
    f=chain();h=pd.Series({0:0.,3:10.});pos=pd.Series({0:300.,3:0.})
    result=check_slopes(f,h,{0,3},pos)
    assert result['status']=='nonpositive_fixed_head_path'
    assert result['nonpositive_paths'][0]['path_links']==[1,2,3]


def test_floor_failure_is_separate_from_positive_slope_feasibility():
    f=chain();f['link_length_m']=20000.
    result=check_slopes(f,pd.Series({0:10.,3:0.}),{0,3},pd.Series({0:300.,3:0.}))
    assert result['positive_slopes_feasible']
    assert result['status']=='relative_floor_infeasible'
    assert result['maximum_floor_fraction']==pytest.approx(.005)


def test_fixed_fixed_edges_are_checked():
    f=pd.DataFrame(dict(id_link=[1],u=[0],v=[1],link_length_m=[10.]))
    r=check_slopes(f,pd.Series({0:1.,1:1.}),{0,1},pd.Series({0:10.,1:0.}))
    assert r['status']=='nonpositive_fixed_head_path'


def test_longest_parallel_path_controls_required_drop():
    f=pd.DataFrame(dict(id_link=[1,2,3,4],u=[0,0,1,2],v=[1,1,2,3],link_length_m=[100.,500.,100.,100.]))
    paths=path_constraints(f,pd.Series({0:10.,3:0.}),f.link_length_m)
    assert paths[0]['path_links']==[2,3,4] and paths[0]['path_weight']==700


def test_path_criterion_matches_independent_linear_program():
    rng=np.random.default_rng(123)
    for _ in range(60):
        edges=[(i,i+1) for i in range(7)]+[(i,j) for i in range(8) for j in range(i+2,8) if rng.random()<.2]
        f=pd.DataFrame(edges,columns=['u','v']);f['id_link']=np.arange(len(f));w=rng.uniform(.01,2,len(f))
        fixed={0:10.,3:float(rng.uniform(0,10)),7:0.};heads=pd.Series(fixed)
        paths=path_constraints(f,heads,w);expected=all(p['available_drop_m']>=p['path_weight']-1e-9 for p in paths)
        free=[v for v in range(8) if v not in fixed];ix={v:i for i,v in enumerate(free)};A=np.zeros((len(f),len(free)));b=-w.copy()
        for j,r in enumerate(f.itertuples()):
            for v,sign in [(r.u,1),(r.v,-1)]:
                if v in ix:A[j,ix[v]]-=sign
                else:b[j]+=sign*fixed[v]
        lp=linprog(np.zeros(len(free)),A_ub=A,b_ub=b,bounds=[(None,None)]*len(free),method='highs')
        assert lp.success==expected
