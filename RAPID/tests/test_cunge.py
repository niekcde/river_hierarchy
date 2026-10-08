"""Physical invariants and independent sparse/cascade parity for Cunge prep."""
import json
import numpy as np
import pandas as pd
import pytest
from scipy.io import netcdf_file
from rapid_tools.cunge import freeze_grid, parameters, common_clock, route, write_prepared_state
from rapid_tools.engine import run_prepared_state, ccc_mat, rte_mat, mus_rte
from rapid_tools.k_values import compute_dt_bounds
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import factorized


def example():
    return pd.DataFrame(dict(id_link=[10,11,12,13],u=[0,1,1,2],v=[1,2,2,3],
                             link_length_m=[100.,5000.,300.,700.],width=[10.,6.,4.,10.],
                             slope=[.001]*4,depth_m=[2.]*4))


def test_frozen_grid_preserves_lengths_discharge_and_diffusivity_across_n():
    grid=freeze_grid(example(),20.)
    np.testing.assert_allclose(grid.reference_discharge_cms,[20.,12.,8.,20.])
    for n in [.01,.035,.1]:
        f=parameters(grid,n)
        np.testing.assert_array_equal(f['count'],grid['count'])
        np.testing.assert_allclose(f['count']*f.rapid_effective_length_m,f.link_length_m)
        np.testing.assert_allclose(f.rapid_celerity_mps*f.rapid_effective_length_m*(.5-f.rapid_x),
                                   f.reference_discharge_cms/(2*f.width*f.slope))
        assert f.rapid_x.max()<=.25+1e-12
    with pytest.raises(ValueError,match='bounds'):parameters(grid,.005)
    with pytest.raises(ValueError,match='positive'):freeze_grid(example().assign(slope=0),20.)


@pytest.mark.parametrize('method',['cunge','fixed_x'])
def test_prepared_engine_matches_independent_native_cascade_at_every_report(tmp_path,method):
    grid=freeze_grid(example(),20.);f=parameters(grid,.035,method=method)
    dt,nsub=common_clock(grid,300.)
    flow=2.+18.*np.sin(np.linspace(0,np.pi,160))**2
    flow=np.r_[np.full(5,2.),flow,np.full(100,2.)]
    expected,audit=route(f,flow,nsub,dt)
    m=write_prepared_state(f,flow,300.,dt,tmp_path)
    _,q=run_prepared_state(tmp_path,return_qout=True)
    actual=q[:,np.array(m['outlet_reach_ids'])-1].sum(axis=1)
    np.testing.assert_allclose(actual,expected,rtol=2e-11,atol=2e-11)
    np.testing.assert_allclose(q[:5],np.broadcast_to(q[0],q[:5].shape),atol=1e-12)
    assert audit['minimum_native_flow_cms']>0
    assert m['represented_length_m']==pytest.approx(m['physical_length_m'])


def test_two_time_boundary_only_corrects_first_native_substep():
    k=np.array([100.]);x=np.array([-.4]);dt=10.
    c1,c2,c3=ccc_mat(k,x,dt);mats=rte_mat(csc_matrix((1,1)),c1,c2,c3)
    solve=factorized(mats[0]);state=np.array([2.]);q=np.array([10.])
    mean,end=mus_rte(*mats,3,state,q,solve,previous_external=state,previous_inflow_matrix=c2)
    manual=[];old=2.;prev=2.
    for _ in range(3):
        old=c1.diagonal()[0]*10+c2.diagonal()[0]*prev+c3.diagonal()[0]*old
        manual.append(old);prev=10
    assert end[0]==pytest.approx(old)
    assert mean[0]==pytest.approx(np.mean(manual))
    legacy,_=mus_rte(*mats,3,state,q,solve)
    assert not np.isclose(mean[0],legacy[0])


def test_negative_x_requires_explicit_policy_and_checks_every_edge():
    k=pd.Series([100.,50.]);x=np.array([-.4,.2])
    assert compute_dt_bounds(k,x_value=x,coefficient_policy='signed_inflow')==(0.,80.)
    with pytest.raises(ValueError,match='Negative X'):
        compute_dt_bounds(k,x_value=x,coefficient_policy='signed_current_inflow')
    with pytest.raises(ValueError,match='finite'):
        compute_dt_bounds(k,x_value=[np.nan,.1],coefficient_policy='signed_inflow')


def test_prepared_boundary_rejects_lateral_source(tmp_path):
    grid=freeze_grid(example(),20.);f=parameters(grid,.035);dt,_=common_clock(grid,300.)
    write_prepared_state(f,[2.,3.,2.],300.,dt,tmp_path)
    with netcdf_file(tmp_path/'inflow.nc','a') as ds:ds.variables['m3_riv'][1,-1]=1.
    with pytest.raises(ValueError,match='only at inlet'):
        run_prepared_state(tmp_path)
