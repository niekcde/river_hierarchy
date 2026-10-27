import numpy as np
import pytest
from dataclasses import replace
from synthetic_experiments.design import Design, FAMILIES, build_original
from synthetic_experiments.networks import make_states, validate_graph
from synthetic_experiments.hydraulics import HydraulicSettings, mct_cell, prepare, route, forcing

PARAMS=dict(total_width_m=500.,reference_slope=1e-4,outer_a_fraction=.5,inner_fraction=.5,
            length_multiplier=1.,total_length_m=20000.,internal_location='central')

@pytest.mark.parametrize('factor',[1.,1.1,1.2])
def test_width_expansion_all_seven_pairs(factor):
    states,pairs=make_states(dict(PARAMS,branch_width_factor=factor))
    assert len(pairs)==7
    for key,g in states.items():
        validate_graph(g)
        root=next(n for n in g if not g.in_degree(n));sink=next(n for n in g if not g.out_degree(n))
        assert sum(d['w'] for *_,d in g.out_edges(root,data=True))==pytest.approx(500)
        assert sum(d['w'] for *_,d in g.in_edges(sink,data=True))==pytest.approx(500)
        for x in [6000,9000,13000]:
            total=sum(d['w'] for u,v,d in g.edges(data=True) if g.nodes[u]['x']<x<g.nodes[v]['x'])
            assert total==pytest.approx(500*(1 if key.startswith('centerline/') else factor))
    for pair in pairs:
        area=lambda g:sum(d['w']*d['length_base_m'] for *_,d in g.edges(data=True))
        assert area(states[pair['parent_state']])==pytest.approx(area(states[pair['child_state']]))


def test_unexpected_width_change_rejected():
    states,_=make_states(dict(PARAMS,branch_width_factor=1.2));g=states['nested/original']
    edge=next(e for e in g.edges(keys=True) if g.edges[e].get('kind')=='loop')
    g.edges[edge]['w']*=2
    with pytest.raises(ValueError):validate_graph(g)


def test_steady_state_and_storage():
    q=np.full(2000,50.)
    out,a=mct_cell(q,25,1e-4,.025,200,15)
    np.testing.assert_allclose(out,q,rtol=1e-12,atol=1e-12)
    h=(.025*50/(25*np.sqrt(1e-4)))**.6
    assert a['initial_storage_m3']==pytest.approx(25*h*200)


def test_unsteady_conservation_including_storage_and_refinement():
    dt=15.;t=np.arange(0,5*86400+dt,dt)
    q=np.interp(t,[0,3600,7200,14400,t[-1]],[50,50,100,50,50])
    out,a=mct_cell(q,25,3.333333333e-5,.025,1000,dt)
    assert out.min()>0
    assert abs(np.trapz(q-out,t)-(a['final_storage_m3']-a['initial_storage_m3'])) < .01
    tt=np.arange(0,t[-1]+dt/2,dt/2)
    refined,_=mct_cell(np.interp(tt,t,q),25,3.333333333e-5,.025,1000,dt/2)
    assert np.max(abs(refined[::2]-out))<.02


@pytest.mark.parametrize('method',['fixed_manning','variable_manning'])
def test_equal_split_invariance(method):
    params=dict(PARAMS,branch_width_factor=1.2,total_length_m=2000.)
    states,_=make_states(params)
    s=HydraulicSettings(method=method,rise_hours=1,fall_hours=2,spinup_hours=1,tail_hours=24)
    t,q=forcing(s);outputs=[]
    for key in ['simple/original','simple/full']:
        w,a=route(prepare(states[key],params,s),q,s)
        assert abs(a['network_mass_residual_m3'])<.1
        outputs.append(w)
    np.testing.assert_allclose(*outputs,atol=1e-7,rtol=1e-10)


def test_small_perturbation_matches_fixed_cunge():
    from scipy.signal import lfilter
    dt=15;t=np.arange(0,2*86400,dt);base=100.;q=base+.001*np.exp(-((t-30000)/6000)**2)
    q[0]=base
    variable,_=mct_cell(q,100,1e-4,.025,200,dt)
    h=(.025*base/(100*.01))**.6;c=5/3*h**(2/3)*.01/.025
    k=200/c;x=.5-base/(2*100*1e-4*c*200);den=dt+2*k*(1-x)
    a=(dt-2*k*x)/den;b=(dt+2*k*x)/den;d=(2*k*(1-x)-dt)/den
    fixed,_=lfilter([a,b],[1,-d],q,zi=[(b+d)*base])
    assert max(abs(variable-fixed))<2e-7


def test_current_runner_persists_seven_pairs_and_resumes(tmp_path):
    from synthetic_experiments.depth_pipeline import run_group
    from dataclasses import asdict
    import pandas as pd
    s=HydraulicSettings(method='variable_manning',rise_hours=1,fall_hours=2,
                        spinup_hours=1,tail_hours=12,segment_length_m=200)
    job=dict(id='test',case='equal',parameters=dict(PARAMS,total_length_m=2000,branch_width_factor=1.2),settings=asdict(s))
    limits=dict(maximum_tail_hours=192,mass_balance_tolerance=1e-7,volume_tolerance=1e-6,
                centroid_seconds=1,final_excess_fraction=1e-7)
    result=run_group(job,{},limits,tmp_path)
    assert len(result)==7
    assert set(result.method)=={'variable_manning'}
    assert set(result.branch_width_factor)=={1.2}
    again=run_group(job,{},limits,tmp_path)
    np.testing.assert_allclose(result.centroid_shift_minutes,again.centroid_shift_minutes,atol=1e-12)
    with pytest.raises(ValueError,match='provenance'):
        run_group(job,{'changed':True},limits,tmp_path)


def test_shared_state_cache_preserves_outputs_and_skips_routing(tmp_path,monkeypatch):
    import synthetic_experiments.depth_pipeline as dp
    import pandas as pd
    from dataclasses import asdict
    s=HydraulicSettings(method='variable_manning',rise_hours=1,fall_hours=2,spinup_hours=1,tail_hours=12)
    job=dict(id='cache_test',case='equal',parameters=dict(PARAMS,total_length_m=2000,branch_width_factor=1.2),settings=asdict(s))
    limits=dict(maximum_tail_hours=192,mass_balance_tolerance=1e-7,volume_tolerance=1e-6,centroid_seconds=1,final_excess_fraction=1e-7)
    a=dp.run_group(job,{},limits,tmp_path/'uncached')
    b=dp.run_group(job,{},limits,tmp_path/'first',tmp_path/'cache')
    def forbidden(*args):raise AssertionError('Identical state should come from cache')
    monkeypatch.setattr(dp,'route',forbidden)
    z=dp.run_group(job,{},limits,tmp_path/'second',tmp_path/'cache')
    pd.testing.assert_frame_equal(a,b,check_exact=True)
    pd.testing.assert_frame_equal(b,z,check_exact=True)
    for name in ['first','second']:
        pd.testing.assert_frame_equal(pd.read_parquet(tmp_path/'uncached/hydrographs.parquet'),pd.read_parquet(tmp_path/name/'hydrographs.parquet'),check_exact=True)
    with pytest.raises(AssertionError,match='Identical state'):
        dp.run_group(job,{'changed':True},limits,tmp_path/'changed',tmp_path/'cache')
