from dataclasses import replace
import json

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from synthetic_experiments.networks import make_states, canonical_network
from synthetic_experiments.routing import Settings, schedule, compact_k, prepare, forcing, route
from synthetic_experiments.resistance import predictor
from synthetic_experiments.pipeline import plan, run_jobs, analyse, reuse_runs

BASE=dict(total_width_m=500,reference_slope=.0001,outer_a_fraction=.8,inner_fraction=.5,
          length_multiplier=1.,total_length_m=20000,internal_location='central')


def test_seven_distinct_baseline_networks_and_length_adjustment():
    states,pairs=make_states(BASE)
    keys={json.dumps(canonical_network(g),sort_keys=True) for g in states.values()}
    assert len(keys)==7
    assert len(pairs)==7
    changed,_=make_states(dict(BASE,length_multiplier=1.5))
    assert canonical_network(changed['simple/full'])==canonical_network(states['centerline/original'])
    assert canonical_network(changed['nested/inner_only'])!=canonical_network(changed['simple/original'])
    assert sum(d['length_m'] for *_,d in changed['nested/original'].edges(data=True)) > sum(d['length_m'] for *_,d in states['nested/original'].edges(data=True))


def test_shared_segmentation_matches_planning_and_forcing_volume():
    states,_=make_states(dict(BASE,total_length_m=10000,length_multiplier=1.25))
    s=Settings()
    nets=[canonical_network(g) for g in states.values()]
    clock=schedule(nets,s)
    for net in nets:
        estimated=compact_k(net,s); actual=prepare(net,s)
        assert actual.rapid_k.min()==pytest.approx(estimated.rapid_k.min())
        assert actual.rapid_k.max()==pytest.approx(estimated.rapid_k.max())
        assert len(actual)==estimated['count'].sum()
    t,q=forcing(clock,s)
    expected=.5*(s.peak_cms-s.baseflow_cms)*(s.rise_hours+s.fall_hours)*3600
    assert np.sum(q-s.baseflow_cms)*clock['dt_seconds']==pytest.approx(expected)


def test_resistance_domains_and_width_scale_invariance():
    s=Settings(); states,pairs=make_states(BASE)
    larger,_=make_states(dict(BASE,total_width_m=1000))
    for p in pairs:
        r=predictor(states[p['parent_state']],states[p['child_state']],p,s)
        rr=predictor(larger[p['parent_state']],larger[p['child_state']],p,s)
        assert r['Lambda_matched']==pytest.approx(rr['Lambda_matched'],rel=1e-8)
        assert r['Lambda_matched']>0
        if p['family']=='crossover' and p['child_state']!='crossover/full':
            assert r['domain_mode']=='containing_compound_context'
            assert r['width_fraction']==1
        if p['child_state']=='nested/main_island_only':
            assert r['domain_mode']=='containing_unit_island_context'
            assert r['width_fraction']==1
            assert (r['domain_start_m'],r['domain_end_m'])==(5000,15000)
        if p['child_state']=='nested/inner_only':
            assert r['width_fraction']==pytest.approx(.8)


def test_routing_agrees_with_shared_file_engine(tmp_path):
    from rapid_tools.prep import (create_conn_file,create_riv_file,compute_reach_ratios,
        order_prepared_links_for_total_reach_order,write_routing_parameter_files)
    from rapid_tools.forcing import write_inflow_netcdf
    from rapid_tools.engine import run_rapid
    states,_=make_states(BASE); s=Settings()
    net=canonical_network(states['crossover/original']); clock=schedule([net],s)
    h,m,frame=route(net,s,clock)
    g=nx.MultiDiGraph()
    for r in frame.itertuples():
        g.add_edge(r.id_us_node,r.id_ds_node,key=str(r.reach_id),reach_id=r.reach_id,width=r.width)
    _,order=create_conn_file(g,tmp_path)
    _,river_order=create_riv_file(g,tmp_path)
    compute_reach_ratios(g,tmp_path)
    frame['centroid_x']=frame.geometry.centroid.x
    frame['centroid_y']=0.
    ordered=order_prepared_links_for_total_reach_order(frame,order)
    write_routing_parameter_files(ordered,tmp_path)
    t,q=forcing(clock,s)
    f=pd.DataFrame(dict(time=pd.Timestamp('2000-01-01')+pd.to_timedelta(t,unit='s'),discharge_cms=q))
    write_inflow_netcdf(ordered,f,tmp_path/'inflow.nc')
    _,qout=run_rapid(tmp_path,ROUTING_TIMESTEP_SECONDS=clock['dt_seconds'],return_qout=True)
    outlet=int(frame.loc[frame.is_outlet,'reach_id'].iloc[0])
    assert qout[:,river_order.index(outlet)]==pytest.approx(h.outlet_cms.to_numpy(),rel=1e-6,abs=1e-5)


def test_end_to_end_resume_and_completed_comparisons(tmp_path):
    config=tmp_path/'config.json'
    config.write_text(json.dumps(dict(cases=[BASE],routing={})))
    output=tmp_path/'experiment'
    manifest=plan(config,output)
    assert manifest['n_unique_runs']==7
    run_jobs(output)
    paths=list((output/'runs').glob('*/metrics.json'))
    stamps=[p.stat().st_mtime_ns for p in paths]
    run_jobs(output)
    assert [p.stat().st_mtime_ns for p in paths]==stamps
    result=analyse(output)
    assert len(result)==7
    assert (result.parent_volume_error.abs()<1e-4).all()


def test_exact_reuse_and_incompatible_provenance_rejected(tmp_path):
    config=tmp_path/'config.json'
    config.write_text(json.dumps(dict(cases=[BASE],routing={})))
    old,new=tmp_path/'old',tmp_path/'new'
    plan(config,old);run_jobs(old)
    plan(config,new)
    report=reuse_runs(new,old)
    assert report['reused']==7 and report['remaining']==0
    for item in report['jobs']:
        original=old/'runs'/item['source_run_id']
        copied=new/'runs'/item['run_id']
        assert (copied/'hydrograph.parquet').read_bytes()==(original/'hydrograph.parquet').read_bytes()
        assert (copied/'segments.csv').read_bytes()==(original/'segments.csv').read_bytes()
        assert (copied/'segments.csv').stat().st_ino!=(original/'segments.csv').stat().st_ino
    # A different clock is a different routing experiment, even for identical geometry.
    third=tmp_path/'changed_clock';m=plan(config,third)
    for run_id in m['run_ids']:
        path=third/'jobs'/f'{run_id}.json';job=json.loads(path.read_text())
        job['clock']['buffer_hours']+=1
        path.write_text(json.dumps(job))
    assert reuse_runs(third,old)['reused']==0
    path=old/'manifest.json';m=json.loads(path.read_text());m['routing_source']='unverified'
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='compatibility'):
        reuse_runs(new,old)
