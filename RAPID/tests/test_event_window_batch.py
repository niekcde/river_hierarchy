"""Guard against false early settlement and verify resumable adaptive routing."""
import json
import numpy as np
import pandas as pd
import pytest
from rapid_tools import event_window_batch as batch
from rapid_tools.common_grid import route_interval_means


def frame():
    return pd.DataFrame(dict(u=[0,1,1,2,3],v=[1,2,3,4,4],rapid_k=[30.,50.,80.,40.,70.],
        rapid_x=[-2.]*5,count=[2,3,1,2,3],split_fraction=[1.,.3,.7,1.,1.],
        reference_flow_fraction=[1.,.3,.7,.3,.7]))


def test_equilibrium_prefix_does_not_change_event_response():
    f=frame();flow=np.r_[np.full(5,10.),np.full(20,20.),np.full(30,8.)]
    assert batch.equilibrium_check(f,5.)<1e-12
    q,_=route_interval_means(f,flow,2,5.,initial_flow=10.)
    padded,_=route_interval_means(f,np.r_[np.full(24*360,10.),flow],2,5.,initial_flow=10.)
    np.testing.assert_allclose(padded[-len(flow):],q,rtol=1e-12,atol=1e-10)


def test_rejects_inconsistent_equilibrium():
    f=frame();f.loc[1,'reference_flow_fraction']=.4
    with pytest.raises(ValueError,match='initialization'):batch.equilibrium_check(f,5.)


def test_later_excursion_invalidates_temporary_settlement():
    t=pd.date_range('2025-01-01',periods=200*60,freq='min',tz='UTC')+pd.Timedelta(seconds=30)
    q=np.ones(len(t))*10;q[150*60]=11
    required,covered=batch.settling_hours(t,q,t[0]-pd.Timedelta(seconds=30),10,.01)
    assert required>246 and not covered
    q[150*60]=10
    required,covered=batch.settling_hours(t,q,t[0]-pd.Timedelta(seconds=30),10,.01)
    assert required==96 and covered


def test_new_batch_extends_shared_tail_then_resumes_and_rejects_corruption(tmp_path,monkeypatch):
    monkeypatch.setattr(batch,'ROOT',tmp_path)
    p=tmp_path/'params.parquet';frame().to_parquet(p)
    t=pd.date_range('2025-01-01',periods=3,freq='min',tz='UTC')+pd.Timedelta(seconds=30)
    f=pd.DataFrame(dict(time_utc=t,interval_start_utc=t-pd.Timedelta(seconds=30),interval_end_utc=t+pd.Timedelta(seconds=30),forcing_cms=[10.,20.,12.],forcing_endpoint_hold=False,upstream_gap=False))
    fp=tmp_path/'forcing.parquet';f.to_parquet(fp)
    w=dict(start_utc=t[0].isoformat(),end_utc=t[-1].isoformat(),start_cms=10.,end_cms=12.)
    c=dict(example='02',window=w,forcing=str(fp),initial_hours=24,tail_hours=120,initial_cms=10.,final_cms=12.,event_intervals=3,tolerance_cms=.01,simulation_end_utc=(t[-1]+pd.Timedelta(seconds=30,hours=120)).isoformat(),source_sha256={},jobs=[dict(state_id='S000_base',parameters=str(p),native_dt_seconds=5.,aliases=['indep/S000_base'])])
    plan=tmp_path/'plan.json';batch.save(plan,dict(schema='reviewed-event-window-batch-v2',source_sha256={},code_sha256={},cases=[c]))
    out=tmp_path/'RAPID/outputs/test';calls=[]
    def route(f,flow,nsub,dt,initial_flow):
        calls.append(len(flow));q=np.full(len(flow),12.)
        if len(calls)==1:q[-1]=13. # deliberately censored late excursion
        return q,dict(minimum_native_flow_cms=10.)
    monkeypatch.setattr(batch.common_grid,'route_interval_means',route)
    batch.run(plan,out)
    done=json.loads((out/'02/completion.json').read_text())
    assert done['final_tail_hours']>120 and len(calls)==2
    batch.run(plan,out);assert len(calls)==2
    path=next(k for k in done['files'] if k.endswith('outlet.parquet'))
    with open(path,'ab') as handle:handle.write(b'corruption')
    with pytest.raises(ValueError,match='changed'):batch.run(plan,out)
