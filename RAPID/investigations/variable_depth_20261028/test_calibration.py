import numpy as np
import pandas as pd
import calibrate_references as c

def test_calibration_retains_failed_clock_review_and_resumes(tmp_path,monkeypatch):
 c.r.init_worker(tmp_path,'test');c.CTX=c.r.CTX
 monkeypatch.setattr(c,'ensure_space',lambda *args:100.)
 ex='05';case=c.CTX['cases'][ex];goal=pd.Timestamp(case['timing_target']['target_utc'])
 data=pd.DataFrame(dict(time_utc=pd.date_range(goal-pd.Timedelta(minutes=5),periods=11,freq='min'),forcing_cms=np.full(11,100.),downstream_cms=np.full(11,100.)))
 original=pd.read_parquet
 monkeypatch.setattr(pd,'read_parquet',lambda path,*a,**kw:data.copy() if str(path)==case['calibration_forcing'] else original(path,*a,**kw))
 _,dt,_=c.prepare_frame(ex,'S000_base',c.RUN)
 def solve(f,flow,clock,initial):
  q=np.full(11,100.);q[5 if clock==dt else 7]=200.
  return q,dict(minimum_flow_cms=100.,depth_min_m=1.,depth_max_m=2.,residual_pct_excess_volume=0.)
 monkeypatch.setattr(c,'solve_wave',solve)
 result=c.calibrate_task((c.RUN,ex))
 assert result['timing_tolerance_met'] and not result['calibration_accepted']
 assert not result['spatial_convergence_certified']
 assert result['numerical_checks'][0]['timing_change_minutes']==2.
 assert c.calibrate_task((c.RUN,ex))==result
