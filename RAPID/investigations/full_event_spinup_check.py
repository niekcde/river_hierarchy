"""Complete-waveform 0/6/24-hour initial-buffer check for isolated selected states."""
from pathlib import Path
import sys,json,time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
from event_buffer_tests import OUT,cascade,DEFAULT_REFRESH_ROOT,peak


def main():
    selected=pd.read_csv(OUT/'selection.csv',dtype={'example':str});rows=[]
    for ex,g in selected.groupby('example'):
        inv=pd.read_csv(DEFAULT_REFRESH_ROOT/ex/'inventory.csv').drop_duplicates('fingerprint').set_index('fingerprint')
        for r in g.itertuples():
            h=pd.read_parquet(OUT/f'{ex}_{r.fingerprint}.parquet');t=h.time_since_event_start_seconds.to_numpy();flow=h.inflow_cms.to_numpy();ref=h.outlet_cms.to_numpy()
            dt=float(np.diff(t)[0]);audit=json.loads((DEFAULT_REFRESH_ROOT/ex/r.fingerprint/'run/Qout.audit.json').read_text());routing_dt=audit['routing_dt_seconds'];nsub=round(dt/routing_dt)
            f=pd.read_parquet(inv.loc[r.fingerprint,'frame'])
            amplitude=max(1.,np.ptp(ref))
            for hours in [6,24]:
                n=int(np.ceil(hours*3600/dt));begin=time.perf_counter()
                out,_=cascade(f,np.r_[np.full(n,flow[0]),flow],nsub,routing_dt,float(flow[0]));q=out[n:]
                error=float(np.max(abs(q-ref)));assert error<1e-8*amplitude
                rows.append(dict(example=ex,state=r.state,fingerprint=r.fingerprint,spinup_hours=hours,
                                 actual_spinup_hours=n*dt/3600,max_wave_error_cms=error,
                                 max_wave_error_fraction=error/amplitude,
                                 peak_time_difference_seconds=peak(t,q)-peak(t,ref),
                                 peak_height_difference_cms=float(q.max()-ref.max()),
                                 tv_difference_cms=float(abs(np.diff(q)).sum()-abs(np.diff(ref)).sum()),
                                 calculation_seconds=time.perf_counter()-begin))
            pd.DataFrame(rows).to_csv(OUT/'full_waveform_spinup_tests.csv',index=False)
        print(ex,'complete-event spinup tests complete',flush=True)


if __name__=='__main__':main()
