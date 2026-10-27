"""Extend pilot tail and verify centroids against exact linear-network moments."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src')]
import numpy as np
import pandas as pd
from rapid_tools.cunge import route
OUT=ROOT/'RAPID/outputs/local_boundary_20261027_v2';SRC=ROOT/'RAPID/outputs/uncapped_event_window_20261020/parameters/22'

def main():
    dest=OUT/'extended_pulse';dest.mkdir(exist_ok=False)
    previous=pd.read_parquet(OUT/'pulse_hydrographs.parquet');step=300.;t=np.arange(int(np.ceil(3*previous.time_seconds.max()/step))+1)*step
    flow=100+100*np.maximum(0,1-abs(t-24*3600)/(12*3600));saved=pd.DataFrame({'time_seconds':t,'inflow_cms':flow})
    dt=float(pd.read_csv(OUT/'routing_responses.csv').dt.iloc[0]);nsub=round(step/dt)
    frames={'parent':pd.read_parquet(SRC/'S000_base/parameters.parquet')}
    for state in ['S013_unit_8','S296_unit_30','island_22_041']:
        frames[state+'_existing']=pd.read_parquet(SRC/state/'parameters.parquet')
        frames[state+'_local']=pd.read_parquet(OUT/state/('parameters.parquet' if state=='S013_unit_8' else 'port_path_parameters.parquet'))
    rows=[];moment=lambda f:float((f.reference_flow_fraction*f['count']*f.rapid_k).sum());base_mean=moment(frames['parent'])
    for name,f in frames.items():
        q,a=route(f,flow,nsub,dt);e=q-100;mass=e.sum()*step;centroid=float((t*e).sum()/e.sum());exact=moment(f)
        rows.append(dict(name=name,dt=dt,centroid_seconds=centroid,exact_centroid_shift_minutes=(exact-base_mean)/60,peak_cms=float(q.max()),relative_volume_error=float(mass/((flow-100).sum()*step)-1),tail_excess_cms=float(e[-1]),**a));saved[name]=q
        pd.DataFrame(rows).to_csv(dest/'responses.csv',index=False);print(name,rows[-1]['relative_volume_error'],flush=True)
    a=pd.DataFrame(rows).set_index('name');a['pulse_centroid_shift_minutes']=(a.centroid_seconds-a.loc['parent','centroid_seconds'])/60
    a['centroid_contrast_error_seconds']=60*(a.pulse_centroid_shift_minutes-a.exact_centroid_shift_minutes)
    a.to_csv(dest/'responses.csv');saved.to_parquet(dest/'hydrographs.parquet',index=False)
    assert a.relative_volume_error.abs().max()<1e-5
    assert a.centroid_contrast_error_seconds.abs().max()<.1
    assert a.minimum_native_flow_cms.min()>0
    print(a[['exact_centroid_shift_minutes','pulse_centroid_shift_minutes','centroid_contrast_error_seconds']].to_string())

if __name__=='__main__':main()
