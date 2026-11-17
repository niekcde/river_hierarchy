"""MC refinement under frozen hydraulics, using production interval means."""
import sys,json
import numpy as np
import pandas as pd
from verify_inertial56 import ROOT,OUT,PREVIOUS,SAVED,metrics
sys.path.insert(0,str(ROOT/'RAPID/src'))
sys.path.insert(0,str(ROOT))
from rapid_tools import cunge,common_grid

def main():
    forcing=pd.read_parquet(SAVED/'forcing.parquet');origin=forcing.interval_start_utc.iloc[0]
    t=(forcing.time_utc-origin).dt.total_seconds().to_numpy();flow=forcing.forcing_cms.to_numpy();event=forcing[forcing.phase.eq('input_event')]
    baseline=np.interp(t,[(event.interval_start_utc.iloc[0]-origin).total_seconds(),(event.interval_end_utc.iloc[-1]-origin).total_seconds()],[flow[0],flow[-1]])
    records=[]
    for state in ['S000_base','S014_unit_10']:
        f=pd.read_parquet(PREVIOUS/(state+'_mc_parameters.parquet'))
        for factor,dt in [(1,15.),(1,7.5),(2,15.),(4,15.),(8,15.)]:
            g=f.copy();g['count']=f['count']*factor;g['rapid_effective_length_m']=g.link_length_m/g['count'];g=cunge.parameters(g,float(f.n_manning.iloc[0]))
            q,audit=common_grid.route_interval_means(g,flow,round(60/dt),dt,initial_flow=flow[0])
            tag=f'{state}_MC_x{factor}_dt{dt:g}'
            w=forcing[['time_utc','phase']].copy();w['q_outlet_cms']=q;w.to_parquet(OUT/(tag+'.parquet'),index=False)
            if factor==1 and dt==15:
                saved=pd.read_parquet(SAVED/state/'outlet.parquet').q_outlet_cms.to_numpy()
                error=float(abs(q-saved).max());assert error<1e-7,error
            else:error=None
            record=dict(tag=tag,state=state,factor=factor,dt=dt,cells=int(g['count'].sum()),saved_reproduction_error_cms=error,**metrics(t,q,baseline),**audit)
            records.append(record);print(record,flush=True)
    pd.DataFrame(records).to_csv(OUT/'mc_resolution.csv',index=False)

if __name__=='__main__':main()
