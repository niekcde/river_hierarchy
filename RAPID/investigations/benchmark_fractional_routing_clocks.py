"""Full-event sparse-engine verification of the fractional-clock pilot bases."""
from check_fractional_routing_clocks import *

if __name__ == '__main__':
    for ex in sys.argv[1:] or ['37','46']:
        dest=OUT/ex
        inv=pd.read_csv(INV/ex/'inventory.csv')
        base=inv[(inv.family=='indep') & (inv.state=='S000_base')].iloc[0]
        f=pd.read_parquet(base.frame)
        h=pd.read_csv(ROOT/f'network_variants/output/{ex}_indep__anchored/states/S000_base/rapid/run/outlet_hydrograph.csv')
        flow=h.q_inflow_cms.to_numpy();q0=float(flow[0]);start=int(np.flatnonzero(flow!=q0)[0])
        forcing_dt=float(np.median(np.diff(h.time_seconds)))
        rows=[]
        for dt in [CLOCKS[ex][0],CLOCKS[ex][-1]]:
            q,seconds,audit=sparse_benchmark(f,flow[start:],round(forcing_dt/dt),dt,q0)
            reference=pd.read_parquet(dest/f'{base.fingerprint}_dt{dt}.parquet').outflow.to_numpy()[start:]
            error=float(np.max(np.abs(q-reference))/np.ptp(flow))
            assert error<1e-8,error
            row=dict(dt_s=dt,active_intervals=len(q),seconds=seconds,max_sparse_cascade_error_fraction=error,**audit)
            rows.append(row);pd.DataFrame(rows).to_csv(dest/'full_event_sparse_benchmarks.csv',index=False)
            print(ex,row,flush=True)
