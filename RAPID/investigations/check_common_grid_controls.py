"""Small initial-state/gap stress tests on four common-grid calibration pilots."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'RAPID/src')]
from rapid_tools import cunge
from rapid_tools.common_grid import route_interval_means


def main():
    root = ROOT/'RAPID/outputs/common_grid_validation_20261008'
    dest = Path(json.loads((root/'latest.json').read_text())['directory'])
    freeze = dest/'baseline'
    comparison = pd.read_csv(freeze/'RAPID/outputs/cunge_calibration_reviewed_20261006/comparison.csv', dtype={'example':str})
    targets = {r['example']:r for r in json.loads((freeze/'RAPID/reviews/calibration_targets_20261006/targets.json').read_text())['records']}
    metrics = pd.read_csv(dest/'base_metrics.csv', dtype={'example':str})
    records = []
    for ex in ['03','46','54','55']:
        frame = pd.read_parquet(dest/ex/'60s.parquet')
        origin = frame.interval_start_utc.iloc[0]
        edges = np.r_[(frame.interval_start_utc-origin).dt.total_seconds(),
                      (frame.interval_end_utc.iloc[-1]-origin).total_seconds()]
        folder = freeze/Path(comparison[comparison.example==ex].iloc[0].output_dir).relative_to(ROOT)
        grid = pd.read_parquet(folder/'frozen_grid.parquet')
        raw = pd.read_parquet(folder/'observed_upstream.parquet')
        ts = (pd.to_datetime(raw.time,utc=True)-origin).dt.total_seconds().to_numpy()
        q = raw.discharge_cms.to_numpy()
        variants = [('initial_minus10pct',frame.forcing_cms.to_numpy(),q[0]*.9),
                    ('initial_plus10pct',frame.forcing_cms.to_numpy(),q[0]*1.1)]
        gap_ids = np.flatnonzero(np.diff(ts)>2*targets[ex]['upstream_sampling_minutes']*60)
        for direction in ['left','right'] if len(gap_ids) else []:
            flow = frame.forcing_cms.to_numpy().copy()
            for i in gap_ids:
                a, b = np.maximum(edges[:-1],ts[i]), np.minimum(edges[1:],ts[i+1])
                valid = b>a
                x, y = a[valid]-ts[i], b[valid]-ts[i]
                slope = (q[i+1]-q[i])/(ts[i+1]-ts[i])
                linear_integral = q[i]*(y-x)+.5*slope*(y*y-x*x)
                held = q[i if direction=='left' else i+1]*(y-x)
                flow[valid] += (held-linear_integral)/60
            variants.append((f'gap_hold_{direction}',flow,q[0]))
        for r in comparison[comparison.example==ex].itertuples():
            f = cunge.parameters(grid,r.n_manning,method=r.method)
            m = metrics[(metrics.example==ex)&(metrics.method==r.method)&(metrics.variant=='60s')].iloc[0]
            dt = float(m.dt_s); ns = int(round(60/dt))
            base = frame[r.method+'_cms'].to_numpy()
            observed = frame.downstream_cms.notna().to_numpy()
            crest = observed & (frame.time_utc >= pd.Timestamp(targets[ex]['crest_start_utc'])-pd.Timedelta(hours=6)) & (frame.time_utc <= pd.Timestamp(targets[ex]['crest_end_utc'])+pd.Timedelta(hours=6))
            for name,flow,initial in variants:
                routed,audit = route_interval_means(f,flow,ns,dt,initial_flow=initial)
                ids = np.flatnonzero(observed)
                peak = ids[np.argmax(routed[ids])]; oldpeak = ids[np.argmax(base[ids])]
                records.append(dict(example=ex,method=r.method,control=name,
                    global_peak_change_s=(frame.time_utc.iloc[peak]-frame.time_utc.iloc[oldpeak]).total_seconds(),
                    max_event_difference_pct_inlet_amplitude=float(np.max(abs(routed[observed]-base[observed]))/np.ptp(q)*100),
                    max_crest_difference_pct_inlet_amplitude=float(np.max(abs(routed[crest]-base[crest]))/np.ptp(q)*100),
                    input_volume_change_m3=float(np.sum(flow-frame.forcing_cms.to_numpy())*60),
                    minimum_native_flow_cms=audit['minimum_native_flow_cms'],
                    interpretation='stress test, not adopted forcing or statistical uncertainty bound'))
        pd.DataFrame(records).to_csv(dest/'initialization_and_gap_controls.csv',index=False)
        print(ex,'controls completed',flush=True)
    (dest/'control_protocol.json').write_text(json.dumps(dict(
        examples=['03','46','54','55'], initial_state='plus/minus 10%, unchanged forcing',
        gaps='constant previous/next observation across >2-cadence gaps; not a statistical bound',
        production_modified=False),indent=2))


if __name__=='__main__':
    main()
