"""Actual antecedent-forcing control for 46 and local cadence audit for all gauges."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'RAPID/src')]
from rapid_tools import cunge
from rapid_tools.common_grid import route_interval_means
from rapid_tools.time_support import point_series_interval_means


def main():
    root = ROOT/'RAPID/outputs/common_grid_validation_20261008'
    dest = Path(json.loads((root/'latest.json').read_text())['directory'])
    freeze = dest/'baseline'
    comparison = pd.read_csv(freeze/'RAPID/outputs/cunge_calibration_reviewed_20261006/comparison.csv',dtype={'example':str})
    targets = {r['example']:r for r in json.loads((freeze/'RAPID/reviews/calibration_targets_20261006/targets.json').read_text())['records']}
    cadence = []
    for r in comparison[comparison.method=='cunge'].itertuples():
        folder = freeze/Path(r.output_dir).relative_to(ROOT)
        for side in ['upstream','downstream']:
            raw = pd.read_parquet(folder/f'observed_{side}.parquet')
            ts = pd.to_datetime(raw.time,utc=True)
            target = pd.Timestamp(targets[r.example]['target_utc' if side=='downstream' else 'observed_upstream_first_max_utc'])
            local = ts[ts.between(target-pd.Timedelta(hours=3),target+pd.Timedelta(hours=3))]
            intervals = local.diff().dropna().dt.total_seconds()/60
            cadence.append(dict(example=r.example,side=side,selected_time_utc=target.isoformat(),
                nominal_minutes=targets[r.example][side+'_sampling_minutes'],
                local_median_minutes=float(intervals.median()),local_min_minutes=float(intervals.min()),
                local_max_minutes=float(intervals.max()),local_samples=len(local),
                local_spacing_counts=json.dumps({str(k):int(v) for k,v in intervals.value_counts().items()})))
    pd.DataFrame(cadence).to_csv(dest/'local_crest_sampling.csv',index=False)

    source = ROOT/'SWORD_gauge_match/outputs/subdaily_values/CA/subdaily_timeseries.parquet'
    raw = pd.read_parquet(source,filters=[('station_key','==','CA:07HA001')],columns=['time','discharge'])
    raw.time = pd.to_datetime(raw.time,utc=True)
    r = comparison[(comparison.example=='46')&(comparison.method=='cunge')].iloc[0]
    folder = freeze/Path(r.output_dir).relative_to(ROOT)
    grid = pd.read_parquet(folder/'frozen_grid.parquet')
    f = cunge.parameters(grid,r.n_manning,method='cunge')
    event = pd.read_parquet(dest/'46/60s.parquet')
    event_start = event.interval_start_utc.iloc[0]
    end = event.interval_end_utc.iloc[-1]
    base_dt, n30 = cunge.common_clock(grid,30.)
    controls = [(7,1.),(30,1.),(90,1.),(180,1.),(180,.9),(180,1.1)]
    records=[]
    for days, factor in controls:
        origin = event_start-pd.Timedelta(days=days)
        # Keep bracketing points at the left edge. After the event retain the same
        # constant diagnostic tail as the baseline, rather than new observed input.
        before = raw[raw.time <= origin].tail(1)
        through = raw[(raw.time > origin)&(raw.time <= pd.to_datetime(pd.read_parquet(folder/'observed_upstream.parquet').time,utc=True).max())]
        points = pd.concat([before,through]).drop_duplicates('time').sort_values('time')
        assert points.time.min() <= origin
        saved = dest/'46'/f'antecedent_{days}d_points.parquet'
        if not saved.exists():
            points.to_parquet(saved,index=False)
        t = (points.time-origin).dt.total_seconds().to_numpy()
        q = points.discharge.to_numpy()
        duration = (end-origin).total_seconds()
        edges = np.arange(int(round(duration/60))+1)*60.
        flow = point_series_interval_means(t,q,edges,allow_endpoint_hold=True)
        initial = float(np.interp(0,t,q))*factor
        means,audit = route_interval_means(f,flow,n30*2,base_dt,initial_flow=initial)
        offset = days*1440
        aligned = means[offset:]
        assert len(aligned)==len(event)
        mask=event.downstream_cms.notna().to_numpy()
        ids=np.flatnonzero(mask); peak=ids[np.argmax(aligned[mask])]
        base=event.cunge_cms.to_numpy()
        crest=mask&(event.time_utc>=pd.Timestamp(targets['46']['target_utc'])-pd.Timedelta(hours=6))&(event.time_utc<=pd.Timestamp(targets['46']['target_utc'])+pd.Timedelta(hours=6))
        out=event[['time_utc','interval_start_utc','interval_end_utc']].copy()
        out['cunge_cms']=aligned
        out.to_parquet(dest/'46'/f'antecedent_{days}d_initial{factor:.1f}.parquet',index=False)
        records.append(dict(example='46',method='cunge',antecedent_days=days,initial_multiplier=factor,
            peak_utc=event.time_utc.iloc[peak].isoformat(),
            peak_error_min=(event.time_utc.iloc[peak]-pd.Timestamp(targets['46']['target_utc'])).total_seconds()/60,
            max_crest_difference_from_no_antecedent_cms=float(np.max(abs(aligned[crest]-base[crest]))),
            longest_antecedent_sampling_interval_minutes=float(np.diff(t[t<0+days*86400]).max()/60),
            raw_points_sha256=hashlib.sha256(saved.read_bytes()).hexdigest(),**audit))
        pd.DataFrame(records).to_csv(dest/'46_antecedent_controls.csv',index=False)
        print('46',days,factor,records[-1]['peak_error_min'],flush=True)
    evidence = dict(source=str(source),station='CA:07HA001',
        note='Retained actual upstream point records; linear interpolation; long intervals audited, not silently verified',
        raw_event_values_checked=False,calibration_changed=False)
    (dest/'46_antecedent_provenance.json').write_text(json.dumps(evidence,indent=2))


if __name__=='__main__':
    main()
