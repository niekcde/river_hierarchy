"""All-pair TV on UTC-aligned time-averaged bins; proposed windows only."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'RAPID/src'))
from rapid_tools.hydrograph_review import base_directory, event_clock
from rapid_tools.analysis import _load_gauge_quality_sheet

OUT = ROOT / 'RAPID/validation/standard_total_variation_20261005'
DATES = {'02': ('07-29', '08-12'), '03': (None, None),
         '05': ('12-31', None), '11': ('05-04', '05-30'),
         '13': (None, '10-08'), '15': (None, None), '21': (None, None),
         '22': (None, None), '37': ('06-25', None), '41': ('07-05', '07-29'),
         '45': ('06-13', None), '46': ('08-24', '09-11'),
         '48': ('05-23', '06-09'), '54': (None, '06-29'),
         '55': ('04-07', '04-25'), '56': ('06-21', '07-17')}


def bin_means(t, q, edges):
    """Exact bin integral for piecewise-linear saved samples; no extrapolation."""
    assert edges[0] >= t[0] and edges[-1] <= t[-1]
    integral = cumulative_trapezoid(q, t, initial=0)
    i = np.clip(np.searchsorted(t, edges, side='right') - 1, 0, len(t)-2)
    dx = edges - t[i]
    v = integral[i] + q[i]*dx + .5*(q[i+1]-q[i])/(t[i+1]-t[i])*dx**2
    return np.diff(v)/np.diff(edges)


def metrics(q, baseline):
    tv = np.abs(np.diff(q)).sum()
    amplitude = q.max()-baseline
    return tv, amplitude, tv/amplitude if amplitude > 0 else np.nan


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f = pd.read_parquet(ROOT/'RAPID/validation/all_pairs_lag_20261005/features.parquet')
    quality = _load_gauge_quality_sheet()
    results, windows = [], []
    for ex, group in f.groupby('example_id', sort=True):
        h = pd.read_csv(base_directory(ex)/'run/outlet_hydrograph.csv')
        clock, _ = event_clock(ex, h)
        t = (clock-clock.iloc[0]).dt.total_seconds().to_numpy()
        assert np.all(np.diff(t)>0) and np.diff(t).max() <= 3600
        qr = quality[pd.to_numeric(quality.Example, errors='coerce').eq(int(ex))].iloc[0]
        old_start, old_end = pd.to_datetime(qr.Period_start, utc=True), pd.to_datetime(qr.Period_end, utc=True)
        ds, de = DATES[ex]
        start = pd.Timestamp(f'{old_start.year}-{ds}', tz='UTC') if ds else old_start
        # New end dates include the named calendar day. Existing exact boundaries stay exact.
        year_end = start.year if ex == '02' else old_end.year
        end = pd.Timestamp(f'{year_end}-{de}', tz='UTC')+pd.Timedelta(days=1) if de else old_end
        assert start < end
        parent_q = h.q_outlet_cms.to_numpy()
        pk = clock.iloc[int(parent_q.argmax())]
        windows.append(dict(example=ex,previous_start_utc=old_start,previous_end_utc=old_end,
                            proposed_start_utc=start,proposed_end_exclusive_utc=end,
                            change_start=ds is not None,change_end=de is not None,
                            expands_start=start<old_start,parent_peak_utc=pk,
                            parent_peak_in_proposal=start<=pk<end,
                            native_max_step_minutes=np.diff(t).max()/60,
                            within_saved_output=start>=clock.iloc[0] and end<=clock.iloc[-1],
                            status='proposal only; not applied to forcing',
                            note=('July 5 confirmed by user; supersedes original June 5 proposal' if ex=='41' else
                                  'Explicit August 12 assigned to 2025 event; previous configured end was in 2026' if ex=='02' else '')))
        grids = {}
        for window, lo, hi in [('full_routed_event', clock.iloc[0], clock.iloc[-1]),
                               ('proposed_event_window', start, end)]:
            for hours in [1, 2, 4]:
                begin, finish = max(lo,clock.iloc[0]).ceil(f'{hours}h'), min(hi,clock.iloc[-1]).floor(f'{hours}h')
                edges = (pd.date_range(begin, finish, freq=f'{hours}h')-clock.iloc[0]).total_seconds().to_numpy()
                assert len(edges)>2
                pq = bin_means(t,parent_q,edges)
                grids[window,hours] = (edges, metrics(pq,max(parent_q[0],parent_q[-1])))
        for _, r in group.iterrows():
            c = pd.read_csv(Path(r.refreshed_run_dir)/'outlet_hydrograph.csv', usecols=['time_seconds','q_outlet_cms','q_inflow_cms'])
            np.testing.assert_allclose(c.time_seconds,h.time_seconds,rtol=0,atol=1e-8)
            np.testing.assert_allclose(c.q_inflow_cms,h.q_inflow_cms,rtol=1e-10,atol=1e-8)
            q = c.q_outlet_cms.to_numpy()
            child_pk = clock.iloc[int(q.argmax())]
            for (window,hours),(edges,(ptv,pa,pv)) in grids.items():
                tv,a,v = metrics(bin_means(t,q,edges),max(q[0],q[-1]))
                results.append(dict(example=ex,state_id=r.state_id,mode=r['mode'],fingerprint=r.fingerprint,
                                    plot_category=r.plot_category,parent_domain_n_links=r.parent_domain_n_links,
                                    window=window,sampling_hours=hours,n_bins=len(edges)-1,
                                    parent_tv_cms=ptv,child_tv_cms=tv,tv_change_pct=100*(tv/ptv-1),
                                    parent_amplitude_cms=pa,child_amplitude_cms=a,
                                    parent_normalized_tv=pv,child_normalized_tv=v,
                                    normalized_tv_change_pct=100*(v/pv-1),
                                    child_peak_in_proposal=start<=child_pk<end))
        print(ex,len(group),'pairs complete',flush=True)
    d = pd.DataFrame(results)
    assert len(d)==2214*6 and np.isfinite(d[['tv_change_pct','normalized_tv_change_pct']]).all().all()
    d.to_csv(OUT/'comparison_total_variation.csv',index=False)
    pd.DataFrame(windows).to_csv(OUT/'proposed_event_windows.csv',index=False)
    d.groupby(['example','window','sampling_hours']).agg(
        n=('tv_change_pct','size'),median_tv_change_pct=('tv_change_pct','median'),
        min_tv_change_pct=('tv_change_pct','min'),max_tv_change_pct=('tv_change_pct','max'),
        median_normalized_change_pct=('normalized_tv_change_pct','median'),
        max_normalized_change_pct=('normalized_tv_change_pct','max')).to_csv(OUT/'network_summary.csv')


if __name__ == '__main__':
    main()
