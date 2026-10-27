"""Event-excess moment definitions for saved interval means.

Caller must supply a reviewed event window and explicit baseline. This module
does not infer a baseline, clip negative excess, or conflate centroid with median.
"""
import numpy as np


def event_distribution(interval_edges_seconds, discharge_cms, baseline_cms):
    """Exact moments/quantiles of a piecewise-constant interval-mean excess.

    Baseline may be scalar or one interval mean per bin. Returned times use the
    supplied clock. Tail closure and baseline adequacy remain caller QC duties.
    """
    edges=np.asarray(interval_edges_seconds,float)
    q=np.asarray(discharge_cms,float)-np.asarray(baseline_cms,float)
    if edges.ndim!=1 or q.ndim!=1 or len(edges)!=len(q)+1:
        raise ValueError('Need N+1 interval edges and N excess-discharge values.')
    dt=np.diff(edges)
    if not np.isfinite(edges).all() or not np.isfinite(q).all() or (dt<=0).any():
        raise ValueError('Finite values and increasing interval edges required.')
    if (q<0).any():
        raise ValueError('Negative excess: review baseline/window; do not silently clip.')
    volume=float(q@dt)
    if volume<=0: raise ValueError('Positive event-excess volume required.')
    mid=(edges[:-1]+edges[1:])/2
    centroid=float(np.average(mid,weights=q*dt))
    variance=float(np.average((mid-centroid)**2+dt**2/12,weights=q*dt))
    cumulative=np.r_[0.,np.cumsum(q*dt)]
    def quantile(p):
        target=p*volume
        i=min(int(np.searchsorted(cumulative,target,side='right')-1),len(q)-1)
        return float(edges[i]+(target-cumulative[i])/q[i])
    t10,t25,t50,t75,t90=[quantile(p) for p in [.1,.25,.5,.75,.9]]
    return dict(volume_m3=volume,centroid_seconds=centroid,temporal_sd_seconds=variance**.5,
                t10_seconds=t10,t25_seconds=t25,t50_seconds=t50,t75_seconds=t75,t90_seconds=t90,
                volume_width80_seconds=t90-t10,volume_width50_seconds=t75-t25,
                quantile_asymmetry=((t90-t50)-(t50-t10))/(t90-t10),
                peak_excess_cms=float(q.max()))


def analytic_checks():
    # A uniform 10-second pulse has exact, nonzero within-bin temporal variance.
    a=event_distribution([0,10],[5],2)
    np.testing.assert_allclose([a['volume_m3'],a['centroid_seconds'],a['temporal_sd_seconds'],
        a['volume_width80_seconds'],a['volume_width50_seconds'],a['quantile_asymmetry']],
        [30,5,10/np.sqrt(12),8,5,0],atol=1e-12)
    b=event_distribution([0,2,10],[5,5],2)
    for key in a: np.testing.assert_allclose(a[key],b[key],atol=1e-12)
    c=event_distribution([100,110],[8],2)
    np.testing.assert_allclose(c['centroid_seconds']-a['centroid_seconds'],100)
    np.testing.assert_allclose(c['temporal_sd_seconds'],a['temporal_sd_seconds'])
    try: event_distribution([0,10],[1],2)
    except ValueError: pass
    else: raise AssertionError('Negative excess was accepted')


if __name__=='__main__':
    analytic_checks()
    print('Analytic interval moments, subdivision/translation/amplitude invariance, and invalid-baseline checks passed.')
