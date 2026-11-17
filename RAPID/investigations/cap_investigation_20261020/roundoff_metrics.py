"""Strict distribution metrics with bounded, recorded floating-point cleanup.

Raw discharge and the frozen baseline never change. Only negative excess samples
within 1e-12 of the discharge scale may be zeroed in the metric working array;
the added volume must also be <=1e-12 of signed event excess volume.
"""
import numpy as np
from manning_network_pilot import state_metrics as strict_metrics
from rapid_tools.hydrograph_response import distribution


def state_metrics(t,q,window,state,aliases):
    try:
        return strict_metrics(t,q,window,state,aliases)
    except ValueError as error:
        if str(error)!='Negative excess under frozen baseline; needs explicit review':raise
        original_error=error
    a,b,base=window.rise_seconds,window.fall_seconds,window.baseline_cms
    tt=np.r_[a,t[(t>a)&(t<b)],b]
    excess=np.interp(tt,t,q)-base
    scale=max(1.,abs(float(base)),float(np.max(np.abs(q))))
    tolerance=1e-12*scale
    corrected=np.maximum(excess,0.)
    volume=float(np.trapz(excess,tt));added=float(np.trapz(corrected-excess,tt))
    if not np.isfinite(excess).all() or excess.min() < -tolerance or volume<=0 or added>1e-12*volume:
        raise original_error
    d=distribution(tt,corrected);peak=int(corrected.argmax())
    return dict(example='22',state=state,aliases=aliases,window=window.window,
        rise_seconds=a,fall_seconds=b,baseline_cms=base,rise_utc=window.rise_utc,fall_utc=window.fall_utc,
        threshold_fraction=window.threshold_fraction,peak_seconds=tt[peak],peak_cms=corrected[peak]+base,
        peak_excess_cms=corrected[peak],duration_hours=(b-a)/3600,**d,
        metrics_status='available',metrics_reason='Roundoff-only negative excess zeroed in metric working array; raw discharge and baseline unchanged.',
        roundoff_negative_samples=int((excess<0).sum()),roundoff_minimum_excess_cms=float(excess.min()),
        roundoff_tolerance_cms=tolerance,roundoff_added_volume_m3=added,roundoff_added_volume_fraction=added/volume)
