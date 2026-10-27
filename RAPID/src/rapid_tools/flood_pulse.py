"""Local discharge intensity for synchronized flood-pulse animations."""
import numpy as np


def interpolate_discharge(times, qout, time):
    times = np.asarray(times)
    if time < times[0] or time > times[-1]:
        raise ValueError('Animation timestamp is outside routed discharge support')
    right = min(int(np.searchsorted(times, time, side='right')), len(times)-1)
    left = max(0, right-1)
    if left == right:
        return qout[left]
    weight = (time-times[left])/(times[right]-times[left])
    return (1-weight)*qout[left] + weight*qout[right]


def prepare_pulse_field(reaches, qout, times, start, end, baseline_cms, *, background='reviewed'):
    """Normalize local excess to each subreach's maximum in the reviewed window.

    Scale the reviewed outlet baseline by reference flow fraction. Lower it to
    the local window minimum if needed; do not clip a negative excess silently.
    With local_endpoints, show only positive departure from a straight line
    between each local window endpoint. This display returns to zero at both
    endpoints; it does not alter routing or reviewed response metrics.
    This visual normalization is not a comparison of absolute branch discharge.
    """
    times = np.asarray(times)
    if not times[0] <= start < end <= times[-1]:
        raise ValueError('Pulse window is outside routed discharge support')
    if qout.shape != (len(times), len(reaches)) or not np.all(np.diff(times)>0):
        raise ValueError('Pulse discharge columns/time do not match reaches and clock')
    first = interpolate_discharge(times, qout, start)
    last = interpolate_discharge(times, qout, end)
    inside = qout[np.searchsorted(times,start):np.searchsorted(times,end,side='right')]
    low = np.minimum(first,last); high = np.maximum(first,last)
    if len(inside):
        low = np.minimum(low, inside.min(axis=0)); high = np.maximum(high, inside.max(axis=0))
    proposed = float(baseline_cms)*reaches.reference_flow_fraction.to_numpy(float)
    baseline = np.minimum(proposed,low)
    scale = high-baseline
    if background not in {'reviewed', 'local_endpoints'}:
        raise ValueError('Unknown pulse background')
    if background == 'local_endpoints':
        # Visual pulse isolation: positive departure from a line through each
        # reach's own window endpoints. This is not physical baseflow separation.
        scale = np.zeros(len(reaches), dtype=float)
        lo, hi = np.searchsorted(times, start), np.searchsorted(times, end, side='right')
        for offset in range(lo, hi, 512):
            stop = min(offset+512, hi)
            weight = ((times[offset:stop]-start)/(end-start))[:,None]
            local_background = first[None,:] + weight*(last-first)[None,:]
            scale = np.maximum(scale, (qout[offset:stop]-local_background).max(axis=0))
    paths=[];columns=[]
    for column,geometry in enumerate(reaches.geometry):
        for part in geometry.geoms if geometry.geom_type=='MultiLineString' else [geometry]:
            paths.append(np.asarray(part.coords));columns.append(column)
    return dict(paths=paths,columns=np.asarray(columns),qout=qout,times=times,
                baseline=baseline,scale=scale,background=background,start=start,end=end,
                first=first,last=last,baseline_lowered_count=int(np.sum(proposed>low+1e-8)))


def pulse_intensity(field, time):
    q = interpolate_discharge(field['times'],field['qout'],time)
    baseline = field['baseline']
    if field['background'] == 'local_endpoints':
        weight = (time-field['start'])/(field['end']-field['start'])
        baseline = field['first'] + weight*(field['last']-field['first'])
    intensity=np.divide(q-baseline,field['scale'],out=np.zeros_like(q,dtype=float),where=field['scale']>0)
    return np.clip(intensity,0,1)[field['columns']]
