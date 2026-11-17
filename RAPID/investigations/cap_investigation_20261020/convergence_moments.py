"""Signed moments for resolution diagnostics only; production metrics stay strict.

No hydrograph clipping or baseline adjustment. Tiny negative excess can be
retained in the integral if both amplitude and volume bounds pass. Quantiles
are deliberately not calculated for a signed diagnostic waveform.
"""
import numpy as np
import pandas as pd


def metrics(ex,state,force,q,windows):
    t=force.time_utc.astype('int64').to_numpy()/1e9
    rows=[]
    for _,w in windows.iterrows():
        a,b=float(w.rise_seconds),float(w.fall_seconds)
        if not t[0]<=a<b<=t[-1]:raise ValueError('Convergence window outside waveform support')
        tt=np.r_[a,t[(t>a)&(t<b)],b];excess=np.interp(tt,t,q)-w.baseline_cms
        if not np.isfinite(excess).all():raise ValueError('Nonfinite convergence excess')
        h=np.diff(tt);x=tt-tt[0];left,right=excess[:-1],excess[1:]
        mass=h*(left+right)/2;volume=float(mass.sum())
        # Integrate the negative part with exact zero-crossings, not a clipped
        # approximation that would overestimate crossing-interval deficits.
        deficit=0.
        for l,r,dt in zip(left,right,h):
            if l<=0 and r<=0:deficit-=dt*(l+r)/2
            elif l<0:deficit+=dt*l*l/(2*(r-l))
            elif r<0:deficit+=dt*r*r/(2*(l-r))
        scale=max(float(np.max(np.abs(q))),abs(float(w.baseline_cms)),1.)
        amplitude_tolerance=1e-8*scale
        if volume<=0 or excess.min()< -amplitude_tolerance or deficit>1e-10*volume:
            raise ValueError(f'Convergence baseline review required: {ex}/{state}/{w.window}; min_excess={excess.min():.12g} cms; deficit={deficit:.12g} m3; amplitude_limit={amplitude_tolerance:.12g} cms')
        centroid=tt[0]+float(np.sum(x[:-1]*mass+h*h*(left/2+(right-left)/3))/volume)
        peak=int(excess.argmax())
        rows.append(dict(example=ex,state=state,window=w.window,centroid_seconds=centroid,peak_seconds=float(tt[peak]),
            peak_cms=float(w.baseline_cms+excess[peak]),minimum_excess_cms=float(excess.min()),
            negative_excess_samples=int((excess<0).sum()),negative_excess_volume_m3=float(deficit),
            signed_excess_volume_m3=volume,negative_volume_fraction=float(deficit/volume),
            amplitude_tolerance_cms=amplitude_tolerance,diagnostic_method='signed_exact_moments_no_clipping'))
    return pd.DataFrame(rows)
