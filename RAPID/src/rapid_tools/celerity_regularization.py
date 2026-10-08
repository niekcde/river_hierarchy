"""Explicit numerical routing celerity, separate from hydraulic estimates."""
import numpy as np
import pandas as pd


def regularize_routing_celerity(frame: pd.DataFrame, *, dt_seconds: int,
                               x_value: float = .1, margin: float = .01,
                               min_celerity_mps: float = .05,
                               max_celerity_mps: float = 5.) -> pd.DataFrame:
    """Nearest admissible celerity at an explicitly shared comparison clock.

    Input celerity includes the ordinary common caps. Preserve raw hydraulic
    celerity and effective length; never alter slope, depth, n, X or geometry.
    The caller must use one clock for its parent-child comparison and check
    forcing divisibility. Incompatible common caps raise rather than expand.
    """
    if not np.isfinite(dt_seconds) or dt_seconds <= 0 or int(dt_seconds) != dt_seconds:
        raise ValueError('dt_seconds must be a positive integer')
    if not 0 < x_value < .5 or not 0 <= margin < 1-2*x_value:
        raise ValueError('Invalid X or interior stability margin')
    if not 0 < min_celerity_mps <= max_celerity_mps < np.inf:
        raise ValueError('Invalid common celerity caps')
    out=frame.copy()
    if 'rapid_celerity_pre_regularization_mps' in out:
        raise ValueError('Already regularized: recompute from hydraulic inputs before changing clocks')
    length=out['rapid_effective_length_m'].to_numpy(dtype=float)
    current=out['rapid_celerity_mps'].to_numpy(dtype=float)
    if not (np.isfinite(length)&(length>0)&np.isfinite(current)&(current>0)).all():
        raise ValueError('Effective length and celerity must be finite and positive')
    if ((current<min_celerity_mps-1e-12)|(current>max_celerity_mps+1e-12)).any():
        raise ValueError('Apply the common celerity caps before regularization')
    lower=np.maximum(min_celerity_mps,(1+margin)*2*x_value*length/dt_seconds)
    upper=np.minimum(max_celerity_mps,(1-margin)*2*(1-x_value)*length/dt_seconds)
    if (lower>upper).any():
        raise ValueError('Selected clock is incompatible with common celerity caps')
    routing=np.clip(current,lower,upper)
    changed=~np.isclose(current,routing,rtol=1e-12,atol=1e-12)
    out['rapid_celerity_pre_regularization_mps']=current
    out['rapid_celerity_routing_mps']=routing
    out['rapid_celerity_mps']=routing
    out['rapid_k_pre_regularization']=length/current
    out['rapid_k']=length/routing
    out['rapid_celerity_regularization_enabled']=True
    out['rapid_celerity_regularization_applied']=changed
    out['rapid_celerity_regularization_factor']=routing/current
    out['rapid_celerity_regularization_reason']=np.select(
        [current<lower,current>upper],['slow_link_routing_floor','fast_link_routing_ceiling'],
        default='not_required')
    out['rapid_celerity_regularization_dt_seconds']=int(dt_seconds)
    out['rapid_celerity_regularization_margin']=float(margin)
    out['rapid_celerity_regularization_lower_mps']=lower
    out['rapid_celerity_regularization_upper_mps']=upper
    source=out.get('rapid_k_source_method',pd.Series('hydraulic_celerity',index=out.index)).astype(str)
    out['rapid_k_source_method']=np.where(changed,source+'+explicit_routing_regularization',source)
    return out
