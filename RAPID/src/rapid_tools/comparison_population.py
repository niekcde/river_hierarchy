"""One observation per routed child and matched hydraulic comparison domain."""
import json
import numpy as np
import pandas as pd

COMPARISON_KEY = ['example_id', 'fingerprint', 'matched_base_link_ids', 'child_link_ids']


def canonical_comparisons(frame):
    """Prefer topology to island aliases; preserve every selection in an alias table.

    Call before display filters so an island alias cannot reintroduce an excluded
    topological comparison. Equal scatter coordinates alone never imply identity.
    """
    data = frame.loc[~frame.technical_alias.fillna(False)].copy()
    if data[COMPARISON_KEY].isna().any().any():
        raise ValueError('Incomplete comparison identity')
    data['original_plot_category'] = data.plot_category
    data['_priority'] = data['mode'].map({'indep': 0, 'compound': 1, 'island': 2})
    if data._priority.isna().any():
        raise ValueError('Unknown comparison mode')
    data = data.sort_values(COMPARISON_KEY + ['_priority', 'state_id'])
    checks = [c for c in ['Lambda_matched', 'length_fraction', 'width_fraction',
              'len_fraction_width_weighted_log_lambda', 'interpolated_lag_shift_pct',
              'interpolated_peak_shift_hours', 'event_duration_change_pct',
              'peak_excess_change_pct'] if c in data]
    keep=[]; aliases=[]
    for key, group in data.groupby(COMPARISON_KEY, sort=False):
        first=group.iloc[0].copy()
        if len(group)>1:
            for c in checks:
                values=pd.to_numeric(group[c],errors='raise').to_numpy(float)
                if not np.allclose(values, values[0], rtol=1e-10, atol=1e-12, equal_nan=True):
                    raise ValueError(f'Conflicting duplicate comparison: {key}, {c}')
        first['selection_count']=len(group)
        first['selection_modes']=','.join(sorted(group['mode'].unique()))
        first['selection_aliases']=json.dumps(group[['mode','state_id','target_id']].to_dict('records'),default=str)
        if first['mode']=='island':first['plot_category']='island-only collapse'
        keep.append(first)
        for _,row in group.iterrows():
            aliases.append(dict(zip(COMPARISON_KEY,key),mode=row['mode'],state_id=row.state_id,
                retained_mode=first['mode'],retained_state_id=first.state_id,
                is_retained=row['mode']==first['mode'] and row.state_id==first.state_id,
                original_plot_category=row.original_plot_category,
                retained_plot_category=first.plot_category))
    canonical=pd.DataFrame(keep).drop(columns='_priority',errors='ignore').reset_index(drop=True)
    return canonical,pd.DataFrame(aliases)


def plotted_comparisons(frame, *, exclude_edge_compounds=False, require_shape_metrics=False):
    """Canonical lag comparisons; optional historical filters must be explicit."""
    data,aliases=canonical_comparisons(frame)
    if exclude_edge_compounds:
        # Preserve the original full inventory's physical order extrema.
        maxima=frame.groupby('example_id').compound_downstream_order.max()
        last=data.example_id.map(maxima)
        edge=data.compound_downstream_order.notna() & (
            data.compound_downstream_order.le(1) | data.compound_downstream_order.ge(last))
        data=data.loc[~edge].copy()
    cols=['len_fraction_width_weighted_log_lambda','interpolated_lag_shift_pct',
          'interpolated_peak_shift_hours']
    if require_shape_metrics:
        cols += ['event_duration_change_pct','peak_excess_change_pct']
    valid=np.isfinite(data[cols].apply(pd.to_numeric,errors='coerce')).all(axis=1)
    data=data.loc[valid & data.peak_interpolation_status.eq('ok')].copy()
    return data.reset_index(drop=True),aliases
