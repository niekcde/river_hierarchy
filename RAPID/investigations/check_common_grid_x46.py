"""Isolate X from Manning n for network 46; diagnostic controls, not calibration."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'RAPID/src')]
from rapid_tools import cunge
from rapid_tools.common_grid import route_interval_means

def main():
    dest = Path(json.loads((ROOT/'RAPID/outputs/common_grid_validation_20261008/latest.json').read_text())['directory'])
    freeze = dest/'baseline'
    comp = pd.read_csv(freeze/'RAPID/outputs/cunge_calibration_reviewed_20261006/comparison.csv', dtype={'example':str})
    rows = comp[comp.example=='46']
    folder = freeze/Path(rows.iloc[0].output_dir).relative_to(ROOT)
    grid = pd.read_parquet(folder/'frozen_grid.parquet')
    data = pd.read_parquet(dest/'46/60s.parquet')
    initial = json.loads((dest/'46/60s.json').read_text())[0]['initial_cms']
    valid = data.downstream_cms.notna().to_numpy()
    ids = np.flatnonzero(valid)
    records = []
    fig = go.Figure()
    for r in rows.itertuples():
        for method in ['cunge','fixed_x']:
            params = cunge.parameters(grid,r.n_manning,method=method)
            q,audit = route_interval_means(params,data.forcing_cms.to_numpy(),6,10.,initial_flow=initial)
            peak = ids[np.argmax(q[ids])]
            label = f'n={r.n_manning:.8f}, X={method}'
            records.append(dict(n_source=r.method, x_method=method,n_manning=r.n_manning,peak_cms=q[peak],peak_utc=str(data.time_utc.iloc[peak]),minimum_native_cms=audit['minimum_native_flow_cms']))
            data[r.method+'_n_'+method+'_x_cms'] = q
            fig.add_trace(go.Scatter(x=data.time_utc[valid],y=q[valid],name=label))
            if r.method == 'cunge' and method=='cunge':
                p = params.copy()
                coefficient = (2*p.rapid_k*(1-p.rapid_x)-10)/(2*p.rapid_k*(1-p.rapid_x)+10)
                p['local_decay_days'] = -10/np.log(abs(coefficient))/86400
                p.sort_values('local_decay_days',ascending=False).to_csv(dest/'46_x_reach_parameters.csv',index=False)
    for key in ['upstream_cms','downstream_cms']:
        fig.add_trace(go.Scatter(x=data.time_utc[valid],y=data[key][valid],name=key,line=dict(dash='dot')))
    fig.update_layout(title='46: X-only controls at each existing Manning n — equilibrium start, 60 s means',xaxis_title='UTC',yaxis_title='Discharge (m³/s)',template='plotly_white',height=750,legend=dict(orientation='h',y=1.15))
    fig.write_html(dest/'46_x_attribution.html',include_plotlyjs=True)
    data.to_parquet(dest/'46/x_attribution.parquet',index=False)
    pd.DataFrame(records).to_csv(dest/'46_x_attribution.csv',index=False)
    print(pd.DataFrame(records).to_string(index=False))
if __name__=='__main__':main()
