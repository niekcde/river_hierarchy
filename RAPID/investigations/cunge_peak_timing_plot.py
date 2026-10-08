"""Plot archived case 11 peaks without changing calibration or routing inputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'RAPID/outputs/cunge_calibration_alpha2'


def main():
    results = pd.read_csv(OUT / 'comparison.csv', dtype={'example': str})
    selected = results.loc[results.example.eq('11')].set_index('method')
    folder = Path(selected.loc['cunge', 'output_dir'])
    observed = {name: pd.read_parquet(folder / f'observed_{name}.parquet')
                for name in ['upstream', 'downstream']}
    waves = {name: pd.read_parquet(folder / f'{name}_best_wave.parquet')
             for name in ['fixed_x', 'cunge']}
    start = pd.to_datetime(observed['downstream'].time, utc=True).min()
    end = pd.to_datetime(observed['downstream'].time, utc=True).max()
    series = [
        ('Observed upstream', observed['upstream'], 'time', 'discharge_cms', '#16836b', 'solid'),
        ('Observed downstream', observed['downstream'], 'time', 'discharge_cms', '#222222', 'solid'),
        ('Fixed X', waves['fixed_x'], 'time_utc', 'q_outlet_cms', '#5473a4', 'solid'),
        ('Cunge', waves['cunge'], 'time_utc', 'q_outlet_cms', '#cf7443', 'solid'),
        ('Prepared inlet', waves['cunge'], 'time_utc', 'q_inflow_cms', '#9655b0', 'dot'),
    ]
    fig = go.Figure()
    rows = []
    for name, frame, timecol, qcol, color, dash in series:
        times = pd.to_datetime(frame[timecol], utc=True)
        q = frame[qcol].to_numpy(dtype=float)
        mask = (times >= start) & (times <= end)
        window_q = q[mask]
        lo, hi = float(window_q.min()), float(window_q.max())
        yn = (q - lo) / (hi - lo)
        # Peak search on the entire archived series, as in calibration.
        peaks = np.flatnonzero(q == q.max())
        first, last = times.iloc[peaks[0]], times.iloc[peaks[-1]]
        assert start <= first <= end
        rows.append(dict(series=name, first_max_utc=first.isoformat(),
                         last_equal_max_utc=last.isoformat(), maximum_cms=float(q.max()),
                         equal_max_samples=len(peaks), equal_max_span_h=(last-first).total_seconds()/3600))
        x = times[mask].dt.strftime('%Y-%m-%dT%H:%M:%SZ').tolist()
        fig.add_trace(go.Scatter(x=x, y=yn[mask], name=name, legendgroup=name,
                                mode='lines', line=dict(color=color, width=2, dash=dash),
                                customdata=window_q,
                                hovertemplate=name+'<br>%{x|%Y-%m-%d %H:%M:%S} UTC<br>Normalized Q: %{y:.5f}<br>Q: %{customdata:.2f} m³/s<extra></extra>'))
        if len(peaks) > 1:
            fig.add_trace(go.Scatter(x=times.iloc[peaks].dt.strftime('%Y-%m-%dT%H:%M:%SZ').tolist(),
                                    y=yn[peaks], mode='markers', name=name+' — equal maxima',
                                    legendgroup=name, showlegend=False,
                                    marker=dict(color=color, size=8, symbol='circle-open'),
                                    hovertemplate=name+' — equal absolute maximum<br>%{x|%Y-%m-%d %H:%M:%S} UTC<extra></extra>'))
        fig.add_trace(go.Scatter(x=[first.strftime('%Y-%m-%dT%H:%M:%SZ')], y=[yn[peaks[0]]],
                                mode='markers', name=name+' — selected peak', legendgroup=name,
                                showlegend=False, marker=dict(color=color, size=14, symbol='diamond', line=dict(color='white', width=1)),
                                hovertemplate=name+' — FIRST ABSOLUTE MAXIMUM<br>%{x|%Y-%m-%d %H:%M:%S} UTC<br>Q: '+f'{q.max():.2f}'+' m³/s<extra></extra>'))
        fig.add_shape(type='line', x0=first.isoformat(), x1=first.isoformat(), y0=0, y1=1,
                      xref='x', yref='y', line=dict(color=color, width=1, dash='dot'))
        offsets = {'Observed upstream':(-110,-105), 'Observed downstream':(95,-100),
                   'Fixed X':(110,-60), 'Cunge':(45,-30), 'Prepared inlet':(-110,-65)}
        ax, ay = offsets[name]
        fig.add_annotation(x=first.isoformat(), y=1, text=f'{name}<br>{first:%d May %H:%M:%S} UTC',
                           showarrow=True, arrowcolor=color, font=dict(color=color, size=11),
                           ax=ax, ay=ay, bgcolor='rgba(255,255,255,0.85)')
    peaks = pd.DataFrame(rows).set_index('series')
    t = pd.to_datetime(peaks.first_max_utc, utc=True)
    peaks['lag_from_observed_upstream_h'] = (t-t.loc['Observed upstream']).dt.total_seconds()/3600
    peaks['lag_from_prepared_inlet_h'] = (t-t.loc['Prepared inlet']).dt.total_seconds()/3600
    for method, label in [('fixed_x','Fixed X'), ('cunge','Cunge')]:
        np.testing.assert_allclose(peaks.loc[label,'lag_from_prepared_inlet_h'], selected.loc[method,'lag_h'], atol=1e-9)
    manifest = json.loads((folder/'case.json').read_text())
    np.testing.assert_allclose(peaks.loc['Observed downstream','lag_from_observed_upstream_h'], manifest['target']['observed_lag_h'])
    fig.update_layout(template='plotly_white', height=760,
                      title=dict(text='Case 11: normalized hydrographs and selected absolute peaks<br><sup>Diamonds = first absolute maximum; open circles = all equal maxima. No time shifting.</sup>', x=.03),
                      margin=dict(t=190,b=100,l=75,r=40),
                      xaxis=dict(title='Time (UTC)', type='date', range=[start.isoformat(),end.isoformat()], rangeslider=dict(visible=True,thickness=.10)),
                      yaxis=dict(title='Independently normalized discharge', range=[-.05,1.08]),
                      legend=dict(orientation='h',y=-.29,x=0), hovermode='closest',
                      updatemenus=[dict(type='buttons',direction='right',x=0,y=1.40,
                          buttons=[dict(label='Full event',method='relayout',args=[{'xaxis.range':[start.isoformat(),end.isoformat()]}]),
                                   dict(label='Peak detail (16–22 May)',method='relayout',args=[{'xaxis.range':['2024-05-16T00:00:00Z','2024-05-22T00:00:00Z']}])])])
    target = OUT/'case_11_normalized_peak_timing.html'
    table=peaks.reset_index().to_html(index=False,float_format=lambda x:f'{x:.4f}',border=0)
    html=fig.to_html(full_html=False,include_plotlyjs=True,config={'responsive':True,'displaylogo':False,'scrollZoom':True})
    target.write_text('<!doctype html><html><head><meta charset="utf-8"><title>Case 11 peak timing</title>'
                     '<style>body{font:15px system-ui;margin:24px;color:#222}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}p{max-width:1100px}</style></head><body>'
                     +html+'<p><b>Timing definition:</b> the first sampled absolute maximum within the reviewed observation window; model maxima are searched over the archived simulation. This is not the start of the rising limb. A rising-limb onset requires a separately defined criterion and was not the calibration target.</p>'
                     '<p>Each curve uses its own minimum and maximum within the displayed event window for normalization. Hover shows original discharge. The prepared inlet is included because model lag is measured from its peak, which is five minutes earlier than the observed upstream peak. Model Manning n: fixed X = '
                     +f'{selected.loc["fixed_x","n_manning"]:.8f}; Cunge = {selected.loc["cunge","n_manning"]:.8f}.'
                     '</p><p>Equal maxima need not form a continuous flat plateau: open markers show the actual tied samples. Lines connect available archived samples; they do not fill observational gaps.</p>'
                     +table+'</body></html>')
    peaks.to_csv(OUT/'case_11_peak_timing.csv')
    fig.write_json(OUT/'case_11_normalized_peak_timing.plotly.json')
    print(peaks.to_string())
    print(target)


if __name__ == '__main__':
    main()
