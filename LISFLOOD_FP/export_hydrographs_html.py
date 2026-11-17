"""Export saved example 56 runs as a standalone interactive HTML figure."""
import json
import re

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from summarize_boundaries56 import ROOT, DATA, OUT, SCENARIOS, STATES, get_run


def main():
    titles = ['A · Normal flow', 'B · Half outlet friction slope',
              'C · Double outlet friction slope', 'D · 10 km downstream extension']
    fits = {s: json.loads((OUT / s / 'calibration.json').read_text()) for s in SCENARIOS}
    fig = make_subplots(rows=2, cols=2, shared_xaxes='all', shared_yaxes='all',
                        vertical_spacing=.15, horizontal_spacing=.08,
                        subplot_titles=[f'{title} · n = {fits[s]["n"]:.6f}'
                                        for title, s in zip(titles, SCENARIOS)])
    colors = ['#176d9c', '#d37522', '#25866a', '#8753a3']
    mc_root = ROOT / 'RAPID/outputs/event_window_runs_20261013/56/tail_216h'
    mc = {}
    for state in STATES:
        saved = 'S000_base' if state == 'base' else state
        completion = json.loads((mc_root / saved / 'completion.json').read_text())
        assert completion['complete'] and completion['settled']
        reference = pd.read_parquet(mc_root / saved / 'outlet.parquet').sort_values('time_utc').reset_index(drop=True)
        # Retain each five-minute bin's extrema at their original timestamps.
        # This keeps peaks while avoiding oversized browser SVG traces.
        bins = pd.to_datetime(reference.time_utc, utc=True).dt.floor('5min')
        grouped = reference.groupby(bins).q_outlet_cms
        indices = sorted(set(grouped.idxmin()) | set(grouped.idxmax()) | {0, len(reference)-1})
        mc[state] = reference.loc[indices]
    observed = pd.read_parquet(DATA / 'observed_downstream.parquet').sort_values('time')
    # Insert missing values to prevent lines spanning observation gaps >15 min.
    ox, oy = [], []
    previous = None
    for r in observed.itertuples():
        if previous is not None and (r.time - previous).total_seconds() > 900:
            ox.append(None)
            oy.append(None)
        ox.append(pd.Timestamp(r.time).tz_convert('UTC').strftime('%Y-%m-%d %H:%M:%S'))
        oy.append(float(r.discharge_cms))
        previous = r.time
    for i, s in enumerate(SCENARIOS):
        row, col = i // 2 + 1, i % 2 + 1
        fig.add_trace(go.Scatter(x=ox, y=oy, name='Downstream gauge',
                                legendgroup='observed', showlegend=i == 0,
                                line=dict(color='#777777', width=1), connectgaps=False), row, col)
        for state, color in zip(STATES, colors):
            _, wave = get_run(s, state, fits[s]['n'])
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(wave.time_utc, utc=True).dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                y=wave.gauge_cms.tolist(),
                name='Diffusive · ' + ('Original' if state == 'base' else state),
                legendgroup='diffusive_' + state, showlegend=i == 0,
                line=dict(color=color, width=2, dash='solid'),
                connectgaps=False), row, col)
            reference = mc[state]
            assert reference.q_outlet_cms.notna().all()
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(reference.time_utc, utc=True).dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                y=reference.q_outlet_cms.tolist(),
                name='MC · ' + ('Original' if state == 'base' else state),
                legendgroup='mc_' + state, showlegend=i == 0,
                line=dict(color=color, width=1.8, dash='dash'), connectgaps=False), row, col)
    fig.update_traces(mode='lines', hovertemplate='%{x|%d %b %Y %H:%M} UTC<br>%{y:.3f} m³/s<extra>%{fullData.name}</extra>')
    fig.update_xaxes(type='date', title_text='Time (UTC)', range=['2025-06-25', '2025-07-01'])
    fig.update_yaxes(title_text='Discharge (m³/s)', col=1)
    values = [float(y) for trace in fig.data for y in trace.y if y is not None and pd.notna(y)]
    padding = (max(values) - min(values)) * .05
    fig.update_yaxes(type='linear', range=[min(values)-padding, max(values)+padding], autorange=False)
    fig.update_layout(template='plotly_white', height=1050,
                      title=dict(text='Example 56 · diffusive-wave and Musking–Cunge comparison', y=.98),
                      hovermode='x unified', margin=dict(t=110, b=220),
                      legend=dict(orientation='h', y=-.13, yanchor='top', x=0, groupclick='togglegroup'))
    note = ('<p style="font:14px sans-serif;margin:16px 4%">Drag to zoom; double-click to reset. '
            'Click a legend entry to toggle that curve in all panels. Full simulation data are embedded. '
            'Discharge is measured at the original outlet, including for the extension scenario.<br>'
            'Fixed measured widths and proxy-bed profiles; n calibrated on the original only. '
            'Historical cached geometry; one-event calibration; no lateral inflow. '
            'Custom diffusive-wave prototype, not the official LISFLOOD-FP executable.<br>'
            '<b>Solid: diffusive-wave; dashed: saved fixed-depth Musking–Cunge (n = 0.020446875).</b> '
            'Color identifies the original or collapse state. The same MC references are repeated in each panel. '
            'MC uses a different forcing window/preprocessing; this is a contextual comparison, '
            'not an isolated comparison of the equations. MC is not recalibrated for these outlet scenarios. '
            'For display, MC retains the minimum and maximum of each five-minute bin at their original timestamps; '
            'source results are unchanged.</p>')
    html = fig.to_html(include_plotlyjs=True, full_html=True,
                       config=dict(responsive=True, displaylogo=False,
                                   toImageButtonOptions=dict(format='png', scale=2)))
    html = html.replace('<head>', '<head><title>Example 56 — calibrated hydrographs</title>')
    html = html.replace('</body>', note + '</body>')
    path = OUT / 'calibrated_hydrographs.html'
    path.write_text(html)
    (OUT / 'hydrographs56_plotly.html').write_text(html)
    (OUT / 'hydrographs56_with_mc_plotly.html').write_text(html)
    assert len(fig.data) == 36
    for trace in fig.data:
        assert len(trace.x) == len(trace.y) and len(trace.x) > 1
    assert len({trace.legendgroup for trace in fig.data}) == 9
    assert not re.search(r'<script\b[^>]*\bsrc\s*=', html, re.IGNORECASE)
    print(f'{path}\n{path.stat().st_size:,} bytes; 36 traces; Plotly embedded for offline use')


if __name__ == '__main__':
    main()
