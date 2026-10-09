"""Fifteen-panel common-grid review with unchanged Excel target markers."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'RAPID/outputs/common_grid_validation_20261008'
COLORS = dict(upstream_cms='#2580a5', downstream_cms='#18212b', fixed_x_cms='#dc8a16', cunge_cms='#ce3d45')


def main():
    dest = Path(json.loads((OUT/'latest.json').read_text())['directory'])
    freeze = dest/'baseline'
    targets = {r['example']: r for r in json.loads((freeze/'RAPID/reviews/calibration_targets_20261006/targets.json').read_text())['records']}
    rows = pd.read_csv(freeze/'RAPID/outputs/cunge_calibration_reviewed_20261006/comparison.csv', dtype={'example': str})
    examples = sorted(set(rows.example)-{'11'})
    if any(not (dest/ex/'60s.parquet').exists() for ex in examples):
        raise ValueError('All 15 one-minute pilots are required for the review figure.')
    fig, axes = plt.subplots(5, 3, figsize=(18, 20))
    zoom, zaxes = plt.subplots(5, 3, figsize=(18, 20))
    review = []
    for ex, ax, za in zip(examples, axes.flat, zaxes.flat):
        f = pd.read_parquet(dest/ex/'60s.parquet')
        target = targets[ex]
        origin = pd.Timestamp(target['plot_origin_utc'])
        nominal = pd.Timestamp(target['target_utc'])
        lo, hi = pd.Timestamp(target['crest_start_utc']), pd.Timestamp(target['crest_end_utc'])
        folder = freeze/Path(rows[rows.example == ex].iloc[0].output_dir).relative_to(ROOT)
        raw = pd.read_parquet(folder/'observed_downstream.parquet')
        rt = pd.to_datetime(raw.time, utc=True)
        tt = (pd.to_datetime(f.time_utc, utc=True)-origin).dt.total_seconds().to_numpy()/86400
        valid = f.downstream_cms.notna().to_numpy()
        rawx = (rt-origin).dt.total_seconds().to_numpy()/86400
        xr = [(rt.min()-origin).total_seconds()/86400, (rt.max()-origin).total_seconds()/86400]
        tx = (nominal-origin).total_seconds()/86400
        lx, hx = (lo-origin).total_seconds()/86400, (hi-origin).total_seconds()/86400
        pad = max(.15, (hx-lx)*.4)
        zx = (max(xr[0], lx-pad), min(xr[1], hx+pad))
        zoom_values = []
        for key, color in COLORS.items():
            q = f[key].to_numpy(float)
            usable = valid & np.isfinite(q)
            baseline, amplitude = np.min(q[usable]), max(np.ptp(q[usable]), 1e-12)
            y = (q-baseline)/amplitude
            if key != 'upstream_cms':
                zoom_values.extend(y[usable & (tt>=zx[0]) & (tt<=zx[1])].tolist())
            if key == 'upstream_cms':
                diagnostic = np.where(f.upstream_gap.to_numpy(), y, np.nan)
                for panel in (ax, za):
                    panel.plot(tt[valid], diagnostic[valid], color=color, lw=1, ls=':', alpha=.45)
                y[f.upstream_gap.to_numpy()] = np.nan
            if key == 'downstream_cms':
                diagnostic = np.where(f.downstream_gap.to_numpy(), y, np.nan)
                for panel in (ax, za):
                    panel.plot(tt[valid], diagnostic[valid], color=color, lw=1, ls=':', alpha=.45)
                y[f.downstream_gap.to_numpy()] = np.nan
                ry = (raw.discharge_cms.to_numpy()-baseline)/amplitude
                # Raw points reveal observation cadence near the target. Full-event
                # marker thinning is display-only; all points remain in the zoom.
                ax.scatter(rawx[::max(1, len(raw)//300)], ry[::max(1, len(raw)//300)],
                           s=3, color='#637080', alpha=.3, rasterized=True)
                rawzoom = (rawx>=zx[0]) & (rawx<=zx[1])
                za.scatter(rawx[rawzoom], ry[rawzoom], s=9, color='#637080', alpha=.65, rasterized=True)
            for panel in (ax, za):
                panel.plot(tt[valid], y[valid], color=color, lw=1.15,
                           ls='--' if key == 'upstream_cms' else '-')
        zy = (min(zoom_values), max(zoom_values))
        zpad = max((zy[1]-zy[0])*.15, .0001)
        for panel in (ax, za):
            limits = (-.05, 1.12) if panel is ax else (zy[0]-zpad, zy[1]+zpad)
            if hi > lo:
                panel.axvspan(lx, hx, color='#8353ad', alpha=.12, zorder=0)
            panel.axvline(tx, color='#8353ad', lw=1.5, ls=':')
            panel.scatter([tx], [limits[1]-.045*(limits[1]-limits[0])], marker='v', s=40, color='#8353ad', zorder=5)
            panel.set_ylim(*limits)
            panel.grid(alpha=.15)
            panel.set_xlabel('Days from panel origin (UTC)', fontsize=8)
            panel.tick_params(labelsize=8)
            if ex in examples[::3]:
                panel.set_ylabel('Normalized discharge', fontsize=9)
            issue = ' · clock unresolved' if ex in ['21', '22'] else ''
            panel.set_title(f'{ex}  |  target {nominal:%d %b %H:%M:%S}{issue}', loc='left', fontsize=10, pad=8)
        ax.set_xlim(xr)
        za.set_xlim(*zx)
        za.ticklabel_format(axis='y', style='plain', useOffset=False)
        # Retain the chosen target unchanged. Global maximum shown here is only
        # an audit diagnostic and never substitutes for a plateau/crest selection.
        ids = np.flatnonzero(valid & ~f.downstream_gap.to_numpy())
        obs_peak = ids[np.argmax(f.downstream_cms.to_numpy()[ids])]
        review.append(dict(example=ex, target_utc=nominal.isoformat(),
            selected_span_start_utc=lo.isoformat(), selected_span_end_utc=hi.isoformat(),
            selection_rule=target['selection_rule'], target_changed=False,
            resampled_observed_first_global_max_utc=f.time_utc.iloc[obs_peak].isoformat(),
            downstream_sampling_minutes=target['downstream_sampling_minutes'],
            status='Existing selection preserved; model operator requires review before fit'))
    handles = [Line2D([0], [0], color=COLORS[k], lw=2, ls='--' if k=='upstream_cms' else '-', label=label)
               for k, label in [('upstream_cms','Upstream gauge'), ('downstream_cms','Downstream gauge'),
                                ('fixed_x_cms','Fixed-X model'), ('cunge_cms','Cunge model')]]
    handles += [Line2D([0], [0], color='#8353ad', ls=':', lw=2, label='Existing Excel target'),
                Patch(facecolor='#8353ad', alpha=.12, label='Selected crest span')]
    for canvas, filename, subtitle in [(fig, '15_panel_common_grid_targets', 'Complete observed events'),
                                       (zoom, '15_panel_common_grid_target_crests', 'Detail around each target; vertical axes zoomed to downstream and model crests')]:
        canvas.suptitle('Common-grid calibration review · 15 retained examples', x=.055, y=.985, ha='left', fontsize=20)
        canvas.text(.055, .965, f'{subtitle} · one-minute interval means · existing parameters, NOT recalibrated', fontsize=12)
        canvas.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5,.951), ncol=6, frameon=False, fontsize=10)
        canvas.text(.055,.019,'Curves normalized independently over the full observation window; no time alignment. Purple targets are unchanged.',fontsize=10)
        canvas.text(.055,.007,'Faint dotted gauge segments span >2 nominal sampling intervals (diagnostic interpolation). Example 11 excluded. 21/22 clocks provisional.',fontsize=9,color='#555555')
        canvas.subplots_adjust(top=.91, bottom=.052, left=.055, right=.985, hspace=.40, wspace=.20)
        canvas.savefig(dest/f'{filename}.png', dpi=160, facecolor='white')
        canvas.savefig(dest/f'{filename}.svg', facecolor='white')
        plt.close(canvas)
    pd.DataFrame(review).to_csv(dest/'target_review.csv', index=False)
    print(dest/'15_panel_common_grid_targets.png')


if __name__ == '__main__':
    main()
