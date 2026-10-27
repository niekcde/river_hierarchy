"""Replace the compact October 14 plot table using its own saved curves.

No routing or calibration is performed. The archived calibration is deliberately
used instead of the later uncapped refits, which do not belong to this plot.
"""
from pathlib import Path
import base64
from datetime import datetime, timezone
import gzip
import html
import json
import math
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'RAPID/outputs/completed_event_review_20261014'
BUNDLE = ROOT / 'RAPID/outputs/bounded_beta_calibration_20261009'


def centroid(panel, series):
    """Exact first moment of piecewise-linear excess Q on the review window."""
    x = panel['shared_arrays'][series['x_ref']]
    start, end = panel['input_bounds'][0], panel['cutoffs']['0.1pct']
    pairs = [(t, q - panel['summary']['baseline_cms'])
             for t, q in zip(x, series['y']) if start <= t <= end]
    assert pairs[0][0] == start and pairs[-1][0] == end
    assert all(q >= 0 for _, q in pairs)
    mass, moment = [], []
    for (a, qa), (b, qb) in zip(pairs, pairs[1:]):
        dt = (b - a) * 86400
        area = dt * (qa + qb) / 2
        mass.append(area)
        moment.append((a - start) * 86400 * area + dt * dt * (qa / 2 + (qb - qa) / 3))
    elapsed = math.fsum(moment) / math.fsum(mass)
    epoch = datetime.fromisoformat(panel['origin']).timestamp() + start * 86400 + elapsed
    return epoch, elapsed / 3600


def main():
    path = OUT / '15_calibration_review_email.html'
    document = path.read_text()
    payload = re.search(r'<script id="payload"[^>]*>(.*?)</script>', document, re.S)[1]
    panels = json.loads(gzip.decompress(base64.b64decode(payload)))
    rows, records = [], []
    stamp = lambda t: datetime.fromtimestamp(round(t / 60) * 60, timezone.utc).strftime('%Y-%m-%d %H:%M')
    for p in panels:
        ex = p['example']
        r = json.loads((BUNDLE / ex / 'result.json').read_text())
        series = {s['key']: s for s in p['calibration_series']}
        full, full_hours = centroid(p, series['base'])
        center, center_hours = centroid(p, series['centerline'])
        low, high, peak = r['height_low_cms'], r['height_high_cms'], r['peak_cms']
        signed = peak - (low if peak < low else high if peak > high else peak)
        assert math.isclose(abs(signed), r['height_distance_cms'], abs_tol=1e-8)
        record = dict(example=ex, full_centroid_utc=stamp(full), single_centroid_utc=stamp(center),
                      full_centroid_hours_from_input_start=full_hours,
                      single_centroid_hours_from_input_start=center_hours,
                      centroid_shift_hours=(center-full)/3600,
                      n_manning=r['n_manning'], beta=r['beta'],
                      calibration_timing_error_minutes=r['timing_error_minutes'],
                      calibration_height_error_cms=signed,
                      calibration_height_error_pct=100*r['height_signed_residual'],
                      height_target_low_cms=low, height_target_high_cms=high,
                      analysis_end_utc=p['summary']['cutoff_utc'],
                      baseline_cms=p['summary']['baseline_cms'])
        records.append(record)
        target = f'{low:.1f}' if low == high else f'{low:.1f}–{high:.1f}'
        cells = [ex, stamp(full), stamp(center), f'{(center-full)/3600:+.2f}',
                 f"{r['n_manning']:.6f}", f"{r['beta']:.4f}", target,
                 f"{r['timing_error_minutes']:+.2f}", f'{signed:+.2f}',
                 f"{100*r['height_signed_residual']:+.2f}"]
        rows.append('<tr>' + ''.join('<td>'+html.escape(v)+'</td>' for v in cells) + '</tr>')
    headers = ['Example', 'Centroid · Full Complexity (UTC)', 'Centroid · single centerline (UTC)',
               'Centroid shift · single − full (h)', 'Manning n', 'Beta',
               'Calibration height target (m³/s)', 'Calibration timing error (min)',
               'Calibration height error (m³/s)', 'Calibration height error (%)']
    table = '<table><thead><tr>'+''.join('<th>'+html.escape(v)+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table>'
    notes = '''<h2>Centroid timing and calibration · 15 examples</h2>
<p>Centroids describe the plotted reruns: the time-weighted centre of discharge above this review’s shared minimum baseline, from input-event start to its shared 0.1% settling cutoff. This is the historical review window, not a recalculation with later analysis definitions. Times are UTC, rounded to the nearest minute; negative single − full means the single centerline arrives earlier.</p>
<p>Manning n and beta are the original fitted parameters (9 October calibration bundle). Timing and height errors below belong to that calibration, not a new fit of these cropped-input reruns or of the peak-normalized gauge. Timing error is model feature minus target: negative = early. Height error is signed distance from the target point or nearest edge of the target interval; zero means inside the interval. Percent error uses the target value or interval midpoint.</p>
<p>Beta is the diffusion multiplier. X is derived per reach: X = 0.5 − beta × (0.5 − X<sub>Cunge</sub>); there is no single fitted X per example. The older fixed-X benchmark used X = 0.1 and is not a curve in this figure. The table stays fixed when display controls change.</p>'''
    # Display rounded minutes, consistently with the table description.
    for record, row in zip(records, rows):
        assert record['example'] in row
    replacement = '<section id="calibration-summary">'+notes+'<div style="overflow-x:auto">'+table+'</div></section>'
    if '<section id="calibration-summary">' in document:
        document, count = re.subn(r'<section id="calibration-summary">.*?</section>', lambda _: replacement, document, count=1, flags=re.S)
    else:
        document, count = re.subn(r'<table\b.*?</table>', lambda _: replacement, document, count=1, flags=re.S)
    assert count == 1 and len(records) == 15
    assert re.search(r'<script id="payload"[^>]*>(.*?)</script>', document, re.S)[1] == payload
    assert len(document.encode()) < 34_000_000
    path.write_text(document)
    (OUT / 'email_calibration_table.json').write_text(json.dumps(records, indent=2))
    print(f'Updated 15 rows; plot payload unchanged; HTML {path.stat().st_size/1e6:.3f} MB')


if __name__ == '__main__':
    main()
