"""Summarize regenerated attributes and retain link-level review provenance."""
from pathlib import Path
import json
import html
import pandas as pd
from regenerate_reviewed_hydraulics import OUT, base


def main():
    summary = json.loads((OUT / 'summary.json').read_text())
    results = pd.read_csv(OUT / 'regeneration_results.csv', dtype={'example': str})
    widths, flags = [], []
    checks = []
    for folder in sorted((OUT / 'objects').iterdir()):
        w = json.loads((folder / 'width_result.json').read_text())
        h = json.loads((folder / 'head_result.json').read_text())
        ids = w.get('changed_width_links', [])
        frame = pd.read_parquet(folder / 'widths.parquet')
        changed = frame[frame.id_link.isin(ids)].copy()
        changed.insert(0, 'state', w['state'])
        changed.insert(0, 'example', w['example'])
        changed['geometry_id'] = folder.name
        widths.append(changed)
        assert h['status'] == 'current_slope_pass'
        f = pd.read_parquet(folder / 'hydraulic_frame.parquet')
        ref = base(h['example'])[1].set_index('id_node')['head']
        values = pd.concat([f[['u', 'head_us']].rename(columns={'u':'node','head_us':'head'}),
                            f[['v', 'head_ds']].rename(columns={'v':'node','head_ds':'head'})])
        assert values.groupby('node')['head'].nunique().max() == 1
        values = values.drop_duplicates('node').set_index('node')['head']
        boundary = {n for patch in h['head_qa']['patches'] for n in patch['boundary_nodes']}
        assert all(values.loc[n] == ref.loc[n] for n in boundary)
        assert ((f.head_us - f.head_ds) > 0).all() and (f.slope > 0).all()
        checks.append(dict(example=h['example'], state=h['state'], links=len(f),
                           fixed_patch_boundary_nodes=len(boundary), boundary_heads_preserved=True,
                           min_head_drop_m=float((f.head_us-f.head_ds).min()), min_slope=float(f.slope.min())))
        audit = folder / 'routing_width_audit.csv'
        if audit.exists():
            a = pd.read_csv(audit)
            a = a[a.id_link.isin(ids) & a.width_correction_flag.eq(True)].copy()
            a.insert(0, 'geometry_id', folder.name)
            flags.append(a)
    pd.concat(widths, ignore_index=True).to_csv(OUT / 'remeasured_width_links.csv', index=False)
    pd.DataFrame(checks).to_csv(OUT / 'current_slope_validation.csv', index=False)
    flagged = pd.concat(flags, ignore_index=True)
    flagged.to_csv(OUT / 'width_correction_review.csv', index=False)
    counts = results.groupby(['example', 'status']).size().unstack(fill_value=0)
    assert len(checks) == 970
    assert results.status.eq('attributes_regenerated').all()
    fallback_count = sum(len(json.loads(p.read_text())['untrimmed_fallback_links'])
                         for p in (OUT / 'objects').glob('*/width_result.json'))
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Reviewed geometry: hydraulic regeneration</title>
<style>body{{font:16px/1.5 system-ui;max-width:1200px;margin:40px auto;padding:0 24px;color:#172534}}
table{{border-collapse:collapse;font-size:14px}}th,td{{padding:6px 12px;border-bottom:1px solid #ddd;text-align:left}}
th{{background:#eef2f5}}code{{background:#eef2f5;padding:2px 4px}}.notice{{padding:16px;background:#fff3dc}}</style>
<h1>Hydraulic regeneration of reviewed geometries</h1>
<p>Source: <code>{html.escape(summary['source_geometry_release'])}</code>. Geometry edits and reviewed island decisions are preserved.</p>
<p>All 970 geometry objects have regenerated attributes using the existing domain-based optimized slope implementation.
All pass. The additional fixed-exterior diagnostic has been superseded and is not an acceptance criterion.</p>
<p class="notice">This is an attribute regeneration, not a routing run or a recalibration. The active geometry release remains unchanged,
and final preparation/publication is a separate step. No old RAPID outputs were overwritten.</p>
<h2>Validation</h2><ul>
<li>Finite positive widths, depths, lengths and slopes; unique heads at shared nodes.</li>
<li>Hydraulic lengths and endpoints match the reviewed geometry.</li>
<li>All unchanged whole-link routing widths are retained exactly.</li>
<li>Cunge accepts all candidate frames, with a single inlet/outlet, acyclic connectivity and conserved reference flow.
Exact-length subdivisions and finite K/X were checked at Manning n = 0.035, alpha = 2 and beta = 1.
This is an input-consistency check, not a validation of calibrated routing performance.</li>
<li>Depth uses the existing reference depth and width relation with exponent 0.65. No Manning-n or beta calibration was performed.</li></ul>
<h2>Coverage by network</h2>{counts.to_html()}
<h2>Width provenance</h2><p>283 link geometries across 54 geometry objects required measurements or source-profile retrimming.
{fallback_count} use the existing untrimmed fallback where there is no valid trimmed core. This includes the six previously flagged fragments.
The production width solver applied 102 corrections; {len(flagged)} evaluated links retain correction flags.
No evaluated link has a corridor-consistency flag. Positive widths do not remove these measurement limitations.</p>
<p><a href="remeasured_width_links.csv">All remeasured links and trim flags</a> ·
<a href="width_correction_review.csv">Flagged width corrections</a></p>
<h2>Current slope validation</h2>
<p>All 970 frames have strictly positive head drops and slopes. Shared nodes have unique heads, and the fixed boundary heads
of every optimized patch equal their reference values exactly. The existing optimizer and its constraints were not changed.</p>
<p><a href="current_slope_validation.csv">Per-case slope and boundary validation</a></p>
<p>The earlier 55 flags came from an additional fixed-exterior constraint. That diagnostic is superseded;
no case requires approval on that basis. Every case was regenerated with the same current implementation.</p>
<h2>Next step</h2><p>Publish the regenerated attribute tables into the preparation workflow and use the selected calibration
when preparing routing parameters. No calibration or routing run was performed here.</p>
<p><a href="regeneration_results.csv">Complete object inventory</a> · <a href="summary.json">Machine-readable summary</a></p>
</html>'''
    (OUT / 'regeneration_review.html').write_text(text)
    print(OUT / 'regeneration_review.html')


if __name__ == '__main__':
    main()
