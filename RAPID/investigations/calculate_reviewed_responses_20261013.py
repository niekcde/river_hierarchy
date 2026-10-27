"""Export peak responses supported by reviewed references, without inventing B(t)."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'RAPID/src'))
from rapid_tools.current_review import build_current_analysis_frame

OUT = ROOT / 'RAPID/outputs/reviewed_responses_20261013'
OUT.mkdir(exist_ok=True)
selection_path = ROOT / 'RAPID/outputs/calibration_review/plateau_review_20261013.json'
cases = json.loads(selection_path.read_text())['cases']
frame = build_current_analysis_frame(list(cases))
frame['state_aliases'] = frame.groupby(['example_id', 'fingerprint']).state_id.transform(lambda s: ' | '.join(dict.fromkeys(s)))
frame = frame.loc[~frame.technical_alias].copy()
starts, ends, refs, durations = {}, {}, {}, {}
for ex, case in cases.items():
    candidate = next(c for c in case['candidates'] if c['key'] == case['default_candidate'])
    level = next(l for l in candidate['levels'] if l['raise_pct'] == case['default_raise_pct'])
    starts[ex] = pd.Timestamp(case['start']['start_utc'])
    ends[ex] = pd.Timestamp(level['crossing_utc'])
    refs[ex] = max(case['start']['start_cms'], level['plateau_cms'])
    durations[ex] = (ends[ex] - starts[ex]).total_seconds() / 3600
frame['review_start_utc'] = frame.example_id.map(starts)
frame['review_end_utc'] = frame.example_id.map(ends)
frame['base_review_window_hours'] = frame.example_id.map(durations)
frame['shared_scalar_qref_cms'] = frame.example_id.map(refs)
frame['parent_scalar_peak_excess_cms'] = frame.parent_peak_cms - frame.shared_scalar_qref_cms
frame['child_scalar_peak_excess_cms'] = frame.child_peak_cms - frame.shared_scalar_qref_cms
eligible = frame.parent_scalar_peak_excess_cms.gt(0) & frame.child_scalar_peak_excess_cms.gt(0)
for label in ['parent', 'child']:
    peak = pd.to_datetime(frame[label + '_peak_utc'], utc=True)
    eligible &= peak.ge(frame.review_start_utc) & peak.le(frame.review_end_utc)
frame['peak_within_review_window_and_positive_excess'] = eligible
frame['scalar_peak_excess_change_pct'] = np.where(
    eligible, 100 * (frame.child_scalar_peak_excess_cms / frame.parent_scalar_peak_excess_cms - 1), np.nan)
frame['scalar_peak_excess_definition'] = 'Qref=max(reviewed Q0, reviewed Qf); separate from Q-B(t) catalogue peak'
frame['event_distribution_status'] = 'Pending explicit B(t); no clipping or assumed background'
frame['reviewed_duration_response_status'] = 'Pending per-state departure/end rule; base window is not a child duration'
frame.to_csv(OUT / 'peak_responses.csv', index=False)
frame.to_parquet(OUT / 'peak_responses.parquet', index=False)
summary = frame.groupby('example_id').agg(pairs=('state_id', 'size'),
    scalar_peak_excess_valid=('scalar_peak_excess_change_pct', 'count'))
summary.to_csv(OUT / 'coverage.csv')
registry_paths = sorted(set(frame.registry_path))
manifest = dict(
    status='Partial: peak responses calculated; volume-based catalogue requires B(t)',
    selected_references_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),
    registries={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in registry_paths},
    source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    pairs=len(frame), networks=len(summary),
    timing='Existing sampled and reviewed production features retained as distinct columns',
    duration='Manual base review window is reported; not substituted for per-state event duration',
    background='Not yet assigned; scalar peak excess is not catalogue time-varying-background excess')
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
assert not frame.duplicated(['example_id', 'fingerprint']).any()
assert frame.loc[eligible, 'scalar_peak_excess_change_pct'].notna().all()
print(summary.to_string())
print(f'{len(frame)} unique parent-child pairs exported to {OUT}')
