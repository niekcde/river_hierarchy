"""Record user-proposed ranges with the confirmed shifted-downstream policy."""
from pathlib import Path
import json,html
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
DEST=Path(json.loads((ROOT/'RAPID/outputs/common_grid_validation_20261008/latest.json').read_text())['directory'])
OUT=ROOT/'RAPID/reviews/common_grid_production_20261008'
def main():
 draft=json.loads((DEST/'beta_followups/height_target_review.json').read_text())
 bounds={'15':(2100,2200),'21':(14450,14500),'22':(20300,20400),'41':(7600,7700),'45':(1380,1420),'55':(560,580),'56':(1800,1900)}
 current=['13','37','48'];downstream=['03','05','46','54'];records=[]
 for original in draft['records']:
  ex=original['example'];data=pd.read_parquet(DEST/ex/'60s.parquet');valid=data.downstream_cms.notna();meta=json.loads((DEST/ex/'60s.json').read_text())[0]
  peak=float(data.cunge_cms[valid].max());up=float(data.forcing_cms.max());raw=float(data.downstream_cms[valid].max());offset=meta['initial_cms']-float(data.downstream_cms[valid].iloc[0]);shifted=raw+offset
  rec=dict(example=ex,units='m3/s',priority='soft height; timing primary',height_objective_basis='user-defined modeling range',timing_target_utc=original['time_target_utc'],feature_rule=original['feature_rule'],original_cunge_peak_cms=peak,forcing_peak_cms=up,downstream_raw_peak_cms=raw,downstream_shifted_peak_cms=shifted,downstream_display_offset_cms=offset,low_cms=None,high_cms=None,status='user_proposed',notes=[])
  if ex in bounds:rec['low_cms'],rec['high_cms']=bounds[ex]
  elif ex=='02':
   rec['low_cms']=2650;rec['high_cms']=1.01*up
   rec['notes'].append('Interpretation: lower 2650, upper common-grid input peak plus 1%.')
  elif ex in current:
   rec['low_cms']=.99*peak;rec['high_cms']=1.01*peak;rec['status']='suggested_small_range_pending_review'
   rec['notes'].append('Proposed +/-1% around ORIGINAL common-grid Cunge beta=1 peak, not fixed-X or a new fitted result. User endorsed current height, not these exact endpoints.')
  elif ex in downstream:
   rec['height_objective_basis']='baseline-adjusted downstream selected feature; user confirmed shifted curve'
   rec['status']='downstream_basis_confirmed'
   rec['notes'].append('User confirmed shifted downstream curve on 2026-10-08. Offset is frozen from the original forcing plateau and first supported downstream mean; never recompute it to improve a fit or from antecedent model output. Original measurements remain unchanged.')
   start=pd.Timestamp(original['feature_start_utc']);end=pd.Timestamp(original['feature_end_utc'])
   feature=valid & (data.interval_end_utc>=start) & (data.interval_start_utc<=end)
   assert feature.any()
   rec['height_target_nominal_cms']=float(data.downstream_cms[feature].max()+offset)
   rec['height_feature_operator']='maximum common-grid downstream mean overlapping the reviewed crest span'
   rec['soft_target_type']='point; no hard height acceptance band specified'
   if ex=='05':
    rec['low_cms']=rec['height_target_nominal_cms'];rec['high_cms']=1100
    assert rec['low_cms']<=rec['high_cms']
    rec['soft_target_type']='interval'
    rec['notes'].append('Use shifted downstream selected peak through 1100 m3/s. No extra percentage margin added.')
   if ex=='54':rec['notes'].append('Two-crest timing target remains unchanged. Nominal height is the taller selected crest, not the intervening midpoint discharge; compare both crest heights during shape review. Crest-span min/max are not uncertainty bounds.')
  if ex=='21':rec['notes'].append('User allows widening if not achievable. Test joint timing/height feasibility first; record any original/revised interval and reason, rather than silently expanding it.')
  if rec['low_cms'] is not None:
   rec['original_peak_inside_range']=rec['low_cms']<=peak<=rec['high_cms']
   rec['range_intersects_values_at_or_below_input_peak']=rec['low_cms']<=up
  if raw>up:rec['notes'].append('Raw downstream peak exceeds input peak. A direct raw-height fit may require boundary/flow comparability treatment; never generate numerical amplification to satisfy it.')
  records.append(rec)
 payload=dict(status='Shifted downstream basis confirmed; manual ranges recorded; suggested small ranges and point-target tolerances remain explicitly identified',date='2026-10-08',policy='No calibration run against these targets yet. Manual ranges are modeling acceptance criteria, not measurement confidence intervals. No autonomous widening or new hydraulic correction.',records=records)
 (OUT/'manual_height_targets_proposal.json').write_text(json.dumps(payload,indent=2))
 rows=[]
 for r in records:
  interval=f"Soft point: {r['height_target_nominal_cms']:.2f}" if r['low_cms'] is None else f"{r['low_cms']:.2f} – {r['high_cms']:.2f}"
  rows.append(dict(example=r['example'],soft_range_cms=interval,current_Cunge=round(r['original_cunge_peak_cms'],2),input_peak=round(r['forcing_peak_cms'],2),downstream_raw=round(r['downstream_raw_peak_cms'],2),downstream_shifted=round(r['downstream_shifted_peak_cms'],2),status=r['status']))
 doc='<html><meta charset="utf-8"><title>Proposed manual height targets</title><style>body{font:15px Arial;margin:30px}td,th{padding:8px}table{border-collapse:collapse}</style><h1>User-proposed soft height targets — 8 October 2026</h1><p>Timing remains primary. These are modelling targets, not confidence intervals. No fitting run against this proposal. Original Cunge means beta=1 at existing n.</p>'+pd.DataFrame(rows).to_html(index=False)
 for r in records:doc+='<p><b>'+r['example']+'</b>: '+html.escape(' '.join(r['notes']))+'</p>'
 (OUT/'manual_height_targets_proposal.html').write_text(doc+'</html>')
 print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
