"""Prepare measured evidence and explicitly separated interpretations for Excel."""
from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'RAPID/validation/hydrograph_match_review_20261005'

NOTES={
'02': ('Clear peaks. Downstream much higher/broader; additional input? Overall not a bad match.',
       'Model maximum 9.1 h early; downstream peak is 1.59× upstream. Three large external tributary candidates support a lateral-input hypothesis.',
       'Separate routing timing from downstream event growth; tributary timing remains unmeasured.'),
'03': ('Report start and middle of downstream plateau, and whether the model peak lies inside it.',
       'Model maximum lies inside the exact 17:45–23:00 plateau. First-maximum error overstates an unambiguous timing discrepancy.',
       'Retain first, midpoint and full plateau window. Midpoint is a sensitivity convention, not a uniquely observed peak.'),
'05': ('Substantial SWORD tributary may explain the extra peak. Report maximum and earlier 7 Jan 19:00–8 Jan 09:00 shoulder.',
       'Model is 17.8 h before the recorded maximum but inside the proposed earlier shoulder. Tributary area supports additional input as plausible.',
       'Keep both interpretations. Do not select the shoulder solely because it agrees with the model.'),
'11': ('Consider event start. Tributaries and broadening may delay the maximum; 18 May 09:45 feature closer to model. Overall not bad.',
       'Model is 34.8 h early versus first global maximum, but inside the broad 95%-of-maximum window. Candidate 18 May 09:45 feature is 4.9 h before model.',
       'Record the feature as exploratory. Legacy event start is only 11.5 min before forcing peak; do not use it as flood onset.'),
'13': ('Observed peak around 2400 versus 800 model; thought model about 22 h later. Not terrible, not great.',
       'Actual maxima: 2446 versus 861 m³/s. Model is 47.6 h late (not 22 h); 95% window centers still differ 40.9 h. Five tributary candidates.',
       'Large amplitude growth and substantial timing error remain. Lateral input is plausible but does not establish the cause of the delay.'),
'15': ('Higher observed peak; similar shapes; model late and broader, which might explain some of the timing.',
       'Observed peak 4230 versus 2137 m³/s model. Model is 23.8 h late; window-center comparison still 23.0 h late.',
       'Broadening alone does not resolve the discrepancy. Three tributary candidates support amplitude-growth investigation.'),
'21': ('About 10 h late; perhaps not terrible and still defensible?',
       'Model is 10.1 h late and inside a 63.3 h observed 95% window. Window-center error 8.4 h. Broad-event agreement is better than a single time suggests.',
       'Report timing error and breadth together. No universal acceptable-hours threshold is assumed.'),
'22': ('Similar shape/behaviour to 21; is delay also similar?',
       'Delay is 49.8 h, substantially larger than 21. Window-center error 52.7 h. Observed peak is 1.29× upstream; seven tributary candidates.',
       'Keep shape resemblance separate from timing skill. Investigate length/celerity and lateral-event timing.'),
'37': ('Model slightly earlier visually; input/model sharp rise and flat decline may match downstream plateau.',
       'Global model maximum is 7.5 h later, not earlier, than first downstream maximum. It lies inside the 20 h 95% window. Feature choice explains conflicting impressions.',
       'Keep rise/shoulder matching separate from the global maximum. Observed peak is 1.76× upstream.'),
'41': ('Sharper/higher and earlier observed peak. Long network? A 95% peak window may help.',
       'Reference network is 508 km. First-maximum error 47.3 h falls to 30.8 h for paired 95% window centers; model maximum lies in observed window.',
       'A broader window shows overlap but does not erase delayed modeled timing. Use the same rule in every case.'),
'45': ('Shapes align well but model is quite late. Can we find another explanation?',
       'Model maximum 27.4 h late; window-center error 28.7 h. Modeled lag 72.3 h agrees with 71.3 h geometric routing residence diagnostic.',
       'Timing reflects accumulated modeled travel time, not merely peak picking. Check depth/celerity assumptions without calibrating to this event.'),
'46': ('Most problematic and much slower. Could slope behaviour explain it?',
       'Model maximum 204.2 h late (8.5 days). A 95% window does not help. Long 775 km corridor and widespread low celerities dominate, not one isolated link.',
       'Treat as a material timing failure for this event. Inspect distributed slope/depth evidence and event correspondence before changing parameters.'),
'48': ('Review four highest peaks individually; repeated/coarse samples make first maximum misleading. First peak seems good.',
       'Global model maximum is 4.8 h late; exact-max span midpoint reduces it to 3.1 h. Multiple daily shoulders require event-by-event pairing; coarse values are repeated.',
       'Candidate peaks are supplied without automatic match claims. Inspect the full event plot and plateau spans; processed sample spacing is not native measurement precision.'),
'54': ('About 6 h late; not too terrible?',
       'Measured error is 5.1 h. Window-center comparison 4.4 h. Downstream peak is slightly lower than upstream.',
       'Report absolute and event-relative discrepancy; a small hours error alone is not validation.'),
'55': ('Fair disagreement but not terrible?',
       'Model maximum 22.4 h late; 95% window-center error 24.2 h. First observed maximum is a sharp point; downstream peak is 1.44× upstream.',
       'Test feature correspondence. No represented external tributary was identified, but SWORD flags unmapped inputs and reach accumulation grows.'),
'56': ('Broad peaks need a broader comparison. Percentage difference seems large.',
       'Model maximum 10.8 h late and inside the 34.7 h observed 95% window. Paired window-center error is 13.0 h, so broadening does not uniformly improve agreement.',
       'Report window overlap alongside lag error. The short observed lag amplifies percentages.'),
}


def read(name):
    return pd.read_csv(SRC/f'{name}.csv',dtype={'example':str,'tributary_reach_id':str,'receiving_reach_id':str},keep_default_na=False)


def main():
    summary=read('summary'); starts=read('event_starts'); routing=read('routing_summary'); corridors=read('corridors'); tributaries=read('tributaries'); windows=read('peak_windows'); events=read('48_peak_events')
    cases=[]
    for r in summary.to_dict('records'):
        user,assessment,action=NOTES[r['example']]
        cases.append(dict(**r,user_note=user,assessment=assessment,next_check=action))
    links=read('routing_links');f=links[links.example.eq('46')].copy(); f['band']=pd.cut(f.reference_mid_dist_out_km,np.arange(0,851,50),include_lowest=True)
    bands=f.groupby('band',observed=True).agg(links=('id_link','count'),residence_h=('residence_contribution_h','sum'),median_slope=('slope','median'),median_celerity_mps=('rapid_celerity_mps','median')).reset_index();bands['band']=bands.band.astype(str);bands['residence_fraction']=bands.residence_h/bands.residence_h.sum()
    cols=['id_link','reference_mid_dist_out_km','link_length_m','width','depth_m','slope','rapid_celerity_mps','flow_fraction','residence_contribution_h']
    payload=dict(cases=cases,starts=starts.to_dict('records'),routing=routing.to_dict('records'),corridors=corridors.to_dict('records'),tributaries=tributaries.to_dict('records'),windows=windows.to_dict('records'),events48=events.to_dict('records'),bands46=bands.to_dict('records'),links46=f.nlargest(20,'residence_contribution_h')[cols].to_dict('records'))
    (SRC/'workbook_evidence.json').write_text(json.dumps(payload,indent=2,allow_nan=False))

if __name__=='__main__':main()
