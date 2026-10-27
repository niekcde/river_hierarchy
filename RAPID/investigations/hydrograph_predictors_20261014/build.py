"""Link finalized responses to freshly computed geometry; never reroute models."""
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'RAPID/src'), str(ROOT/'RAPID/investigations/parent_child_predictor_exploration')]
from rapid_tools.hydrograph_response import load, digest, pulse, METRICS as RESPONSES
from rapid_tools.current_domains import load_domain_pair, allocation_tables
from features import describe, HYDRAULIC, METRICS

OUT = ROOT/'RAPID/outputs/hydrograph_predictors_20261014'
FOUNDATION = ROOT/'RAPID/outputs/geometry_response_foundation_20261012T120140'


def descriptors(frame):
    if 'id_us_node' not in frame:
        frame=frame.rename(columns={'u':'id_us_node','v':'id_ds_node'})
    m, meta = describe(frame, False)
    if meta['status'] != 'ok':
        raise ValueError(meta['status'])
    m = {k:float(v) for k,v in m.items() if k not in HYDRAULIC}
    if not m['bifurcations']:
        for k in m:
            if k.startswith(('dominance_', 'effective_branches_', 'split_entropy_', 'split_exposure_', 'split_impact_', 'branch_length_contrast_')) and k.endswith(('_mean','_sd')):
                m[k] = np.nan
    if not m['route_length_sd']: m['route_length_skew'] = np.nan
    return m


def pair_features(p,c,prefix):
    result = {}
    bounded = {'dominance_mean','dominance_sd','split_entropy_mean','split_entropy_sd','split_exposure_mean','split_exposure_sd','route_length_skew','log1p_paths'}
    for k in p:
        result[prefix+'parent_'+k] = p[k]
        result[prefix+'child_'+k] = c[k]
        result[prefix+'delta_'+k] = c[k]-p[k]
        if k not in bounded:
            result[prefix+'ratio_'+k] = c[k]/p[k] if p[k]>0 and c[k]>=0 else np.nan
    return result


def alias_lookup(rows):
    lookup = {}
    for r in rows.itertuples():
        for a in r.aliases.split(' | '):
            key = (str(r.example).zfill(2), a)
            if key in lookup and lookup[key] != r.state:
                raise ValueError(f'Ambiguous alias {key}')
            lookup[key] = r.state
    return lookup


def crests(t,q,baseline):
    # Deliberately no prominence filter: the first crest in 54 has a shallow dip.
    dt = np.median(np.diff(t))
    peaks,_ = find_peaks(q, distance=max(1,int(2*3600/dt)))
    return peaks[q[peaks]-baseline >= .99*(q.max()-baseline)]


def crest_diagnostic(t,q,reference_times,baseline,radius_hours=4.5):
    peaks = crests(t,q,baseline)
    matches = []
    for time in reference_times:
        candidates = peaks[np.abs(t[peaks]-time)<=radius_hours*3600]
        if len(candidates)!=1:
            return dict(status='unresolved',crest_switch_flag=None)
        matches.append(int(candidates[0]))
    if matches[0]>=matches[1] or int(q.argmax()) not in matches:
        return dict(status='unresolved',crest_switch_flag=None)
    dominant = matches.index(int(q.argmax()))
    return dict(status='resolved_two_crests',crest_switch_flag=bool(dominant!=1),
        dominant_crest=dominant+1,left_peak_seconds=t[matches[0]],right_peak_seconds=t[matches[1]],
        left_peak_cms=q[matches[0]],right_peak_cms=q[matches[1]])


def main():
    OUT.mkdir(exist_ok=True)
    states,responses = load('0.5%')  # Validate response source/input/table hashes.
    plan_path=ROOT/'RAPID/outputs/event_window_plan_20261013/plan.json'
    plan=json.loads(plan_path.read_text()); hashes={str(plan_path):digest(plan_path)}
    lookup=alias_lookup(responses)
    old_alias=pd.read_csv(FOUNDATION/'state_aliases.csv',dtype={'example':str})
    old_alias['response_state']=[lookup.get((r.example,r.state_id)) for r in old_alias.itertuples()]
    old_alias['join_status']=np.where(old_alias.state_id.eq('S000_base'),'parent_only',np.where(old_alias.response_state.notna(),'matched','unmatched'))
    old_alias.to_csv(OUT/'historical_alias_join_audit.csv',index=False)
    rows=[]; audits=[]; jobs=[]; domain_hashes={}
    for case in plan['cases']:
        ex=case['example'];basejob=next(j for j in case['jobs'] if j['state_id']=='S000_base')
        parent=pd.read_parquet(basejob['parameters']);pm=descriptors(parent)
        nodes,edges=allocation_tables(parent)
        visits=nodes.set_index('node').visit_probability
        for job in case['jobs']:
            path=job['parameters']; h=digest(path)
            assert h==case['source_sha256'][path],f'Geometry changed since response run: {path}'
            hashes[path]=h
            if job['state_id']=='S000_base':continue
            state='fully_collapsed' if job['state_id']=='single_centerline' else job['state_id']
            aliases=[a.split('/')[-1] for a in job['aliases']]
            mapped={lookup[(ex,a)] for a in [state,*aliases] if (ex,a) in lookup}
            assert mapped=={state},(ex,state,mapped)
            jobs.append(dict(example=ex,state=state,fingerprint=job['fingerprint'],parameters=path))
            child=pd.read_parquet(path);cm=descriptors(child)
            row=dict(example=ex,state=state,fingerprint=job['fingerprint'],**pair_features(pm,cm,'network_'))
            oldfps=set(old_alias.loc[(old_alias.example==ex)&(old_alias.response_state==state),'fingerprint'])
            assert not oldfps or oldfps=={job['fingerprint']},(ex,state,oldfps)
            try:
                if state=='fully_collapsed':
                    lp,lc=parent,child
                    meta=dict(source=int(nodes.iloc[0].node),domain_type='whole_network_single_centerline',membership_path=None)
                else:
                    family,sid=job['aliases'][0].split('/')
                    rec=pd.Series(dict(registry_path=str(ROOT/f'network_variants/output/{ex}_{family}__anchored/rapid_run_registry.csv'),state_id=sid,
                        parameters_parquet=path,parent_parameters_parquet=basejob['parameters']))
                    lp,lc,meta=load_domain_pair(rec)
                    member=meta['membership_path'];domain_hashes[member]=digest(member)
                    comparison=str(Path(member).parent/'bubble_equivalent_state_comparison.csv')
                    domain_hashes[comparison]=digest(comparison)
                a,b=descriptors(lp),descriptors(lc)
                row.update(pair_features(a,b,'domain_'))
                exposure=float(visits.loc[meta['source']])
                row.update(domain_entry_exposure=exposure,domain_length_fraction=a['total_length']/pm['total_length'],
                    domain_parent_length_over_network=a['route_length_mean']/pm['route_length_mean'],
                    exposed_length_change=exposure*(b['route_length_mean']-a['route_length_mean'])/pm['route_length_mean'],
                    exposed_route_sd_change=exposure*(b['route_length_sd']-a['route_length_sd'])/pm['route_length_mean'])
                audits.append(dict(example=ex,state=state,status='validated',domain_type=meta['domain_type'],reason='',membership_path=meta['membership_path']))
            except (ValueError,FileNotFoundError,KeyError,AssertionError) as error:
                audits.append(dict(example=ex,state=state,status='unavailable',domain_type=None,reason=str(error),membership_path=None))
            rows.append(row)
        print(ex,'geometry complete',flush=True)
    predictors=pd.DataFrame(rows);audit=pd.DataFrame(audits);jobs=pd.DataFrame(jobs)
    assert not predictors.duplicated(['example','state']).any()
    merged=responses.merge(predictors,on=['example','state'],how='outer',validate='one_to_one',indicator=True)
    merged[['example','state','aliases','_merge']].to_csv(OUT/'response_predictor_join_audit.csv',index=False)
    merged[merged._merge.ne('both')].to_csv(OUT/'unmatched_records.csv',index=False)
    assert merged._merge.eq('both').all()
    merged=merged.drop(columns='_merge').merge(audit.drop(columns=['membership_path','reason']),on=['example','state'],validate='one_to_one')
    assert len(merged)==2187 and np.isfinite(merged[list(RESPONSES)]).all().all()
    predictors.to_parquet(OUT/'predictors.parquet',index=False)
    merged.to_parquet(OUT/'linked_responses.parquet',index=False)
    audit.to_csv(OUT/'local_domain_audit.csv',index=False);jobs.to_csv(OUT/'geometry_sources.csv',index=False)
    # All candidate pairs have one canonical response; historical hydraulic columns are never joined.
    diagnostic=[];s54=states[states.example.eq('54')];parent54=s54[s54.state.eq('S000_base')].iloc[0]
    t,e,_=pulse(parent54);q=e+parent54.baseline_cms;peaks=crests(t,q,parent54.baseline_cms)
    assert len(peaks)==2 and int(q.argmax())==peaks[1]
    refs=t[peaks]
    for r in s54[~s54.state.eq('S000_base')].to_dict('records'):
        tt,ee,_=pulse(r);qq=ee+r['baseline_cms']
        for radius in [2.,4.5,6.]:
            diagnostic.append(dict(example='54',state=r['state'],matching_radius_hours=radius,
                **crest_diagnostic(tt,qq,refs,r['baseline_cms'],radius)))
    pd.DataFrame(diagnostic).to_csv(OUT/'crest_diagnostic_54.csv',index=False)
    pd.DataFrame(dict(crest=[1,2],time_utc=pd.to_datetime(refs,unit='s',utc=True),peak_cms=q[peaks])).to_csv(OUT/'parent_crests_54.csv',index=False)
    pd.DataFrame([dict(metric=k,family=v[2],description=v[3],length_exponent=v[0],time_exponent=v[1]) for k,v in METRICS.items() if k not in HYDRAULIC]).to_csv(OUT/'geometry_metric_catalog.csv',index=False)
    hashes.update(domain_hashes)
    for path in [ROOT/'RAPID/src/rapid_tools/hydrograph_response.py',ROOT/'RAPID/src/rapid_tools/current_domains.py',ROOT/'RAPID/investigations/parent_child_predictor_exploration/features.py',Path(__file__),FOUNDATION/'state_aliases.csv',ROOT/'RAPID/outputs/hydrograph_response/manifest.json',ROOT/'RAPID/outputs/hydrograph_response/responses.parquet',ROOT/'RAPID/outputs/hydrograph_response/sensitivity_responses.parquet']:
        hashes[str(path)]=digest(path)
    assert all(digest(p)==h for p,h in hashes.items())
    summary=dict(comparisons=len(merged),examples=merged.example.nunique(),matched=int(len(merged)),unmatched=0,
        historical_unmatched_aliases=int(old_alias.join_status.eq('unmatched').sum()),local_domains=audit.status.value_counts().to_dict(),
        window='0.5%',source_hashes=hashes,
        scope='Fresh physical length/width/connectivity descriptors. No old response/hydraulic columns. No routing or calibration.',
        crest_diagnostic='54 only; two local maxima within 99% of maximum excess, separated by >=2h; each must match exactly one parent crest within radius; global maximum must be one of them. Null means unresolved, not no switch. Default 4.5h; 2h and 6h sensitivity. Diagnostic, never predictor.',
        unresolved_work='Resistance requires new full-cohort static calculation under fixed exponent; not silently copied from historical tables.')
    (OUT/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
    print({k:v for k,v in summary.items() if k!='source_hashes'},flush=True)


if __name__=='__main__': main()
