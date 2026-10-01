"""Read-only reference-section audit; no changes to preparation or routing.

Compare stored peak-reference BASED depth and Q with the wide-channel Manning
relation behind RAPID celerity. The selected examples match the earlier real
pilot, not a statistical sample of rivers. Synthetic BASED evaluation is new
and diagnostic; the completed grid still uses its saved fixed kb.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'RAPID/src'))
from rapid_tools.reference_section import predict_based_depth,DEFAULT_BASED_MODEL_PATH
from rapid_tools.k_values import compute_k_values,KValueConfig
OUT=ROOT/'synthetic_experiments/outputs/full_grid_island/applicability/based_reference_audit'
EXAMPLES=['03','21','55','56']

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def evaluate(label,group,ref,config,**extra):
    w,h,q,s=[float(ref[k]) for k in ['reference_width_m','reference_depth_m','reference_discharge_cms','reference_slope']]
    kb=float(ref['kb_value']);n=float(config['n_manning'])
    assert np.isclose(w/h,kb)
    u=q/(w*h);um=h**(2/3)*np.sqrt(s)/n;qm=w*h*um
    implied_n=w*h**(5/3)*np.sqrt(s)/q
    frame=compute_k_values(pd.DataFrame([dict(width=w,slope_used=s,link_length_m=1000.)]),config=KValueConfig(
        width_field='width',kb_value=kb,n_manning=n,use_celerity_capping=config['use_celerity_capping'],
        min_celerity_mps=config['min_celerity_mps'],max_celerity_mps=config['max_celerity_mps']))
    assert np.isclose(frame.rapid_celerity_mps_raw.iloc[0],5*um/3)
    assert np.isclose(qm/q,um/u) and np.isclose(qm/q,implied_n/n)
    assert np.isclose(w*h**(5/3)*np.sqrt(s)/implied_n,q)
    return dict(case=label,group=group,**ref,n_manning=n,velocity_from_q_mps=u,velocity_manning_mps=um,
        manning_discharge_cms=qm,q_manning_over_reference=qm/q,reference_over_manning=q/qm,
        n_implied_by_reference=implied_n,celerity_raw_mps=frame.rapid_celerity_mps_raw.iloc[0],
        celerity_used_mps=frame.rapid_celerity_mps.iloc[0],reference_celerity_clipped=bool(frame.rapid_celerity_capped.iloc[0]),
        f2_from_q=u*u/(9.80665*h),f2_from_manning=um*um/(9.80665*h),**extra)

def main():
    OUT.mkdir(parents=True,exist_ok=True);rows=[];sources=[]
    keys=['reference_width_m','reference_depth_m','reference_discharge_cms','reference_slope','kb_value']
    for ex in EXAMPLES:
        directory=ROOT/'network_variants/output'/f'{ex}_indep__anchored'
        mp=directory/'rapid_prep_manifest.json';rp=directory/'reference_section/reference_kb_summary.json'
        manifest=json.loads(mp.read_text());stored=json.loads(rp.read_text())
        ref=manifest['reference_kb_summary'];config=manifest['prep_config']
        assert config['kb_mode']=='based_reference_section'
        matches=all(np.isclose(ref[k],stored[k],rtol=1e-10,atol=1e-10) for k in keys)
        assert matches, f'{ex}: cached and prepared reference conditions differ'
        marker=directory/'rapid_width_refresh_required.json'
        marker_data=json.loads(marker.read_text()) if marker.exists() else {}
        rows.append(evaluate(ex,'Real stored BASED', {k:ref[k] for k in keys},config,
            depth_method='stored BASED at reference peak Q',cached_reference_matches_prep=matches,
            width_refresh_marker_present=marker.exists(),width_refresh_marker_status=marker_data.get('status','absent')))
        sources.append(dict(case=ex,prep_manifest=str(mp.relative_to(ROOT)),prep_sha256=sha(mp),
            reference_summary=str(rp.relative_to(ROOT)),reference_sha256=sha(rp),
            refresh_marker=marker_data,station=manifest['forcing_config'].get('station_key')))
    run=ROOT/'synthetic_experiments/outputs/full_grid_island'
    configs=json.loads((run/'configurations.json').read_text())
    baseline=dict(total_width_m=500,reference_slope=1e-4,outer_a_fraction=.8,inner_fraction=.5,length_multiplier=1.,total_length_m=20000,internal_location='central')
    cid=next(c['configuration_id'] for c in configs if c['parameters']==baseline)
    states=pd.read_csv(run/'states.csv')
    rid=states.loc[(states.configuration_id==cid)&(states.state_id=='centerline/original'),'run_id'].iloc[0]
    jobpath=run/'jobs'/f'{rid}.json';job=json.loads(jobpath.read_text());settings=job['settings']
    config=dict(n_manning=settings['n_manning'],use_celerity_capping=True,min_celerity_mps=settings['min_celerity_mps'],max_celerity_mps=settings['max_celerity_mps'])
    q=settings['peak_cms'];w=baseline['total_width_m'];s=baseline['reference_slope']
    with threadpool_limits(limits=1):
        depth=predict_based_depth(discharge_cms=q,width_m=w,slope=s)
        # Verify current model reproduces the stored real estimates; never overwrite them.
        for row in rows:
            predicted=predict_based_depth(discharge_cms=row['reference_discharge_cms'],width_m=row['reference_width_m'],slope=row['reference_slope'])
            row['recomputed_based_depth_m']=predicted
            row['based_reproduction_relative_error']=predicted/row['reference_depth_m']-1
            assert np.isclose(predicted,row['reference_depth_m'],rtol=1e-5)
    for label,method,h,kb in [('Synthetic: saved kb=20','saved fixed kb',w/settings['kb'],settings['kb']),('Synthetic: BASED reference','diagnostic BASED at same peak Q',depth,w/depth)]:
        rows.append(evaluate(label,'Synthetic',dict(reference_width_m=w,reference_depth_m=h,reference_discharge_cms=q,reference_slope=s,kb_value=kb),config,depth_method=method))
    d=pd.DataFrame(rows);d.to_csv(OUT/'reference_comparison.csv',index=False)
    provenance=dict(reference='HYD-2026-09-29-BASED-MANNING',date='2026-09-29',
        sample_reason='Examples 03, 21, 55, 56 from the earlier real pilot; independent roots only; one reference per example.',
        scope='Stored preparation reference sections, not verification of current branch geometry or final refreshed run products.',
        definition='r_Q=Q_Manning/Q_reference=U_Manning/U_Q=n_implied/n_configured; unclipped quantities.',
        reference_discharge='Stored peak-reference Q for real; saved 1000 m3/s synthetic peak for both synthetic rows.',
        source_files=sources,synthetic_job=str(jobpath.relative_to(ROOT)),synthetic_job_sha256=sha(jobpath),
        based_model=str(DEFAULT_BASED_MODEL_PATH.relative_to(ROOT)),based_model_sha256=sha(DEFAULT_BASED_MODEL_PATH),
        script_sha256=sha(Path(__file__)),routing_changed=False,
        caveat='n_implied is an algebraic consistency diagnostic, not a fitted or recommended replacement roughness. Existing width-refresh markers are recorded without judging whether obsolete.')
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print(d[['case','reference_discharge_cms','reference_depth_m','kb_value','n_manning','manning_discharge_cms','q_manning_over_reference','n_implied_by_reference']].round(5).to_string(index=False))
if __name__=='__main__':main()
