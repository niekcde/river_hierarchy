"""Versioned shared-window pilot responses and held-network baseline diagnostics."""
import json
import numpy as np
import pandas as pd
from pilot import ROOT,OUT,read,save,sha,descriptors,pair_features
from rapid_tools.hydrograph_response import load,distribution,response_rows
from rapid_tools.event_window_batch import verify
import drop_exposure_check as baseline


def response_window(q,input_start,input_end,initial,final,tolerance):
    departure=abs(q[input_start:]-initial)>tolerance
    sums=np.r_[0,np.cumsum(departure)];sustained=np.flatnonzero(sums[30:]-sums[:-30]==30)
    if not len(sustained):raise ValueError('No sustained event response')
    onset=input_start+int(sustained[0]);outside=np.flatnonzero((np.arange(len(q))>input_end)&(abs(q-final)>tolerance))
    end=int(outside[-1]+1) if len(outside) else input_end+1
    if end>=len(q):raise ValueError('Response outside saved support')
    return onset,end


def main():
    prepared=json.loads((OUT/'preparation_manifest.json').read_text());verify(prepared['sources']);verify(prepared['code_sha256'])
    plan=json.loads((OUT/'routing_plan.json').read_text());previous={w:load(w)[0] for w in ['0.5%','1%','2%']}
    states=[];windows=[];reproduction=[];audit=[];files={}
    for case in plan['cases']:
        ex=case['example'];receipt=OUT/'runs'/ex/'completion.json';done=json.loads(receipt.read_text());verify(done['files']);folder=ROOT/PathFix(done['final_directory'])
        files.update(done['files']);files[str(receipt)]=sha(receipt)
        wave=pd.read_parquet(folder/'forcing.parquet');t=wave.time_utc.astype('int64').to_numpy()/1e9;ii=np.flatnonzero(wave.phase.eq('input_event'));curves={}
        for job in case['jobs']:
            state=job['state_id'];path=folder/state/'outlet.parquet';f=pd.read_parquet(path);assert f.time_utc.equals(wave.time_utc)
            q=f.q_outlet_cms.to_numpy();assert np.isfinite(q).all();curves[state]=(q,path)
            if state in ['S000_base','single_centerline']:
                oldstate='fully_collapsed' if state=='single_centerline' else state
                record=previous['0.5%'].query('example==@ex and state==@oldstate').iloc[0]
                assert sha(record.outlet)==record.outlet_sha256
                old=pd.read_parquet(record.outlet);comparison=f[['time_utc','q_outlet_cms']].merge(old[['time_utc','q_outlet_cms']],on='time_utc',suffixes=('_new','_old'),validate='one_to_one')
                assert len(comparison)==min(len(f),len(old))
                np.testing.assert_allclose(comparison.q_outlet_cms_new,comparison.q_outlet_cms_old,rtol=1e-12,atol=1e-8)
                reproduction.append(dict(example=ex,state=state,max_absolute_discharge_difference=float(abs(comparison.q_outlet_cms_new-comparison.q_outlet_cms_old).max())))
        oldbase=float(previous['0.5%'].query('example==@ex').baseline_cms.iloc[0]);base=min([oldbase]+[float(q[ii[0]:].min()) for q,path in curves.values()])
        for window in previous:
            original=previous[window].query('example==@ex').iloc[0];tol=float(original.tolerance_cms)
            onset=float(original.rise_seconds);end=float(original.fall_seconds)
            for state,(q,path) in curves.items():
                a,b=response_window(q,ii[0],ii[-1],case['initial_cms'],case['final_cms'],tol)
                onset=min(onset,float(t[a]));end=max(end,float(t[b]));audit.append(dict(example=ex,state=state,window=window,response_onset_seconds=float(t[a]),response_end_seconds=float(t[b])))
            windows.append(dict(example=ex,window=window,baseline_cms=base,previous_baseline_cms=oldbase,rise_seconds=onset,fall_seconds=end,previous_rise_seconds=float(original.rise_seconds),previous_fall_seconds=float(original.fall_seconds),expanded=onset<float(original.rise_seconds) or end>float(original.fall_seconds),tolerance_cms=tol))
            ia=int(np.searchsorted(t,onset));ib=int(np.searchsorted(t,end));assert t[ia]==onset and t[ib]==end
            for state,(q,path) in curves.items():
                excess=q[ia:ib+1]-base;assert excess.min()>=0
                m=distribution(t[ia:ib+1],excess);peak=ia+int(q[ia:ib+1].argmax());assert np.isclose(q[peak],q[ii[0]:].max(),atol=1e-8,rtol=0)
                states.append(dict(example=ex,state=state,aliases=state,window=window,baseline_cms=base,threshold_fraction=float(window[:-1])/100,tolerance_cms=tol,rise_seconds=onset,fall_seconds=end,rise_utc=pd.Timestamp(onset,unit='s',tz='UTC').isoformat(),fall_utc=pd.Timestamp(end,unit='s',tz='UTC').isoformat(),outlet=str(path),outlet_sha256=sha(path),peak_seconds=float(t[peak]),peak_cms=float(q[peak]),peak_excess_cms=float(q[peak]-base),duration_hours=(end-onset)/3600,**m))
    st=pd.DataFrame(states);responses=response_rows(st)
    kinds={j['state']:j['kind'] for j in json.loads((OUT/'prepared_jobs.json').read_text())};kinds['single_centerline']='legacy_uniform_whole_network'
    responses['comparison']=responses.state.map(kinds)
    st.to_parquet(OUT/'state_metrics.parquet',index=False);responses.to_parquet(OUT/'responses.parquet',index=False);responses.to_csv(OUT/'responses.csv',index=False)
    pd.DataFrame(windows).to_csv(OUT/'window_audit.csv',index=False);pd.DataFrame(audit).to_csv(OUT/'state_window_audit.csv',index=False);pd.DataFrame(reproduction).to_csv(OUT/'reference_reproduction.csv',index=False)
    # Additive diagnostics: controlled single edits, same parent, same common window.
    pairs=json.loads((OUT/'selection.json').read_text());additive=[]
    for pair in pairs:
        for window in previous:
            frame=responses[(responses.example==pair['example'])&(responses.window==window)].set_index('state')
            combined=frame.loc[pair['state']];a=frame.loc['control_'+pair['state_a']];b=frame.loc['control_'+pair['state_b']]
            for metric in ['centroid_shift_hours','peak_shift_hours','peak_excess_change_pct','volume_change_pct','duration80_change_pct','asymmetry_change']:
                additive.append(dict(example=pair['example'],state=pair['state'],window=window,response=metric,combined=float(combined[metric]),single_a=float(a[metric]),single_b=float(b[metric]),nonadditivity=float(combined[metric]-a[metric]-b[metric])))
    pd.DataFrame(additive).to_csv(OUT/'additivity_diagnostics.csv',index=False)
    # Apply original seven-input definition only; no edited-length substitutions.
    d=pd.read_parquet(baseline.SOURCE/'input.parquet');pred=pd.read_parquet(OUT/'predictors.parquet');info=json.loads((baseline.OUT/'final_model.json').read_text());params=info['params'];oof=pd.read_parquet(baseline.OUT/'predictions.parquet');oof=oof[oof.variant=='seven_inputs'];prediction=[];support=[]
    for ex,g in pred.groupby('example'):
        tr=d[d.example.ne(ex)];te=d[d.example.eq(ex)];model=baseline.fit(tr,params)
        expected=te[['example','state']].merge(oof[['example','state','predicted_hours']],on=['example','state'],validate='one_to_one')
        np.testing.assert_allclose(baseline.forest.predict(model,te),expected.predicted_hours,rtol=1e-10,atol=1e-12)
        estimate=baseline.forest.predict(model,g)
        for r,value in zip(g.itertuples(),estimate):prediction.append(dict(example=ex,state=r.state,predicted_hours=float(value)))
        for _,r in g.iterrows():
            for c in baseline.COLUMNS:
                v=float(r[c]);lo=float(tr[c].min());hi=float(tr[c].max());support.append(dict(example=ex,state=r.state,feature=c,value=v,training_min=lo,training_max=hi,outside=v<lo-1e-12 or v>hi+1e-12))
    linked=responses.merge(pred,on=['example','state'],how='left',validate='many_to_one').merge(pd.DataFrame(prediction),on=['example','state'],how='left',validate='many_to_one')
    linked['centroid_error_minutes']=60*(linked.predicted_hours-linked.centroid_shift_hours)
    linked.to_parquet(OUT/'linked_diagnostics.parquet',index=False);pd.DataFrame(support).to_csv(OUT/'predictor_support.csv',index=False)
    manifest=dict(status='pilot_completed',examples=sorted(st.example.unique()),new_states=21,pairs=6,segmented_whole_network=3,controlled_single_edits=12,parent_and_legacy_reference_reruns=6,run_count=27,baseline_policy='Old minimum retained unless a new state lowers it; all pilot metrics recomputed consistently',window_policy='Union of existing all-state shared boundaries with new-state 30-minute onset and final-equilibrium tolerance boundaries; independent 0.5%,1%,2% versions; no overwrites of old results',prediction_policy='Seven-input frozen settings, train only other networks original partial collapses; reproduce original OOF predictions; no pilot training or edited-length substitution',sources={**files,str(OUT/'routing_plan.json'):sha(OUT/'routing_plan.json'),str(ROOT/'RAPID/outputs/hydrograph_response/manifest.json'):sha(ROOT/'RAPID/outputs/hydrograph_response/manifest.json'),str(Path(__file__)):sha(__file__)},limitations=['Purposeful 3-network pilot, not independent validation','Network 22 clocks retain existing provisional status','Network 13 excluded after 7–14m geometric endpoint mismatches','No calibration or legacy reference changes','No augmented predictor fit performed'])
    save(OUT/'analysis_manifest.json',manifest)
    print(responses[(responses.window=='0.5%')&~responses.comparison.eq('single_control')][['example','state','centroid_shift_hours','peak_shift_hours','peak_excess_change_pct']].round(4).to_string(index=False))
    print(pd.DataFrame(windows)[['example','window','expanded','baseline_cms','previous_baseline_cms']].to_string(index=False))

from pathlib import Path
# Path composition preserves already absolute saved directories.
def PathFix(value):return Path(value)
if __name__=='__main__':main()
