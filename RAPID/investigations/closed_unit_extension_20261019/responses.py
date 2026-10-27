"""Harmonize old and new comparisons to one versioned window per network."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from geometry_audit import OUT,ROOT,pilot
from rapid_tools.hydrograph_response import load,distribution,response_rows
from rapid_tools.event_window_batch import verify
from analyze_pilot import response_window


def calculate(row,start,end,base,window):
    path=Path(row['outlet']);assert pilot.sha(path)==row['outlet_sha256']
    f=pd.read_parquet(path);t=f.time_utc.astype('int64').to_numpy()/1e9;q=f.q_outlet_cms.to_numpy(float)
    if not t[0]<=start<end<=t[-1]:raise ValueError('New common window beyond historical support; reroute this state: '+str(path))
    ids=(t>start)&(t<end);tt=np.r_[start,t[ids],end];qq=np.interp(tt,t,q);e=qq-base
    assert e.min()>=-1e-9
    if e.min()<0:raise ValueError('Baseline requires review')
    m=distribution(tt,e);peak=int(qq.argmax())
    return dict(example=row['example'],state=row['state'],aliases=row.get('aliases',row['state']),window=window,baseline_cms=base,threshold_fraction=float(window[:-1])/100,tolerance_cms=row['tolerance_cms'],rise_seconds=start,fall_seconds=end,rise_utc=pd.Timestamp(start,unit='s',tz='UTC').isoformat(),fall_utc=pd.Timestamp(end,unit='s',tz='UTC').isoformat(),outlet=str(path),outlet_sha256=row['outlet_sha256'],peak_seconds=float(tt[peak]),peak_cms=float(qq[peak]),peak_excess_cms=float(e[peak]),duration_hours=(end-start)/3600,**m)


def main():
    plan=pilot.read(OUT/'routing_plan.json');original={w:load(w)[0] for w in ['0.5%','1%','2%']};original_responses={w:load(w)[1] for w in original};rows=[];windowrows=[];replication=[];sourcehashes={}
    for case in plan['cases']:
        ex=case['example'];receipt=OUT/'runs'/ex/'completion.json';done=pilot.read(receipt);verify(done['files']);folder=Path(done['final_directory']);wave=pd.read_parquet(folder/'forcing.parquet');ii=np.flatnonzero(wave.phase.eq('input_event'));t=wave.time_utc.astype('int64').to_numpy()/1e9
        curves={};base=float(original['0.5%'].loc[original['0.5%'].example.eq(ex),'baseline_cms'].iloc[0])
        for job in case['jobs']:
            state=job['state_id'];path=folder/state/'outlet.parquet';f=pd.read_parquet(path);assert f.time_utc.equals(wave.time_utc);q=f.q_outlet_cms.to_numpy();curves[state]=(q,path);base=min(base,float(q[ii[0]:].min()))
            if state in ['S000_base','single_centerline']:
                oldstate='fully_collapsed' if state=='single_centerline' else state;ref=original['0.5%'].query('example==@ex and state==@oldstate').iloc[0];old=pd.read_parquet(ref.outlet);comp=f[['time_utc','q_outlet_cms']].merge(old[['time_utc','q_outlet_cms']],on='time_utc',suffixes=('_new','_old'),validate='one_to_one');np.testing.assert_allclose(comp.q_outlet_cms_new,comp.q_outlet_cms_old,rtol=1e-12,atol=1e-8);replication.append(dict(example=ex,state=state,max_difference_cms=float(abs(comp.q_outlet_cms_new-comp.q_outlet_cms_old).max())))
        for window,old in original.items():
            old=old[old.example.eq(ex)];ref=old.iloc[0];start=float(ref.rise_seconds);end=float(ref.fall_seconds);tol=float(ref.tolerance_cms)
            for state,(q,path) in curves.items():
                a,b=response_window(q,ii[0],ii[-1],case['initial_cms'],case['final_cms'],tol);start=min(start,float(t[a]));end=max(end,float(t[b]))
            changed=start!=float(ref.rise_seconds) or end!=float(ref.fall_seconds) or base!=float(ref.baseline_cms)
            windowrows.append(dict(example=ex,window=window,old_start_seconds=float(ref.rise_seconds),new_start_seconds=start,old_end_seconds=float(ref.fall_seconds),new_end_seconds=end,old_baseline_cms=float(ref.baseline_cms),new_baseline_cms=base,earlier_start_minutes=(float(ref.rise_seconds)-start)/60,later_end_minutes=(end-float(ref.fall_seconds))/60,recomputed_old_states=bool(changed),tolerance_cms=tol))
            # Recompute ALL original states in changed networks: same targets for baseline/augmentation.
            for record in old.to_dict('records'):
                if changed:rows.append(calculate(record,start,end,base,window))
                else:rows.append(record)
            for state,(q,path) in curves.items():
                if state in ['S000_base','single_centerline']:continue
                record=dict(example=ex,state=state,aliases=state,outlet=str(path),outlet_sha256=pilot.sha(path),tolerance_cms=tol)
                rows.append(calculate(record,start,end,base,window))
        sourcehashes.update(done['files']);sourcehashes[str(receipt)]=pilot.sha(receipt)
        print(ex,'response cohort harmonized',flush=True)
    st=pd.DataFrame(rows);assert not st.duplicated(['example','state','window']).any();responses=response_rows(st)
    jobs=json.loads((OUT/'prepared_jobs.json').read_text());newkind={j['state']:'segmented' for j in jobs};newkind.update({j['state']:j['kind'] for j in json.loads((pilot.OUT/'prepared_jobs.json').read_text()) if j['kind']!='segmented_whole_network'})
    responses['cohort']=responses.state.map(newkind).fillna('original');responses.loc[responses.state.eq('fully_collapsed'),'cohort']='legacy_uniform'
    st.to_parquet(OUT/'state_metrics.parquet',index=False);responses.to_parquet(OUT/'responses.parquet',index=False);pd.DataFrame(windowrows).to_csv(OUT/'window_audit.csv',index=False);pd.DataFrame(replication).to_csv(OUT/'reference_reproduction.csv',index=False)
    change=[]
    for window,old in original_responses.items():
        new=responses[responses.window.eq(window)];j=old.merge(new,on=['example','state'],validate='one_to_one',suffixes=('_old','_new'))
        for metric in ['centroid_shift_hours','peak_shift_hours','peak_excess_change_pct','duration80_change_pct','volume_change_pct','asymmetry_change']:
            x=j[['example','state']].copy();x['window']=window;x['metric']=metric;x['old_value']=j[metric+'_old'];x['new_value']=j[metric+'_new'];x['change']=x.new_value-x.old_value;change.append(x)
    pd.concat(change,ignore_index=True).to_parquet(OUT/'window_response_changes.parquet',index=False)
    new=responses[responses.cohort.isin(['pair','segmented'])].merge(pd.read_parquet(OUT/'edited_predictors.parquet'),on=['example','state'],validate='many_to_one');new.to_parquet(OUT/'new_comparisons.parquet',index=False)
    pilot.save(OUT/'response_manifest.json',dict(status='completed',networks=15,primary='0.5%',sensitivity=['1%','2%'],common_window='Historical all-state boundaries unioned with new-state boundaries; all original targets recomputed where boundaries or baseline change. No per-state windows.',source_hashes={**sourcehashes,str(OUT/'routing_plan.json'):pilot.sha(OUT/'routing_plan.json'),str(ROOT/'RAPID/outputs/hydrograph_response/manifest.json'):pilot.sha(ROOT/'RAPID/outputs/hydrograph_response/manifest.json'),str(Path(__file__)):pilot.sha(__file__)},note='Preserves original response tables. Segmented states use edited-region-only predictors.'))
    print(responses[responses.window.eq('0.5%')].cohort.value_counts().to_string())
if __name__=='__main__':main()
