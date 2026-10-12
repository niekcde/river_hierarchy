"""Manifest-driven compact RAPID PREP/RUN for the published master inventory.

Physical state parameters and shared forcing are prepared once. RUN uses the
validated DAG cascade, avoiding full reach-by-time disk arrays. Native states are
audited; only common-interval outlet means are persisted.
"""
from pathlib import Path
import argparse,json,math,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from . import cunge
from .common_grid import route_interval_means
from .common_grid_production import shared_clock,model_feature
from .run_manifest import load_run_manifest,master_path,digest


def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def inventory_summary(inventory,examples):
    selected=inventory[inventory.example.isin(examples)&inventory.status.eq('ready')]
    table=selected.groupby(['example','family']).size().unstack(fill_value=0)
    for family in ['indep','compound_bubbles','island']:
        if family not in table:table[family]=0
    table=table[['indep','compound_bubbles','island']]
    table['total_entries']=table.sum(axis=1)
    table['unique_states']=selected.groupby('example').fingerprint.nunique()
    return table


def prepare(manifest_path,*,examples=None,max_states=None):
    manifest_path=Path(manifest_path);m,params=load_run_manifest(manifest_path)
    root=Path(m['master_root']);dest=master_path(root,m['outputs']['prep'])
    receipt_path=master_path(root,m['outputs']['prep_receipt'])
    if receipt_path.exists():raise FileExistsError('PREP already completed; use a new run manifest for a new preparation.')
    inv=pd.read_csv(master_path(root,m['hydraulic_inventory']),dtype={'example':str})
    from network_variants.reviewed_geometry import ReviewedGeometry
    release=ReviewedGeometry(master_path(root,m['geometry_manifest']))
    release.assert_hydraulics_current(inv)
    if (~inv.status.isin(['ready','excluded_duplicate','excluded_topology_review'])).any():raise ValueError('Unresolved inventory status')
    examples=list(examples or m['examples'])
    if not set(examples)<=set(m['examples']) or (max_states is not None and max_states<1):raise ValueError('Invalid PREP selection')
    dest.mkdir(parents=True,exist_ok=True)
    inventory_summary(inv,m['examples']).to_csv(dest/'inventory_summary.csv')
    inv[~inv.status.eq('ready')].to_csv(dest/'excluded_entries.csv',index=False)
    source_hashes={};records=[];selected_aliases=[];started=time.monotonic()
    from . import common_grid
    code_hashes={str(Path(module.__file__)):digest(module.__file__) for module in [cunge,common_grid]}
    code_hashes[str(Path(__file__))]=digest(__file__)
    for ex in examples:
        selected=inv[inv.example.eq(ex)&inv.status.eq('ready')].copy()
        # Base first for smoke selections; preserve every alias of selected fingerprints.
        ordered=selected.assign(_base=selected.state.ne('S000_base')).sort_values('_base',kind='stable').drop_duplicates('fingerprint')
        if max_states is not None:ordered=ordered.head(max_states)
        selected=selected[selected.fingerprint.isin(ordered.fingerprint)]
        selected_aliases.append(selected)
        p=params[ex];case=m['calibration_cases'][ex]
        frozen=pd.read_parquet(master_path(root,case['frozen_grid.parquet']))
        base=inv[inv.example.eq(ex)&inv.family.eq('indep')&inv.state.eq('S000_base')].iloc[0]
        current=pd.read_parquet(base.frame)
        columns=['id_link','u','v','link_length_m','width','slope','depth_m']
        pd.testing.assert_frame_equal(current[columns].sort_values('id_link').reset_index(drop=True),frozen[columns].sort_values('id_link').reset_index(drop=True),check_dtype=False,rtol=1e-10,atol=1e-8)
        # Content checks are against the published release, not only mutable path labels.
        for row in selected.itertuples():
            path=Path(row.frame)
            if str(path) in source_hashes:continue
            record=release.canonical(release.lookup(ex,row.family,row.state))
            expected=record.get('retained_source_checksums',{}).get(str(path))
            actual=digest(path)
            if expected is None or actual!=expected:raise ValueError('Published hydraulic frame changed: '+str(path))
            source_hashes[str(path)]=actual
        frames={};physical={}
        for row in ordered.itertuples():
            grid=cunge.freeze_grid(pd.read_parquet(row.frame),p['reference_discharge_cms'])
            physical[row.fingerprint]=grid
            for method in m['methods']:
                g=grid.copy()
                if method=='cunge':
                    g['count']*=p['cunge']['subdivision_multiplier'];g['rapid_effective_length_m']=g.link_length_m/g['count']
                    f=cunge.parameters(g,p['cunge']['n_manning'],beta=p['cunge']['beta'])
                elif method=='single_x':
                    if p['single_x']['x']!=.1:raise ValueError('Unreviewed single-X benchmark')
                    f=cunge.parameters(g,p['single_x']['n_manning'],method='fixed_x')
                else:raise ValueError('Unknown method')
                frames[(row.fingerprint,method)]=f
        dt=shared_clock(list(frames.values()),[30,60],p['cunge']['native_dt_cap_seconds'])
        wave=pd.read_parquet(master_path(root,case['60s.parquet']))
        if not m['quality_policy']['diagnostic_gap_fill'] and wave.upstream_gap.any():raise ValueError('Unapproved forcing gap')
        exdir=dest/ex;exdir.mkdir(parents=True,exist_ok=True)
        wave.to_parquet(exdir/'forcing.parquet',index=False)
        for (fp,method),frame in frames.items():
            target=exdir/fp/(method+'.parquet');target.parent.mkdir(parents=True,exist_ok=True);frame.to_parquet(target,index=False)
            records.append(dict(example=ex,fingerprint=fp,method=method,parameters=str(target.relative_to(root)),forcing=str((exdir/'forcing.parquet').relative_to(root)),native_dt_s=dt,nsub=round(60/dt),initial_cms=p['initial_discharge_cms'],base=fp==base.fingerprint))
        print('PREP',ex,len(ordered),'unique states',len(frames),'method jobs; dt',dt,flush=True)
    pd.concat(selected_aliases).to_csv(dest/'aliases.csv',index=False)
    save(dest/'jobs.json',records)
    for path,expected in source_hashes.items():
        if digest(path)!=expected:raise ValueError('Source changed during preparation: '+path)
    files={str(p.relative_to(root)):digest(p) for p in dest.rglob('*') if p.is_file() and p!=receipt_path and p.suffix!='.tmp'}
    receipt=dict(complete=True,manifest_sha256=digest(manifest_path),files=files,source_frame_sha256=source_hashes,
                 scope=dict(examples=examples,max_states_per_example=max_states,full_batch=set(examples)==set(m['examples']) and max_states is None),jobs=len(records),
                 runtime_seconds=time.monotonic()-started,code_sha256=code_hashes,format='compact hydraulic frames and shared interval forcing; no full reach-time arrays')
    save(receipt_path,receipt)
    return receipt


def route_job(job,root,forcing_path,output_dir,target):
    started=time.monotonic();root=Path(root);frame=pd.read_parquet(master_path(root,job['parameters']));wave=pd.read_parquet(forcing_path)
    q,audit=route_interval_means(frame,wave.forcing_cms.to_numpy(),job['nsub'],job['native_dt_s'],initial_flow=job['initial_cms'])
    if audit['minimum_native_flow_cms'] < -1e-9:raise ValueError('Negative native discharge')
    valid=wave.downstream_cms.notna().to_numpy();t=wave.time_utc.astype('int64').to_numpy()/1e9
    tm,height,kind=model_feature(q[valid],t[valid],job['example'],target)
    amp=max(float(np.ptp(wave.forcing_cms)),1e-12)
    # Require the final two days to be artificial held tail and near equilibrium.
    tail_bins=2*1440
    settled=(len(q)>=tail_bins and bool(wave.forcing_endpoint_hold.iloc[-tail_bins:].all()) and
             bool(np.all(abs(q[-tail_bins:]-wave.forcing_cms.iloc[-1])<=.001*amp)))
    dest=Path(output_dir)/job['fingerprint']/job['method'];dest.mkdir(parents=True,exist_ok=True)
    out=wave[['interval_start_utc','interval_end_utc','time_utc']].copy();out['q_outlet_cms']=q;out.to_parquet(dest/'outlet.parquet',index=False)
    result=dict(**job,feature_seconds=tm,feature_utc=pd.Timestamp(tm,unit='s',tz='UTC').isoformat(),peak_cms=height,feature_kind=kind,
                target_error_minutes=(tm-pd.Timestamp(target['target_utc']).timestamp())/60,tail_settled=settled,**audit,runtime_seconds=time.monotonic()-started)
    save(dest/'result.json',result);return result


def _work(args):return route_job(*args)


def run(manifest_path,*,workers=2,max_tail_extensions=7):
    if workers<1 or max_tail_extensions<0:raise ValueError('Invalid run limits')
    manifest_path=Path(manifest_path);m,params=load_run_manifest(manifest_path,stage='run');root=Path(m['master_root'])
    prep=master_path(root,m['outputs']['prep']);dest=master_path(root,m['outputs']['results']);dest.mkdir(parents=True,exist_ok=True)
    receipt=json.loads(master_path(root,m['outputs']['prep_receipt']).read_text())
    for p,h in {**receipt['source_frame_sha256'],**receipt['code_sha256']}.items():
        if digest(p)!=h:raise ValueError('Hydraulics changed since PREP: '+p)
    if (dest/'completion.json').exists():raise FileExistsError('RUN already complete')
    jobs=json.loads((prep/'jobs.json').read_text());aliases=pd.read_csv(prep/'aliases.csv',dtype={'example':str})
    targetfile=Path(m['calibration_parameters']).parent/'timing_targets.json'
    targets={r['example']:r for r in json.loads(master_path(root,str(targetfile)).read_text())['records']}
    results=[];started=time.monotonic()
    run_manifest_sha=digest(manifest_path);prep_receipt_sha=digest(master_path(root,m['outputs']['prep_receipt']))
    for ex in sorted({j['example'] for j in jobs}):
        group=[j for j in jobs if j['example']==ex];exdir=dest/ex;exdir.mkdir(parents=True,exist_ok=True)
        checkpoint=exdir/'completion.json'
        if checkpoint.exists():
            done=json.loads(checkpoint.read_text())
            if done.get('manifest_sha256')!=run_manifest_sha or done.get('prep_receipt_sha256')!=prep_receipt_sha:raise ValueError('Existing checkpoint belongs to different inputs')
            for rel,h in done['files'].items():
                if digest(exdir/rel)!=h:raise ValueError('Checkpoint file changed: '+rel)
            results.extend(json.loads((exdir/j['fingerprint']/j['method']/'result.json').read_text()) for j in group)
            print('RUN',ex,'resumed completed example',flush=True);continue
        wave=pd.read_parquet(master_path(root,group[0]['forcing']));flowpath=exdir/'shared_forcing.parquet'
        for attempt in range(max_tail_extensions+1):
            wave.to_parquet(flowpath,index=False)
            with ProcessPoolExecutor(max_workers=workers) as pool:
                rows=list(pool.map(_work,[(j,str(root),str(flowpath),str(exdir),targets[ex]) for j in group]))
            if all(r['tail_settled'] for r in rows):break
            if attempt==max_tail_extensions:raise ValueError('Tail did not settle; incomplete RUN retained for diagnosis')
            n=3*1440*2**attempt;end=wave.interval_end_utc.iloc[-1]
            tail=pd.DataFrame({c:np.nan for c in wave.columns},index=range(n))
            tail['interval_start_utc']=end+pd.to_timedelta(np.arange(n)*60,unit='s');tail['interval_end_utc']=end+pd.to_timedelta((np.arange(n)+1)*60,unit='s');tail['time_utc']=end+pd.to_timedelta((np.arange(n)+.5)*60,unit='s')
            tail['forcing_cms']=float(wave.forcing_cms.iloc[-1]);tail['forcing_endpoint_hold']=True
            wave=pd.concat([wave,tail],ignore_index=True)
        base_checks=[]
        expected=pd.read_parquet(master_path(root,m['calibration_cases'][ex]['wave.parquet']))
        for j in group:
            if not j['base']:continue
            actual=pd.read_parquet(exdir/j['fingerprint']/j['method']/'outlet.parquet').q_outlet_cms.to_numpy()[:len(expected)]
            column='calibrated_cunge_cms' if j['method']=='cunge' else 'fixed_x_cms'
            difference=float(np.max(abs(actual-expected[column].to_numpy())))
            relative=difference/max(float(np.ptp(expected.forcing_cms)),1e-12)*100
            if relative>.1:raise ValueError('Base replay differs by >0.1% of forcing amplitude; review before batch continuation')
            base_checks.append(dict(method=j['method'],maximum_difference_cms=difference,maximum_difference_pct_input=relative))
        hashes={str(p.relative_to(exdir)):digest(p) for p in exdir.rglob('*') if p.is_file() and p.name!='completion.json'}
        save(checkpoint,dict(complete=True,tail_extensions=attempt,shared_report_bins=len(wave),jobs=len(rows),base_replay=base_checks,manifest_sha256=run_manifest_sha,prep_receipt_sha256=prep_receipt_sha,files=hashes))
        results.extend(rows);pd.DataFrame(results).to_csv(dest/'unique_results.csv',index=False)
        print('RUN',ex,len(rows),'jobs complete; tail extensions',attempt,flush=True)
    merged=aliases.merge(pd.DataFrame(results),on=['example','fingerprint'],suffixes=('_inventory',''))
    merged.to_csv(dest/'state_results.csv',index=False)
    comparisons=[]
    for row in merged.itertuples():
        if pd.isna(row.parent_state) or row.parent_state==row.state:continue
        p=merged[merged.example.eq(row.example)&merged.family.eq(row.family)&merged.state.eq(row.parent_state)&merged.method.eq(row.method)]
        if p.empty and row.parent_state=='S000_base':p=merged[merged.example.eq(row.example)&merged.state.eq('S000_base')&merged.method.eq(row.method)].drop_duplicates('fingerprint')
        if len(p)!=1:
            if receipt['scope']['full_batch']:raise ValueError('Missing or ambiguous parent in full batch: '+str(row.parent_state))
            continue
        parent=p.iloc[0];comparisons.append(dict(example=row.example,family=row.family,method=row.method,parent_state=row.parent_state,child_state=row.state,delta_feature_seconds=row.feature_seconds-parent.feature_seconds,delta_peak_cms=row.peak_cms-parent.peak_cms,feature_kind_same=row.feature_kind==parent.feature_kind))
    pd.DataFrame(comparisons).to_csv(dest/'parent_child_comparison.csv',index=False)
    load_run_manifest(manifest_path,stage='run')
    for p,h in {**receipt['source_frame_sha256'],**receipt['code_sha256']}.items():
        if digest(p)!=h:raise ValueError('Source changed during RUN: '+p)
    save(dest/'completion.json',dict(complete=True,scope=receipt['scope'],manifest_sha256=digest(manifest_path),jobs=len(results),parent_child_comparisons=len(comparisons),runtime_seconds=time.monotonic()-started,final_scientific_acceptance=False))
    return results


def cli(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prep','run','inventory','publish'])
    parser.add_argument('--manifest',type=Path);parser.add_argument('--examples',nargs='+');parser.add_argument('--max-states',type=int);parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args(argv)
    pointer=Path(__file__).resolve().parents[2]/'config/production_run.json'
    path=args.manifest or Path(json.loads(pointer.read_text())['manifest'])
    if args.stage=='prep':
        prepare(path,examples=args.examples,max_states=args.max_states)
        if args.max_states is None:
            from .state_publication import publish
            publish(path,stage='prep')
    elif args.stage=='run':
        if args.examples or args.max_states:parser.error('RUN uses the exact PREP selection; select scope during PREP.')
        run(path,workers=args.workers)
        m,_=load_run_manifest(path,stage='run')
        receipt=json.loads(master_path(m['master_root'],m['outputs']['prep_receipt']).read_text())
        if receipt['scope']['max_states_per_example'] is None:
            from .state_publication import publish
            publish(path,stage='run')
    elif args.stage=='publish':
        from .state_publication import publish
        publish(path,stage='prep')
        publish(path,stage='run')
    else:
        m,_=load_run_manifest(path);inv=pd.read_csv(master_path(m['master_root'],m['hydraulic_inventory']),dtype={'example':str});print(inventory_summary(inv,m['examples']).to_string())

if __name__=='__main__':cli()
