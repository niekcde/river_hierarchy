"""Fully collapsed reference-section Cunge cases and combined batch entry point."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
from . import cunge
from .common_grid_production import refined_parameters,shared_clock
from .master_workflow import save,route_job
from .run_manifest import load_run_manifest,master_path,digest


def hydraulic_frame(base, links, reference):
    """One collapsed reach, preserving current boundary heads and anchored depth."""
    if len(links)!=1:raise ValueError('Reference section must contain exactly one collapsed link')
    _,_,inlet,outlet=cunge.topology(base)
    upper=base.loc[base.u.eq(inlet),'head_us'].to_numpy();lower=base.loc[base.v.eq(outlet),'head_ds'].to_numpy()
    if not np.allclose(upper,upper[0]) or not np.allclose(lower,lower[0]):raise ValueError('Inconsistent boundary heads')
    expected=reference['reference_depth_m']*(base.width/reference['reference_width_m'])**.65
    if not np.allclose(expected,base.depth_m,rtol=1e-9):raise ValueError('Reference depth does not match current base hydraulics')
    if links.crs is None or not links.crs.is_projected or not np.isclose(links.crs.axis_info[0].unit_conversion_factor,1):raise ValueError('Reference geometry must use projected metres')
    row=links.iloc[0];length=float(row.geometry.length);width=float(row.wid_adj_routing)
    if not np.isclose(width,reference['reference_width_m'],rtol=1e-9):raise ValueError('Reference width changed')
    frame=pd.DataFrame([dict(id_link=int(row.id_link),u=int(row.id_us_node),v=int(row.id_ds_node),link_length_m=length,width=width,depth_m=float(reference['reference_depth_m']),head_us=float(upper[0]),head_ds=float(lower[0]),slope=float((upper[0]-lower[0])/length))])
    if not np.isfinite(frame[['link_length_m','width','depth_m','slope']]).all().all() or (frame[['link_length_m','width','depth_m','slope']]<=0).any().any():raise ValueError('Invalid collapsed hydraulics')
    return frame


def case_inputs(manifest, ex):
    m,params=load_run_manifest(manifest);root=Path(m['master_root']);exp=root/f'{ex}_indep__anchored';ref=exp/'reference_section'
    paths=dict(base=exp/'states/S000_base/hydraulic_context/hydraulic_frame.parquet',links=ref/'variant/widths/links_with_width_families.gpkg',reference=ref/'reference_kb_summary.json')
    import geopandas as gpd
    base=pd.read_parquet(paths['base']);frozen=pd.read_parquet(master_path(root,m['calibration_cases'][ex]['frozen_grid.parquet']))
    cols=['id_link','u','v','link_length_m','width','depth_m','slope']
    pd.testing.assert_frame_equal(base[cols].sort_values('id_link').reset_index(drop=True),frozen[cols].sort_values('id_link').reset_index(drop=True),check_dtype=False,rtol=1e-10,atol=1e-8)
    before={str(p):digest(p) for p in paths.values()}
    frame=hydraulic_frame(base,gpd.read_file(paths['links']),json.loads(paths['reference'].read_text()))
    grid=refined_parameters(cunge.freeze_grid(frame,params[ex]['reference_discharge_cms']),params[ex]['cunge'])
    for p,h in before.items():
        if digest(p)!=h:raise ValueError('Source changed while reading: '+p)
    return m,params[ex],ref,frame,grid,before


def prepare(manifest, examples, *, destination_root=None):
    for ex in examples:
        m,p,ref,frame,grid,sources=case_inputs(manifest,ex)
        dest=Path(destination_root)/ex if destination_root else ref/'rapid/cunge'
        prep=dest/'prep';receipt=prep/'completion.json'
        if receipt.exists():
            done=json.loads(receipt.read_text());verify(done,manifest)
            print('CENTERLINE PREP',ex,'already complete',flush=True);continue
        prep.mkdir(parents=True,exist_ok=True)
        cap=p['cunge']['native_dt_cap_seconds']
        network_registry=ref.parent/'rapid_prep_registry.csv'
        if network_registry.exists():
            registry=pd.read_csv(network_registry)
            if 'method' in registry and registry.method.eq('cunge').all():
                cap=min(cap,float(registry.routing_dt_seconds.min()))
                sources[str(network_registry)]=digest(network_registry)
        dt=shared_clock([grid],[30,60],cap)
        frame.to_parquet(prep/'hydraulic_frame.parquet',index=False);grid.to_parquet(prep/'parameters.parquet',index=False)
        wave=pd.read_parquet(master_path(m['master_root'],m['calibration_cases'][ex]['60s.parquet']));wave.to_parquet(prep/'forcing.parquet',index=False)
        save(receipt,dict(complete=True,example=ex,method='cunge',manifest=str(Path(manifest).resolve()),manifest_sha256=digest(manifest),source_sha256=sources,files={str(f):digest(f) for f in prep.iterdir() if f.is_file()},native_dt_s=dt,nsub=round(60/dt),initial_cms=p['initial_discharge_cms'],n_manning=p['cunge']['n_manning'],beta=p['cunge']['beta'],slope_policy='current base boundary head drop / collapsed geometry length',depth_policy='same reference depth as current base anchored width-depth relationship',code_sha256={str(Path(mod.__file__)):digest(mod.__file__) for mod in [cunge,sys.modules[__name__]]}))
        print('CENTERLINE PREP',ex,'dt',dt,'cells',int(grid['count'].sum()),flush=True)


def verify(receipt,manifest):
    load_run_manifest(manifest)
    if not receipt['complete'] or receipt['manifest_sha256']!=digest(manifest):raise ValueError('Centerline PREP belongs to different manifest')
    for p,h in {**receipt['source_sha256'],**receipt['files'],**receipt['code_sha256']}.items():
        if digest(p)!=h:raise ValueError('Centerline input changed: '+p)


def run(manifest,examples,*,destination_root=None):
    m,_=load_run_manifest(manifest);root=Path(m['master_root'])
    targets={r['example']:r for r in json.loads((root/Path(m['calibration_parameters']).parent/'timing_targets.json').read_text())['records']}
    for ex in examples:
        ref=root/f'{ex}_indep__anchored/reference_section';dest=Path(destination_root)/ex if destination_root else ref/'rapid/cunge'
        prep=dest/'prep';out=dest/'run';receipt=json.loads((prep/'completion.json').read_text());verify(receipt,manifest)
        if (out/'completion.json').exists():
            done=json.loads((out/'completion.json').read_text())
            if done['prep_sha256']!=digest(prep/'completion.json'):raise ValueError('Completed centerline run has different PREP')
            for p,h in done['files'].items():
                if digest(p)!=h:raise ValueError('Completed centerline result changed: '+p)
            print('CENTERLINE RUN',ex,'already complete',flush=True);continue
        out.mkdir(parents=True,exist_ok=True);wave=pd.read_parquet(prep/'forcing.parquet');job=dict(example=ex,fingerprint='fully_collapsed',method='cunge',parameters='prep/parameters.parquet',native_dt_s=receipt['native_dt_s'],nsub=receipt['nsub'],initial_cms=receipt['initial_cms'],base=False)
        # route_job's output layout is a work area; publish final files directly under run/.
        for attempt in range(8):
            wave.to_parquet(out/'forcing.parquet',index=False)
            result=route_job(job,dest,out/'forcing.parquet',out/'work',targets[ex])
            if result['tail_settled']:break
            if attempt==7:raise ValueError('Centerline tail failed to settle: '+ex)
            n=3*1440*2**attempt;end=wave.interval_end_utc.iloc[-1];tail=pd.DataFrame({c:np.nan for c in wave.columns},index=range(n))
            for col,offset in [('interval_start_utc',0),('interval_end_utc',1),('time_utc',.5)]:tail[col]=end+pd.to_timedelta((np.arange(n)+offset)*60,unit='s')
            tail['forcing_cms']=float(wave.forcing_cms.iloc[-1]);tail['forcing_endpoint_hold']=True;wave=pd.concat([wave,tail],ignore_index=True)
        import shutil
        work=out/'work/fully_collapsed/cunge'
        for name in ['outlet.parquet','result.json']:shutil.copy2(work/name,out/name)
        hydro=pd.read_parquet(out/'outlet.parquet');hydro['q_inflow_cms']=wave.forcing_cms.to_numpy();hydro['time_seconds']=(hydro.time_utc-hydro.time_utc.iloc[0]).dt.total_seconds();hydro.to_csv(out/'outlet_hydrograph.csv',index=False)
        verify(receipt,manifest)
        save(out/'completion.json',dict(complete=True,example=ex,prep_sha256=digest(prep/'completion.json'),tail_extensions=attempt,files={str(f):digest(f) for f in out.iterdir() if f.is_file()},final_scientific_acceptance=False))
        if destination_root is None:
            pd.DataFrame([dict(state_id='fully_collapsed',method='cunge',status='ran',rapid_prep_dir=str(prep),rapid_run_dir=str(out),outlet_hydrograph_csv=str(out/'outlet_hydrograph.csv'),result_json=str(out/'result.json'))]).to_csv(ref/'rapid_run_registry.csv',index=False)
        print('CENTERLINE RUN',ex,'complete; target difference',result['target_error_minutes'],'minutes (comparison, not a calibration acceptance test)',flush=True)


def cli():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prep','run']);parser.add_argument('--examples',nargs='+',required=True);parser.add_argument('--centerline-only',action='store_true');parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();repo=Path(__file__).resolve().parents[3];pointer=json.loads((repo/'RAPID/config/production_run.json').read_text());source=Path(pointer['manifest']);m,_=load_run_manifest(source)
    if len(set(args.examples))!=len(args.examples) or not set(args.examples)<=set(m['examples']):parser.error('Select unique retained study examples')
    if args.centerline_only:
        (prepare if args.stage=='prep' else run)(source,args.examples);return
    # New future batches have a distinct name; active earlier manifests are untouched.
    run_id='common_grid_20261009_with_centerline_'+'_'.join(args.examples);old=m['run_id'];m['run_id']=run_id
    m['outputs']={k:v.replace('rapid_runs/'+old+'/', 'rapid_runs/'+run_id+'/') for k,v in m['outputs'].items()}
    path=repo/'RAPID/config'/f'{run_id}.json';content=json.dumps(m,indent=2)+'\n'
    if path.exists() and path.read_text()!=content:raise ValueError('Existing batch manifest differs')
    if not path.exists():path.write_text(content)
    from .master_workflow import cli as network_cli
    if args.stage=='prep':
        network_cli(['prep','--manifest',str(path),'--examples',*args.examples]);prepare(source,args.examples)
    else:
        network_cli(['run','--manifest',str(path),'--workers',str(args.workers)]);run(source,args.examples)
