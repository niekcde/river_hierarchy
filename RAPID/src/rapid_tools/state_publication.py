"""Publish compact Cunge products beside their reviewed geometry states.

Batch receipts remain recovery/provenance records. Existing constant-X products
remain untouched; their original registries are archived before switching active
registries. Aliases reference one canonical state product.
"""
from pathlib import Path
import json
import shutil
import pandas as pd
from .run_manifest import load_run_manifest, master_path, digest


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2)+'\n');tmp.replace(path)


def copy_verified(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(source)!=digest(target):raise ValueError('Refusing to replace different product: '+str(target))
    else:
        tmp=target.with_suffix(target.suffix+'.tmp');shutil.copy2(source,tmp)
        if digest(source)!=digest(tmp):raise ValueError('Copy checksum mismatch')
        tmp.replace(target)
    return digest(target)


def experiment(root, row):
    return root/f'{row.example}_{row.family}__anchored'


def publish(manifest_path, *, stage='run'):
    manifest_path=Path(manifest_path)
    m,_=load_run_manifest(manifest_path,stage='run') # Both publications require completed PREP.
    if m['methods']!=['cunge']:raise ValueError('State publication currently requires Cunge-only scope')
    root=Path(m['master_root']);prep=master_path(root,m['outputs']['prep']);results=master_path(root,m['outputs']['results'])
    receipt=json.loads((prep/'completion.json').read_text())
    if receipt['scope']['max_states_per_example'] is not None:raise ValueError('Do not publish partial example smoke tests')
    if stage not in ('prep','run'):raise ValueError('Unknown publication stage')
    if stage=='run':
        done=json.loads((results/'completion.json').read_text())
        if not done['complete'] or done['manifest_sha256']!=digest(manifest_path):raise ValueError('RUN is not complete for this manifest')
        for ex in receipt['scope']['examples']:
            checkpoint=json.loads((results/ex/'completion.json').read_text())
            for name,h in checkpoint['files'].items():
                if digest(results/ex/name)!=h:raise ValueError('Completed result changed: '+name)
    aliases=pd.read_csv(prep/'aliases.csv',dtype={'example':str})
    jobs=json.loads((prep/'jobs.json').read_text())
    inventory=pd.read_csv(master_path(root,m['hydraulic_inventory']),dtype={'example':str})
    # Plan and check all destinations before changing active registries.
    products={};copies=[]
    for job in jobs:
        group=aliases[aliases.example.eq(job['example'])&aliases.fingerprint.eq(job['fingerprint'])]
        chosen=group.sort_values(['context_alias','family','state'],kind='stable').iloc[0]
        # A base shared across families always belongs with the independent base.
        base=group[group.family.eq('indep')&group.state.eq('S000_base')]
        if len(base):chosen=base.iloc[0]
        dest=experiment(root,chosen)/'states'/chosen.state/'rapid'/'cunge'
        owner=dest/'publication.json'
        if owner.exists() and json.loads(owner.read_text())['manifest_sha256']!=digest(manifest_path):
            raise ValueError('A different Cunge run is already published: '+str(dest))
        forcing=root/f'{job["example"]}_indep__anchored'/'rapid'/'cunge'/'forcing.parquet'
        entries=[(root/job['parameters'],dest/'prep'/'parameters.parquet'),(root/job['forcing'],forcing)]
        if stage=='run':
            source=results/job['example']/job['fingerprint']/'cunge'
            entries += [(source/'outlet.parquet',dest/'run'/'outlet.parquet'),(source/'result.json',dest/'run'/'result.json'),(results/job['example']/'shared_forcing.parquet',forcing.with_name('run_forcing.parquet'))]
        for src,tgt in entries:
            if tgt.exists() and digest(src)!=digest(tgt):raise ValueError('Existing product differs: '+str(tgt))
        copies.extend(entries);products[(job['example'],job['fingerprint'])]=(dest,forcing,job)
    # Preserve byte-identical legacy registry snapshots; all their data paths stay valid.
    for (ex,family), group in aliases.groupby(['example','family']):
        exp=root/f'{ex}_{family}__anchored';archive=exp/'rapid'/'legacy_constant_x'
        for name in ['rapid_prep_registry.csv','rapid_run_registry.csv','rapid_prep_manifest.json','rapid_run_manifest.json']:
            p=exp/name
            if not p.exists():continue
            active=exp/'rapid'/'cunge'/'active.json'
            if not active.exists():copy_verified(p,archive/name)
        write_json(archive/'README.json',dict(method='legacy_constant_x',storage='Original state rapid/prep and rapid/run files retained in place; archived registries preserve their original paths. No legacy hydrographs deleted.'))
    hashes={}
    for src,tgt in copies:hashes[str(tgt)]=copy_verified(src,tgt)
    for key,(dest,forcing,job) in products.items():
        write_json(dest/'publication.json',dict(manifest=str(manifest_path),manifest_sha256=digest(manifest_path),stage=stage,method='cunge',fingerprint=key[1],forcing=str(forcing),run_forcing=str(forcing.with_name('run_forcing.parquet')) if stage=='run' else None))
        if stage=='run':
            wave=pd.read_parquet(dest/'run'/'outlet.parquet');flow=pd.read_parquet(forcing.with_name('run_forcing.parquet'))
            assert len(wave)==len(flow)
            wave['time_seconds']=(wave.time_utc-wave.time_utc.iloc[0]).dt.total_seconds()
            wave['q_inflow_cms']=flow.forcing_cms.to_numpy()
            wave.to_csv(dest/'run'/'outlet_hydrograph.csv',index=False)
    for (ex,family), group in aliases.groupby(['example','family']):
        exp=root/f'{ex}_{family}__anchored';rows=[]
        for row in group.itertuples():
            dest,forcing,job=products[(ex,row.fingerprint)]
            local=exp/'states'/row.state/'rapid'/'cunge';local.mkdir(parents=True,exist_ok=True)
            if local!=dest:write_json(local/'reference.json',dict(canonical_product=str(dest),fingerprint=row.fingerprint,manifest_sha256=digest(manifest_path)))
            record=dict(state_id=row.state,method='cunge',fingerprint=row.fingerprint,rapid_prep_dir=str(dest/'prep'),rapid_run_dir=str(dest/'run'),parameters_parquet=str(dest/'prep'/'parameters.parquet'),forcing_parquet=str(forcing),routing_dt_seconds=job['native_dt_s'],forcing_dt_seconds=60,status='prepared' if stage=='prep' else 'ran',output_format='compact_outlet_interval_means',qout_nc=None)
            if stage=='run':
                result=json.loads((dest/'run'/'result.json').read_text())
                record.update(outlet_hydrograph_csv=str(dest/'run'/'outlet_hydrograph.csv'),outlet_parquet=str(dest/'run'/'outlet.parquet'),result_json=str(dest/'run'/'result.json'),calibration_feature_utc=result['feature_utc'],calibration_feature_kind=result['feature_kind'],target_error_minutes=result['target_error_minutes'])
                wave=pd.read_parquet(dest/'run'/'outlet.parquet');peak=wave.loc[wave.q_outlet_cms.idxmax()]
                record.update(peak_time_utc=peak.time_utc.isoformat(),peak_discharge_cms=float(peak.q_outlet_cms),peak_time_seconds=float((peak.time_utc-wave.time_utc.iloc[0]).total_seconds()))
            rows.append(record)
        registry=pd.DataFrame(rows)
        registry.to_csv(exp/('rapid_prep_registry.csv' if stage=='prep' else 'rapid_run_registry.csv'),index=False)
        info=dict(method='cunge',manifest=str(manifest_path),manifest_sha256=digest(manifest_path),states=len(rows),stage=stage,legacy_registry_directory=str(exp/'rapid'/'legacy_constant_x'),reach_time_outputs_available=False)
        write_json(exp/('rapid_prep_manifest.json' if stage=='prep' else 'rapid_run_manifest.json'),info)
        write_json(exp/'rapid'/'cunge'/'active.json',info)
    report=dict(complete=True,stage=stage,examples=receipt['scope']['examples'],unique_states=len(products),family_entries=len(aliases),manifest_sha256=digest(manifest_path),published_sha256=hashes,legacy_deleted=False)
    write_json(master_path(root,m['outputs']['reports'])/f'state_publication_{stage}.json',report)
    print(f'Published {len(products)} unique Cunge states / {len(aliases)} family entries beside state geometry ({stage}); legacy retained.',flush=True)
    return report
