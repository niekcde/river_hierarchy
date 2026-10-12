import json
from pathlib import Path
import pandas as pd
import pytest
from rapid_tools.run_manifest import digest
from rapid_tools.state_publication import publish


def fixture(tmp):
    def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d))
    prep=tmp/'batch/prep';run=tmp/'batch/results';prep.mkdir(parents=True)
    params=tmp/'params.json';save(params,{'02':{'cunge':{'beta':.5}}})
    aliases=pd.DataFrame([dict(example='02',family=f,state='S000_base',fingerprint='fp',context_alias=f!='indep') for f in ['indep','compound_bubbles']]);aliases.to_csv(prep/'aliases.csv',index=False);aliases.to_csv(tmp/'inventory.csv',index=False)
    wave=pd.DataFrame({'time_utc':pd.date_range('2026-01-01',periods=3,freq='min',tz='UTC'),'q_outlet_cms':[1.,2.,1.],'forcing_cms':[1.,3.,1.]})
    wave.to_parquet(prep/'forcing.parquet');pd.DataFrame({'x':[.1]}).to_parquet(prep/'parameters.parquet')
    job=dict(example='02',fingerprint='fp',parameters='batch/prep/parameters.parquet',forcing='batch/prep/forcing.parquet',native_dt_s=5,method='cunge');save(prep/'jobs.json',[job])
    m=dict(schema_version='rapid-master-run-v1',master_root=str(tmp),examples=['02'],methods=['cunge'],calibration_parameters='params.json',hydraulic_inventory='inventory.csv',input_sha256={'params.json':digest(params)},time_policy={'report_interval_seconds':60},outputs=dict(prep='batch/prep',results='batch/results',reports='batch/reports',prep_receipt='batch/prep/completion.json'))
    path=tmp/'manifest.json';save(path,m)
    save(prep/'completion.json',dict(complete=True,manifest_sha256=digest(path),scope=dict(examples=['02'],max_states_per_example=None),files={str(p.relative_to(tmp)):digest(p) for p in prep.iterdir()}))
    out=run/'02/fp/cunge';out.mkdir(parents=True);wave.to_parquet(out/'outlet.parquet');wave.to_parquet(run/'02/shared_forcing.parquet')
    save(out/'result.json',dict(feature_utc='2026-01-01T00:01:00Z',feature_kind='single_smoothed_crest',target_error_minutes=0))
    save(run/'02/completion.json',dict(files={str(p.relative_to(run/'02')):digest(p) for p in (run/'02').rglob('*') if p.is_file()}))
    save(run/'completion.json',dict(complete=True,manifest_sha256=digest(path)))
    for f in ['indep','compound_bubbles']:
        e=tmp/f'02_{f}__anchored';e.mkdir();(e/'rapid_run_registry.csv').write_text('old legacy registry\n');old=e/'states/S000_base/rapid/run';old.mkdir(parents=True);(old/'old.nc').write_bytes(b'legacy')
    return path


def test_publish_preserves_legacy_and_shares_base(tmp_path):
    path=fixture(tmp_path)
    publish(path,stage='prep');report=publish(path)
    assert report['unique_states']==1 and report['family_entries']==2
    for f in ['indep','compound_bubbles']:
        e=tmp_path/f'02_{f}__anchored'
        assert (e/'rapid/legacy_constant_x/rapid_run_registry.csv').read_text()=='old legacy registry\n'
        assert (e/'states/S000_base/rapid/run/old.nc').read_bytes()==b'legacy'
        row=pd.read_csv(e/'rapid_run_registry.csv').iloc[0]
        assert Path(row.outlet_hydrograph_csv).exists()
        assert '02_indep__anchored' in row.rapid_run_dir
    publish(path)


def test_corrupt_output_rejected_before_registry_change(tmp_path):
    path=fixture(tmp_path)
    (tmp_path/'batch/results/02/fp/cunge/result.json').write_text('{}')
    with pytest.raises(ValueError,match='Completed result changed'):publish(path)
    assert (tmp_path/'02_indep__anchored/rapid_run_registry.csv').read_text()=='old legacy registry\n'
