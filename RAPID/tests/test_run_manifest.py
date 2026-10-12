import json
import pytest
from rapid_tools.run_manifest import digest,load_run_manifest,master_path


def test_manifest_rejects_changed_calibration_and_requires_matching_prep(tmp_path):
    params=tmp_path/'parameters.json';params.write_text(json.dumps({'02':{'cunge':{'beta':.5}}}))
    manifest=tmp_path/'run_manifest.json'
    data=dict(schema_version='rapid-master-run-v1',master_root=str(tmp_path),examples=['02'],
              calibration_parameters='parameters.json',input_sha256={'parameters.json':digest(params)},
              time_policy={'report_interval_seconds':60},outputs={'prep_receipt':'prep.json'})
    manifest.write_text(json.dumps(data))
    load_run_manifest(manifest)
    with pytest.raises(ValueError,match='PREP is not complete'):load_run_manifest(manifest,stage='run')
    prepared=tmp_path/'prepared.dat';prepared.write_text('prepared')
    receipt=tmp_path/'prep.json';receipt.write_text(json.dumps(dict(complete=True,manifest_sha256=digest(manifest),files={'prepared.dat':digest(prepared)})))
    load_run_manifest(manifest,stage='run')
    prepared.write_text('changed')
    with pytest.raises(ValueError,match='Prepared input changed'):load_run_manifest(manifest,stage='run')
    params.write_text('{}')
    with pytest.raises(ValueError,match='Input changed'):load_run_manifest(manifest)


@pytest.mark.parametrize('path',['../outside','/absolute'])
def test_paths_must_stay_in_master_folder(tmp_path,path):
    with pytest.raises(ValueError):master_path(tmp_path,path)
