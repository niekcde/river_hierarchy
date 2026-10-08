from pathlib import Path
import sys,json
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'investigations'))
import cunge_full_hierarchy_reviewed as runner


def test_reviewed_flow_gate_requires_exact_evidence_and_unchanged_kinematics(tmp_path,monkeypatch):
 p=tmp_path/'evidence.json';p.write_text(json.dumps({'sources':[dict(example='41',key='a_b',parent_frame='parent',child_frame='child',parent_sha256='hash_a',child_sha256='hash_b')]}));monkeypatch.setattr(runner,'REVIEW',p)
 check={k+'_mismatches':0 for k in ['rapid_celerity_mps','rapid_k','count']};hashes={'parent':'hash_a','child':'hash_b'}
 assert runner.reviewed_transition('41','a','b',check,hashes)
 assert not runner.reviewed_transition('41','a','unknown',check,hashes)
 assert not runner.reviewed_transition('41','a','b',check,{'parent':'changed','child':'hash_b'})
 check['rapid_k_mismatches']=1;assert not runner.reviewed_transition('41','a','b',check,hashes)


def test_internal_conservation_failure_is_rejected():
 f=pd.DataFrame(dict(u=[0,1],v=[1,2],reference_discharge_cms=[10.,9.],rapid_k=[1.,1.],rapid_x=[-.1,-.1]))
 with pytest.raises(AssertionError):runner.verify_conservation(f,10.)
 f['reference_discharge_cms']=10.;runner.verify_conservation(f,10.)


def test_completed_stage_replaces_bulk_and_preserves_old_small_results(tmp_path):
 final=tmp_path/'45';stage=tmp_path/'.reviewed_staging'/'45'
 for p in [final,stage]:
  (p/'states'/'fp'/'run').mkdir(parents=True);(p/'completion.json').write_text('{"all_routed":true}');(p/'experiment.json').write_text('{}');(p/'states'/'fp'/'outlet_hydrograph.parquet').write_text('small')
 (final/'states'/'fp'/'run'/'Qout.nc').write_text('old large');(stage/'states'/'fp'/'run'/'Qout.nc').write_text('new large');(stage/'states'/'fp'/'result.json').write_text(json.dumps({'output_dir':str(stage/'states'/'fp')}))
 runner.publish_completed(stage,final)
 assert not stage.exists();assert (final/'states'/'fp'/'run'/'Qout.nc').read_text()=='new large'
 assert json.loads((final/'states'/'fp'/'result.json').read_text())['output_dir']==str(final/'states'/'fp')
 archived=list((tmp_path/'_previous_summaries'/'45').iterdir());assert len(archived)==1
 assert (archived[0]/'completion.json').exists();assert (archived[0]/'states'/'fp'/'outlet_hydrograph.parquet').exists();assert not (archived[0]/'states'/'fp'/'run').exists()


def test_incomplete_stage_cannot_replace_existing_results(tmp_path):
 stage=tmp_path/'stage';stage.mkdir();(stage/'completion.json').write_text('{"all_routed":false}');final=tmp_path/'final';final.mkdir();(final/'sentinel').write_text('keep')
 with pytest.raises(AssertionError):runner.publish_completed(stage,final)
 assert (final/'sentinel').read_text()=='keep'
