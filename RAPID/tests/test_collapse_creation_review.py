from pathlib import Path
import sys
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rapid_tools import collapse_creation_review as review


def test_priority_uses_failing_alias_not_first_alias(monkeypatch):
    boundaries=pd.DataFrame(dict(example=['41','41'],state=['CB_good','S_bad'],nearest_child_node_distance_m=[0.,800.],nearest_child_line_distance_m=[0.,0.],missing_canonical_node=[False,True]))
    units=pd.DataFrame(dict(example=['41'],states=['CB_good;S_bad'],review_class=['projection_cut_unresolved_over_100m']))
    monkeypatch.setattr(review,'load_review_data',lambda:(pd.DataFrame(),boundaries,units))
    queue=review.cut_queue('priority')
    assert queue.iloc[0].representative_state=='S_bad'
    assert review.cut_queue('priority','05').empty


def test_missing_historical_stage_is_not_replaced_with_final(tmp_path):
    r=pd.Series(dict(variant_manifest=str(tmp_path/'absent.json'),base_links_path='base.gpkg',base_nodes_path='base_nodes.gpkg',child_links_path='final.gpkg',child_nodes_path='final_nodes.gpkg'))
    stages=review.stage_paths(r)
    assert stages[1]==('Pre-anchor geometry unavailable',None,None)
    assert stages[2][1]=='final.gpkg'


def test_review_record_does_not_change_source_registry(tmp_path,monkeypatch):
    monkeypatch.setattr(review,'OUT',tmp_path)
    monkeypatch.setattr(review,'case_row',lambda ex,state:pd.Series(dict(example='41',fingerprint='abc')))
    with pytest.raises(ValueError,match='Reviewer'):
        review.save_review('41','S016_unit_4','cut','missing_cut','','')
    assert not list(tmp_path.iterdir())
    entry=review.save_review('41','S016_unit_4','cut','missing_cut','Boundary 5 absent from the final line.','reviewer')
    assert [p.name for p in tmp_path.iterdir()]==['human_reviews.jsonl']
    assert entry['effect']=='review record only; no production deletion/exclusion'


def test_reuse_requires_target_match_and_no_partial_edits(tmp_path,monkeypatch):
    monkeypatch.setattr(review,'OUT',tmp_path)
    monkeypatch.setattr(review,'case_row',lambda ex,state:pd.Series(dict(example='41',fingerprint='abc')))
    pd.DataFrame(dict(example=['41'],island_state=['island_41_006'],independent_state=['S016_unit_4'],target_island=['island_41_006'],island_partial_count=[0],independent_partial_count=[1])).to_csv(tmp_path/'island_independent_target_pairs.csv',index=False)
    with pytest.raises(ValueError,match='No clean singleton'):
        review.save_review('41','island_41_006','island','reuse_validated_unit_output','Checked geometry','reviewer','S016_unit_4')
    assert not (tmp_path/'human_reviews.jsonl').exists()
