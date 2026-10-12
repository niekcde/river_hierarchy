"""Exercise publication backups and refusal to overwrite changed master files."""
import importlib.util
import json
from pathlib import Path
import sys
import pytest


def module(tmp_path, monkeypatch):
    investigations=Path(__file__).resolve().parents[1]/'investigations'
    monkeypatch.syspath_prepend(str(investigations))
    import publish_master_networks as publication
    root=tmp_path/'repo';master=tmp_path/'master';stage=tmp_path/'stage'
    for p in [root,master,stage]:p.mkdir()
    monkeypatch.setattr(publication,'ROOT',root)
    monkeypatch.setattr(publication,'MASTER',master)
    monkeypatch.setattr(publication,'STAGE',stage)
    monkeypatch.setattr(publication,'BACKUP',master/'backups')
    monkeypatch.setattr(publication,'safe_target',lambda p: Path(p) if Path(p).is_relative_to(master) else (_ for _ in ()).throw(ValueError('Outside master')))
    return publication


def plan(m):
    target=m.MASTER/'network.gpkg';target.write_bytes(b'old network')
    source=m.STAGE/'new.gpkg';source.write_bytes(b'new network')
    actions=[dict(source=str(source),target=str(target),sha256=m.digest(source),previous_sha256=m.digest(target),bytes=source.stat().st_size,sidecars={})]
    m.save(m.STAGE/'installation_plan.json',dict(master_root=str(m.MASTER),actions=actions))
    return target


def test_publication_backs_up_and_rolls_back(tmp_path,monkeypatch):
    m=module(tmp_path,monkeypatch);target=plan(m)
    m.publish();assert target.read_bytes()==b'new network'
    assert (m.BACKUP/'files/network.gpkg').read_bytes()==b'old network'
    m.rollback();assert target.read_bytes()==b'old network'


def test_publication_refuses_concurrent_master_change(tmp_path,monkeypatch):
    m=module(tmp_path,monkeypatch);target=plan(m);target.write_bytes(b'newer user edit')
    with pytest.raises(AssertionError,match='Source changed'):
        m.publish()
    assert target.read_bytes()==b'newer user edit'
    assert not (m.BACKUP/'completion.json').exists()


def test_reviewed_duplicate_island_stays_excluded_on_context_refresh(tmp_path):
    import pandas as pd
    from network_variants.island_experiment import generate_island_hydraulic_contexts
    analysis=tmp_path/'analysis.csv'
    pd.DataFrame([dict(island_id='island_41_006',analysis_backend='independent_unit_alias',analysis_state_dir=str(tmp_path/'canonical'),analysis_directed_links_path='unused',analysis_directed_nodes_path='unused',separate_observation_excluded=True)]).to_csv(analysis,index=False)
    links=tmp_path/'links.csv';nodes=tmp_path/'nodes.csv';links.write_text('id_link\n');nodes.write_text('id_node\n')
    result=generate_island_hydraulic_contexts(analysis,base_link_context_path=links,base_node_context_path=nodes,output_dir=tmp_path/'output')
    assert result.hydraulic_context_status.iloc[0]=='excluded_from_primary_analysis'
    assert result.hydraulic_context_action.iloc[0]=='excluded_duplicate_use_independent'
