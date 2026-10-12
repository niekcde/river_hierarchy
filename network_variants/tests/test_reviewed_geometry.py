import json
from types import SimpleNamespace
import pandas as pd
import pytest
from network_variants.reviewed_geometry import ReviewedGeometry


def registry(tmp_path):
    original = tmp_path/'old_nodes.gpkg'
    records = [dict(example='41',family='indep',state='S016_unit_4',geometry_status='accepted',
                    geometry_id='new-geometry',requires_hydraulic_regeneration=True,nodes=str(original)),
               dict(example='41',family='island',state='island_41_006',canonical_key=['41','indep','S016_unit_4'],separate_observation_excluded=True),
               dict(example='41',family='island',state='existing_alias',canonical_key=['41','indep','S016_unit_4'])]
    p=tmp_path/'manifest.json';p.write_text(json.dumps(dict(release='reviewed-test',records=records)))
    return ReviewedGeometry(p)


def test_dependent_alias_cannot_route_old_hydraulics(tmp_path):
    release=registry(tmp_path)
    inv=pd.DataFrame([dict(example='41',family='island',state='existing_alias',status='ready')])
    with pytest.raises(ValueError,match='predating'):
        release.assert_hydraulics_current(inv)
    inv['geometry_release']='reviewed-test';inv['geometry_id']='new-geometry'
    with pytest.raises(ValueError,match='Regenerate'):
        release.assert_hydraulics_current(inv)
    release.hydraulic_regeneration_complete=True
    release.assert_hydraulics_current(inv)
    inv['geometry_id']='other-geometry'
    with pytest.raises(ValueError,match='predating'):
        release.assert_hydraulics_current(inv)


def test_duplicate_island_must_be_excluded_as_separate_observation(tmp_path):
    release=registry(tmp_path)
    inv=pd.DataFrame([dict(example='41',family='island',state='island_41_006',status='ready')])
    with pytest.raises(ValueError,match='duplicate island'):
        release.assert_hydraulics_current(inv)
    inv['status']='excluded_duplicate'
    release.assert_hydraulics_current(inv)
    assert release.canonical(release.lookup('41','island','island_41_006'))['state']=='S016_unit_4'


def test_preparation_rejects_stale_geometry_before_writing(tmp_path):
    release=registry(tmp_path)
    context=SimpleNamespace(directed_nodes_path=tmp_path/'old_nodes.gpkg')
    with pytest.raises(ValueError,match='Regenerate hydraulic'):
        release.assert_preparation_current(context)
    context.geometry_release='reviewed-test';context.geometry_id='new-geometry'
    with pytest.raises(ValueError,match='Regenerate'):
        release.assert_preparation_current(context)
    release.hydraulic_regeneration_complete=True
    release.assert_preparation_current(context)


def test_alias_cycle_is_rejected(tmp_path):
    release=registry(tmp_path)
    independent=release.lookup('41','indep','S016_unit_4')
    independent['canonical_key']=['41','island','existing_alias']
    with pytest.raises(ValueError,match='Cycle'):
        release.canonical(independent)


def test_active_manifest_tampering_is_rejected(tmp_path, monkeypatch):
    from network_variants import reviewed_geometry as module
    p=tmp_path/'active.json';p.write_text(json.dumps(dict(manifest=str(tmp_path/'manifest.json'),sha256='wrong')))
    registry(tmp_path)
    monkeypatch.setattr(module,'ACTIVE',p)
    with pytest.raises(ValueError,match='checksum'):
        ReviewedGeometry()


def test_pending_attributes_are_not_accepted_by_release_stamp_alone(tmp_path):
    release=registry(tmp_path)
    inv=pd.DataFrame([dict(example='41',family='indep',state='S016_unit_4',
                          geometry_release='reviewed-test',geometry_id='new-geometry',
                          status='pending_reviewed_geometry_hydraulics')])
    with pytest.raises(ValueError,match='Regenerate'):
        release.assert_hydraulics_current(inv)


def test_normal_registry_exposes_integrated_identity_without_relabelling_old_hydraulics(tmp_path):
    release=registry(tmp_path)
    frame=pd.DataFrame([dict(state_id='S016_unit_4',status='complete',directed_nodes_path='old')])
    annotated=release.annotate_state_registry('41_indep__anchored',frame)
    assert annotated.integrated_geometry_id.iloc[0]=='new-geometry'
    assert annotated.integrated_hydraulics_pending.iloc[0]
    assert annotated.directed_nodes_path.iloc[0]=='old'
    assert 'integrated_geometry_id' not in frame


def test_integrated_geometry_path_also_requires_hydraulic_regeneration(tmp_path):
    release=registry(tmp_path)
    record=release.lookup('41','indep','S016_unit_4')
    record['bundle']=str(tmp_path/'bundle')
    context=SimpleNamespace(directed_nodes_path=tmp_path/'bundle/nodes.parquet')
    with pytest.raises(ValueError,match='Regenerate'):
        release.assert_preparation_current(context)


def test_unregistered_state_cannot_silently_bypass_active_review(tmp_path):
    release=registry(tmp_path)
    inv=pd.DataFrame([dict(example='41',family='indep',state='.promotion-temporary',status='ready')])
    with pytest.raises(ValueError,match='not in the reviewed registry'):
        release.assert_hydraulics_current(inv)


def test_excluded_state_cannot_be_marked_ready(tmp_path):
    release=registry(tmp_path)
    record=release.lookup('41','indep','S016_unit_4')
    record['geometry_status']='excluded';record['requires_hydraulic_regeneration']=False
    inv=pd.DataFrame([dict(example='41',family='indep',state='S016_unit_4',status='ready')])
    with pytest.raises(ValueError,match='excluded state'):
        release.assert_hydraulics_current(inv)


def test_retained_geometry_preserves_identity_and_detects_source_changes(tmp_path):
    import geopandas as gpd
    from shapely.geometry import LineString, Point
    from network_variants.reviewed_geometry import digest
    lp=tmp_path/'links.gpkg';np=tmp_path/'nodes.gpkg';fp=tmp_path/'old.parquet'
    gpd.GeoDataFrame(dict(id_link=[1],id_us_node=[10],id_ds_node=[11]),
                     geometry=[LineString([(0,0),(1,0)])],crs=32631).to_file(lp)
    gpd.GeoDataFrame(dict(id_node=[10,11]),geometry=[Point(0,0),Point(1,0)],crs=32631).to_file(np)
    pd.DataFrame(dict(id_link=[1],id_us_node=[10],id_ds_node=[11],u=[3],v=[4])).to_parquet(fp)
    record=dict(example='41',family='indep',state='retained',geometry_status='accepted',
                links=str(lp),nodes=str(np),original_frame=str(fp),
                retained_source_checksums={str(p):digest(p) for p in [lp,np,fp]})
    p=tmp_path/'release.json';p.write_text(json.dumps(dict(release='test',records=[record])))
    release=ReviewedGeometry(p)
    links,nodes,_,_,_=release.load('41','indep','retained')
    assert nodes.parent_identity.tolist()==[3,4]
    assert links.u.tolist()==[3] and links.v.tolist()==[4]
    pd.DataFrame(dict(changed=[True])).to_parquet(fp)
    with pytest.raises(ValueError,match='source changed'):
        release.load('41','indep','retained')


def test_published_master_inventory_rejects_old_frame_with_new_labels(tmp_path):
    release=registry(tmp_path);release.hydraulic_regeneration_complete=True
    rec=release.lookup('41','indep','S016_unit_4')
    rec.update(master_products_published=True,original_frame=str(tmp_path/'current.parquet'),original_fingerprint='current')
    row=dict(example='41',family='indep',state='S016_unit_4',status='ready',geometry_release='reviewed-test',geometry_id='new-geometry',frame=str(tmp_path/'old.parquet'),fingerprint='current')
    with pytest.raises(ValueError,match='published master'):
        release.assert_hydraulics_current(pd.DataFrame([row]))
    row['frame']=rec['original_frame'];release.assert_hydraulics_current(pd.DataFrame([row]))


def test_published_master_preparation_rejects_mixed_width_sources(tmp_path):
    release=registry(tmp_path);release.hydraulic_regeneration_complete=True
    rec=release.lookup('41','indep','S016_unit_4')
    rec.update(master_products_published=True,links=str(tmp_path/'current.gpkg'),original_manifest=str(tmp_path/'hydraulic_context/manifest.json'))
    context=SimpleNamespace(directed_nodes_path=tmp_path/'old_nodes.gpkg',link_widths_path=tmp_path/'stale.gpkg',hydraulic_context_output_dir=tmp_path/'hydraulic_context',geometry_release='reviewed-test',geometry_id='new-geometry')
    with pytest.raises(ValueError,match='master width'):
        release.assert_preparation_current(context)
    context.link_widths_path=tmp_path/'current.gpkg';release.assert_preparation_current(context)
    frame=release.annotate_state_registry('41_indep__anchored',pd.DataFrame([dict(state_id='S016_unit_4')]))
    assert not frame.integrated_hydraulics_pending.iloc[0]
