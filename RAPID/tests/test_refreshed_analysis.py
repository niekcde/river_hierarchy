from pathlib import Path
import sys,json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rapid_tools.refreshed_analysis import domain_geometry,plateau_event_duration,_verify_completed


def test_flat_outlet_under_varying_forcing_is_rejected(tmp_path):
    from rapid_tools.refreshed_analysis import _wave_metrics
    pd.DataFrame(dict(time_seconds=[0,1,2],q_outlet_cms=[0,0,0],
        q_inflow_cms=[10,20,10])).to_csv(tmp_path/'outlet_hydrograph.csv',index=False)
    with pytest.raises(ValueError,match='flat outlet'):_wave_metrics(tmp_path)


def networks():
    parent=pd.DataFrame(dict(id_link=range(1,10),id_us_node=[0,1,1,2,2,3,4,5,6],
        id_ds_node=[1,2,6,3,4,5,5,6,7],width=[10,2,8,1,1,1,1,2,10],
        link_length_m=[100,100,100,200,100,100,100,100,100]))
    child=pd.concat([parent[~parent.id_link.isin([4,5,6,7])],pd.DataFrame([dict(id_link=10,id_us_node=2,id_ds_node=5,width=2,link_length_m=200)])],ignore_index=True)
    return parent,child


def test_geometry_uses_actual_domain_and_width_flow_share():
    p,c=networks()
    local=domain_geometry(p,c,{4,5,6,7},{10})
    assert local['matched_len_frac']==pytest.approx(.5)
    assert local['width_fraction_of_total_raw']==pytest.approx(.2)
    assert local['child_parent_len_frac']==pytest.approx(.4)
    expanded=domain_geometry(p,c,{2,3,4,5,6,7,8},{2,3,8,10})
    assert expanded['matched_len_frac']==pytest.approx(.8)
    assert expanded['width_fraction_of_total_raw']==pytest.approx(1.)


def test_unclosed_or_missing_domain_is_rejected():
    p,c=networks()
    with pytest.raises(ValueError,match='external flow'):domain_geometry(p,c,{4,6},{10})
    with pytest.raises(KeyError):domain_geometry(p,c,{999},{10})


def test_duration_preserves_translation_and_detects_late_peak():
    t=np.arange(0.,101.,.01)
    pulse=lambda center,amplitude:amplitude*np.maximum(1-abs(t-center)/5,0)
    q=pulse(20,100)
    first=plateau_event_duration(t,q,0,0,1,1,1)
    shifted=plateau_event_duration(t,pulse(40,100),0,0,1,1,1)
    assert first==pytest.approx(9.9)
    assert shifted==pytest.approx(first)
    assert plateau_event_duration(t,q+pulse(60,2),0,0,1,1,1)>40
    assert plateau_event_duration(t,q+pulse(60,.5),0,0,1,1,1)==pytest.approx(first)
    with pytest.raises(ValueError,match='Censored'):plateau_event_duration(t[:2100],q[:2100],0,0,1,1,1)


def test_mixed_policy_is_rejected(tmp_path):
    d=tmp_path/'41'/'fingerprint';d.mkdir(parents=True);q=d/'Qout.nc.gz';q.write_bytes(b'dummy')
    (d/'completed.json').write_text(json.dumps(dict(status='routed',qout=str(q))))
    (d/'policy_manifest.json').write_text(json.dumps(dict(policy_version='old')))
    with pytest.raises(ValueError,match='Mixed hydraulic policy'):
        _verify_completed(tmp_path,'41',SimpleNamespace(fingerprint='fingerprint',state='child'))


@pytest.mark.parametrize('target',[None,200.])
def test_zero_source_link_id_cannot_become_routing_sentinel(target):
    import geopandas as gpd
    from shapely.geometry import LineString,Point
    from rapid_tools.prep import split_links_into_subreaches
    from rapid_tools.engine import net_mat
    from scipy.sparse import identity
    from scipy.sparse.linalg import factorized
    links=gpd.GeoDataFrame(dict(id_link=[1,0,2],id_us_node=[1,2,3],id_ds_node=[2,3,4],
        link_length_m=[100.,100.,100.]),geometry=[LineString([(x,0),(x+100,0)]) for x in [0,100,200]],crs='EPSG:3857')
    nodes=gpd.GeoDataFrame(dict(id_node=[1,2,3,4]),geometry=[Point(x,0) for x in [0,100,200,300]],crs=links.crs)
    routed,_=split_links_into_subreaches(links,nodes,target_length_m=target)
    assert routed.reach_id.ne(0).all() and routed.reach_id.is_unique
    assert 0 in set(routed.parent_link_id)
    routed=routed.sort_values('id_us_node');ids=routed.reach_id.tolist();mapping={rid:i for i,rid in enumerate(ids)}
    matrix=net_mat([[ids[1]],[ids[2]],[]],[1,1,0],[[1.],[1.],[]],mapping,np.array(ids),mapping)
    steady=factorized(identity(3,format='csc')-matrix)(np.array([7.,0.,0.]))
    np.testing.assert_allclose(steady,[7.,7.,7.])


def test_interpolated_peak_recovers_subsample_vertex_and_flags_ties():
    from rapid_tools.refreshed_analysis import interpolated_peak
    t=np.arange(5.)*10;q=100-(t-23.)**2/100
    value,status=interpolated_peak(t,q)
    assert status=='ok' and value==pytest.approx(23.)
    assert interpolated_peak(t,[0,1,2,2,0])[1]=='tied_peak'
    assert interpolated_peak(t,[4,3,2,1,0])[1]=='boundary'


def test_peak_excess_uses_shared_parent_high_plateau():
    from rapid_tools.refreshed_analysis import paired_responses
    keys=['peak_time_seconds','rise95_time_seconds','plateau995_center_seconds','centroid50_seconds','centroid75_seconds','peak_window_center95_seconds','peak_window_centroid95_seconds','peak_window_width95_seconds','rise_time_seconds','fall_time_50_seconds','fall_time_10_seconds','e_folding_time_seconds']
    p={k:100. for k in keys};p.update(lag_seconds=100.,peak_discharge=30.,peak_excess=10.,peak_reference_cms=20.,event_duration_seconds=100.,event_duration_status='ok',interpolated_peak_seconds=100.,interpolated_peak_status='ok',inflow_peak_seconds=0.)
    c=dict(p,peak_discharge=31.,peak_reference_cms=25.,peak_excess=6.)
    assert paired_responses(p,c)['peak_excess_change_pct']==pytest.approx(10.)
