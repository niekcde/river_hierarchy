import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString
from rapid_tools.centerline_batch import hydraulic_frame


def test_collapsed_reach_inherits_current_boundary_drop_and_reference_depth():
    base=pd.DataFrame(dict(id_link=[1,2],u=[0,1],v=[1,2],head_us=[100.,98.],head_ds=[98.,90.],width=[10.,20.],depth_m=3*(np.array([10.,20.])/10)**.65))
    links=gpd.GeoDataFrame(dict(id_link=[0],id_us_node=[4],id_ds_node=[5],wid_adj_routing=[10.]),geometry=[LineString([(0,0),(100,0)])],crs='EPSG:32631')
    frame=hydraulic_frame(base,links,dict(reference_width_m=10,reference_depth_m=3))
    assert frame.slope.iloc[0]==.1
    assert frame.depth_m.iloc[0]==3
    assert frame.link_length_m.iloc[0]==100
    base.loc[0,'depth_m']=4
    with pytest.raises(ValueError,match='Reference depth'):hydraulic_frame(base,links,dict(reference_width_m=10,reference_depth_m=3))

@pytest.mark.parametrize('only', [False,True])
def test_cli_includes_reference_by_default_and_catchup_skips_networks(tmp_path,monkeypatch,only):
    import json,sys
    import rapid_tools.centerline_batch as module
    import rapid_tools.master_workflow as network
    config=tmp_path/'RAPID/config';config.mkdir(parents=True)
    (config/'production_run.json').write_text(json.dumps({'manifest':str(tmp_path/'source.json')}))
    monkeypatch.setattr(module,'__file__',str(tmp_path/'RAPID/src/rapid_tools/centerline_batch.py'))
    monkeypatch.setattr(module,'load_run_manifest',lambda path:({'examples':['02'],'run_id':'original','outputs':{'prep':'rapid_runs/original/prep'}},{}))
    calls=[]
    monkeypatch.setattr(network,'cli',lambda args:calls.append(('network',args)))
    monkeypatch.setattr(module,'prepare',lambda path,examples:calls.append(('reference',examples)))
    monkeypatch.setattr(sys,'argv',['batch','prep','--examples','02']+(['--centerline-only'] if only else []))
    module.cli()
    assert [c[0] for c in calls]==(['reference'] if only else ['network','reference'])
