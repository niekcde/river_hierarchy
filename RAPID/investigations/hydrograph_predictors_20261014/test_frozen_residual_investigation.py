import numpy as np
import pandas as pd
import frozen_residual_investigation as m


def test_inventory_excludes_responses_and_frozen_features():
    keys=[f'B_{kind}_{metric}' for metric in m.engine.METRICS for kind in ['parent','ratio','delta']]
    d=pd.DataFrame(columns=keys+['predicted_hours','centroid_shift_hours','duration80_change_pct'])
    c=m.inventory(d)
    assert c.feature.is_unique
    assert not set(c.feature)&{'predicted_hours','centroid_shift_hours','duration80_change_pct',*m.COLUMNS}
    assert 'B_delta_total_length' in set(c.feature)


def test_correlation_and_partial_remove_existing_rank_information():
    assert np.isnan(m.correlation([1,1],[2,3],[1,1]))
    rng=np.random.default_rng(4);d=pd.DataFrame(rng.uniform(0,1,(90,7)),columns=m.COLUMNS)
    d['example']=np.repeat(['01','02','03'],30);d['edit_type']='Simple'
    d['candidate']=d[m.COLUMNS[0]]*10
    d['error']=rng.normal(size=len(d))
    assert np.isnan(m.rank_partial(d,'candidate','error'))
