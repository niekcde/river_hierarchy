import numpy as np
import pandas as pd
import drop_exposure_check as m


def test_seven_input_fit_preserves_settings_and_has_no_exposure_dependency():
    rng=np.random.default_rng(17)
    d=pd.DataFrame(rng.uniform(.1,2,(60,7)),columns=m.COLUMNS)
    d['B_exposure']=1.;d['example']=np.repeat(['01','02','03'],20)
    d[m.forest.TARGET]=d.domain_size/5-d.log1p__B_ratio_width_mean/7
    params=dict(m.forest.candidates()[0],n_estimators=8)
    tr,te=d.iloc[:40],d.iloc[40:].copy()
    a=m.fit(tr,params,1);b=m.forest.fit(tr,params,1)
    assert a['estimator'].n_estimators==8 and len(a['columns'])==7
    p=m.forest.predict(a,te[m.COLUMNS])
    np.testing.assert_allclose(p,m.forest.predict(b,te))
    te.B_exposure=999;te[m.forest.TARGET]=999
    np.testing.assert_allclose(p,m.forest.predict(a,te))
