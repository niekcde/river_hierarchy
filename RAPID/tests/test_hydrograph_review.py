from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rapid_tools import hydrograph_review as review


def test_utc_restoration_checks_forcing_before_overlay(tmp_path,monkeypatch):
    ex=tmp_path/'03';ex.mkdir();original=tmp_path/'old.csv'
    saved=pd.DataFrame(dict(time_seconds=[0.,60.,120.],q_inflow_cms=[10.,20.,10.],time_utc=pd.date_range('2024-01-01',periods=3,freq='min',tz='UTC')));saved.to_csv(original,index=False)
    (ex/'forcing_provenance.json').write_text(json.dumps(dict(source='saved_actual_event',path=str(original))))
    monkeypatch.setattr(review,'DEFAULT_REFRESH_ROOT',tmp_path)
    h=saved.copy();h.time_utc=pd.date_range('2020-01-01',periods=3,freq='min',tz='UTC')
    t,_=review.event_clock('03',h);assert t.iloc[0].year==2024
    h.loc[1,'q_inflow_cms']=21
    with pytest.raises(AssertionError):review.event_clock('03',h)


def test_native_gauge_event_keeps_its_utc(tmp_path,monkeypatch):
    ex=tmp_path/'13';ex.mkdir();(ex/'forcing_provenance.json').write_text(json.dumps(dict(source='recorded_gauge_event_window')))
    monkeypatch.setattr(review,'DEFAULT_REFRESH_ROOT',tmp_path)
    h=pd.DataFrame(dict(time_utc=['2024-10-01T00:00:00Z']))
    t,_=review.event_clock('13',h);assert t.iloc[0].year==2024
