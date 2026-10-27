import numpy as np
import pandas as pd
import pytest
from pilot import compose,adjacent_pairs,cunge


def fixture(connector=True):
    # Two parallel-edge closed units in series, with an optional connector.
    start=3 if connector else 2
    rows=[(0,0,1,100.,2.),(1,1,2,100.,2.),(2,0,2,250.,1.),(3,start,4,100.,1.),(4,4,5,100.,1.),(5,start,5,240.,2.)]
    if connector:rows.append((6,2,3,50.,3.))
    f=pd.DataFrame(rows,columns=['id_link','u','v','link_length_m','width'])
    f['id_us_node']=f.u;f['id_ds_node']=f.v
    f['depth_m']=2.;f['head_us']=10-f.u;f['head_ds']=10-f.v;f['slope']=(f.head_us-f.head_ds)/f.link_length_m
    p=cunge.parameters(cunge.freeze_grid(f,20),.04,beta=.5)
    blocks=[dict(example='test',state='A',kind='simple',source=0,sink=2,parent_ids=[0,1,2],child_source=10,child_sink=12),dict(example='test',state='B',kind='simple',source=start,sink=5,parent_ids=[3,4,5],child_source=13,child_sink=15)]
    sections={}
    for b in blocks:
        f=pd.DataFrame([dict(id_link=0,u=b['child_source'],v=b['child_sink'],id_us_node=b['child_source'],id_ds_node=b['child_sink'],link_length_m=180.,width=3.,depth_m=2.,head_us=10-b['source'],head_ds=10-b['sink'],slope=(b['sink']-b['source'])/180.)])
        sections[b['state']]=cunge.parameters(cunge.freeze_grid(f,20),.04,beta=.5)
    return p,blocks,sections

@pytest.mark.parametrize('connector',[True,False])
def test_boundary_connector_and_order(connector):
    p,b,s=fixture(connector);pairs=adjacent_pairs(p,b)
    assert len(pairs)==1 and pairs[0]['connector_ids']==([6] if connector else [])
    c,l=compose(p,b,s);reverse,_=compose(p,b[::-1],s)
    pd.testing.assert_frame_equal(c,reverse)
    assert len(l)==len(p) and l.parent_link_id.is_unique
    assert len(c)==(3 if connector else 2)
    if connector:pd.testing.assert_series_equal(c.set_index('id_link').loc[6,p.columns.drop('id_link')],p.set_index('id_link').loc[6,p.columns.drop('id_link')],check_names=False)


def test_overlap_rejected():
    p,b,s=fixture()
    with pytest.raises(ValueError,match='overlap'):compose(p,[b[0],b[0]],s)


def test_head_discontinuity_rejected():
    p,b,s=fixture();s['A'].loc[:,'head_ds']+=1
    with pytest.raises(AssertionError):compose(p,b,s)


def test_external_exchange_rejected():
    p,b,s=fixture();extra=p.iloc[[0]].copy();extra['id_link']=99;extra['u']=extra['id_us_node']=1;extra['v']=extra['id_ds_node']=4;extra['head_us']=9.;extra['head_ds']=6.
    with pytest.raises(ValueError,match='External'):compose(pd.concat([p,extra],ignore_index=True),b,s)
