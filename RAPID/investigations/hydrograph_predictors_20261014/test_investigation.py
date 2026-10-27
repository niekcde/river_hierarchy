"""Meaningful checks for alias cardinality and diagnostic/target separation."""
import numpy as np
import pandas as pd
import pytest
from build import alias_lookup, crest_diagnostic, pair_features


def test_aliases_resolve_once_and_ambiguity_fails():
    rows=pd.DataFrame([dict(example='46',state='S139_unit_13',aliases='S139_unit_13 | CB006_bubble_101')])
    lookup=alias_lookup(rows)
    assert lookup['46','CB006_bubble_101']=='S139_unit_13'
    with pytest.raises(ValueError,match='Ambiguous alias'):
        alias_lookup(pd.concat([rows,pd.DataFrame([dict(example='46',state='other',aliases='CB006_bubble_101')])]))


def test_crest_switch_and_unresolved_are_distinct():
    t=np.arange(0,24*3600,60.)
    first=np.exp(-((t-6*3600)/(3600))**2)
    second=np.exp(-((t-15*3600)/(3600))**2)
    refs=np.array([6,15])*3600
    assert crest_diagnostic(t,100+9.99*first+10*second,refs,100)['crest_switch_flag'] is False
    assert crest_diagnostic(t,100+10*first+9.99*second,refs,100)['crest_switch_flag'] is True
    assert crest_diagnostic(t,100+10*first,refs,100)['crest_switch_flag'] is None
    # Translation outside the match radius must not be mislabeled as a confirmed switch.
    shifted=100+10*np.exp(-((t-12*3600)/3600)**2)
    assert crest_diagnostic(t,shifted,refs,100)['status']=='unresolved'


def test_ratio_and_undefined_conditional_mean():
    f=pair_features(dict(route_length_mean=10.,dominance_mean=.9,route_length_sd=0.),
                    dict(route_length_mean=8.,dominance_mean=np.nan,route_length_sd=0.),'domain_')
    assert f['domain_ratio_route_length_mean']==.8
    assert np.isnan(f['domain_delta_dominance_mean'])
    assert np.isnan(f['domain_ratio_route_length_sd'])
