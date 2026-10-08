import pandas as pd
import pytest
from rapid_tools.comparison_population import canonical_comparisons,plotted_comparisons


def rows():
    common=dict(example_id='05',fingerprint='same',matched_base_link_ids='1,2',child_link_ids='3',
                technical_alias=False,Lambda_matched=.5,interpolated_lag_shift_pct=-1.)
    return pd.DataFrame([
        dict(common,mode='island',state_id='island1',target_id='a',plot_category='island collapse'),
        dict(common,mode='indep',state_id='unit1',target_id=1,plot_category='nested unit'),
        dict(common,mode='island',state_id='island2',target_id='b',plot_category='island collapse',child_link_ids='4'),
    ])


def test_identity_not_coordinates_and_topology_preference():
    canonical,aliases=canonical_comparisons(rows())
    assert canonical.state_id.tolist()==['unit1','island2']
    assert canonical.plot_category.tolist()==['nested unit','island-only collapse']
    assert canonical.selection_count.tolist()==[2,1]
    assert len(aliases)==3 and aliases.is_retained.sum()==2
    shuffled,_=canonical_comparisons(rows().iloc[::-1])
    pd.testing.assert_frame_equal(canonical,shuffled)


def test_conflicting_measurements_cannot_be_silently_deduplicated():
    data=rows();data.loc[0,'Lambda_matched']=.6
    with pytest.raises(ValueError,match='Conflicting duplicate'):canonical_comparisons(data)


def test_edge_compounds_are_retained_by_default_and_shape_is_not_required():
    data=rows().assign(compound_downstream_order=1,
        len_fraction_width_weighted_log_lambda=.1,
        interpolated_peak_shift_hours=-.2,peak_interpolation_status='ok',
        event_duration_change_pct=float('nan'),peak_excess_change_pct=float('nan'))
    selected,_=plotted_comparisons(data)
    assert len(selected)==2
    assert plotted_comparisons(data,exclude_edge_compounds=True)[0].empty
    assert plotted_comparisons(data,require_shape_metrics=True)[0].empty
