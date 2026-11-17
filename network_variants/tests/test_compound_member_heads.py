import numpy as np
import pandas as pd
import pytest
from network_variants.compound_member_heads import (
    member_contract,
    reconstruct_compound_member_heads,
)
from network_variants.run_compound_member_heads import lp_feasible, verify


def inputs(length=200.0):
    parent = pd.DataFrame(
        dict(
            id_link=[1, 2, 3, 4],
            u=[0, 1, 2, 3],
            v=[1, 2, 3, 4],
            link_length_m=[100.0] * 4,
            head_us=[4.0, 3.0, 2.0, 1.0],
            head_ds=[3.0, 2.0, 1.0, 0.0],
            slope=[0.01] * 4,
        )
    )
    child = pd.DataFrame(
        dict(
            id_link=[10, 20, 40],
            u=[0, 1, 3],
            v=[1, 3, 4],
            link_length_m=[100.0, length, 100.0],
        )
    )
    selection = dict(
        selected_parent_links=[2, 3], boundary_nodes=[1, 3], valid=True, target_ok=True
    )
    floors = pd.Series(0.0001, index=[1, 2, 3, 4])
    return child, parent, floors, selection


def test_workflow_matches_renumbered_exterior_and_preserves_it():
    f, p, s, c = inputs()
    result, audit, contract = reconstruct_compound_member_heads(f, p, s, c)
    assert contract.exterior_parent_by_child == {10: 1, 40: 4}
    np.testing.assert_allclose(result.slope, [0.01, 0.01, 0.01])
    assert audit["primary"]["strict_fixed_feasible"]


def test_workflow_requires_explicit_flexibility_policy():
    f, p, s, c = inputs(25000.0)
    with pytest.raises(ValueError, match="explicit review"):
        reconstruct_compound_member_heads(f, p, s, c)
    result, audit, contract = reconstruct_compound_member_heads(
        f, p, s, c, allow_boundary_flexibility=True
    )
    assert set(audit["primary"]["candidates"]) == {1, 3}
    np.testing.assert_allclose(result.head_us, [4.0, 3.25, 0.75])
    _, qa = verify(result, result, p, s, contract, audit)
    assert not qa["strict_lp_feasible"] and qa["final_lp_feasible"]


def test_no_silent_path_subdivision_or_missing_coverage():
    f, p, s, c = inputs()
    with pytest.raises(ValueError, match="coverage"):
        member_contract(f, p, c, exterior_paths=[])
    paths = [
        dict(parent_link=1, child_links=[10, 11], path_unique=True),
        dict(parent_link=4, child_links=[40], path_unique=True),
    ]
    with pytest.raises(ValueError, match="path needs review"):
        member_contract(f, p, c, exterior_paths=paths)


def test_independent_lp_detects_infeasible_path_with_internal_node():
    f = pd.DataFrame(dict(id_link=[1, 2], u=[0, 1], v=[1, 2]))
    assert lp_feasible(f, {0: 1.0, 2: 0.0}, [0.4, 0.6])
    assert not lp_feasible(f, {0: 1.0, 2: 0.0}, [0.6, 0.6])


def test_independent_verifier_detects_unreleased_exterior_motion():
    f, p, s, c = inputs()
    result, audit, contract = reconstruct_compound_member_heads(f, p, s, c)
    result.loc[result.u.eq(0), "head_us"] += 0.01
    result["slope"] = (result.head_us - result.head_ds) / result.link_length_m
    with pytest.raises(ValueError, match="Exterior head moved"):
        verify(result, result, p, s, contract, audit)


def test_one_eligible_endpoint_can_resolve_exterior_floor_conflict():
    f, p, s, c = inputs()
    # Exterior reach 1 is lengthened while its upstream node 0 must stay fixed.
    f.loc[f.id_link.eq(10), "link_length_m"] = 11000.0
    paths = [
        dict(parent_link=1, child_links=[10], path_unique=True),
        dict(parent_link=4, child_links=[40], path_unique=True),
    ]
    result, audit, contract = reconstruct_compound_member_heads(
        f, p, s, c, exterior_paths=paths, allow_boundary_flexibility=True
    )
    assert audit["primary"]["candidates"] == [1]
    np.testing.assert_allclose(result.head_us, [4.0, 2.9, 1.0], atol=1e-8)
    _, qa = verify(result, result, p, s, contract, audit)
    assert qa["final_lp_feasible"]


def test_conflict_with_no_eligible_boundary_still_blocks():
    f, p, s, c = inputs()
    # The lengthened exterior reach is remote from the selected member [3].
    f = pd.DataFrame(
        dict(
            id_link=[10, 20, 30, 40],
            u=[0, 1, 2, 3],
            v=[1, 2, 3, 4],
            link_length_m=[11000.0, 100.0, 100.0, 100.0],
        )
    )
    c = dict(
        selected_parent_links=[3], boundary_nodes=[2, 3], valid=True, target_ok=True
    )
    paths = [
        dict(parent_link=i, child_links=[i * 10], path_unique=True) for i in [1, 2, 4]
    ]
    with pytest.raises(ValueError, match="explicit review"):
        reconstruct_compound_member_heads(
            f, p, s, c, exterior_paths=paths, allow_boundary_flexibility=True
        )
