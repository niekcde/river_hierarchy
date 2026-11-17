from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from network_variants.local_head_reconstruction import (
    LocalHeadContract,
    reconstruct_local_heads,
)


def fixture():
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
            id_link=[1, 20, 4],
            u=[0, 1, 3],
            v=[1, 3, 4],
            link_length_m=[100.0, 200.0, 100.0],
        )
    )
    contract = LocalHeadContract(frozenset([2, 3]), frozenset([1, 3]), {1: 1, 4: 4})
    return child, parent, pd.Series(0.0001, index=[1, 2, 3, 4]), contract


def test_exact_locality_and_input_immutability():
    f, p, floors, c = fixture()
    before = f.copy(deep=True)
    out, a = reconstruct_local_heads(f, p, floors, c)
    np.testing.assert_array_equal(out.head_us, [4.0, 3.0, 1.0])
    np.testing.assert_array_equal(out.head_ds, [3.0, 1.0, 0.0])
    pd.testing.assert_frame_equal(f, before)
    assert a["primary"]["strict_fixed_feasible"]


def test_wrong_boundary_is_rejected():
    f, p, s, c = fixture()
    with pytest.raises(ValueError, match="Declared boundaries"):
        reconstruct_local_heads(f, p, s, replace(c, boundary_nodes=frozenset([1])))


def test_missing_lineage_is_rejected():
    f, p, s, c = fixture()
    with pytest.raises(ValueError, match="cover every exterior"):
        reconstruct_local_heads(f, p, s, replace(c, exterior_parent_by_child={1: 1}))


def test_infeasibility_never_silently_releases_boundary():
    f, p, s, c = fixture()
    f.loc[f.id_link.eq(20), "link_length_m"] = 25000.0
    with pytest.raises(ValueError, match="explicit review"):
        reconstruct_local_heads(f, p, s, c)
    out, a = reconstruct_local_heads(
        f, p, s, replace(c, flexible_nodes=frozenset([1, 3]))
    )
    np.testing.assert_allclose(out.head_us, [4.0, 3.25, 0.75], atol=1e-8)
    assert not a["primary"]["strict_fixed_feasible"]


def test_unapproved_direction_is_rejected():
    f, p, s, c = fixture()
    f.loc[0, ["u", "v"]] = [1, 0]
    with pytest.raises(ValueError, match="direction change"):
        reconstruct_local_heads(f, p, s, c)
