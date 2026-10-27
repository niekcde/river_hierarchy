import pandas as pd
from combination_feasibility_inventory import closed_regions, adjacent


def frame(edges):
    return pd.DataFrame([
        dict(id_link=i, id_us_node=u, id_ds_node=v, link_length_m=100.,
             reference_flow_fraction=1.)
        for i, (u, v) in enumerate(edges)
    ])


def test_parallel_closed_subregions_are_not_adjacent():
    g, regions = closed_regions(frame([(1,2),(2,3),(2,3),(3,6),
                                       (1,4),(4,5),(4,5),(5,6)]))
    r = {(x["source"],x["sink"]):x for x in regions}
    assert set(r) == {(1,6),(2,3),(4,5)}
    assert all(x["primitive"] for x in regions)
    assert not adjacent(g,r[2,3],r[4,5])
    assert not r[2,3]["ids"] & r[4,5]["ids"]


def test_serial_envelope_is_not_an_additional_primitive():
    g, regions = closed_regions(frame([(0,1),(0,1),(1,2),(2,3),(2,3)]))
    r = {(x["source"],x["sink"]):x for x in regions}
    assert set(r) == {(0,1),(0,3),(2,3)}
    assert not r[0,3]["primitive"]
    assert adjacent(g,r[0,1],r[2,3])


def test_external_branch_rejects_apparent_subunit():
    _, regions = closed_regions(frame([(0,1),(0,2),(1,3),(2,3),(1,4),(4,5),(3,5)]))
    # 0 bifurcates and 3 is a confluence, but 1->4 leaves the apparent unit.
    assert (0,3) not in {(r["source"],r["sink"]) for r in regions}
    assert (0,5) in {(r["source"],r["sink"]) for r in regions}
