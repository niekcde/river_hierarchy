from __future__ import annotations

import numpy as np

from rivgraph_compat import rivgraph_centerline_compat, sanitize_centerline_link_fields


def test_sanitizer_retains_only_per_link_sequences() -> None:
    links = {
        "id": [0, 1, 2],
        "idx": [[1], [2], [3]],
        "conn": [[0, 1], [1, 2], [2, 3]],
        "len": [1.0, 2.0, 3.0],
        "n_networks": 1,
        "_ids_finalized": False,
        "_ids_finalization_method": "topology not finalized",
        "future_scalar_metadata": 7,
        "future_short_sequence": [1, 2],
    }

    sanitize_centerline_link_fields(links)

    assert set(links) == {"id", "idx", "conn", "len", "n_networks"}


def test_context_sanitizes_temporary_links_and_restores_rivgraph_function(monkeypatch) -> None:
    import rivgraph.ln_utils as lnu

    def fake_widths(*_args, **_kwargs):
        return {
            "id": [0, 1],
            "idx": [[10], [20]],
            "conn": [[0, 1], [1, 2]],
            "n_networks": 1,
            "_ids_finalized": False,
            "_ids_finalization_method": "topology not finalized",
        }

    monkeypatch.setattr(lnu, "link_widths_and_lengths", fake_widths)
    original_at_entry = lnu.link_widths_and_lengths

    with rivgraph_centerline_compat():
        assert lnu.link_widths_and_lengths is not original_at_entry
        links = lnu.link_widths_and_lengths(None, None)
        assert set(links) == {"id", "idx", "conn", "n_networks"}

    assert lnu.link_widths_and_lengths is original_at_entry


def test_metadata_compatibility_does_not_change_centerline_geometry(monkeypatch) -> None:
    import rivgraph.rivers.river_utils as ru

    mask = np.zeros((31, 61), dtype=bool)
    mask[11:20, :] = True
    original_skel_to_graph = ru.m2g.skel_to_graph

    def graph_without_metadata(*args, **kwargs):
        links, nodes = original_skel_to_graph(*args, **kwargs)
        links.pop("_ids_finalized", None)
        links.pop("_ids_finalization_method", None)
        return links, nodes

    monkeypatch.setattr(ru.m2g, "skel_to_graph", graph_without_metadata)
    expected_coords, expected_widths = ru.mask_to_centerline(mask, "we")

    monkeypatch.setattr(ru.m2g, "skel_to_graph", original_skel_to_graph)
    with rivgraph_centerline_compat():
        actual_coords, actual_widths = ru.mask_to_centerline(mask, "we")

    np.testing.assert_array_equal(actual_coords, expected_coords)
    np.testing.assert_allclose(actual_widths, expected_widths)
