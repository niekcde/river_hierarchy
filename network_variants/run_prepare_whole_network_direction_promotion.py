"""Recover review provenance and approve a geometry-preserving direction fix.

This is an intentionally narrow preparatory step.  It never replaces an
experiment root.  It makes the evidence needed for a later staged rebuild
durable beside the raw RivGraph run:

* restores a missing permanent width-review archive from the completed base;
* fingerprints that historical archive against the original raw graph; and
* writes an approval record proving the supplied corrected graph only reverses
  stable link directions, without changing mask, geometry, links, or nodes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network_variants.width_review_archive import recover_base_width_review_archive, review_archive_dir
from network_variants.width_review_fingerprint import write_direction_only_review_continuity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an auditable, whole-network direction-correction promotion. "
            "Dry-run by default; --execute writes only review/archive provenance, "
            "not experiment outputs."
        )
    )
    parser.add_argument("base_variant_dir", help="Existing S000_base/variant directory.")
    parser.add_argument("--source-links", required=True, help="Original raw <run>/rivgraph/<id>_links.gpkg.")
    parser.add_argument("--source-nodes", required=True, help="Original raw <run>/rivgraph/<id>_nodes.gpkg.")
    parser.add_argument("--cleaned-mask", required=True, help="Original cleaned base mask.")
    parser.add_argument("--corrected-links", required=True, help="Validated corrected raw links GPKG.")
    parser.add_argument("--corrected-nodes", required=True, help="Validated corrected raw nodes GPKG.")
    parser.add_argument("--prior-exit-sides", required=True)
    parser.add_argument("--corrected-exit-sides", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    source_links = Path(args.source_links).expanduser().resolve()
    archive = review_archive_dir(source_links)
    plan = {
        "base_variant_dir": str(Path(args.base_variant_dir).expanduser().resolve()),
        "source_links": str(source_links),
        "source_nodes": str(Path(args.source_nodes).expanduser().resolve()),
        "cleaned_mask": str(Path(args.cleaned_mask).expanduser().resolve()),
        "corrected_links": str(Path(args.corrected_links).expanduser().resolve()),
        "corrected_nodes": str(Path(args.corrected_nodes).expanduser().resolve()),
        "prior_exit_sides": str(args.prior_exit_sides),
        "corrected_exit_sides": str(args.corrected_exit_sides),
        "review_archive": str(archive),
        "continuity_record": str(archive / "direction_only_continuity.json"),
        "operations": [
            "recover missing permanent review archive from completed base product",
            "fingerprint the historical raw mask/network",
            "validate corrected raw graph has only directed-endpoint reversals",
            "write immutable promotion-continuity record",
        ],
        "does_not": [
            "replace raw RivGraph inputs",
            "replace, rename, or delete experiment folders",
            "modify mask or measured width products",
        ],
    }
    if not args.execute:
        print(json.dumps({"execute_requested": False, **plan}, indent=2, sort_keys=True))
        return 0

    recovered = None
    if not (archive / "base_width_review_manifest.json").exists():
        recovered = recover_base_width_review_archive(
            args.base_variant_dir,
            source_links_path=source_links,
            source_nodes_path=args.source_nodes,
            cleaned_mask_path=args.cleaned_mask,
        )
    continuity = write_direction_only_review_continuity(
        archive,
        mask_path=args.cleaned_mask,
        links_path=args.corrected_links,
        nodes_path=args.corrected_nodes,
        prior_exit_sides=args.prior_exit_sides,
        corrected_exit_sides=args.corrected_exit_sides,
    )
    print(json.dumps({"execute_requested": True, **plan, "recovered_archive": recovered, "continuity": continuity}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
