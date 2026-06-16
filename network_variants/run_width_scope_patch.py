from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from network_variants.variant_generation import (
    DEFAULT_FOOTPRINT_BUFFER_SCALE,
    DEFAULT_FOOTPRINT_WIDTH_FIELD,
    DEFAULT_MIN_TRANSECT_PIXELS,
    DEFAULT_TRANSECT_SCALE,
    refresh_variant_width_scope_and_families,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild origin_mask.tif and width-family outputs for an existing variant state "
            "without rerunning collapse/RivGraph."
        )
    )
    parser.add_argument("variant_output_dir", help="Path to an existing variant output directory (the folder containing summary/variant_manifest.json).")
    parser.add_argument("--reviewed-links", default=None, help="Optional override for the parent reviewed links path.")
    parser.add_argument("--reviewed-nodes", default=None, help="Optional override for the parent reviewed nodes path.")
    parser.add_argument("--cleaned-mask", default=None, help="Optional override for the parent cleaned mask path.")
    parser.add_argument("--wet-reference-mask", default=None, help="Optional override for the wet-reference mask path.")
    parser.add_argument("--preferred-width-field", default=DEFAULT_FOOTPRINT_WIDTH_FIELD, help="Preferred reviewed-link width field for scope buffering.")
    parser.add_argument("--footprint-buffer-scale", type=float, default=DEFAULT_FOOTPRINT_BUFFER_SCALE, help="Multiplier applied to footprint width / 2 buffering.")
    parser.add_argument("--transect-scale", type=float, default=DEFAULT_TRANSECT_SCALE, help="Multiplier for transect half-length relative to local total width.")
    parser.add_argument("--min-transect-pixels", type=float, default=DEFAULT_MIN_TRANSECT_PIXELS, help="Minimum transect half-length in raster pixels.")
    parser.add_argument("--max-path-cutoff", type=int, default=100, help="Unit-detection max simple-path cutoff when rehydrating units.")
    parser.add_argument("--max-paths", type=int, default=5000, help="Unit-detection maximum number of simple paths when rehydrating units.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    results = refresh_variant_width_scope_and_families(
        variant_output_dir=args.variant_output_dir,
        reviewed_links_path=args.reviewed_links,
        reviewed_nodes_path=args.reviewed_nodes,
        cleaned_mask_path=args.cleaned_mask,
        wet_reference_mask_path=args.wet_reference_mask,
        preferred_width_field=args.preferred_width_field,
        footprint_buffer_scale=args.footprint_buffer_scale,
        transect_scale=args.transect_scale,
        min_transect_pixels=args.min_transect_pixels,
        max_path_cutoff=args.max_path_cutoff,
        max_paths=args.max_paths,
    )

    print(f"Patched two-bank width inputs for {Path(args.variant_output_dir).resolve()}")
    print(f"origin_mask: {results['origin_mask']}")
    print(f"link_width_families: {results['link_width_families']}")
    print(f"link_width_samples: {results['link_width_samples']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
