"""Legacy example wrapper for the generic smoke workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "rivgraph_centerline" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rivgraph_centerline.smoke_workflow import run_all_for_run  # noqa: E402


DEFAULT_MASK = Path(
    "/Users/6256481/Desktop/PhD_icloud/projecten/river_hierarchy/"
    "niek_review_package/water_masks_sarl/sarl_river_07.tif"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "rivgraph_centerline" / "outputs" / "smoke_tests" / "sarl_river_07"


def main() -> None:
    args = parse_args()
    payload = run_all_for_run(
        name=args.name,
        source_mask=args.mask.expanduser().resolve(),
        exit_sides=args.exit_sides,
        run_dir=args.output_dir.expanduser().resolve(),
        threshold=args.threshold,
        dst_crs=args.dst_crs,
        force_prepare=args.force_prepare,
        verbose=not args.quiet,
    )
    print(json.dumps(payload["rivgraph"], indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="sarl_river_07", help="Run name used in output file names.")
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK, help="Source river mask GeoTIFF.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for prepared mask and RivGraph products.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Water threshold for binary mask creation.")
    parser.add_argument("--dst-crs", default=None, help="Projected CRS override.")
    parser.add_argument("--exit-sides", default="NS", help="Two-letter RivGraph river exit sides, upstream first.")
    parser.add_argument("--force-prepare", action="store_true", help="Rebuild the prepared mask even if it exists.")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose RivGraph logging.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
