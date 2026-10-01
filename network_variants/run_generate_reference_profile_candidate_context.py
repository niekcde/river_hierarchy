from __future__ import annotations

"""Write a non-destructive base candidate context from a frozen WSE profile."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network_variants.reference_profile_candidate import generate_reference_profile_candidate_context


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a candidate base hydraulic context from a frozen reference profile. "
            "Current hydraulic contexts, resistance results, and RAPID products are not modified."
        )
    )
    parser.add_argument("experiment_dir", help="Independent anchored experiment root.")
    parser.add_argument("profile_dir", help="Immutable directory from RAPID/run_generate_reference_profile.py.")
    parser.add_argument("output_dir", help="New, empty candidate-context directory.")
    args = parser.parse_args(argv)
    result = generate_reference_profile_candidate_context(args.experiment_dir, args.profile_dir, args.output_dir)
    print(f"Candidate output: {result.output_dir}")
    print(f"Links/nodes: {result.summary['n_links']}/{result.summary['n_nodes']}")
    print(f"Solved domains: {result.summary['n_domains_solved']}/{result.summary['n_domains']}")
    print(f"Direct-profile links: {result.summary['n_profile_direct_links']}")
    print(
        "Maximum solved-domain path residual [m]: "
        f"{result.summary['max_solved_domain_path_residual_m']:.8g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
