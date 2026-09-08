from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network_variants.island_experiment import generate_island_hydraulic_contexts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate hydraulic contexts for a strict island analysis manifest. "
            "Independent aliases are referenced and topology-review exclusions are skipped."
        )
    )
    parser.add_argument("analysis_manifest", help="island_analysis_manifest_strict.csv")
    parser.add_argument("--base-link-context", required=True)
    parser.add_argument("--base-node-context", required=True)
    args = parser.parse_args(argv)
    registry = generate_island_hydraulic_contexts(
        args.analysis_manifest,
        base_link_context_path=args.base_link_context,
        base_node_context_path=args.base_node_context,
    )
    print(f"Generated island contexts: {int(registry['hydraulic_context_action'].eq('generated_island_context').sum())}")
    print(f"Alias references: {int(registry['hydraulic_context_action'].eq('reference_existing_independent_state').sum())}")
    print(f"Topology-review exclusions: {int(registry['hydraulic_context_action'].eq('excluded_topology_review').sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
