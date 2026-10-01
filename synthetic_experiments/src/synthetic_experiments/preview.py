"""Write baseline topology previews, not routed experiment results."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .design import Design, FAMILIES, build_original, planned_states
from .plotting import plot_network


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    design = Design()
    states = []
    for family in FAMILIES:
        net = build_original(family, design)
        fig = plot_network(net, title=f"{family}: original — schematic lateral positions")
        fig.update_xaxes(title_text="Downstream position (m)")
        fig.update_yaxes(title_text="Schematic lateral position")
        fig.write_html(args.output / f"{family}.html", include_plotlyjs=True)
        states.extend(planned_states(family, f"{family}_baseline"))
    (args.output / "design_manifest.json").write_text(json.dumps(
        {"design": asdict(design), "states": states,
         "stage": "baseline_preview_only"}, indent=2)+"\n")


if __name__ == "__main__":
    main()
