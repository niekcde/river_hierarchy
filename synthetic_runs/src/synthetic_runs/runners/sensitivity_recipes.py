"""Scripted builder for the preserved first sensitivity recipe set."""

from __future__ import annotations

import argparse
import gzip
import json
from copy import deepcopy
from pathlib import Path


if __package__ in {None, ""}:
    import sys

    _SRC_DIR = Path(__file__).resolve().parents[2]
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))

from synthetic_runs.core import _json_default


DEFAULT_META = {
    "L": 12000,
    "W_total": 650.0,
    "xs": 3000,
    "xe": 9000,
    "jump": 500,
    "min_width": 10.0,
    "width_step": 10,
    "x_stability": 0.3,
    "max_breaks": 5,
    "Y0": 1.0,
    "amp_corr": 1.5,
    "amp_loop": 0.7,
}

DEFAULT_SAMPLE_PLAN = {
    "initial": {"ratio": [0.5, 0.5], "flip": False},
    "per_break": [{"ratio": [], "flip": False}],
}


def _base_recipe(*, wa: float, wb: float, geometry_id: int, breaks: list[dict]) -> dict:
    return {
        "meta": deepcopy(DEFAULT_META),
        "initial_split": {"WA": float(wa), "WB": float(wb)},
        "breaks": deepcopy(breaks),
        "geometry_id": int(geometry_id),
        "sample_id": 0,
        "sample_mode": "benchmark",
        "sample_plan": deepcopy(DEFAULT_SAMPLE_PLAN),
    }


def build_default_sensitivity_recipes() -> list[dict]:
    recipes = []
    geometry_id = 1000001

    for w1, w2 in [(416.0, 104.0), (260.0, 260.0)]:
        recipes.append(
            _base_recipe(
                wa=520.0,
                wb=130.0,
                geometry_id=geometry_id,
                breaks=[
                    {
                        "kind": "loop",
                        "from_branch": "A",
                        "to_branch": "A",
                        "xb": 5000.0,
                        "xr": 7000.0,
                        "w1": float(w1),
                        "w2": float(w2),
                        "replace_corridor": True,
                    }
                ],
            )
        )
        geometry_id += 1

    for w_cross in [416.0, 260.0, 104.0]:
        recipes.append(
            _base_recipe(
                wa=520.0,
                wb=130.0,
                geometry_id=geometry_id,
                breaks=[
                    {
                        "kind": "cross",
                        "from_branch": "A",
                        "to_branch": "B",
                        "xb": 5000.0,
                        "xr": 7000.0,
                        "w_cross": float(w_cross),
                    }
                ],
            )
        )
        geometry_id += 1

    recipes.append(_base_recipe(wa=520.0, wb=130.0, geometry_id=geometry_id, breaks=[]))
    geometry_id += 1
    recipes.append(_base_recipe(wa=325.0, wb=325.0, geometry_id=geometry_id, breaks=[]))

    return recipes


def write_recipes(recipes: list[dict], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        for recipe in recipes:
            f.write(json.dumps(recipe, default=_json_default) + "\n")
    return out_path


def load_recipes(path: str | Path) -> list[dict]:
    path = Path(path)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def validate_recipes(recipes: list[dict], reference_path: str | Path) -> bool:
    reference = load_recipes(reference_path)
    return recipes == reference


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write the preserved first sensitivity recipe set as networks_sensitivity.jsonl.gz."
    )
    parser.add_argument("--out", required=True, help="Output .jsonl.gz path")
    parser.add_argument(
        "--validate-against",
        help="Optional reference .jsonl.gz path to compare against after writing",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    recipes = build_default_sensitivity_recipes()
    out_path = write_recipes(recipes, args.out)
    print(f"wrote {out_path} rows {len(recipes)}")

    if args.validate_against:
        ok = validate_recipes(recipes, args.validate_against)
        if not ok:
            print(f"validation failed against {args.validate_against}")
            return 1
        print(f"validated against {args.validate_against}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
