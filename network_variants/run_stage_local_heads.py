"""Reproduce reviewed candidates with the maintained local solver, never promote.

python -m network_variants.run_stage_local_heads --output /new/staging/directory
"""
import argparse
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .local_head_reconstruction import LocalHeadContract, reconstruct_local_heads
from .reviewed_geometry import ReviewedGeometry

ROOT = Path(__file__).resolve().parents[1]
INDEX = (
    ROOT / "RAPID/outputs/accepted_boundary_completion_20261028/candidate_index.json"
)


def stage(output, index=INDEX):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    rows = []
    cache = {}
    release = ReviewedGeometry()

    def track(path):
        path = Path(path)
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def read_json(path):
        return json.loads(track(path).read_text())

    track(release.manifest)
    for item in read_json(index):
        key = item["case"]
        example, state = key.split("/")
        folder = Path(item["candidate_folder"])
        dest = output / example / state
        dest.mkdir(parents=True)
        geometry = track(folder / "geometry_candidate.parquet")
        expected = track(folder / "slope_candidate.parquet")
        if (
            hashes[str(geometry)] != item["geometry_sha256"]
            or hashes[str(expected)] != item["slope_sha256"]
        ):
            raise ValueError(f"Accepted source fingerprint changed: {key}")
        if example not in cache:
            ref = release.lookup(example, "indep", "S000_base")
            parent = pd.read_parquet(track(ref["original_frame"]))
            floors = (
                pd.read_parquet(
                    track(
                        ROOT
                        / f"RAPID/assets/bounded_robust_slopes_20261002/{example}/floor_0.05/old/links.parquet"
                    )
                )
                .set_index("id_link")
                .lower_bound
            )
            cache[example] = parent, floors
        parent, floors = cache[example]
        data = gpd.read_parquet(geometry)
        spec = read_json(folder / "boundary_contract.json")
        primary = read_json(folder / "primary_head_fit.json")
        # Only previously reviewed releases are permitted, not all boundaries.
        contract = LocalHeadContract(
            frozenset(spec["approved_scope"]),
            frozenset(spec["boundary_nodes"]),
            {
                int(r.id_link): int(r.parent_link)
                for r in data.itertuples()
                if pd.notna(r.parent_link)
            },
            frozenset(map(int, spec["accepted_reversed_connectors"])),
            frozenset(primary["candidates"]),
        )
        result, audit = reconstruct_local_heads(data, parent, floors, contract)
        prior = pd.read_parquet(expected).set_index("id_link").loc[result.id_link]
        errors = {}
        for field in ("head_us", "head_ds", "slope", "floor"):
            errors[field] = float(
                np.max(abs(result[field].to_numpy() - prior[field].to_numpy()))
            )
            np.testing.assert_allclose(
                result[field].to_numpy(),
                prior[field].to_numpy(),
                rtol=1e-10,
                atol=1e-10,
            )
        result.to_parquet(dest / "slope_candidate.parquet", index=False)
        (dest / "audit.json").write_text(json.dumps(audit, indent=2))
        (dest / "boundary_contract.json").write_text(
            json.dumps(
                dict(spec, permitted_flexible_nodes=primary["candidates"]), indent=2
            )
        )
        rows.append(
            dict(
                case=key,
                parity=True,
                errors=errors,
                fixed_boundary_feasible=audit["primary"]["strict_fixed_feasible"],
            )
        )
        print(key, "parity passed", flush=True)
    for path, digest in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source modified during stage: {path}")
    for path in Path(__file__).parent.glob("*local_head*.py"):
        track(path)
    summary = dict(
        cases=rows,
        source_hashes=hashes,
        production_modified=False,
        routing_runs=0,
        rf_training_runs=0,
    )
    (output / "manifest.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=Path, default=INDEX)
    args = parser.parse_args()
    stage(args.output, args.index)
