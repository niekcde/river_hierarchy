"""Independent network partitions of the slope-only batch, combined with coverage checks."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import pandas as pd
from .run_compound_member_heads import run, INVENTORY, save


def run_parallel(output, workers):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    inventory = pd.read_parquet(INVENTORY)
    inventory = inventory[inventory.edit_type.eq("Compound member")]
    examples = sorted(inventory.example.astype(str).str.zfill(2).unique())
    partitions = out / "partitions"
    partitions.mkdir()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run, partitions / ex, None, [ex]): ex for ex in examples}
        for future in as_completed(futures):
            result = future.result()
            print(
                "NETWORK COMPLETE", futures[future], result["status_counts"], flush=True
            )
    rows = []
    comparisons = []
    sources = {}
    for ex in examples:
        part = partitions / ex
        m = json.loads((part / "manifest.json").read_text())
        for path, h in m["source_hashes"].items():
            if path in sources and sources[path] != h:
                raise ValueError("Inconsistent source versions across partitions")
            sources[path] = h
        rows.extend(json.loads((part / "case_results.json").read_text()))
        comparisons.append(pd.read_parquet(part / "exterior_comparison.parquet"))
        shutil.move(str(part / ex), str(out / ex))
    keys = {r["case"] for r in rows}
    expected = {str(r.example).zfill(2) + "/" + r.state for r in inventory.itertuples()}
    if len(rows) != len(keys) or keys != expected:
        raise ValueError("Partition coverage or uniqueness failed")
    for path, h in sources.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != h:
            raise ValueError("Source changed during final batch: " + path)
    rows.sort(key=lambda r: r["case"])
    save(out / "case_results.json", rows)
    pd.DataFrame(rows).to_csv(out / "case_results.csv", index=False)
    pd.concat(comparisons, ignore_index=True).to_parquet(
        out / "exterior_comparison.parquet", index=False
    )
    summary = dict(
        cases=len(rows),
        networks=len(examples),
        status_counts=pd.Series([r["status"] for r in rows]).value_counts().to_dict(),
        production_modified=False,
        routing_runs=0,
        rf_training_runs=0,
        workers=workers,
        source_hashes=sources,
    )
    save(out / "manifest.json", summary)
    print(
        "ALL NETWORKS",
        {k: v for k, v in summary.items() if k != "source_hashes"},
        flush=True,
    )
    return summary
