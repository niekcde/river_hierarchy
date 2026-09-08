from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable).resolve()


def _run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {' '.join(command)}\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            env={
                **os.environ,
                "MPLCONFIGDIR": "/tmp/mpl-width-repair",
                "XDG_CACHE_HOME": "/tmp/width-repair-cache",
            },
        )
    if result.returncode:
        raise RuntimeError(f"Command failed with exit code {result.returncode}; inspect {log_path}")


def _eq_is_complete(experiment: Path) -> bool:
    registry = experiment / "state_registry.csv"
    comparison = experiment / "equivalent_resistance/equivalent_resistance_comparison.csv"
    summary = experiment / "equivalent_resistance/state_hierarchy_recompute_summary.json"
    return registry.exists() and comparison.exists() and summary.exists()


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage, promote, and refresh width repairs by example.")
    parser.add_argument("--example", action="append", required=True)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "network_variants/output")
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    args = parser.parse_args()

    records = []
    for example in args.example:
        example = str(example).zfill(2)
        print(f"[{example}] starting", flush=True)
        for suffix in ("indep__anchored", "compound_bubbles__anchored"):
            name = f"{example}_{suffix}"
            target = args.output_root / name
            staging = args.staging_root / name
            log = args.log_root / f"{name}.log"
            promotion = target / "width_product_repair_promotion_manifest.json"
            if not promotion.exists():
                print(f"[{name}] staging all states", flush=True)
                command = [
                    str(PYTHON), "-B", "network_variants/run_width_product_repair.py",
                    str(target), "--output-root", str(args.staging_root),
                    "--include-all-children",
                ]
                if staging.exists():
                    command.append("--overwrite")
                _run(command, log)
                print(f"[{name}] promoting with backup", flush=True)
                _run([
                    str(PYTHON), "-B", "network_variants/run_promote_width_products.py",
                    str(staging), "--target-experiment", str(target),
                    "--backup-root", str(args.backup_root), "--apply",
                ], log)
                # External macOS volumes can remove AppleDouble ``._`` files
                # asynchronously while rmtree is traversing them. The repair
                # has already been promoted and the directory is disposable,
                # so tolerate a concurrently disappearing metadata entry.
                shutil.rmtree(staging, ignore_errors=True)
            else:
                print(f"[{name}] promotion already complete", flush=True)

            print(f"[{name}] recomputing EQ", flush=True)
            _run([
                str(PYTHON), "-B", "network_variants/run_equivalent_resistance_experiment.py",
                str(target), "--recompute-state-hierarchy", "--reuse-manifest-config",
            ], log)
            if not _eq_is_complete(target):
                raise RuntimeError(f"Equivalent-resistance outputs are incomplete: {target}")
            records.append({
                "example": example,
                "experiment": name,
                "status": "complete",
                "promotion_manifest": str(promotion),
                "rapid_refresh_required": (target / "rapid_width_refresh_required.json").exists(),
                "log": str(log),
            })
            print(f"[{name}] complete", flush=True)
        print(f"[{example}] complete", flush=True)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "records": records,
    }
    report_path = args.log_root / "width_repair_batch_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Batch complete: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
