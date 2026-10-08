"""Sequential, fail-closed full refresh under the accepted hydraulic policy."""
from pathlib import Path
import argparse
import json
from datetime import datetime, timezone
import pandas as pd
from run_hydraulic_policy_refresh import run as route, DEFAULT_OUTPUT
from refresh_selected_domain_resistance import run as resistance

EXAMPLES = ["02", "03", "05", "11", "13", "15", "21", "22", "37", "41", "45", "46", "48", "54", "55", "56"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--journal", default="batch_status.json", help="Separate progress journal for a worker assigned disjoint cases.")
    p.add_argument("examples", nargs="*", default=EXAMPLES)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/args.journal
    journal = json.loads(path.read_text()) if path.exists() else []
    for example in args.examples:
        record = dict(example=example, started_at=datetime.now(timezone.utc).isoformat(), status="running")
        existing = next((i for i, item in enumerate(journal) if item["example"] == example), None)
        if existing is not None:
            record["previous_attempts"] = [*journal[existing].get("previous_attempts", []),
                                           {key: value for key, value in journal[existing].items() if key != "previous_attempts"}]
            journal[existing] = record
        else:
            journal.append(record)
        path.write_text(json.dumps(journal, indent=2))
        try:
            route(example, args.output, route=True, compact=True, resume=True)
            resistance(example, args.output)
            table = pd.read_csv(args.output/example/"selected_domain_resistance/comparisons.csv")
            unresolved = table[~table.status.eq("recomputed")]
            if len(unresolved):
                raise RuntimeError(f"{len(unresolved)} unresolved resistance comparisons; inspect the case report.")
            record.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat())
        except Exception as error:
            record.update(status="failed", error=str(error))
            raise
        finally:
            path.write_text(json.dumps(journal, indent=2))


if __name__ == "__main__":
    main()
