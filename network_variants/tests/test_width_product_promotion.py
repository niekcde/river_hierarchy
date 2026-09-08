from __future__ import annotations

import hashlib
import json
from pathlib import Path

from network_variants.width_product_promotion import (
    apply_width_promotion_plan,
    build_width_promotion_plan,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_promotes_complete_validated_width_repair_with_backup(tmp_path: Path) -> None:
    target = tmp_path / "experiment"
    state = target / "states" / "S000_base"
    widths = state / "variant" / "widths"
    summary = state / "variant" / "summary"
    widths.mkdir(parents=True)
    summary.mkdir(parents=True)
    (target / "experiment_manifest.json").write_text('{"example_id": "x"}')
    (summary / "variant_manifest.json").write_text(
        json.dumps({"output_paths": {}, "routing_width_application": {"old": True}})
    )

    repair = tmp_path / "repair"
    repair_widths = repair / "states" / "S000_base" / "variant" / "widths"
    repair_widths.mkdir(parents=True)
    names = {
        "links_with_width_families": "links_with_width_families.gpkg",
        "link_width_families": "link_width_families.csv",
        "link_width_samples": "link_width_samples.csv",
        "routing_width_application": "routing_width_application.csv",
        "routing_width_application_manifest": "routing_width_application_manifest.json",
        "differences": "width_product_repair_differences.csv",
        "validation": "width_product_repair_validation.json",
    }
    output_paths = {}
    hashes = {}
    for key, name in names.items():
        staged = repair_widths / name
        content = '{"schema_version": "new-routing"}' if key == "routing_width_application_manifest" else f"new-{key}"
        staged.write_text(content)
        output_paths[key] = str(staged)
        hashes[key] = _sha256(staged)
        production = widths / name
        if key not in {"differences", "validation"}:
            production.write_text(f"old-{key}")

    repair_manifest = {
        "source_experiment": str(target.resolve()),
        "is_valid": True,
        "states": [{
            "state_id": "S000_base",
            "material_change_count": 2,
            "validation": {"is_valid": True, "checks": {"ok": True}},
            "output_paths": output_paths,
            "file_sha256": hashes,
        }],
    }
    (repair / "width_product_repair_manifest.json").write_text(json.dumps(repair_manifest))

    plan = build_width_promotion_plan(repair)
    backup = apply_width_promotion_plan(plan, backup_root=tmp_path / "backups")

    assert (widths / "links_with_width_families.gpkg").read_text() == "new-links_with_width_families"
    assert (backup / "states/S000_base/variant/widths/links_with_width_families.gpkg").read_text() == "old-links_with_width_families"
    promoted_variant = json.loads((summary / "variant_manifest.json").read_text())
    assert promoted_variant["width_product_repair"]["validation"]["is_valid"] is True
    assert promoted_variant["routing_width_application"]["schema_version"] == "new-routing"
    assert json.loads((target / "experiment_manifest.json").read_text())["width_product_repair"]["state_count"] == 1
    assert json.loads((target / "width_product_repair_promotion_manifest.json").read_text())["is_valid"] is True
