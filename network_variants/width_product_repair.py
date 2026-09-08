from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd

from network_variants.routing_width_application import (
    apply_routing_width_model,
    write_routing_width_audit,
)
from network_variants.width_product_inheritance import inherit_unchanged_link_width_products
from network_variants.width_product_inheritance import WIDTH_INHERITANCE_FIELDS
from network_variants.width_samples import read_width_samples
from network_variants.width_trim_selection import (
    TRIM_SELECTION_VERSION,
    apply_authoritative_width_trim_selection,
    build_parent_child_width_diagnostics,
    resolve_closed_replacement_exclusions,
)


REPAIR_SCHEMA_VERSION = "width-product-repair-v1"


def _one(directory: Path, pattern: str) -> Path:
    matches = sorted(path for path in directory.glob(pattern) if not path.name.startswith("._"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {pattern!r} in {directory}; found {len(matches)}.")
    return matches[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_state_products(state: Path) -> tuple[gpd.GeoDataFrame, pd.DataFrame, gpd.GeoDataFrame]:
    variant = state / "variant"
    widths = variant / "widths"
    links = gpd.read_file(widths / "links_with_width_families.gpkg")
    samples = read_width_samples(widths, low_memory=False)
    nodes = gpd.read_file(_one(variant / "directed", "*directed_nodes.gpkg"))
    return links, samples, nodes


def _attach_diagnostics(
    links: gpd.GeoDataFrame,
    routing_audit: pd.DataFrame,
    parent_links: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    diagnostics = build_parent_child_width_diagnostics(links, parent_links)
    if diagnostics.empty:
        return links, routing_audit
    link_columns = [
        column for column in diagnostics.columns if column != "id_link" and column not in links.columns
    ]
    if link_columns:
        links = links.merge(
            diagnostics[["id_link", *link_columns]],
            on="id_link",
            how="left",
            validate="1:1",
        )
    if not routing_audit.empty:
        routing_audit = routing_audit.merge(
            diagnostics,
            on="id_link",
            how="left",
            validate="1:1",
        )
    return links, routing_audit


def _write_width_products(
    *,
    output_dir: Path,
    links: gpd.GeoDataFrame,
    samples: pd.DataFrame,
    routing_audit: pd.DataFrame,
    routing_summary: dict[str, Any],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    links_path = output_dir / "links_with_width_families.gpkg"
    families_path = output_dir / "link_width_families.csv"
    samples_path = output_dir / "link_width_samples.csv"
    links.to_file(links_path, driver="GPKG")
    pd.DataFrame(links.drop(columns=links.geometry.name)).to_csv(families_path, index=False)
    samples.to_csv(samples_path, index=False)
    routing_paths = write_routing_width_audit(
        routing_audit,
        routing_summary,
        output_dir=output_dir,
    )
    paths = {
        "links_with_width_families": str(links_path.resolve()),
        "link_width_families": str(families_path.resolve()),
        "link_width_samples": str(samples_path.resolve()),
        "routing_width_application": routing_paths["link_audit"],
        "routing_width_application_manifest": routing_paths["manifest"],
    }
    return paths


def _difference_audit(
    old_links: gpd.GeoDataFrame,
    new_links: gpd.GeoDataFrame,
) -> pd.DataFrame:
    fields = ("wid_adj_wet", "wid_adj_total", "wid_adj_routing")
    old = old_links.set_index(pd.to_numeric(old_links["id_link"], errors="raise").astype(int))
    new = new_links.set_index(pd.to_numeric(new_links["id_link"], errors="raise").astype(int))
    records: list[dict[str, Any]] = []
    for link_id in sorted(new.index):
        record: dict[str, Any] = {
            "id_link": int(link_id),
            "width_trim_method": new.at[link_id, "width_trim_method"],
            "width_trim_valid_core": bool(new.at[link_id, "width_trim_valid_core"]),
        }
        material = False
        for field in fields:
            old_value = pd.to_numeric(pd.Series([old.at[link_id, field]]), errors="coerce").iloc[0]
            new_value = pd.to_numeric(pd.Series([new.at[link_id, field]]), errors="coerce").iloc[0]
            record[f"old_{field}"] = old_value
            record[f"new_{field}"] = new_value
            record[f"{field}_relative_change"] = (
                float(new_value / old_value - 1.0)
                if pd.notna(old_value) and pd.notna(new_value) and old_value > 0
                else np.nan
            )
            if pd.notna(old_value) and pd.notna(new_value):
                material |= not bool(np.isclose(old_value, new_value, rtol=1e-6, atol=1e-6))
        record["material_change"] = bool(material)
        records.append(record)
    return pd.DataFrame.from_records(records)


def _validate_repaired_products(
    *,
    source_links: gpd.GeoDataFrame,
    repaired_links: gpd.GeoDataFrame,
    repaired_samples: pd.DataFrame,
    repaired_parent_links: gpd.GeoDataFrame | None = None,
) -> dict[str, Any]:
    source = source_links.set_index(source_links["id_link"].astype(int), drop=False)
    repaired = repaired_links.set_index(repaired_links["id_link"].astype(int), drop=False)
    same_ids = set(source.index) == set(repaired.index)
    geometry_equal = same_ids and all(
        bool(source.at[link_id, "geometry"].equals(repaired.at[link_id, "geometry"]))
        for link_id in source.index
    )
    width_fields = ("wid_adj_wet", "wid_adj_total", "wid_adj_routing")
    finite_positive = {
        field: bool(
            field in repaired.columns
            and np.isfinite(pd.to_numeric(repaired[field], errors="coerce")).all()
            and pd.to_numeric(repaired[field], errors="coerce").gt(0).all()
        )
        for field in width_fields
    }
    sample_ids = set(
        pd.to_numeric(repaired_samples.get("id_link", pd.Series(dtype=float)), errors="coerce")
        .dropna()
        .astype(int)
    )
    samples_reference_known_links = sample_ids.issubset(set(repaired.index))
    trim_audit_complete = bool(
        "width_trim_method" in repaired.columns
        and repaired["width_trim_method"].notna().all()
        and repaired["width_trim_method"].astype(str).ne("").all()
    )

    inherited_values_exact = True
    changed_links_evaluated = True
    if "width_inheritance_status" in repaired.columns:
        inherited_rows = repaired.loc[
            repaired["width_inheritance_status"].eq("inherited_unchanged_1to1")
        ]
        changed_rows = repaired.loc[
            repaired["width_inheritance_status"].eq("changed_evaluated_width_repair")
        ]
        changed_links_evaluated = bool(
            len(changed_rows) > 0
            and np.isfinite(pd.to_numeric(changed_rows["wid_adj_routing"], errors="coerce")).all()
            and pd.to_numeric(changed_rows["wid_adj_routing"], errors="coerce").gt(0).all()
        )
        if repaired_parent_links is None:
            inherited_values_exact = False
        else:
            parent = repaired_parent_links.set_index(
                repaired_parent_links["id_link"].astype(int), drop=False
            )
            fields = [field for field in WIDTH_INHERITANCE_FIELDS if field in repaired and field in parent]
            for row in inherited_rows.itertuples(index=False):
                parent_id = int(getattr(row, "width_inherited_from_parent_link_id"))
                child_id = int(getattr(row, "id_link"))
                if parent_id not in parent.index:
                    inherited_values_exact = False
                    break
                for field in fields:
                    child_value = repaired.at[child_id, field]
                    parent_value = parent.at[parent_id, field]
                    if pd.isna(child_value) and pd.isna(parent_value):
                        continue
                    if child_value != parent_value:
                        inherited_values_exact = False
                        break
                if not inherited_values_exact:
                    break

    checks = {
        "link_ids_preserved": bool(same_ids),
        "link_geometries_preserved": bool(geometry_equal),
        "wet_widths_finite_positive": finite_positive["wid_adj_wet"],
        "total_widths_finite_positive": finite_positive["wid_adj_total"],
        "routing_widths_finite_positive": finite_positive["wid_adj_routing"],
        "samples_reference_known_links": bool(samples_reference_known_links),
        "trim_audit_complete": bool(trim_audit_complete),
        "inherited_values_exact": bool(inherited_values_exact),
        "changed_links_evaluated": bool(changed_links_evaluated),
    }
    return {
        "is_valid": bool(all(checks.values())),
        "checks": checks,
    }


def repair_base_state(
    *,
    source_state: Path,
    output_width_dir: Path,
    example_id: str,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, dict[str, Any]]:
    old_links, old_samples, nodes = _read_state_products(source_state)
    selected, marked, trim_audit = apply_authoritative_width_trim_selection(
        old_links,
        old_samples,
    )
    repaired, routing_audit, routing_summary = apply_routing_width_model(
        selected,
        marked,
        nodes,
        example_id=str(example_id),
        variant_id="S000_base",
        apply_known_base_exclusions=True,
    )
    paths = _write_width_products(
        output_dir=output_width_dir,
        links=repaired,
        samples=marked,
        routing_audit=routing_audit,
        routing_summary=routing_summary,
    )
    differences = _difference_audit(old_links, repaired)
    difference_path = output_width_dir / "width_product_repair_differences.csv"
    differences.to_csv(difference_path, index=False)
    validation = _validate_repaired_products(
        source_links=old_links,
        repaired_links=repaired,
        repaired_samples=marked,
    )
    validation_path = output_width_dir / "width_product_repair_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    summary = {
        "state_id": "S000_base",
        "state_type": "base",
        "link_count": int(len(repaired)),
        "native_fallback_count": int(trim_audit["width_trim_method"].eq("native_rivgraph_fallback").sum()),
        "rejected_or_unavailable_core_count": int(
            trim_audit["width_trim_method"].eq("untrimmed_no_valid_core").sum()
        ),
        "material_change_count": int(differences["material_change"].sum()),
        "material_change_link_ids": differences.loc[
            differences["material_change"], "id_link"
        ].astype(int).tolist(),
        "validation": validation,
        "output_paths": {
            **paths,
            "differences": str(difference_path.resolve()),
            "validation": str(validation_path.resolve()),
        },
    }
    return repaired, marked, summary


def repair_child_state(
    *,
    source_state: Path,
    output_width_dir: Path,
    example_id: str,
    repaired_parent_links: gpd.GeoDataFrame,
    repaired_parent_samples: pd.DataFrame,
) -> dict[str, Any]:
    old_links, old_samples, nodes = _read_state_products(source_state)
    selected, marked, _ = apply_authoritative_width_trim_selection(old_links, old_samples)
    lineage = pd.read_csv(source_state / "variant" / "matching" / "link_lineage.csv")
    inherited, inherited_samples, inheritance_summary = inherit_unchanged_link_width_products(
        child_links=selected,
        child_width_samples=marked,
        parent_links=repaired_parent_links,
        parent_width_samples=repaired_parent_samples,
        link_lineage=lineage,
    )
    changed_ids = inherited.loc[
        inherited["width_inheritance_status"].eq("changed_requires_evaluation"), "id_link"
    ].astype(int).tolist()
    excluded_ids, exclusion_audit = resolve_closed_replacement_exclusions(
        inherited,
        repaired_parent_links,
        changed_ids,
    )
    if not exclusion_audit.empty:
        columns = [
            column for column in exclusion_audit.columns
            if column != "id_link" and column not in inherited.columns
        ]
        inherited = inherited.merge(
            exclusion_audit[["id_link", *columns]],
            on="id_link",
            how="left",
            validate="1:1",
        )
    repaired, routing_audit, routing_summary = apply_routing_width_model(
        inherited,
        inherited_samples,
        nodes,
        example_id=str(example_id),
        variant_id=source_state.name,
        apply_known_base_exclusions=False,
        evaluation_link_ids=changed_ids,
        excluded_evaluation_link_ids=excluded_ids,
        exclusion_reason="closed_two_link_replacement",
    )
    repaired, routing_audit = _attach_diagnostics(
        repaired,
        routing_audit,
        repaired_parent_links,
    )
    repaired.loc[
        repaired["id_link"].astype(int).isin(changed_ids), "width_inheritance_status"
    ] = "changed_evaluated_width_repair"
    paths = _write_width_products(
        output_dir=output_width_dir,
        links=repaired,
        samples=inherited_samples,
        routing_audit=routing_audit,
        routing_summary=routing_summary,
    )
    differences = _difference_audit(old_links, repaired)
    difference_path = output_width_dir / "width_product_repair_differences.csv"
    differences.to_csv(difference_path, index=False)
    validation = _validate_repaired_products(
        source_links=old_links,
        repaired_links=repaired,
        repaired_samples=inherited_samples,
        repaired_parent_links=repaired_parent_links,
    )
    validation_path = output_width_dir / "width_product_repair_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    return {
        "state_id": source_state.name,
        "state_type": "child",
        "link_count": int(len(repaired)),
        "changed_geometry_link_count": int(len(changed_ids)),
        "changed_geometry_link_ids": changed_ids,
        "excluded_closed_replacement_link_ids": excluded_ids,
        "inheritance": inheritance_summary,
        "validation": validation,
        "material_change_count": int(differences["material_change"].sum()),
        "material_change_link_ids": differences.loc[
            differences["material_change"], "id_link"
        ].astype(int).tolist(),
        "output_paths": {
            **paths,
            "differences": str(difference_path.resolve()),
            "validation": str(validation_path.resolve()),
        },
    }


def repair_experiment_width_products(
    experiment: str | Path,
    *,
    output_root: str | Path,
    child_state_ids: Sequence[str] = (),
    include_all_children: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    experiment = Path(experiment).expanduser().resolve()
    states_root = experiment / "states"
    if not states_root.exists():
        raise FileNotFoundError(f"Experiment states directory is missing: {states_root}")
    manifest = json.loads((experiment / "experiment_manifest.json").read_text(encoding="utf-8"))
    example_id = str(manifest["example_id"])
    target = Path(output_root).expanduser().resolve() / experiment.name
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"Repair workspace already exists: {target}")
        shutil.rmtree(target)
    target.mkdir(parents=True)

    available_children = sorted(
        state.name for state in states_root.iterdir()
        if state.is_dir() and state.name != "S000_base"
    )
    requested = available_children if include_all_children else list(child_state_ids)
    unknown = sorted(set(requested).difference(available_children))
    if unknown:
        raise ValueError(f"Requested child states are absent: {unknown}")

    base_width_dir = target / "states" / "S000_base" / "variant" / "widths"
    parent_links, parent_samples, base_summary = repair_base_state(
        source_state=states_root / "S000_base",
        output_width_dir=base_width_dir,
        example_id=example_id,
    )
    state_summaries = [base_summary]
    for state_id in requested:
        state_summaries.append(
            repair_child_state(
                source_state=states_root / state_id,
                output_width_dir=target / "states" / state_id / "variant" / "widths",
                example_id=example_id,
                repaired_parent_links=parent_links,
                repaired_parent_samples=parent_samples,
            )
        )

    repair_manifest = {
        "schema_version": REPAIR_SCHEMA_VERSION,
        "trim_selection_version": TRIM_SELECTION_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_experiment": str(experiment),
        "repair_workspace": str(target),
        "example_id": example_id,
        "requested_child_state_ids": requested,
        "all_children_included": bool(include_all_children),
        "source_experiment_unchanged": True,
        "is_valid": bool(all(state["validation"]["is_valid"] for state in state_summaries)),
        "states": state_summaries,
    }
    for state in repair_manifest["states"]:
        state["file_sha256"] = {
            key: _sha256(Path(path))
            for key, path in state["output_paths"].items()
        }
    manifest_path = target / "width_product_repair_manifest.json"
    manifest_path.write_text(json.dumps(repair_manifest, indent=2), encoding="utf-8")
    return repair_manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild width products in an isolated repair workspace without raster reruns."
    )
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--child-state-id", action="append", default=[])
    parser.add_argument("--include-all-children", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = repair_experiment_width_products(
        args.experiment,
        output_root=args.output_root,
        child_state_ids=args.child_state_id,
        include_all_children=args.include_all_children,
        overwrite=args.overwrite,
    )
    print(json.dumps({
        "repair_workspace": result["repair_workspace"],
        "state_count": len(result["states"]),
        "states": [
            {
                "state_id": state["state_id"],
                "material_change_count": state["material_change_count"],
            }
            for state in result["states"]
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
