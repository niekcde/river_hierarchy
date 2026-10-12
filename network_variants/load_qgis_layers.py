"""Run in the QGIS Python console/editor to load units and unique islands."""

import csv
from pathlib import Path


# SETTINGS
ex = "02"
unit_type = "indep"  # "indep" or "compound"
repository = Path("/Volumes/PhD/river_hierarchy")
load_islands = True


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def numeric_id(value):
    # CSV registries can store integer IDs as e.g. "24.0".
    number = float(value)
    if not number.is_integer():
        raise ValueError(f"Expected an integer unit ID, got {value!r}")
    return int(number)


def collect_layers():
    """Return (group, label, path, style) entries in numeric unit-ID order."""
    if unit_type not in {"indep", "compound"}:
        raise ValueError("unit_type must be 'indep' or 'compound'")

    output = repository / "network_variants" / "output"
    suffix = "indep" if unit_type == "indep" else "compound_bubbles"
    experiment = output / f"{ex}_{suffix}__anchored"
    group = "units" if unit_type == "indep" else "compound"
    base_style = repository / f"base_{unit_type}_unit.qml"
    regular_style = repository / "regular_unit.qml"

    entries = []
    for row in read_rows(experiment / "state_registry.csv"):
        state = row["state_id"]
        if state == "S000_base":
            key = (-1, state)
            label = state
            style = base_style
        else:
            field = "selected_unit_ids" if unit_type == "indep" else "bubble_root_unit_id"
            unit_id = numeric_id(row[field])
            key = (unit_id, state)
            label = f"unit_{unit_id} | {state}"
            style = regular_style
        # Use the published registry path, not a recursive search of backups.
        path_text = row.get("directed_links_path", "").strip()
        if not path_text:
            print(f"No directed links recorded for {state}; skipped")
            continue
        entries.append((key, group, label, Path(path_text), style))

    result = [entry[1:] for entry in sorted(entries)]
    if not load_islands:
        return result

    island_root = output / f"{ex}_island__anchored"
    manifest = island_root / "island_observation_manifest.csv"
    islands = []
    seen = set()
    for row in sorted(
        read_rows(manifest),
        key=lambda row: (numeric_id(row["source_unit_id"]), row["island_id"]),
    ):
        # These aliases already have a unit-state geometry.
        if row["equivalence_status"] != "requires_full_island_state":
            continue
        if row.get("separate_observation_excluded", "").lower() == "true":
            continue
        island_id = row["island_id"]
        path_text = row.get("analysis_directed_links_path", "").strip()
        if not path_text:
            print(f"No current directed links recorded for {island_id}; skipped")
            continue
        path = Path(path_text)
        identity = row.get("geometry_id") or str(path.resolve())
        if identity in seen:
            continue
        seen.add(identity)
        unit_id = numeric_id(row["source_unit_id"])
        islands.append(("islands", f"unit_{unit_id} | {island_id}", path, regular_style))
    return result + islands


def load_into_qgis(entries):
    from qgis.core import QgsProject, QgsVectorLayer

    project = QgsProject.instance()
    root = project.layerTreeRoot()
    main_group = root.findGroup(ex)
    if main_group is None:
        main_group = root.addGroup(ex)

    for group_name, label, path, style in entries:
        if not path.is_file():
            print(f"File not found: {path}")
            continue
        layer = QgsVectorLayer(str(path), label, "ogr")
        if not layer.isValid():
            print(f"Could not load: {path}")
            continue
        subgroup = main_group.findGroup(group_name)
        if subgroup is None:
            subgroup = main_group.addGroup(group_name)
        project.addMapLayer(layer, False)
        subgroup.addLayer(layer).setItemVisibilityChecked(False)

        if style.is_file():
            # PyQGIS returns (message, success), in that order.
            message, success = layer.loadNamedStyle(str(style))
            if not success:
                print(f"Style failed for {label}: {message}")
            layer.triggerRepaint()
        else:
            print(f"Style not found: {style}")
        print(f"Loaded: {group_name} / {label}")


def main():
    print(f"Loading QGIS layers for example {ex} ({unit_type})...")
    entries = collect_layers()
    print(f"Found {len(entries)} layer entries.")
    load_into_qgis(entries)
    print("Finished. Layers are unchecked; enable them in the Layers panel.")


# This is a QGIS console/editor script. Run explicitly because the console's
# execution namespace does not necessarily use __name__ == "__main__".
main()
