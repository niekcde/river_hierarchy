from __future__ import annotations

import re
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence

import geopandas as gpd
import pandas as pd

from .sword_io import SwordFileCatalog, load_reaches
from .utils import DEFAULT_CRS

DEFAULT_TOPOLOGY_COLUMNS = (
    "reach_id",
    "rch_id_up",
    "rch_id_dn",
    "rch_id_up_main",
    "rch_id_dn_main",
    "rch_id_up_1",
    "rch_id_dn_1",
    "rch_id_up_2",
    "rch_id_dn_2",
    "rch_id_up_3",
    "rch_id_dn_3",
    "rch_id_up_4",
    "rch_id_dn_4",
    "dist_out",
    "reach_length",
    "river_name",
    "facc",
    "stream_order",
    "subnetwork_id",
    "n_chan_max",
    "n_chan_mod",
)


def build_example_reach_paths(
    examples: pd.DataFrame,
    station_matches: pd.DataFrame,
    *,
    catalog: SwordFileCatalog | None = None,
    topologies_by_region: Mapping[str, pd.DataFrame] | None = None,
    example_col: str = "example_id",
    station_key_col: str = "station_key",
    reach_id_col: str = "reach_id",
    region_col: str = "sword_region",
    downstream_col: str | Sequence[str] | None = None,
    upstream_station_cols: Sequence[str] = ("station_key_up",),
    downstream_station_cols: Sequence[str] = ("station_key_dn",),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return reaches on any directed path from example upstream stations to downstream stations."""
    if catalog is None and topologies_by_region is None:
        raise ValueError("Provide either catalog or topologies_by_region.")
    if example_col not in examples.columns:
        raise ValueError(f"examples is missing required column '{example_col}'.")

    match_lookup = _prepare_station_matches(
        station_matches,
        station_key_col=station_key_col,
        reach_id_col=reach_id_col,
        region_col=region_col,
    )

    topology_cache: dict[str, pd.DataFrame] = {}
    records: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    for example_id, group in examples.groupby(example_col, sort=False, dropna=False):
        upstream_keys = _unique_values(group, upstream_station_cols)
        downstream_keys = _unique_values(group, downstream_station_cols)

        upstream_matches = match_lookup[match_lookup[station_key_col].isin(upstream_keys)].copy()
        downstream_matches = match_lookup[match_lookup[station_key_col].isin(downstream_keys)].copy()

        missing_upstream = sorted(set(upstream_keys) - set(upstream_matches[station_key_col]))
        missing_downstream = sorted(set(downstream_keys) - set(downstream_matches[station_key_col]))

        regions = sorted(
            {
                str(value)
                for value in pd.concat(
                    [upstream_matches[region_col], downstream_matches[region_col]],
                    ignore_index=True,
                ).dropna()
            }
        )

        if not upstream_keys or not downstream_keys or not regions:
            summaries.append(
                _summary_record(
                    example_col=example_col,
                    example_id=example_id,
                    region=None,
                    upstream_keys=upstream_keys,
                    downstream_keys=downstream_keys,
                    upstream_reach_ids=[],
                    downstream_reach_ids=[],
                    selected_reach_ids=[],
                    missing_upstream=missing_upstream,
                    missing_downstream=missing_downstream,
                    route_status="missing_station_matches",
                )
            )
            continue

        for region in regions:
            region_upstream = upstream_matches[upstream_matches[region_col] == region].copy()
            region_downstream = downstream_matches[downstream_matches[region_col] == region].copy()
            upstream_reach_ids = _unique_reach_ids(region_upstream[reach_id_col])
            downstream_reach_ids = _unique_reach_ids(region_downstream[reach_id_col])

            if not upstream_reach_ids or not downstream_reach_ids:
                summaries.append(
                    _summary_record(
                        example_col=example_col,
                        example_id=example_id,
                        region=region,
                        upstream_keys=_sorted_station_keys(region_upstream[station_key_col]),
                        downstream_keys=_sorted_station_keys(region_downstream[station_key_col]),
                        upstream_reach_ids=upstream_reach_ids,
                        downstream_reach_ids=downstream_reach_ids,
                        selected_reach_ids=[],
                        missing_upstream=missing_upstream,
                        missing_downstream=missing_downstream,
                        route_status="no_region_pair",
                    )
                )
                continue

            topology = _get_region_topology(
                region,
                catalog=catalog,
                topologies_by_region=topologies_by_region,
                topology_cache=topology_cache,
            )
            selected_reach_ids = find_reaches_between(
                topology,
                upstream_reach_ids,
                downstream_reach_ids,
                reach_id_col=reach_id_col,
                downstream_col=downstream_col,
            )
            ordered_reach_ids = _sort_reach_ids(selected_reach_ids, topology, reach_id_col=reach_id_col)

            summaries.append(
                _summary_record(
                    example_col=example_col,
                    example_id=example_id,
                    region=region,
                    upstream_keys=_sorted_station_keys(region_upstream[station_key_col]),
                    downstream_keys=_sorted_station_keys(region_downstream[station_key_col]),
                    upstream_reach_ids=upstream_reach_ids,
                    downstream_reach_ids=downstream_reach_ids,
                    selected_reach_ids=ordered_reach_ids,
                    missing_upstream=missing_upstream,
                    missing_downstream=missing_downstream,
                    route_status="ok" if ordered_reach_ids else "no_directed_route",
                )
            )

            upstream_set = set(upstream_reach_ids)
            downstream_set = set(downstream_reach_ids)
            upstream_station_keys = _station_keys_for_reaches(region_upstream, station_key_col, reach_id_col)
            downstream_station_keys = _station_keys_for_reaches(region_downstream, station_key_col, reach_id_col)

            for path_order, reach_id in enumerate(ordered_reach_ids):
                records.append(
                    {
                        example_col: example_id,
                        region_col: region,
                        reach_id_col: reach_id,
                        "path_order": path_order,
                        "is_upstream_station_reach": reach_id in upstream_set,
                        "is_downstream_station_reach": reach_id in downstream_set,
                        "upstream_station_keys": _join(upstream_station_keys.get(reach_id, [])),
                        "downstream_station_keys": _join(downstream_station_keys.get(reach_id, [])),
                        "all_upstream_station_keys": _join(_sorted_station_keys(region_upstream[station_key_col])),
                        "all_downstream_station_keys": _join(_sorted_station_keys(region_downstream[station_key_col])),
                        "upstream_reach_ids": _join(str(value) for value in upstream_reach_ids),
                        "downstream_reach_ids": _join(str(value) for value in downstream_reach_ids),
                    }
                )

    reach_paths = pd.DataFrame.from_records(records)
    summaries_frame = pd.DataFrame.from_records(summaries)
    if not reach_paths.empty:
        reach_paths = reach_paths.sort_values(
            [example_col, region_col, "path_order"],
            kind="mergesort",
        ).reset_index(drop=True)
    if not summaries_frame.empty:
        summaries_frame = summaries_frame.sort_values(
            [example_col, "sword_region"],
            kind="mergesort",
        ).reset_index(drop=True)
    return reach_paths, summaries_frame


def build_example_reach_geometries(
    example_reaches: pd.DataFrame,
    catalog: SwordFileCatalog,
    *,
    reach_id_col: str = "reach_id",
    region_col: str = "sword_region",
    columns: Sequence[str] = (
        "reach_id",
        "river_name",
        "facc",
        "stream_order",
        "reach_length",
        "dist_out",
        "subnetwork_id",
        "n_chan_max",
        "n_chan_mod",
    ),
) -> gpd.GeoDataFrame:
    if example_reaches.empty:
        return gpd.GeoDataFrame({reach_id_col: []}, geometry=gpd.GeoSeries([], crs=DEFAULT_CRS), crs=DEFAULT_CRS)

    frames: list[gpd.GeoDataFrame] = []
    for region, group in example_reaches.groupby(region_col, sort=False):
        reach_ids = _unique_reach_ids(group[reach_id_col])
        reaches = load_reaches(catalog, regions=[region], columns=columns, reach_ids=reach_ids)
        if reaches.empty:
            continue
        reaches[reach_id_col] = reaches[reach_id_col].map(normalize_reach_id)
        reaches = reaches.dropna(subset=[reach_id_col]).drop_duplicates(subset=[region_col, reach_id_col])

        merged = group.merge(reaches, on=[region_col, reach_id_col], how="left")
        frames.append(gpd.GeoDataFrame(merged, geometry="geometry", crs=DEFAULT_CRS))

    if not frames:
        return gpd.GeoDataFrame({reach_id_col: []}, geometry=gpd.GeoSeries([], crs=DEFAULT_CRS), crs=DEFAULT_CRS)
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=DEFAULT_CRS)


def find_reaches_between(
    reaches: pd.DataFrame,
    upstream_reach_ids: Sequence[object],
    downstream_reach_ids: Sequence[object],
    *,
    reach_id_col: str = "reach_id",
    downstream_col: str | Sequence[str] | None = None,
) -> set[int]:
    """Return all reaches that sit on at least one directed path from upstream to downstream."""
    upstream_ids = set(_unique_reach_ids(upstream_reach_ids))
    downstream_ids = set(_unique_reach_ids(downstream_reach_ids))
    if not upstream_ids or not downstream_ids:
        return set()

    adjacency = build_downstream_adjacency(reaches, reach_id_col=reach_id_col, downstream_col=downstream_col)
    reverse_adjacency = _reverse_adjacency(adjacency)

    reachable_from_upstream = _traverse(adjacency, upstream_ids)
    can_reach_downstream = _traverse(reverse_adjacency, downstream_ids)
    return reachable_from_upstream & can_reach_downstream


def build_downstream_adjacency(
    reaches: pd.DataFrame,
    *,
    reach_id_col: str = "reach_id",
    downstream_col: str | Sequence[str] | None = None,
) -> dict[int, set[int]]:
    if reach_id_col not in reaches.columns:
        raise ValueError(f"reaches is missing required column '{reach_id_col}'.")

    downstream_cols = _resolve_downstream_columns(reaches, downstream_col)

    adjacency: dict[int, set[int]] = defaultdict(set)
    for row in reaches[[reach_id_col, *downstream_cols]].itertuples(index=False):
        reach_id = normalize_reach_id(row[0])
        if reach_id is None:
            continue
        adjacency.setdefault(reach_id, set())
        for downstream_value in row[1:]:
            for downstream_id in parse_reach_id_list(downstream_value):
                adjacency[reach_id].add(downstream_id)
                adjacency.setdefault(downstream_id, set())
    return dict(adjacency)


def load_reach_topology(
    catalog: SwordFileCatalog,
    region: str,
    *,
    columns: Sequence[str] = DEFAULT_TOPOLOGY_COLUMNS,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for item in catalog.select_reach_files(regions=[region]):
        read_columns = [column for column in columns if column in item.columns]
        if "reach_id" not in read_columns:
            continue
        frame = pd.read_parquet(item.path, columns=read_columns)
        frame["sword_region"] = item.region
        frame["source_file"] = str(item.path)
        frame["reach_id"] = frame["reach_id"].map(normalize_reach_id)
        frames.append(frame.dropna(subset=["reach_id"]))

    if not frames:
        return pd.DataFrame(columns=["sword_region", *columns, "source_file"])
    return pd.concat(frames, ignore_index=True)


def parse_reach_id_list(value: object) -> list[int]:
    if isinstance(value, (list, tuple, set)):
        return _unique_reach_ids(value)
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        return parse_reach_id_list(value.tolist())
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"", "nan", "none", "<na>", "[]"}:
            return []
        tokens = re.findall(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", text, flags=re.IGNORECASE)
        return _unique_reach_ids(tokens)
    normalized = normalize_reach_id(value)
    return [] if normalized is None else [normalized]


def normalize_reach_id(value: object) -> int | None:
    if isinstance(value, (list, tuple, set)):
        return None
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        value = value.tolist()
        if isinstance(value, (list, tuple, set)):
            return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None

    try:
        reach_id = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None
    return reach_id if reach_id > 0 else None


def _resolve_downstream_columns(
    reaches: pd.DataFrame,
    downstream_col: str | Sequence[str] | None,
) -> list[str]:
    if downstream_col is None:
        candidates = [
            "rch_id_dn",
            "rch_id_dn_main",
            *_sorted_numbered_columns(reaches.columns, "rch_id_dn_"),
        ]
    elif isinstance(downstream_col, str):
        candidates = [downstream_col]
    else:
        candidates = list(downstream_col)

    columns = [column for column in candidates if column in reaches.columns]
    if not columns:
        raise ValueError(
            "reaches is missing downstream topology columns. "
            f"Tried: {', '.join(candidates)}"
        )
    return columns


def _sorted_numbered_columns(columns: Sequence[object], prefix: str) -> list[str]:
    matched: list[tuple[int, str]] = []
    for column in columns:
        name = str(column)
        match = re.fullmatch(rf"{re.escape(prefix)}(\d+)", name)
        if match:
            matched.append((int(match.group(1)), name))
    return [name for _, name in sorted(matched)]


def _prepare_station_matches(
    station_matches: pd.DataFrame,
    *,
    station_key_col: str,
    reach_id_col: str,
    region_col: str,
) -> pd.DataFrame:
    required = {station_key_col, reach_id_col, region_col}
    missing = required - set(station_matches.columns)
    if missing:
        raise ValueError(f"station_matches is missing required columns: {', '.join(sorted(missing))}")

    matches = station_matches.dropna(subset=[station_key_col, reach_id_col, region_col]).copy()
    matches[station_key_col] = matches[station_key_col].astype("string")
    matches[region_col] = matches[region_col].astype("string").str.lower()
    matches[reach_id_col] = matches[reach_id_col].map(normalize_reach_id)
    matches = matches.dropna(subset=[reach_id_col])

    confidence_rank = {"high": 0, "medium": 1, "low": 2}
    if "confidence_class" in matches.columns:
        matches["_confidence_rank"] = (
            matches["confidence_class"]
            .astype("string")
            .str.lower()
            .map(confidence_rank)
            .fillna(3)
        )
    else:
        matches["_confidence_rank"] = 0
    if "total_score" not in matches.columns:
        matches["total_score"] = 0.0

    return (
        matches.sort_values(
            [station_key_col, "_confidence_rank", "total_score"],
            ascending=[True, True, False],
            kind="mergesort",
        )
        .drop_duplicates(subset=[station_key_col], keep="first")
        .drop(columns=["_confidence_rank"])
        .reset_index(drop=True)
    )


def _get_region_topology(
    region: str,
    *,
    catalog: SwordFileCatalog | None,
    topologies_by_region: Mapping[str, pd.DataFrame] | None,
    topology_cache: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    normalized_region = str(region).lower()
    if normalized_region in topology_cache:
        return topology_cache[normalized_region]
    if topologies_by_region is not None and normalized_region in topologies_by_region:
        topology = topologies_by_region[normalized_region].copy()
    elif catalog is not None:
        topology = load_reach_topology(catalog, normalized_region)
    else:
        topology = pd.DataFrame()
    topology_cache[normalized_region] = topology
    return topology


def _reverse_adjacency(adjacency: Mapping[int, set[int]]) -> dict[int, set[int]]:
    reverse: dict[int, set[int]] = defaultdict(set)
    for reach_id, downstream_ids in adjacency.items():
        reverse.setdefault(reach_id, set())
        for downstream_id in downstream_ids:
            reverse[downstream_id].add(reach_id)
    return dict(reverse)


def _traverse(adjacency: Mapping[int, set[int]], starts: set[int]) -> set[int]:
    seen: set[int] = set()
    queue: deque[int] = deque(starts)
    while queue:
        reach_id = queue.popleft()
        if reach_id in seen:
            continue
        seen.add(reach_id)
        queue.extend(adjacency.get(reach_id, set()) - seen)
    return seen


def _unique_values(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    values: list[str] = []
    for column in columns:
        if column not in frame.columns:
            continue
        values.extend(str(value) for value in frame[column].dropna().astype("string").tolist())
    return sorted(dict.fromkeys(value for value in values if value and value.lower() != "nan"))


def _unique_reach_ids(values: Sequence[object] | pd.Series) -> list[int]:
    normalized: list[int] = []
    for value in values:
        if isinstance(value, (list, tuple, set)) or (
            hasattr(value, "tolist")
            and not isinstance(value, (str, bytes, bytearray))
        ):
            normalized.extend(parse_reach_id_list(value))
            continue
        reach_id = normalize_reach_id(value)
        if reach_id is not None:
            normalized.append(reach_id)
    return list(dict.fromkeys(normalized))


def _sort_reach_ids(reach_ids: set[int], topology: pd.DataFrame, *, reach_id_col: str) -> list[int]:
    if not reach_ids:
        return []
    if "dist_out" not in topology.columns:
        return sorted(reach_ids)
    order = topology.drop_duplicates(subset=[reach_id_col]).copy()
    order[reach_id_col] = order[reach_id_col].map(normalize_reach_id)
    order["dist_out"] = pd.to_numeric(order["dist_out"], errors="coerce")
    dist_by_reach = order.set_index(reach_id_col)["dist_out"].to_dict()
    return sorted(reach_ids, key=lambda reach_id: (-dist_by_reach.get(reach_id, float("-inf")), reach_id))


def _station_keys_for_reaches(frame: pd.DataFrame, station_key_col: str, reach_id_col: str) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for row in frame[[station_key_col, reach_id_col]].itertuples(index=False):
        reach_id = normalize_reach_id(row[1])
        if reach_id is None:
            continue
        out.setdefault(reach_id, []).append(str(row[0]))
    return {reach_id: sorted(dict.fromkeys(keys)) for reach_id, keys in out.items()}


def _sorted_station_keys(values: pd.Series) -> list[str]:
    return sorted(dict.fromkeys(str(value) for value in values.dropna().astype("string").tolist()))


def _summary_record(
    *,
    example_col: str,
    example_id: object,
    region: str | None,
    upstream_keys: Sequence[str],
    downstream_keys: Sequence[str],
    upstream_reach_ids: Sequence[int],
    downstream_reach_ids: Sequence[int],
    selected_reach_ids: Sequence[int],
    missing_upstream: Sequence[str],
    missing_downstream: Sequence[str],
    route_status: str,
) -> dict[str, object]:
    downstream_set = set(downstream_reach_ids)
    selected_set = set(selected_reach_ids)
    return {
        example_col: example_id,
        "sword_region": region,
        "route_status": route_status,
        "route_found": bool(selected_reach_ids),
        "all_downstream_reaches_reached": downstream_set.issubset(selected_set) if downstream_set else False,
        "n_reaches_between": len(selected_reach_ids),
        "upstream_station_keys": _join(upstream_keys),
        "downstream_station_keys": _join(downstream_keys),
        "upstream_reach_ids": _join(str(value) for value in upstream_reach_ids),
        "downstream_reach_ids": _join(str(value) for value in downstream_reach_ids),
        "reaches_between": _join(str(value) for value in selected_reach_ids),
        "missing_upstream_station_keys": _join(missing_upstream),
        "missing_downstream_station_keys": _join(missing_downstream),
    }


def _join(values: Iterable[object]) -> str:
    return ";".join(str(value) for value in values if str(value))
