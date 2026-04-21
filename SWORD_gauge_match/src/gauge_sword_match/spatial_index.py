from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import geopandas as gpd
from pyproj import Transformer
from shapely import STRtree
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

from .utils import expand_point_bbox


@dataclass(frozen=True, slots=True)
class IndexedDistance:
    row_index: int
    distance_m: float


class GeometrySpatialIndex:
    def __init__(self, frame: gpd.GeoDataFrame):
        self.frame = frame.reset_index(drop=True)
        self._tree = STRtree(self.frame.geometry.to_numpy())

    def query(self, point: BaseGeometry, search_radius_m: float, max_results: int | None = None) -> list[IndexedDistance]:
        min_x, min_y, max_x, max_y = expand_point_bbox(point.x, point.y, search_radius_m)
        hits = self._tree.query(box(min_x, min_y, max_x, max_y))
        if len(hits) == 0:
            return []

        distances = distance_to_geometries(point, [self.frame.geometry.iloc[int(idx)] for idx in hits])
        results = [
            IndexedDistance(row_index=int(row_index), distance_m=float(distance))
            for row_index, distance in zip(hits, distances)
            if distance <= search_radius_m
        ]
        results.sort(key=lambda item: item.distance_m)
        if max_results is not None:
            return results[:max_results]
        return results


def distance_to_geometries(point: BaseGeometry, geometries: Iterable[BaseGeometry]) -> list[float]:
    transformer = _build_local_transformer(round(point.y, 5), round(point.x, 5))
    projected_point = transform(transformer.transform, point)
    distances: list[float] = []
    for geometry in geometries:
        projected_geometry = transform(transformer.transform, geometry)
        distances.append(float(projected_point.distance(projected_geometry)))
    return distances


@lru_cache(maxsize=4096)
def _build_local_transformer(lat: float, lon: float) -> Transformer:
    crs = f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs"
    return Transformer.from_crs("EPSG:4326", crs, always_xy=True)

