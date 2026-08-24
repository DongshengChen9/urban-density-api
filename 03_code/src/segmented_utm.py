from __future__ import annotations

import math
from typing import Any

import geopandas as gpd
from shapely.geometry import box


UTM_MIN_LAT = -80.0
UTM_MAX_LAT = 84.0


def utm_zone_from_longitude(longitude: float) -> int:
    """
    Return standard UTM zone number for a longitude in EPSG:4326 degrees.
    """
    if longitude < -180 or longitude > 180:
        raise ValueError(f"Longitude outside EPSG:4326 range: {longitude}")

    if longitude == 180:
        return 60

    zone = math.floor((longitude + 180) / 6) + 1
    return max(1, min(60, int(zone)))


def utm_epsg(zone: int, hemisphere: str) -> int:
    """
    Return EPSG code for a UTM zone and hemisphere.
    """
    if zone < 1 or zone > 60:
        raise ValueError(f"UTM zone must be between 1 and 60: {zone}")

    if hemisphere == "north":
        return 32600 + zone

    if hemisphere == "south":
        return 32700 + zone

    raise ValueError(f"Unsupported hemisphere: {hemisphere}")


def _zone_longitude_bounds(zone: int) -> tuple[float, float]:
    if zone < 1 or zone > 60:
        raise ValueError(f"UTM zone must be between 1 and 60: {zone}")

    min_lon = -180.0 + (zone - 1) * 6.0
    max_lon = min_lon + 6.0

    return min_lon, max_lon


def _hemispheres_for_bounds(min_lat: float, max_lat: float) -> list[str]:
    hemispheres = []

    if max_lat >= 0:
        hemispheres.append("north")

    if min_lat < 0:
        hemispheres.append("south")

    return hemispheres


def _diagnostics_for_bounds(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
) -> list[str]:
    diagnostics = []

    if min_lat < UTM_MIN_LAT or max_lat > UTM_MAX_LAT:
        diagnostics.append("aoi_outside_standard_utm_latitude_range")

    if max_lon - min_lon > 180:
        diagnostics.append("possible_antimeridian_crossing")

    return diagnostics


def ensure_wgs84(aoi: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Validate AOI and return it in EPSG:4326 for longitude-based segmentation.
    """
    if aoi is None:
        raise ValueError("AOI GeoDataFrame is None.")

    if aoi.empty:
        raise ValueError("AOI GeoDataFrame is empty.")

    if aoi.crs is None:
        raise ValueError("AOI GeoDataFrame has no CRS.")

    return aoi.to_crs("EPSG:4326")


def build_utm_zone_polygons(aoi: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Build UTM zone/hemisphere polygons intersecting an AOI.
    """
    aoi_wgs84 = ensure_wgs84(aoi)
    min_lon, min_lat, max_lon, max_lat = aoi_wgs84.total_bounds
    diagnostics = _diagnostics_for_bounds(
        float(min_lon),
        float(min_lat),
        float(max_lon),
        float(max_lat),
    )

    min_zone = utm_zone_from_longitude(float(min_lon))
    max_zone = utm_zone_from_longitude(float(max_lon))
    hemispheres = _hemispheres_for_bounds(float(min_lat), float(max_lat))

    records: list[dict[str, Any]] = []
    geometries = []

    for zone in range(min_zone, max_zone + 1):
        zone_min_lon, zone_max_lon = _zone_longitude_bounds(zone)

        for hemisphere in hemispheres:
            if hemisphere == "north":
                hem_min_lat = max(0.0, UTM_MIN_LAT)
                hem_max_lat = max(UTM_MAX_LAT, float(max_lat))
            else:
                hem_min_lat = min(UTM_MIN_LAT, float(min_lat))
                hem_max_lat = min(0.0, UTM_MAX_LAT)

            records.append(
                {
                    "utm_zone": zone,
                    "hemisphere": hemisphere,
                    "epsg": utm_epsg(zone, hemisphere),
                    "diagnostics": diagnostics,
                }
            )
            geometries.append(
                box(zone_min_lon, hem_min_lat, zone_max_lon, hem_max_lat)
            )

    zone_polygons = gpd.GeoDataFrame(records, geometry=geometries, crs="EPSG:4326")
    aoi_union = aoi_wgs84.geometry.union_all()
    intersects = zone_polygons.geometry.intersects(aoi_union)

    return zone_polygons.loc[intersects].reset_index(drop=True)


def segment_aoi_by_utm_zone(aoi: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Split/intersect an AOI with standard UTM zone and hemisphere polygons.
    """
    aoi_wgs84 = ensure_wgs84(aoi)
    aoi_union = aoi_wgs84.geometry.union_all()
    zone_polygons = build_utm_zone_polygons(aoi_wgs84)

    records: list[dict[str, Any]] = []
    geometries = []

    for _, zone_row in zone_polygons.iterrows():
        segment_geometry = aoi_union.intersection(zone_row.geometry)

        if segment_geometry.is_empty:
            continue

        records.append(
            {
                "segment_id": len(records) + 1,
                "utm_zone": int(zone_row["utm_zone"]),
                "hemisphere": zone_row["hemisphere"],
                "epsg": int(zone_row["epsg"]),
                "diagnostics": list(zone_row["diagnostics"]),
            }
        )
        geometries.append(segment_geometry)

    return gpd.GeoDataFrame(records, geometry=geometries, crs="EPSG:4326")
