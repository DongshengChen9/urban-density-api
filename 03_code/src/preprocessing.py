from __future__ import annotations

import geopandas as gpd

def estimate_metric_crs(gdf) -> str:
    """
    Estimate a suitable projected CRS for metric area and distance calculations.

    For small AOIs, GeoPandas' estimate_utm_crs() is appropriate because it
    selects a local UTM CRS based on the geometry location.

    Returns
    -------
    str
        CRS string, for example "EPSG:32633" for Vienna.
    """
    if gdf is None:
        raise ValueError("Input GeoDataFrame is None.")

    if gdf.empty:
        raise ValueError("Input GeoDataFrame is empty.")

    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame has no CRS.")

    try:
        metric_crs = gdf.estimate_utm_crs()
    except Exception as exc:
        raise RuntimeError("Could not estimate a metric CRS.") from exc

    if metric_crs is None:
        raise RuntimeError("GeoPandas could not estimate a UTM CRS for this AOI.")

    return metric_crs.to_string()

def reproject_to_metric(buildings, aoi, target_crs="auto"):
    """
    Reproject buildings and AOI to a projected metric CRS.
    """
    if buildings.crs is None:
        raise ValueError("Buildings GeoDataFrame has no CRS.")

    if aoi.crs is None:
        raise ValueError("AOI GeoDataFrame has no CRS.")

    if target_crs == "auto":
        target_crs = estimate_metric_crs(aoi)

    buildings_metric = buildings.to_crs(target_crs)
    aoi_metric = aoi.to_crs(target_crs)

    return buildings_metric, aoi_metric

def clean_building_geometries(buildings):
    """
    Clean building geometries before metric calculations.

    This function removes missing and empty geometries, repairs invalid
    geometries where possible, and keeps only polygonal geometries.
    """
    if buildings is None:
        raise ValueError("Buildings GeoDataFrame is None.")

    if buildings.empty:
        raise ValueError("Buildings GeoDataFrame is empty.")

    if buildings.crs is None:
        raise ValueError("Buildings GeoDataFrame has no CRS.")

    cleaned = buildings.copy()

    valid_geometry_mask = cleaned.geometry.apply(
    lambda geom: geom is not None and not geom.is_empty
    )

    cleaned = cleaned[valid_geometry_mask]

    if cleaned.empty:
        raise ValueError("No non-empty building geometries remain after cleaning.")

    invalid_count = (~cleaned.geometry.is_valid).sum()
    if invalid_count > 0:
        cleaned["geometry"] = cleaned.geometry.make_valid()

    polygon_types = ["Polygon", "MultiPolygon"]
    cleaned = cleaned[cleaned.geometry.geom_type.isin(polygon_types)]

    if cleaned.empty:
        raise ValueError("No polygonal building geometries remain after cleaning.")

    return cleaned


def clip_buildings_to_aoi(buildings, aoi):
    """
    Clip buildings to the AOI boundary.

    Buildings and AOI must be in the same projected CRS.
    """
    if buildings is None:
        raise ValueError("Buildings GeoDataFrame is None.")

    if aoi is None:
        raise ValueError("AOI GeoDataFrame is None.")

    if buildings.crs is None or aoi.crs is None:
        raise ValueError("Buildings and AOI must both have CRS.")

    if buildings.crs != aoi.crs:
        raise ValueError(
            f"CRS mismatch: buildings CRS is {buildings.crs}, AOI CRS is {aoi.crs}."
        )

    clipped = gpd.clip(buildings, aoi)

    valid_geometry_mask = clipped.geometry.apply(
    lambda geom: geom is not None and not geom.is_empty
    )

    clipped = clipped[valid_geometry_mask]

    if clipped.empty:
        raise ValueError("No buildings remain after clipping to AOI.")

    return clipped


def add_footprint_area(buildings):
    """
    Add footprint area in square metres.

    The input must be in a projected metric CRS.
    """
    if buildings is None:
        raise ValueError("Buildings GeoDataFrame is None.")

    if buildings.crs is None:
        raise ValueError("Buildings GeoDataFrame has no CRS.")

    if buildings.crs.is_geographic:
        raise ValueError(
            "Buildings are in a geographic CRS. Reproject to a metric CRS before calculating area."
        )

    buildings_with_area = buildings.copy()
    buildings_with_area["footprint_area_m2"] = buildings_with_area.geometry.area

    return buildings_with_area
