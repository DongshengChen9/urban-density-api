from __future__ import annotations

import geopandas as gpd
from shapely.geometry import box

def create_aoi_from_bbox(name, bounds, crs="EPSG:4326") -> gpd.GeoDataFrame:
    """
    Create an AOI GeoDataFrame from a bounding box.

    Parameters
    ----------
    name : str
        AOI name.
    bounds : dict or tuple
        Either a dictionary with keys minx, miny, maxx, maxy
        or a tuple/list in the order (minx, miny, maxx, maxy).
    crs : str
        Coordinate reference system of the bbox coordinates.

    Returns
    -------
    geopandas.GeoDataFrame
        AOI GeoDataFrame with columns aoi_id, aoi_name, geometry.
    """
    if isinstance(bounds, dict):
        minx = bounds["minx"]
        miny = bounds["miny"]
        maxx = bounds["maxx"]
        maxy = bounds["maxy"]
    else:
        minx, miny, maxx, maxy = bounds

    if minx >= maxx or miny >= maxy:
        raise ValueError("Invalid bbox: minx/miny must be smaller than maxx/maxy.")

    geometry = box(minx, miny, maxx, maxy)

    aoi_gdf = gpd.GeoDataFrame(
        {
            "aoi_id": [1],
            "aoi_name": [name],
        },
        geometry=[geometry],
        crs=crs,
    )

    return aoi_gdf


def validate_aoi(aoi_gdf) -> gpd.GeoDataFrame:
    """
    Validate an AOI GeoDataFrame before it is used in the workflow.
    """
    if aoi_gdf is None:
        raise ValueError("AOI is None.")

    if aoi_gdf.empty:
        raise ValueError("AOI GeoDataFrame is empty.")

    if aoi_gdf.crs is None:
        raise ValueError("AOI has no CRS. Define CRS before running the workflow.")

    aoi = aoi_gdf.copy()

    aoi = aoi[aoi.geometry.notna()]
    aoi = aoi[~aoi.geometry.is_empty]

    if aoi.empty:
        raise ValueError("AOI contains no non-empty geometries.")

    invalid_count = (~aoi.geometry.is_valid).sum()
    if invalid_count > 0:
        aoi["geometry"] = aoi.geometry.make_valid()

    return aoi


def dissolve_aoi(aoi_gdf) -> gpd.GeoDataFrame:
    """
    Dissolve AOI geometries into a single AOI feature.
    """
    aoi = validate_aoi(aoi_gdf)

    dissolved = gpd.GeoDataFrame(
        {
            "aoi_id": [1],
            "aoi_name": ["dissolved_aoi"],
        },
        geometry=[aoi.geometry.union_all()],
        crs=aoi.crs,
    )

    return dissolved


def load_aoi(path, layer=None) -> gpd.GeoDataFrame:
    """
    Load AOI from a vector file and validate it.
    """
    if layer is None:
        aoi = gpd.read_file(path)
    else:
        aoi = gpd.read_file(path, layer=layer)

    return validate_aoi(aoi)
