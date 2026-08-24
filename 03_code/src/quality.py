from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd


def _is_projected(gdf: gpd.GeoDataFrame) -> bool:
    if gdf is None or gdf.crs is None:
        return False
    return not gdf.crs.is_geographic


def _first_value(gdf: gpd.GeoDataFrame, column: str):
    if column not in gdf.columns:
        return None

    values = gdf[column].dropna()

    if values.empty:
        return None

    return values.iloc[0]


def summarize_building_quality(buildings: gpd.GeoDataFrame) -> dict[str, Any]:
    """
    Summarize basic quality properties of a building GeoDataFrame.

    This function does not repair data. It reports whether the data are suitable
    for later indicator calculation.
    """
    if buildings is None:
        raise ValueError("Buildings GeoDataFrame is None.")

    if buildings.empty:
        return {
            "n_buildings": 0,
            "crs": str(buildings.crs),
            "is_projected": _is_projected(buildings),
            "status": "empty",
        }

    if buildings.crs is None:
        raise ValueError("Buildings GeoDataFrame has no CRS.")

    n_buildings = len(buildings)

    invalid_geometry_count = int((~buildings.geometry.is_valid).sum())
    empty_geometry_count = int(buildings.geometry.is_empty.sum())
    missing_geometry_count = int(buildings.geometry.isna().sum())

    duplicate_building_id_count = None
    if "building_id" in buildings.columns:
        duplicate_building_id_count = int(buildings["building_id"].duplicated().sum())

    height_missing_count = None
    height_missing_share = None
    height_valid_area_share = None

    if "height_m" in buildings.columns:
        height_missing_count = int(buildings["height_m"].isna().sum())
        height_missing_share = float(buildings["height_m"].isna().mean())

        if "footprint_area_m2" in buildings.columns:
            total_area = buildings["footprint_area_m2"].sum()
            valid_height_area = buildings.loc[
                buildings["height_m"].notna(), "footprint_area_m2"
            ].sum()

            if total_area > 0:
                height_valid_area_share = float(valid_height_area / total_area)

    floors_missing_count = None
    floors_missing_share = None
    floor_valid_area_share = None

    if "num_floors" in buildings.columns:
        floors_missing_count = int(buildings["num_floors"].isna().sum())
        floors_missing_share = float(buildings["num_floors"].isna().mean())

        if "footprint_area_m2" in buildings.columns:
            total_area = buildings["footprint_area_m2"].sum()
            valid_floor_area = buildings.loc[
                buildings["num_floors"].notna(), "footprint_area_m2"
            ].sum()

            if total_area > 0:
                floor_valid_area_share = float(valid_floor_area / total_area)

    total_footprint_area_m2 = None
    min_footprint_area_m2 = None
    max_footprint_area_m2 = None

    if "footprint_area_m2" in buildings.columns:
        total_footprint_area_m2 = float(buildings["footprint_area_m2"].sum())
        min_footprint_area_m2 = float(buildings["footprint_area_m2"].min())
        max_footprint_area_m2 = float(buildings["footprint_area_m2"].max())

    quality = {
        "n_buildings": int(n_buildings),
        "crs": str(buildings.crs),
        "is_projected": _is_projected(buildings),
        "geometry_types": buildings.geometry.geom_type.value_counts().to_dict(),
        "missing_geometry_count": missing_geometry_count,
        "empty_geometry_count": empty_geometry_count,
        "invalid_geometry_count": invalid_geometry_count,
        "duplicate_building_id_count": duplicate_building_id_count,
        "missing_height_count": height_missing_count,
        "missing_height_share": height_missing_share,
        "height_valid_area_share": height_valid_area_share,
        "missing_num_floors_count": floors_missing_count,
        "missing_num_floors_share": floors_missing_share,
        "floor_valid_area_share": floor_valid_area_share,
        "total_footprint_area_m2": total_footprint_area_m2,
        "min_footprint_area_m2": min_footprint_area_m2,
        "max_footprint_area_m2": max_footprint_area_m2,
        "source": _first_value(buildings, "source"),
        "source_release": _first_value(buildings, "source_release"),
    }

    return quality


def check_indicator_readiness(buildings: gpd.GeoDataFrame) -> dict[str, Any]:
    """
    Check which indicators are supported by the current building data.

    This function does not decide whether the results are scientifically strong.
    It reports technical/data availability readiness.
    """
    if buildings is None:
        raise ValueError("Buildings GeoDataFrame is None.")

    if buildings.empty:
        return {
            "gsi": "not_ready_empty_building_layer",
            "far_fsi": "not_ready_empty_building_layer",
            "built_volume_density": "not_ready_empty_building_layer",
            "neighbor_distance": "not_ready_empty_building_layer",
            "height_to_distance_ratio": "not_ready_empty_building_layer",
        }

    if buildings.crs is None:
        raise ValueError("Buildings GeoDataFrame has no CRS.")

    is_projected = _is_projected(buildings)
    has_valid_geometry = (
        buildings.geometry.notna().all()
        and (~buildings.geometry.is_empty).all()
        and buildings.geometry.is_valid.all()
    )

    has_footprint_area = "footprint_area_m2" in buildings.columns
    has_height = "height_m" in buildings.columns and buildings["height_m"].notna().any()
    has_floors = (
        "num_floors" in buildings.columns and buildings["num_floors"].notna().any()
    )

    n_buildings = len(buildings)

    readiness = {
        "gsi": {
            "status": "ready" if is_projected and has_valid_geometry else "not_ready",
            "reason": "Requires projected CRS, valid building geometries, and footprint areas.",
            "required_data": ["geometry"],
            "available": {
                "is_projected": is_projected,
                "has_valid_geometry": bool(has_valid_geometry),
                "has_footprint_area_m2": has_footprint_area,
            },
        },
        "far_fsi": {
            "status": "conditional_ready" if is_projected and has_floors else "not_ready",
            "reason": "Requires reliable floor-count or floor-area data. Missing floor counts are data gaps, not zero.",
            "required_data": ["geometry", "num_floors"],
            "available": {
                "is_projected": is_projected,
                "has_num_floors": bool(has_floors),
                "missing_num_floors_share": (
                    float(buildings["num_floors"].isna().mean())
                    if "num_floors" in buildings.columns
                    else None
                ),
            },
        },
        "built_volume_density": {
            "status": "conditional_ready" if is_projected and has_height else "not_ready",
            "reason": "Requires reliable height data. Missing heights are data gaps, not zero.",
            "required_data": ["geometry", "height_m"],
            "available": {
                "is_projected": is_projected,
                "has_height_m": bool(has_height),
                "missing_height_share": (
                    float(buildings["height_m"].isna().mean())
                    if "height_m" in buildings.columns
                    else None
                ),
            },
        },
        "neighbor_distance": {
            "status": "ready" if is_projected and n_buildings >= 2 else "not_ready",
            "reason": "Requires projected CRS and at least two buildings.",
            "required_data": ["geometry"],
            "available": {
                "is_projected": is_projected,
                "n_buildings": int(n_buildings),
            },
        },
        "height_to_distance_ratio": {
            "status": (
                "conditional_ready"
                if is_projected and n_buildings >= 2 and has_height
                else "not_ready"
            ),
            "reason": "Requires projected CRS, neighbour distance, and valid height data.",
            "required_data": ["geometry", "height_m"],
            "available": {
                "is_projected": is_projected,
                "n_buildings": int(n_buildings),
                "has_height_m": bool(has_height),
                "missing_height_share": (
                    float(buildings["height_m"].isna().mean())
                    if "height_m" in buildings.columns
                    else None
                ),
            },
        },
    }

    return readiness


def summarize_unit_quality(units: gpd.GeoDataFrame) -> dict[str, Any]:
    """
    Summarize basic quality properties of aggregation units.
    """
    if units is None:
        raise ValueError("Units GeoDataFrame is None.")

    if units.empty:
        return {
            "n_units": 0,
            "status": "empty",
        }

    if units.crs is None:
        raise ValueError("Units GeoDataFrame has no CRS.")

    duplicate_unit_id_count = None
    if "unit_id" in units.columns:
        duplicate_unit_id_count = int(units["unit_id"].duplicated().sum())

    unit_area_min_m2 = None
    unit_area_max_m2 = None
    unit_area_total_m2 = None

    if not units.crs.is_geographic:
        areas = units.geometry.area
        unit_area_min_m2 = float(areas.min())
        unit_area_max_m2 = float(areas.max())
        unit_area_total_m2 = float(areas.sum())

    return {
        "n_units": int(len(units)),
        "crs": str(units.crs),
        "is_projected": _is_projected(units),
        "duplicate_unit_id_count": duplicate_unit_id_count,
        "empty_geometry_count": int(units.geometry.is_empty.sum()),
        "invalid_geometry_count": int((~units.geometry.is_valid).sum()),
        "unit_area_min_m2": unit_area_min_m2,
        "unit_area_max_m2": unit_area_max_m2,
        "unit_area_total_m2": unit_area_total_m2,
    }


def export_quality_report(quality_dict: dict[str, Any], path) -> None:
    """
    Export a simple Markdown quality report.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Data quality report", ""]

    for key, value in quality_dict.items():
        lines.append(f"- **{key}**: {value}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
