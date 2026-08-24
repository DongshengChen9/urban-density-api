from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

INDICATOR_DEFINITION_VERSION = "2"
GSI_NUMERICAL_TOLERANCE = 1e-9
FOOTPRINT_OVERLAP_AREA_TOLERANCE_M2 = 1e-6


@dataclass(frozen=True)
class IndicatorSpec:
    """
    Metadata for one indicator in the workflow registry.
    """

    key: str
    name: str
    category: str
    function: Callable[[gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, Any]], pd.DataFrame]
    required_building_cols: tuple[str, ...]
    required_unit_cols: tuple[str, ...]
    output_cols: tuple[str, ...]
    default_enabled: bool
    conditional: bool
    description: str


def ensure_projected_crs(gdf: gpd.GeoDataFrame, name: str) -> None:
    """
    Ensure that a GeoDataFrame has a projected CRS.

    Area and distance indicators must not be calculated in EPSG:4326.
    """
    if gdf is None:
        raise ValueError(f"{name} GeoDataFrame is None.")

    if gdf.empty:
        raise ValueError(f"{name} GeoDataFrame is empty.")

    if gdf.crs is None:
        raise ValueError(f"{name} GeoDataFrame has no CRS.")

    if gdf.crs.is_geographic:
        raise ValueError(
            f"{name} GeoDataFrame is in a geographic CRS ({gdf.crs}). "
            "Reproject to a metric CRS before calculating indicators."
        )


def ensure_same_projected_crs(
    buildings: gpd.GeoDataFrame,
    units: gpd.GeoDataFrame,
) -> None:
    """
    Check CRS consistency between buildings and aggregation units.
    """
    ensure_projected_crs(buildings, "Buildings")
    ensure_projected_crs(units, "Aggregation units")

    if buildings.crs != units.crs:
        raise ValueError(
            f"CRS mismatch: buildings CRS is {buildings.crs}, "
            f"units CRS is {units.crs}."
        )


def validate_required_columns(
    gdf: gpd.GeoDataFrame,
    required_cols: tuple[str, ...],
    gdf_name: str,
) -> None:
    """
    Check that required columns exist before calculating an indicator.
    """
    missing = [col for col in required_cols if col not in gdf.columns]

    if missing:
        raise ValueError(f"{gdf_name} is missing required columns: {missing}")


def _prepare_units(units: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Prepare aggregation units with unit area.
    """
    validate_required_columns(units, ("unit_id", "geometry"), "units")
    ensure_projected_crs(units, "Aggregation units")

    u = units[["unit_id", "geometry"]].copy()

    if "unit_area_m2" in units.columns:
        u["unit_area_m2"] = units["unit_area_m2"].values
    else:
        u["unit_area_m2"] = u.geometry.area

    return u


def _empty_unit_result(
    units: gpd.GeoDataFrame,
    output_cols: list[str],
) -> pd.DataFrame:
    """
    Return one row per unit with NaN output columns.
    """
    out = units[["unit_id"]].copy()

    for col in output_cols:
        out[col] = np.nan

    return pd.DataFrame(out.drop(columns="geometry", errors="ignore"))


def calculate_gsi(
    buildings: gpd.GeoDataFrame,
    units: gpd.GeoDataFrame,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Calculate Ground Space Index / Building Coverage Ratio.

    GSI = area(union(building footprint intersections inside unit)) / unit area

    Scientific note:
    Buildings are intersected with aggregation units. This avoids assigning
    an entire building to one grid cell when it crosses a grid boundary.
    """
    validate_required_columns(buildings, ("building_id", "geometry"), "buildings")
    validate_required_columns(units, ("unit_id", "geometry"), "units")
    ensure_projected_crs(units, "Aggregation units")
    u = _prepare_units(units)
    b = buildings[["building_id", "geometry"]].copy()
    if not b.empty:
        ensure_same_projected_crs(buildings, units)

    out = u[["unit_id", "unit_area_m2"]].copy()
    diagnostic_columns = [
        "building_footprint_area_m2",
        "building_footprint_area_raw_sum_m2",
        "building_footprint_union_area_m2",
        "gsi",
        "gsi_raw_sum",
        "footprint_overlap_area_m2",
        "gsi_overlap_difference",
        "footprint_overlap_flag",
    ]

    if b.empty:
        for column in diagnostic_columns:
            out[column] = 0.0
        out["footprint_overlap_flag"] = False
        return out.drop(columns=["unit_area_m2"])

    intersections = gpd.overlay(
        u[["unit_id", "geometry"]],
        b[["building_id", "geometry"]],
        how="intersection",
        keep_geom_type=True,
    )

    if intersections.empty:
        for column in diagnostic_columns:
            out[column] = 0.0
        out["footprint_overlap_flag"] = False
        return out.drop(columns=["unit_area_m2"])

    intersections["building_area_in_unit_m2"] = intersections.geometry.area
    raw = (
        intersections.groupby("unit_id")["building_area_in_unit_m2"]
        .sum()
        .reset_index(name="building_footprint_area_raw_sum_m2")
    )
    unioned = intersections[["unit_id", "geometry"]].dissolve(
        by="unit_id", as_index=False
    )
    unioned["building_footprint_union_area_m2"] = unioned.geometry.area
    out = out.merge(raw, on="unit_id", how="left").merge(
        unioned[["unit_id", "building_footprint_union_area_m2"]],
        on="unit_id",
        how="left",
    )
    out["building_footprint_area_raw_sum_m2"] = out[
        "building_footprint_area_raw_sum_m2"
    ].fillna(0.0)
    out["building_footprint_union_area_m2"] = out[
        "building_footprint_union_area_m2"
    ].fillna(0.0)
    out["footprint_overlap_area_m2"] = (
        out["building_footprint_area_raw_sum_m2"]
        - out["building_footprint_union_area_m2"]
    )
    out.loc[
        out["footprint_overlap_area_m2"].abs()
        <= FOOTPRINT_OVERLAP_AREA_TOLERANCE_M2,
        "footprint_overlap_area_m2",
    ] = 0.0
    if (out["footprint_overlap_area_m2"] < -FOOTPRINT_OVERLAP_AREA_TOLERANCE_M2).any():
        raise RuntimeError("Unioned footprint coverage exceeds raw summed coverage.")
    out["building_footprint_area_m2"] = out["building_footprint_union_area_m2"]
    out["gsi"] = out["building_footprint_union_area_m2"] / out["unit_area_m2"]
    out["gsi_raw_sum"] = (
        out["building_footprint_area_raw_sum_m2"] / out["unit_area_m2"]
    )
    out["gsi_overlap_difference"] = out["gsi_raw_sum"] - out["gsi"]
    out["footprint_overlap_flag"] = (
        out["footprint_overlap_area_m2"] > FOOTPRINT_OVERLAP_AREA_TOLERANCE_M2
    )
    if (
        (out["gsi"] < -GSI_NUMERICAL_TOLERANCE)
        | (out["gsi"] > 1 + GSI_NUMERICAL_TOLERANCE)
    ).any():
        raise RuntimeError("Union-based GSI falls outside [0, 1] beyond numerical tolerance.")
    return out.drop(columns=["unit_area_m2"])


def calculate_far_fsi(
    buildings: gpd.GeoDataFrame,
    units: gpd.GeoDataFrame,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Calculate FAR/FSI using number of floors.

    FAR/FSI = sum(footprint area inside unit * num_floors) / unit area

    Missing floor counts remain missing. They are not replaced by 0 or 1.
    """
    ensure_same_projected_crs(buildings, units)
    validate_required_columns(buildings, ("building_id", "geometry", "num_floors"), "buildings")
    validate_required_columns(units, ("unit_id", "geometry"), "units")

    u = _prepare_units(units)
    b = buildings[["building_id", "geometry", "num_floors"]].copy()

    b["num_floors"] = pd.to_numeric(b["num_floors"], errors="coerce")
    b.loc[b["num_floors"] <= 0, "num_floors"] = np.nan

    intersections = gpd.overlay(
        u[["unit_id", "geometry"]],
        b[["building_id", "geometry", "num_floors"]],
        how="intersection",
        keep_geom_type=True,
    )

    out = u[["unit_id", "unit_area_m2"]].copy()

    if intersections.empty:
        out["floor_area_sum_m2"] = 0.0
        out["far_fsi"] = 0.0
        out["floor_data_valid_area_share"] = np.nan
        return out[["unit_id", "floor_area_sum_m2", "far_fsi", "floor_data_valid_area_share"]]

    intersections["area_in_unit_m2"] = intersections.geometry.area
    intersections["floor_area_m2"] = (
        intersections["area_in_unit_m2"] * intersections["num_floors"]
    )

    total_footprint_by_unit = (
        intersections.groupby("unit_id")["area_in_unit_m2"]
        .sum()
        .reset_index(name="building_footprint_area_m2")
    )

    floor_area_by_unit = (
        intersections.groupby("unit_id")["floor_area_m2"]
        .sum(min_count=1)
        .reset_index(name="floor_area_sum_m2")
    )

    valid_floor_area_by_unit = (
        intersections[intersections["num_floors"].notna()]
        .groupby("unit_id")["area_in_unit_m2"]
        .sum()
        .reset_index(name="floor_data_valid_area_m2")
    )

    out = out.merge(total_footprint_by_unit, on="unit_id", how="left")
    out = out.merge(floor_area_by_unit, on="unit_id", how="left")
    out = out.merge(valid_floor_area_by_unit, on="unit_id", how="left")

    out["building_footprint_area_m2"] = out["building_footprint_area_m2"].fillna(0.0)
    out["floor_data_valid_area_m2"] = out["floor_data_valid_area_m2"].fillna(0.0)

    # Units without observed buildings get FAR/FSI = 0.
    no_buildings = out["building_footprint_area_m2"] == 0
    out.loc[no_buildings, "floor_area_sum_m2"] = 0.0

    out["far_fsi"] = out["floor_area_sum_m2"] / out["unit_area_m2"]

    out["floor_data_valid_area_share"] = np.where(
        out["building_footprint_area_m2"] > 0,
        out["floor_data_valid_area_m2"] / out["building_footprint_area_m2"],
        np.nan,
    )

    return out[["unit_id", "floor_area_sum_m2", "far_fsi", "floor_data_valid_area_share"]]


def calculate_built_volume_density(
    buildings: gpd.GeoDataFrame,
    units: gpd.GeoDataFrame,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Calculate Built Volume Density.

    Built Volume Density = sum(footprint area inside unit * height_m) / unit area

    Missing heights remain missing. They are not replaced by zero.
    """
    ensure_same_projected_crs(buildings, units)
    validate_required_columns(buildings, ("building_id", "geometry", "height_m"), "buildings")
    validate_required_columns(units, ("unit_id", "geometry"), "units")

    u = _prepare_units(units)
    b = buildings[["building_id", "geometry", "height_m"]].copy()

    b["height_m"] = pd.to_numeric(b["height_m"], errors="coerce")
    b.loc[b["height_m"] <= 0, "height_m"] = np.nan

    intersections = gpd.overlay(
        u[["unit_id", "geometry"]],
        b[["building_id", "geometry", "height_m"]],
        how="intersection",
        keep_geom_type=True,
    )

    out = u[["unit_id", "unit_area_m2"]].copy()

    if intersections.empty:
        out["built_volume_m3"] = 0.0
        out["built_volume_density"] = 0.0
        out["height_valid_area_share"] = np.nan
        return out[["unit_id", "built_volume_m3", "built_volume_density", "height_valid_area_share"]]

    intersections["area_in_unit_m2"] = intersections.geometry.area
    intersections["built_volume_m3"] = (
        intersections["area_in_unit_m2"] * intersections["height_m"]
    )

    total_footprint_by_unit = (
        intersections.groupby("unit_id")["area_in_unit_m2"]
        .sum()
        .reset_index(name="building_footprint_area_m2")
    )

    volume_by_unit = (
        intersections.groupby("unit_id")["built_volume_m3"]
        .sum(min_count=1)
        .reset_index(name="built_volume_m3")
    )

    valid_height_area_by_unit = (
        intersections[intersections["height_m"].notna()]
        .groupby("unit_id")["area_in_unit_m2"]
        .sum()
        .reset_index(name="height_valid_area_m2")
    )

    out = out.merge(total_footprint_by_unit, on="unit_id", how="left")
    out = out.merge(volume_by_unit, on="unit_id", how="left")
    out = out.merge(valid_height_area_by_unit, on="unit_id", how="left")

    out["building_footprint_area_m2"] = out["building_footprint_area_m2"].fillna(0.0)
    out["height_valid_area_m2"] = out["height_valid_area_m2"].fillna(0.0)

    no_buildings = out["building_footprint_area_m2"] == 0
    out.loc[no_buildings, "built_volume_m3"] = 0.0

    out["built_volume_density"] = out["built_volume_m3"] / out["unit_area_m2"]

    out["height_valid_area_share"] = np.where(
        out["building_footprint_area_m2"] > 0,
        out["height_valid_area_m2"] / out["building_footprint_area_m2"],
        np.nan,
    )

    return out[["unit_id", "built_volume_m3", "built_volume_density", "height_valid_area_share"]]


def calculate_area_indicators_shared(
    buildings: gpd.GeoDataFrame,
    units: gpd.GeoDataFrame,
    enabled: set[str],
    chunk_size: int | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """Calculate GSI, FAR/FSI, and BVD from one exact intersection table.

    The spatial rule and formulas are identical to the individual indicator
    functions. Chunking partitions aggregation units only; every exact
    building-cell intersection is retained once.
    """
    supported = {"gsi", "far_fsi", "built_volume_density"}
    requested = enabled & supported
    if not requested:
        return {}

    ensure_same_projected_crs(buildings, units)
    validate_required_columns(buildings, ("building_id", "geometry"), "buildings")
    validate_required_columns(units, ("unit_id", "geometry"), "units")
    if "far_fsi" in requested:
        validate_required_columns(buildings, ("num_floors",), "buildings")
    if "built_volume_density" in requested:
        validate_required_columns(buildings, ("height_m",), "buildings")

    started = time.perf_counter()
    u = _prepare_units(units)
    attribute_columns = [
        column
        for column, indicator in [
            ("num_floors", "far_fsi"),
            ("height_m", "built_volume_density"),
        ]
        if indicator in requested
    ]
    b = buildings[["building_id", *attribute_columns, "geometry"]].copy()
    if "num_floors" in b.columns:
        b["num_floors"] = pd.to_numeric(b["num_floors"], errors="coerce")
        b.loc[b["num_floors"] <= 0, "num_floors"] = np.nan
    if "height_m" in b.columns:
        b["height_m"] = pd.to_numeric(b["height_m"], errors="coerce")
        b.loc[b["height_m"] <= 0, "height_m"] = np.nan

    effective_chunk_size = int(chunk_size or len(u) or 1)
    if effective_chunk_size <= 0:
        raise ValueError("Area intersection chunk size must be positive.")

    pieces: list[gpd.GeoDataFrame] = []
    for start in range(0, len(u), effective_chunk_size):
        unit_chunk = u.iloc[start : start + effective_chunk_size]
        piece = gpd.overlay(
            unit_chunk[["unit_id", "geometry"]],
            b,
            how="intersection",
            keep_geom_type=True,
        )
        if not piece.empty:
            pieces.append(piece)

    if pieces:
        intersections = gpd.GeoDataFrame(
            pd.concat(pieces, ignore_index=True),
            geometry="geometry",
            crs=u.crs,
        )
        intersections["area_in_unit_m2"] = intersections.geometry.area
    else:
        intersections = gpd.GeoDataFrame(
            columns=[
                "unit_id",
                "building_id",
                *attribute_columns,
                "geometry",
                "area_in_unit_m2",
            ],
            geometry="geometry",
            crs=u.crs,
        )

    if metrics is not None:
        metrics.update(
            {
                "building_grid_candidate_pairs": int(len(intersections)),
                "building_grid_exact_intersections": int(len(intersections)),
                "area_intersection_seconds": float(time.perf_counter() - started),
                "area_intersection_chunk_size": effective_chunk_size,
            }
        )

    out_base = u[["unit_id", "unit_area_m2"]].copy()
    total_area = (
        intersections.groupby("unit_id")["area_in_unit_m2"]
        .sum()
        .rename("building_footprint_area_m2")
        .reset_index()
        if not intersections.empty
        else pd.DataFrame(columns=["unit_id", "building_footprint_area_m2"])
    )
    base = out_base.merge(total_area, on="unit_id", how="left")
    base["building_footprint_area_m2"] = base["building_footprint_area_m2"].fillna(0.0)
    no_buildings = base["building_footprint_area_m2"] == 0
    results: dict[str, pd.DataFrame] = {}

    if "gsi" in requested:
        result = base[["unit_id", "building_footprint_area_m2", "unit_area_m2"]].copy()
        result["gsi"] = result["building_footprint_area_m2"] / result["unit_area_m2"]
        results["gsi"] = result[["unit_id", "building_footprint_area_m2", "gsi"]]

    if "far_fsi" in requested:
        if intersections.empty:
            floor_sum = pd.DataFrame(columns=["unit_id", "floor_area_sum_m2"])
            floor_valid = pd.DataFrame(columns=["unit_id", "floor_data_valid_area_m2"])
        else:
            intersections["floor_area_m2"] = (
                intersections["area_in_unit_m2"] * intersections["num_floors"]
            )
            floor_sum = (
                intersections.groupby("unit_id")["floor_area_m2"]
                .sum(min_count=1)
                .rename("floor_area_sum_m2")
                .reset_index()
            )
            floor_valid = (
                intersections[intersections["num_floors"].notna()]
                .groupby("unit_id")["area_in_unit_m2"]
                .sum()
                .rename("floor_data_valid_area_m2")
                .reset_index()
            )
        result = base.merge(floor_sum, on="unit_id", how="left").merge(
            floor_valid, on="unit_id", how="left"
        )
        result["floor_data_valid_area_m2"] = result["floor_data_valid_area_m2"].fillna(0.0)
        result.loc[no_buildings, "floor_area_sum_m2"] = 0.0
        result["far_fsi"] = result["floor_area_sum_m2"] / result["unit_area_m2"]
        result["floor_data_valid_area_share"] = np.where(
            result["building_footprint_area_m2"] > 0,
            result["floor_data_valid_area_m2"] / result["building_footprint_area_m2"],
            np.nan,
        )
        results["far_fsi"] = result[
            ["unit_id", "floor_area_sum_m2", "far_fsi", "floor_data_valid_area_share"]
        ]

    if "built_volume_density" in requested:
        if intersections.empty:
            volume_sum = pd.DataFrame(columns=["unit_id", "built_volume_m3"])
            height_valid = pd.DataFrame(columns=["unit_id", "height_valid_area_m2"])
        else:
            intersections["built_volume_m3"] = (
                intersections["area_in_unit_m2"] * intersections["height_m"]
            )
            volume_sum = (
                intersections.groupby("unit_id")["built_volume_m3"]
                .sum(min_count=1)
                .reset_index()
            )
            height_valid = (
                intersections[intersections["height_m"].notna()]
                .groupby("unit_id")["area_in_unit_m2"]
                .sum()
                .rename("height_valid_area_m2")
                .reset_index()
            )
        result = base.merge(volume_sum, on="unit_id", how="left").merge(
            height_valid, on="unit_id", how="left"
        )
        result["height_valid_area_m2"] = result["height_valid_area_m2"].fillna(0.0)
        result.loc[no_buildings, "built_volume_m3"] = 0.0
        result["built_volume_density"] = result["built_volume_m3"] / result["unit_area_m2"]
        result["height_valid_area_share"] = np.where(
            result["building_footprint_area_m2"] > 0,
            result["height_valid_area_m2"] / result["building_footprint_area_m2"],
            np.nan,
        )
        results["built_volume_density"] = result[
            ["unit_id", "built_volume_m3", "built_volume_density", "height_valid_area_share"]
        ]

    return results


def _nearest_neighbor_table(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calculate nearest footprint-to-footprint distance for each building.

    The distance is geometric distance between building polygons, not centroid
    distance. Touching buildings may therefore have distance 0.
    """
    ensure_projected_crs(buildings, "Buildings")
    validate_required_columns(buildings, ("building_id", "geometry"), "buildings")

    b = buildings[["building_id", "geometry"]].copy()

    if len(b) < 2:
        return gpd.GeoDataFrame(
            {
                "building_id": b["building_id"],
                "neighbor_building_id": pd.Series(dtype="object"),
                "neighbor_distance_m": pd.Series(dtype="float64"),
            },
            geometry=b.geometry,
            crs=b.crs,
        )

    left = b.copy()
    right = b.rename(columns={"building_id": "neighbor_building_id"}).copy()

    nearest = gpd.sjoin_nearest(
        left,
        right,
        how="left",
        distance_col="neighbor_distance_m",
        exclusive=True,
    )

    # Extra protection against self-matches.
    nearest = nearest[nearest["building_id"] != nearest["neighbor_building_id"]]

    nearest = (
        nearest.sort_values(["building_id", "neighbor_distance_m"])
        .drop_duplicates(subset="building_id")
        .reset_index(drop=True)
    )

    return gpd.GeoDataFrame(
        nearest[["building_id", "neighbor_building_id", "neighbor_distance_m", "geometry"]],
        geometry="geometry",
        crs=buildings.crs,
    )


def _assign_building_values_to_units(
    building_values: gpd.GeoDataFrame,
    units: gpd.GeoDataFrame,
    value_cols: list[str],
) -> gpd.GeoDataFrame:
    """
    Assign building-level values to aggregation units by representative point.
    """
    ensure_same_projected_crs(building_values, units)

    points = building_values[["building_id", *value_cols, "geometry"]].copy()
    points["geometry"] = points.geometry.representative_point()

    assigned = gpd.sjoin(
        points,
        units[["unit_id", "geometry"]],
        how="left",
        predicate="within",
    )

    return assigned


def calculate_neighbor_distance(
    buildings: gpd.GeoDataFrame,
    units: gpd.GeoDataFrame,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Calculate average and median nearest-neighbour distance per aggregation unit.

    This is a contextual morphology indicator, not a primary density indicator.
    """
    ensure_same_projected_crs(buildings, units)
    validate_required_columns(units, ("unit_id", "geometry"), "units")

    u = units[["unit_id"]].copy()

    if len(buildings) < 2:
        u["avg_neighbor_distance_m"] = np.nan
        u["median_neighbor_distance_m"] = np.nan
        u["neighbor_distance_valid_count"] = 0
        return pd.DataFrame(u)

    nearest = _nearest_neighbor_table(buildings)

    return aggregate_neighbor_distance_from_table(nearest, units)


def aggregate_neighbor_distance_from_table(
    nearest: gpd.GeoDataFrame,
    units: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Aggregate a compatible building-level nearest-neighbour table to units."""
    ensure_same_projected_crs(nearest, units)
    validate_required_columns(
        nearest,
        ("building_id", "neighbor_distance_m", "geometry"),
        "nearest",
    )
    validate_required_columns(units, ("unit_id", "geometry"), "units")

    u = units[["unit_id"]].copy()
    if nearest.empty:
        u["avg_neighbor_distance_m"] = np.nan
        u["median_neighbor_distance_m"] = np.nan
        u["neighbor_distance_valid_count"] = 0
        return pd.DataFrame(u)

    assigned = _assign_building_values_to_units(
        nearest,
        units,
        value_cols=["neighbor_distance_m"],
    )

    summary = (
        assigned.groupby("unit_id")
        .agg(
            avg_neighbor_distance_m=("neighbor_distance_m", "mean"),
            median_neighbor_distance_m=("neighbor_distance_m", "median"),
            neighbor_distance_valid_count=("neighbor_distance_m", "count"),
        )
        .reset_index()
    )

    out = u.merge(summary, on="unit_id", how="left")
    out["neighbor_distance_valid_count"] = out["neighbor_distance_valid_count"].fillna(0).astype(int)

    return pd.DataFrame(out)


def calculate_height_to_distance_ratio(
    buildings: gpd.GeoDataFrame,
    units: gpd.GeoDataFrame,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Calculate diagnostic nearest-neighbour height-to-distance ratio per unit.

    For each building:
        ratio = height_m / nearest_neighbor_distance_m

    Missing height values remain missing. Distances smaller than the configured
    minimum threshold are set to missing to avoid infinite or unstable ratios.

    Scientific note:
    This legacy nearest-neighbour ratio is retained as a diagnostic only. The
    official contextual height-width indicator for the thesis is the
    street-profile-based height-to-width ratio calculated by the street_context
    branch, because nearest-neighbour spacing often becomes zero in attached or
    overlapping urban fabric.
    """
    if params is None:
        params = {}

    ensure_same_projected_crs(buildings, units)
    validate_required_columns(buildings, ("building_id", "geometry", "height_m"), "buildings")
    validate_required_columns(units, ("unit_id", "geometry"), "units")

    min_distance = float(params.get("min_distance_for_ratio_m", 0.5))

    u = units[["unit_id"]].copy()

    if len(buildings) < 2:
        u["avg_height_to_distance_ratio"] = np.nan
        u["median_height_to_distance_ratio"] = np.nan
        u["height_distance_valid_count"] = 0
        return pd.DataFrame(u)

    b = buildings[["building_id", "geometry", "height_m"]].copy()
    b["height_m"] = pd.to_numeric(b["height_m"], errors="coerce")
    b.loc[b["height_m"] <= 0, "height_m"] = np.nan

    nearest = _nearest_neighbor_table(b)
    nearest = nearest.merge(
        b[["building_id", "height_m"]],
        on="building_id",
        how="left",
    )

    valid = (
        nearest["height_m"].notna()
        & nearest["neighbor_distance_m"].notna()
        & (nearest["neighbor_distance_m"] > min_distance)
    )

    nearest["height_to_distance_ratio"] = np.nan
    nearest.loc[valid, "height_to_distance_ratio"] = (
        nearest.loc[valid, "height_m"] / nearest.loc[valid, "neighbor_distance_m"]
    )

    assigned = _assign_building_values_to_units(
        nearest,
        units,
        value_cols=["height_to_distance_ratio"],
    )

    summary = (
        assigned.groupby("unit_id")
        .agg(
            avg_height_to_distance_ratio=("height_to_distance_ratio", "mean"),
            median_height_to_distance_ratio=("height_to_distance_ratio", "median"),
            height_distance_valid_count=("height_to_distance_ratio", "count"),
        )
        .reset_index()
    )

    out = u.merge(summary, on="unit_id", how="left")
    out["height_distance_valid_count"] = out["height_distance_valid_count"].fillna(0).astype(int)

    return pd.DataFrame(out)


def calculate_building_neighbor_diagnostics(
    buildings: gpd.GeoDataFrame,
    aoi: gpd.GeoDataFrame | None = None,
    params: dict[str, Any] | None = None,
    nearest_table: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """
    Create building-level diagnostics for nearest-neighbour indicators.

    This function is not a primary density indicator. It is a diagnostic output
    used to interpret zero neighbour distances, near-zero distances, missing
    height values, height-to-distance ratio outliers, and possible AOI boundary
    effects.

    Parameters
    ----------
    buildings:
        Cleaned building GeoDataFrame in projected CRS.
    aoi:
        Optional AOI GeoDataFrame in the same projected CRS. If provided, the
        function calculates the distance from each building to the AOI boundary.
    params:
        Optional parameter dictionary. Supported keys:
        - min_distance_for_ratio_m
        - boundary_distance_threshold_m

    Returns
    -------
    geopandas.GeoDataFrame
        One row per building with nearest-neighbour diagnostics.
    """
    if params is None:
        params = {}

    ensure_projected_crs(buildings, "Buildings")
    validate_required_columns(buildings, ("building_id", "geometry"), "buildings")

    min_distance = float(params.get("min_distance_for_ratio_m", 0.5))
    boundary_threshold = float(params.get("boundary_distance_threshold_m", 50.0))

    b = buildings[["building_id", "geometry"]].copy()

    if "height_m" in buildings.columns:
        b["height_m"] = pd.to_numeric(buildings["height_m"], errors="coerce")
        b.loc[b["height_m"] <= 0, "height_m"] = np.nan
    else:
        b["height_m"] = np.nan

    nearest = (
        nearest_table[[
            "building_id",
            "neighbor_building_id",
            "neighbor_distance_m",
            "geometry",
        ]].copy()
        if nearest_table is not None
        else _nearest_neighbor_table(b)
    )
    ensure_same_projected_crs(nearest, buildings)

    diagnostics = nearest.merge(
        b[["building_id", "height_m"]],
        on="building_id",
        how="left",
    )

    diagnostics["neighbor_distance_m"] = pd.to_numeric(
        diagnostics["neighbor_distance_m"],
        errors="coerce",
    )

    diagnostics["has_neighbor"] = diagnostics["neighbor_distance_m"].notna()

    diagnostics["is_zero_distance"] = (
        diagnostics["neighbor_distance_m"].notna()
        & (diagnostics["neighbor_distance_m"] == 0)
    )

    diagnostics["is_near_zero_distance"] = (
        diagnostics["neighbor_distance_m"].notna()
        & (diagnostics["neighbor_distance_m"] <= min_distance)
    )

    diagnostics["has_valid_height"] = diagnostics["height_m"].notna()

    valid_ratio = (
        diagnostics["has_valid_height"]
        & diagnostics["neighbor_distance_m"].notna()
        & (diagnostics["neighbor_distance_m"] > min_distance)
    )

    diagnostics["height_to_distance_ratio"] = np.nan
    diagnostics.loc[valid_ratio, "height_to_distance_ratio"] = (
        diagnostics.loc[valid_ratio, "height_m"]
        / diagnostics.loc[valid_ratio, "neighbor_distance_m"]
    )

    diagnostics["has_valid_height_to_distance_ratio"] = valid_ratio

        # Classify zero-distance relationships.
    # Distance = 0 can mean touching buildings, overlapping footprints,
    # containment, identical geometries, or another intersection case.
    neighbor_geometries = b[["building_id", "geometry"]].rename(
        columns={
            "building_id": "neighbor_building_id",
            "geometry": "neighbor_geometry",
        }
    )

    diagnostics = diagnostics.merge(
        neighbor_geometries,
        on="neighbor_building_id",
        how="left",
    )

    diagnostics["touches_neighbor"] = False
    diagnostics["intersects_neighbor"] = False
    diagnostics["overlaps_neighbor"] = False
    diagnostics["within_neighbor"] = False
    diagnostics["contains_neighbor"] = False
    diagnostics["equals_neighbor"] = False
    diagnostics["overlap_area_m2"] = 0.0
    diagnostics["shared_boundary_length_m"] = 0.0
    diagnostics["zero_distance_relation"] = pd.NA

    zero_mask = diagnostics["is_zero_distance"] & diagnostics["neighbor_geometry"].notna()

    zero_index = diagnostics.index[zero_mask]
    if len(zero_index) > 0:
        left_geometry = diagnostics.loc[zero_index, "geometry"].array
        right_geometry = diagnostics.loc[zero_index, "neighbor_geometry"].array
        touches = np.asarray(shapely.touches(left_geometry, right_geometry), dtype=bool)
        intersects = np.asarray(shapely.intersects(left_geometry, right_geometry), dtype=bool)
        overlaps = np.asarray(shapely.overlaps(left_geometry, right_geometry), dtype=bool)
        within = np.asarray(shapely.within(left_geometry, right_geometry), dtype=bool)
        contains = np.asarray(shapely.contains(left_geometry, right_geometry), dtype=bool)
        equals = np.asarray(shapely.equals(left_geometry, right_geometry), dtype=bool)

        diagnostics.loc[zero_index, "touches_neighbor"] = touches
        diagnostics.loc[zero_index, "intersects_neighbor"] = intersects
        diagnostics.loc[zero_index, "overlaps_neighbor"] = overlaps
        diagnostics.loc[zero_index, "within_neighbor"] = within
        diagnostics.loc[zero_index, "contains_neighbor"] = contains
        diagnostics.loc[zero_index, "equals_neighbor"] = equals

        intersection_geometry = shapely.intersection(left_geometry, right_geometry)
        diagnostics.loc[zero_index, "overlap_area_m2"] = np.where(
            intersects,
            shapely.area(intersection_geometry),
            0.0,
        )
        shared_boundary = shapely.intersection(
            shapely.boundary(left_geometry),
            shapely.boundary(right_geometry),
        )
        diagnostics.loc[zero_index, "shared_boundary_length_m"] = np.where(
            touches,
            shapely.length(shared_boundary),
            0.0,
        )
        relations = np.select(
            [equals, overlaps, within, contains, touches, intersects],
            [
                "equal_geometry",
                "overlap",
                "within_neighbor",
                "contains_neighbor",
                "touching_boundary",
                "intersects_other",
            ],
            default="zero_distance_other",
        )
        diagnostics.loc[zero_index, "zero_distance_relation"] = relations

    # Optional AOI boundary diagnostics.
    if aoi is not None:
        ensure_projected_crs(aoi, "AOI")

        if aoi.crs != buildings.crs:
            raise ValueError(
                f"CRS mismatch: buildings CRS is {buildings.crs}, AOI CRS is {aoi.crs}."
            )

        aoi_geometry = aoi.geometry.union_all()
        aoi_boundary = aoi_geometry.boundary

        diagnostics["distance_to_aoi_boundary_m"] = diagnostics.geometry.distance(
            aoi_boundary
        )

        diagnostics["is_near_aoi_boundary"] = (
            diagnostics["distance_to_aoi_boundary_m"] <= boundary_threshold
        )
    else:
        diagnostics["distance_to_aoi_boundary_m"] = np.nan
        diagnostics["is_near_aoi_boundary"] = False

    diagnostics = diagnostics[
        [
            "building_id",
            "neighbor_building_id",
            "neighbor_distance_m",
            "height_m",
            "height_to_distance_ratio",
            "has_neighbor",
            "is_zero_distance",
            "is_near_zero_distance",
            "zero_distance_relation",
            "touches_neighbor",
            "intersects_neighbor",
            "overlaps_neighbor",
            "within_neighbor",
            "contains_neighbor",
            "equals_neighbor",
            "overlap_area_m2",
            "shared_boundary_length_m",
            "has_valid_height",
            "has_valid_height_to_distance_ratio",
            "distance_to_aoi_boundary_m",
            "is_near_aoi_boundary",
            "geometry",
        ]
    ]

    return gpd.GeoDataFrame(
        diagnostics,
        geometry="geometry",
        crs=buildings.crs,
    )



INDICATOR_REGISTRY = {
    "gsi": IndicatorSpec(
        key="gsi",
        name="Ground Space Index / Building Coverage Ratio",
        category="core_density",
        function=calculate_gsi,
        required_building_cols=("building_id", "geometry"),
        required_unit_cols=("unit_id", "geometry"),
        output_cols=("building_footprint_area_m2", "gsi"),
        default_enabled=True,
        conditional=False,
        description="Horizontal building coverage per aggregation unit.",
    ),
    "far_fsi": IndicatorSpec(
        key="far_fsi",
        name="Floor Area Ratio / Floor Space Index",
        category="conditional_density",
        function=calculate_far_fsi,
        required_building_cols=("building_id", "geometry", "num_floors"),
        required_unit_cols=("unit_id", "geometry"),
        output_cols=("floor_area_sum_m2", "far_fsi", "floor_data_valid_area_share"),
        default_enabled=True,
        conditional=True,
        description="Estimated total floor area per unit land area.",
    ),
    "built_volume_density": IndicatorSpec(
        key="built_volume_density",
        name="Built Volume Density",
        category="conditional_density",
        function=calculate_built_volume_density,
        required_building_cols=("building_id", "geometry", "height_m"),
        required_unit_cols=("unit_id", "geometry"),
        output_cols=("built_volume_m3", "built_volume_density", "height_valid_area_share"),
        default_enabled=True,
        conditional=True,
        description="Estimated built volume per unit land area.",
    ),
    "neighbor_distance": IndicatorSpec(
        key="neighbor_distance",
        name="Average distance to adjacent / nearest buildings",
        category="contextual_morphology",
        function=calculate_neighbor_distance,
        required_building_cols=("building_id", "geometry"),
        required_unit_cols=("unit_id", "geometry"),
        output_cols=(
            "avg_neighbor_distance_m",
            "median_neighbor_distance_m",
            "neighbor_distance_valid_count",
        ),
        default_enabled=True,
        conditional=False,
        description="Nearest-neighbour building spacing.",
    ),
    "height_to_distance_ratio": IndicatorSpec(
        key="height_to_distance_ratio",
        name="Diagnostic nearest-neighbour height-to-distance ratio",
        category="contextual_morphology",
        function=calculate_height_to_distance_ratio,
        required_building_cols=("building_id", "geometry", "height_m"),
        required_unit_cols=("unit_id", "geometry"),
        output_cols=(
            "avg_height_to_distance_ratio",
            "median_height_to_distance_ratio",
            "height_distance_valid_count",
        ),
        default_enabled=True,
        conditional=True,
        description=(
            "Diagnostic-only relationship between building height and "
            "nearest-neighbour spacing. The official contextual height-width "
            "indicator is the street-profile-based height-to-width ratio."
        ),
    ),
}


def run_indicators(
    buildings: gpd.GeoDataFrame,
    units: gpd.GeoDataFrame,
    config: dict[str, Any],
    precomputed_neighbor_table: gpd.GeoDataFrame | None = None,
    performance_metrics: dict[str, Any] | None = None,
) -> gpd.GeoDataFrame:
    """
    Run enabled indicators from the registry.

    Parameters
    ----------
    buildings:
        Cleaned building GeoDataFrame in projected CRS.
    units:
        Aggregation units in the same projected CRS.
    config:
        Workflow config. Expected keys:
        - indicators
        - indicator_parameters

    Returns
    -------
    geopandas.GeoDataFrame
        Aggregation units with indicator columns joined by unit_id.
    """
    ensure_same_projected_crs(buildings, units)
    validate_required_columns(units, ("unit_id", "geometry"), "units")

    enabled_config = config.get("indicators", {})
    params = config.get("indicator_parameters", config.get("parameters", {}))

    unknown_indicators = set(enabled_config) - set(INDICATOR_REGISTRY)

    if unknown_indicators:
        raise ValueError(f"Unknown indicators in config: {unknown_indicators}")

    results = units.copy()
    performance_config = config.get("performance", {})
    enabled_area = {
        key
        for key in ("gsi", "far_fsi", "built_volume_density")
        if enabled_config.get(key, INDICATOR_REGISTRY[key].default_enabled)
    }
    shared_area_results: dict[str, pd.DataFrame] = {}
    if performance_config.get("shared_area_intersections", True) and enabled_area:
        chunk_size = performance_config.get("area_intersection_chunk_size")
        shared_area_results = calculate_area_indicators_shared(
            buildings=buildings,
            units=units,
            enabled=enabled_area,
            chunk_size=int(chunk_size) if chunk_size is not None else None,
            metrics=performance_metrics,
        )

    for key, spec in INDICATOR_REGISTRY.items():
        enabled = enabled_config.get(key, spec.default_enabled)

        if not enabled:
            continue

        validate_required_columns(buildings, spec.required_building_cols, "buildings")
        validate_required_columns(units, spec.required_unit_cols, "units")

        if key in shared_area_results:
            indicator_result = shared_area_results[key]
        elif key == "neighbor_distance" and precomputed_neighbor_table is not None:
            indicator_result = aggregate_neighbor_distance_from_table(
                precomputed_neighbor_table,
                units,
            )
        else:
            indicator_result = spec.function(
                buildings=buildings,
                units=units,
                params=params,
            )

        expected_cols = ["unit_id", *spec.output_cols]
        missing_output_cols = [
            col for col in expected_cols if col not in indicator_result.columns
        ]

        if missing_output_cols:
            raise RuntimeError(
                f"Indicator '{key}' did not return expected columns: {missing_output_cols}"
            )

        merge_result = indicator_result[expected_cols].copy()

        # Avoid duplicate columns if run_indicators is called on a units layer
        # that already contains previous indicator columns.
        columns_to_replace = [
            col for col in spec.output_cols if col in results.columns
        ]

        if columns_to_replace:
            results = results.drop(columns=columns_to_replace)

        results = results.merge(merge_result, on="unit_id", how="left")

    return results
