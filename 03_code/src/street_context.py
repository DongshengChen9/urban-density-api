from __future__ import annotations

import logging
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from osm_street_acquisition import acquire_osmnx_street_edges


# This numerical tolerance rejects floating-point residues. It is not a
# physical minimum street width.
METRIC_GEOMETRY_TOLERANCE_M = 1e-6
STREET_PROFILE_RATIO_OUTLIER_THRESHOLD = 10.0
STREET_PROFILE_TOPOLOGY_RULE_VERSION = "1"


def add_street_profile_width_diagnostics(
    street_profiles: gpd.GeoDataFrame,
    width_col: str = "street_profile_width_m",
    geometry_tolerance_m: float = METRIC_GEOMETRY_TOLERANCE_M,
) -> gpd.GeoDataFrame:
    """Classify numeric and geometric denominator failures without altering widths."""
    if width_col not in street_profiles.columns:
        raise ValueError(f"Street-profile width column not found: {width_col}")
    if geometry_tolerance_m <= 0 or not np.isfinite(geometry_tolerance_m):
        raise ValueError("geometry_tolerance_m must be finite and positive.")
    _ensure_projected_crs(street_profiles, "street_profiles")
    result = street_profiles.copy()
    width = pd.to_numeric(result[width_col], errors="coerce")
    result[width_col] = width
    geometry_length = result.geometry.length
    result["street_profile_geometry_length_m"] = geometry_length
    reason = pd.Series(pd.NA, index=result.index, dtype="string")
    reason.loc[result.geometry.isna()] = "profile_geometry_missing"
    reason.loc[reason.isna() & result.geometry.is_empty.fillna(False)] = "profile_geometry_empty"
    reason.loc[reason.isna() & geometry_length.notna() & (geometry_length <= geometry_tolerance_m)] = "profile_geometry_zero_length"
    reason.loc[reason.isna() & width.isna()] = "profile_width_missing"
    reason.loc[reason.isna() & ~np.isfinite(width)] = "profile_width_non_finite"
    reason.loc[reason.isna() & (width < 0)] = "profile_width_negative"
    reason.loc[reason.isna() & (width == 0)] = "profile_width_zero"
    reason.loc[reason.isna() & (width > 0) & (width <= geometry_tolerance_m)] = "profile_width_below_metric_tolerance"
    result["street_profile_width_valid"] = reason.isna()
    result["street_profile_width_invalid_reason"] = reason
    return result


def add_street_profile_topology_diagnostics(
    street_profiles: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    distance: float = 10.0,
) -> gpd.GeoDataFrame:
    """Flag sampled profile origins intersecting or touching a building footprint."""
    if distance <= 0 or not np.isfinite(distance):
        raise ValueError("distance must be finite and positive.")
    _ensure_projected_crs(street_profiles, "street_profiles")
    _ensure_projected_crs(buildings, "buildings")
    _same_crs(street_profiles, buildings, "street_profiles", "buildings")
    result = street_profiles.copy()
    valid_buildings = buildings[buildings.geometry.notna() & ~buildings.geometry.is_empty].copy()
    segmented = shapely.segmentize(result.geometry.array, float(distance))
    coordinates, street_rows = shapely.get_coordinates(segmented, return_index=True)
    origin_count = np.bincount(street_rows, minlength=len(result))
    intersecting_count = np.zeros(len(result), dtype=int)
    if len(coordinates) and not valid_buildings.empty:
        pairs = shapely.STRtree(valid_buildings.geometry.array).query(
            shapely.points(coordinates), predicate="intersects"
        )
        if pairs.size:
            intersecting_count = np.bincount(
                street_rows[np.unique(pairs[0])], minlength=len(result)
            )
    result["street_profile_origin_count"] = origin_count
    result["street_profile_origin_intersecting_building_count"] = intersecting_count
    result["street_profile_topology_valid"] = intersecting_count == 0
    result["street_profile_topology_invalid_reason"] = pd.Series(
        np.where(intersecting_count > 0, "profile_origin_intersects_building", pd.NA),
        index=result.index, dtype="string"
    )
    normalized = pd.Series(shapely.to_wkb(shapely.normalize(result.geometry.array), hex=True), index=result.index)
    result["street_profile_coincident_geometry_count"] = normalized.map(normalized.value_counts()).fillna(0).astype(int) - 1
    return result


def _ensure_projected_crs(gdf: gpd.GeoDataFrame, name: str) -> None:
    """
    Ensure that a GeoDataFrame has a projected CRS.

    Street-profile widths, nearest distances, and grid aggregation must be
    calculated in metric units, not in geographic degrees.
    """
    if gdf.crs is None:
        raise ValueError(f"{name} has no CRS.")

    if not gdf.crs.is_projected:
        raise ValueError(
            f"{name} must use a projected CRS for metric distance calculation. "
            f"Current CRS: {gdf.crs}"
        )


def _same_crs(
    left: gpd.GeoDataFrame,
    right: gpd.GeoDataFrame,
    left_name: str,
    right_name: str,
) -> None:
    """
    Ensure that two GeoDataFrames use the same CRS.
    """
    if left.crs != right.crs:
        raise ValueError(
            f"{left_name} and {right_name} must use the same CRS. "
            f"{left_name}: {left.crs}; {right_name}: {right.crs}"
        )


def _to_string_if_list(value: Any) -> Any:
    """
    Normalize non-null OSM attributes to strings for stable cache export.

    OSMnx can return a mixture of scalar and list values in one attribute
    column (notably ``osmid``). PyArrow cannot serialize that mixed object
    dtype, while the workflow uses these fields only as categorical labels.
    """
    if value is None or value is pd.NA:
        return value
    if not isinstance(value, (list, tuple, set)) and pd.isna(value):
        return value
    if isinstance(value, set):
        value = sorted(value, key=str)
    if isinstance(value, (list, tuple)):
        return "|".join(str(item) for item in value)
    return str(value)


def _safe_stat(series: pd.Series, stat: str) -> float | None:
    """
    Calculate a numeric statistic safely.
    """
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return None

    if stat == "min":
        return float(values.min())
    if stat == "median":
        return float(values.median())
    if stat == "max":
        return float(values.max())
    if stat == "mean":
        return float(values.mean())

    raise ValueError(f"Unsupported statistic: {stat}")


def fetch_streets_from_osmnx(
    aoi: gpd.GeoDataFrame,
    network_type: str = "drive",
    target_crs: Any | None = None,
    acquisition_config: dict[str, Any] | None = None,
    query_context: dict[str, Any] | None = None,
    return_provenance: bool = False,
) -> gpd.GeoDataFrame | tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """
    Fetch street centerlines from OpenStreetMap using OSMnx.
    """
    if target_crs is None:
        target_crs = aoi.crs
    logging.info("Fetching OSMnx street network. network_type=%s", network_type)
    edges, provenance = acquire_osmnx_street_edges(
        aoi,
        network_type=network_type,
        acquisition_config=acquisition_config,
        target_crs=target_crs,
        query_context=query_context,
    )

    streets = edges.copy()

    streets = streets[
        streets.geometry.notna()
        & ~streets.geometry.is_empty
        & streets.geometry.geom_type.isin(["LineString", "MultiLineString"])
    ].copy()

    streets = streets.to_crs(target_crs)
    streets = streets.explode(index_parts=False).reset_index(drop=True)

    keep_cols = [
        col
        for col in ["osmid", "highway", "name", "geometry"]
        if col in streets.columns
    ]

    streets = streets[keep_cols].copy()

    for col in ["osmid", "highway", "name"]:
        if col in streets.columns:
            streets[col] = streets[col].apply(_to_string_if_list)

    streets["street_id"] = [f"street_{i:05d}" for i in range(len(streets))]

    _ensure_projected_crs(streets, "streets")

    logging.info("Fetched street segments: %s", len(streets))

    provenance["street_row_count"] = int(len(streets))
    provenance["geometry_type_counts"] = {
        str(key): int(value) for key, value in streets.geometry.geom_type.value_counts().items()
    }
    return (streets, provenance) if return_provenance else streets


def calculate_street_profile_segments(
    streets: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
    height_col: str = "height_m",
    distance: float = 10.0,
    tick_length: float = 60.0,
) -> gpd.GeoDataFrame:
    """
    Calculate street-profile widths using momepy.street_profile.
    """
    import momepy

    if streets.empty:
        raise ValueError("Street layer is empty.")

    if buildings.empty:
        raise ValueError("Building layer is empty.")

    if height_col not in buildings.columns:
        raise ValueError(f"Building height column not found: {height_col}")

    _ensure_projected_crs(streets, "streets")
    _ensure_projected_crs(buildings, "buildings")
    _same_crs(streets, buildings, "streets", "buildings")

    streets_profile = streets.copy().reset_index(drop=True)

    buildings_for_profile = buildings[
        buildings.geometry.notna()
        & ~buildings.geometry.is_empty
    ].copy()

    buildings_for_profile[height_col] = pd.to_numeric(
        buildings_for_profile[height_col],
        errors="coerce",
    )

    logging.info(
        "Calculating street profiles. streets=%s, buildings=%s, distance=%s, tick_length=%s",
        len(streets_profile),
        len(buildings_for_profile),
        distance,
        tick_length,
    )

    profile = momepy.street_profile(
        streets=streets_profile,
        buildings=buildings_for_profile,
        distance=distance,
        tick_length=tick_length,
        height=buildings_for_profile[height_col],
    )

    if len(profile) != len(streets_profile):
        raise ValueError(
            "momepy.street_profile returned a different number of rows than streets. "
            f"streets={len(streets_profile)}, profile={len(profile)}"
        )

    streets_profile = streets_profile.join(profile.reset_index(drop=True))

    required_profile_cols = [
        "width",
        "openness",
        "width_deviation",
        "height",
        "height_deviation",
        "hw_ratio",
    ]

    missing_cols = [
        col for col in required_profile_cols
        if col not in streets_profile.columns
    ]

    if missing_cols:
        raise ValueError(
            "Street profile output does not contain expected columns. "
            f"Missing: {missing_cols}. Available: {streets_profile.columns.tolist()}"
        )

    streets_profile["street_profile_width_m"] = pd.to_numeric(
        streets_profile["width"],
        errors="coerce",
    )

    streets_profile["street_profile_openness"] = pd.to_numeric(
        streets_profile["openness"],
        errors="coerce",
    )

    streets_profile["street_profile_width_deviation_m"] = pd.to_numeric(
        streets_profile["width_deviation"],
        errors="coerce",
    )

    streets_profile["street_profile_momepy_height_m"] = pd.to_numeric(
        streets_profile["height"],
        errors="coerce",
    )

    streets_profile["street_profile_height_deviation_m"] = pd.to_numeric(
        streets_profile["height_deviation"],
        errors="coerce",
    )

    streets_profile["street_profile_hw_ratio_momepy"] = pd.to_numeric(
        streets_profile["hw_ratio"],
        errors="coerce",
    )

    streets_profile["street_profile_width_is_capped"] = (
        streets_profile["street_profile_width_m"] >= float(tick_length) * 0.99
    )

    streets_profile = add_street_profile_width_diagnostics(streets_profile)
    streets_profile = add_street_profile_topology_diagnostics(
        streets_profile,
        buildings_for_profile,
        distance=distance,
    )
    streets_profile["has_opposite_profile_evidence"] = (
        streets_profile["street_profile_width_valid"]
        & streets_profile["street_profile_topology_valid"]
        & (~streets_profile["street_profile_width_is_capped"].fillna(False))
        & (
            (streets_profile["street_profile_openness"] < 1)
            | streets_profile["street_profile_momepy_height_m"].notna()
        )
    )

    logging.info(
        "Street profile calculated. valid_width=%s/%s",
        int(streets_profile["street_profile_width_valid"].sum()),
        len(streets_profile),
    )

    return streets_profile
def assign_buildings_to_street_profiles(
    buildings: gpd.GeoDataFrame,
    streets_profile: gpd.GeoDataFrame,
    building_id_col: str = "building_id",
    height_col: str = "height_m",
    max_distance_m: float | None = None,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """
    Assign each building to the nearest street-profile segment.

    If several street segments are equally nearest, the function keeps one
    deterministic match per building and reports how many duplicate matches
    were removed.
    """
    if buildings.empty:
        raise ValueError("Building layer is empty.")

    if streets_profile.empty:
        raise ValueError("Street-profile layer is empty.")

    if height_col not in buildings.columns:
        raise ValueError(f"Building height column not found: {height_col}")

    if "street_id" not in streets_profile.columns:
        raise ValueError("streets_profile must contain `street_id`.")

    if "street_profile_width_m" not in streets_profile.columns:
        raise ValueError("streets_profile must contain `street_profile_width_m`.")

    _ensure_projected_crs(buildings, "buildings")
    _ensure_projected_crs(streets_profile, "streets_profile")
    _same_crs(buildings, streets_profile, "buildings", "streets_profile")

    buildings_for_join = buildings.copy().reset_index(drop=True)

    if building_id_col not in buildings_for_join.columns:
        buildings_for_join[building_id_col] = [
            f"building_{i:06d}" for i in range(len(buildings_for_join))
        ]

    buildings_for_join["building_row_id"] = buildings_for_join.index

    buildings_for_join[height_col] = pd.to_numeric(
        buildings_for_join[height_col],
        errors="coerce",
    )

    buildings_for_join = buildings_for_join[
        [building_id_col, "building_row_id", height_col, "geometry"]
    ].copy()

    diagnostic_cols = [
        "street_profile_geometry_length_m",
        "street_profile_width_valid",
        "street_profile_width_invalid_reason",
        "street_profile_origin_count",
        "street_profile_origin_intersecting_building_count",
        "street_profile_topology_valid",
        "street_profile_topology_invalid_reason",
        "street_profile_coincident_geometry_count",
    ]
    streets_for_join = streets_profile[
        [
            "street_id",
            "street_profile_width_m",
            "street_profile_openness",
            "street_profile_momepy_height_m",
            "street_profile_hw_ratio_momepy",
            "street_profile_width_is_capped",
            "has_opposite_profile_evidence",
            "geometry",
        ] + [column for column in diagnostic_cols if column in streets_profile.columns]
    ].copy()

    streets_for_join = streets_for_join[
        streets_for_join["street_profile_width_m"].notna()
        & (streets_for_join["street_profile_width_m"] > 0)
    ].copy()

    logging.info(
        "Assigning buildings to nearest street profile. buildings=%s, street_profiles=%s",
        len(buildings_for_join),
        len(streets_for_join),
    )

    join_kwargs = {
        "how": "left",
        "distance_col": "building_to_street_centerline_m",
    }

    if max_distance_m is not None:
        join_kwargs["max_distance"] = max_distance_m

    building_street_raw = gpd.sjoin_nearest(
        buildings_for_join,
        streets_for_join,
        **join_kwargs,
    )

    raw_join_rows = int(len(building_street_raw))
    original_buildings = int(len(buildings_for_join))

    building_street = (
        building_street_raw
        .sort_values(
            by=[
                "building_row_id",
                "building_to_street_centerline_m",
                "street_id",
            ],
            na_position="last",
        )
        .drop_duplicates(subset="building_row_id", keep="first")
        .copy()
        .reset_index(drop=True)
    )

    building_street = building_street.drop(
        columns=["index_right"],
        errors="ignore",
    )

    duplicate_rows_removed = raw_join_rows - int(len(building_street))

    if len(building_street) != original_buildings:
        raise ValueError(
            "Building-to-street assignment did not retain one row per building. "
            f"buildings={original_buildings}, assigned={len(building_street)}"
        )

    summary = {
        "n_buildings": original_buildings,
        "raw_join_rows_before_deduplication": raw_join_rows,
        "duplicate_join_rows_removed": int(duplicate_rows_removed),
        "matched_to_street_count": int(building_street["street_id"].notna().sum()),
        "matched_to_street_share": float(building_street["street_id"].notna().mean()),
    }

    logging.info(
        "Building-street assignment complete. matched=%s/%s, duplicate_rows_removed=%s",
        summary["matched_to_street_count"],
        original_buildings,
        duplicate_rows_removed,
    )

    return building_street, summary


def calculate_building_street_profile_ratio(
    building_street: gpd.GeoDataFrame,
    height_col: str = "height_m",
    geometry_tolerance_m: float = METRIC_GEOMETRY_TOLERANCE_M,
    outlier_threshold: float = STREET_PROFILE_RATIO_OUTLIER_THRESHOLD,
) -> gpd.GeoDataFrame:
    """
    Calculate building-level street-profile height-to-width ratio.

    Two versions are created:
    - preliminary: height and street profile width are valid;
    - strict: additionally requires opposite-profile evidence.
    """
    required_cols = [
        height_col,
        "street_profile_width_m",
        "has_opposite_profile_evidence",
    ]

    missing_cols = [
        col for col in required_cols
        if col not in building_street.columns
    ]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    if geometry_tolerance_m <= 0 or not np.isfinite(geometry_tolerance_m):
        raise ValueError("geometry_tolerance_m must be finite and positive.")
    if outlier_threshold <= 0 or not np.isfinite(outlier_threshold):
        raise ValueError("outlier_threshold must be finite and positive.")
    _ensure_projected_crs(building_street, "building_street")

    result = building_street.copy()

    result[height_col] = pd.to_numeric(result[height_col], errors="coerce")

    result["street_profile_width_m"] = pd.to_numeric(
        result["street_profile_width_m"],
        errors="coerce",
    )
    if "street_profile_width_valid" not in result.columns:
        width = result["street_profile_width_m"]
        reason = pd.Series(pd.NA, index=result.index, dtype="string")
        reason.loc[width.isna()] = "profile_width_missing"
        reason.loc[reason.isna() & ~np.isfinite(width)] = "profile_width_non_finite"
        reason.loc[reason.isna() & (width < 0)] = "profile_width_negative"
        reason.loc[reason.isna() & (width == 0)] = "profile_width_zero"
        reason.loc[reason.isna() & (width > 0) & (width <= geometry_tolerance_m)] = "profile_width_below_metric_tolerance"
        result["street_profile_width_valid"] = reason.isna()
        result["street_profile_width_invalid_reason"] = reason
    else:
        result["street_profile_width_valid"] = result["street_profile_width_valid"].fillna(False).astype(bool)
    if "street_profile_topology_valid" not in result.columns:
        result["street_profile_topology_valid"] = True
    else:
        result["street_profile_topology_valid"] = result["street_profile_topology_valid"].fillna(False).astype(bool)
    if "street_profile_topology_invalid_reason" not in result.columns:
        result["street_profile_topology_invalid_reason"] = pd.Series(pd.NA, index=result.index, dtype="string")

    result["street_profile_height_to_width_ratio_prelim"] = np.nan
    result["street_profile_height_to_width_ratio_strict"] = np.nan

    valid_prelim_mask = (
        result[height_col].notna()
        & np.isfinite(result[height_col])
        & (result[height_col] > 0)
        & result["street_profile_width_valid"]
        & result["street_profile_topology_valid"]
    )

    valid_strict_mask = (
        valid_prelim_mask
        & result["has_opposite_profile_evidence"].fillna(False)
    )

    result.loc[
        valid_prelim_mask,
        "street_profile_height_to_width_ratio_prelim",
    ] = (
        result.loc[valid_prelim_mask, height_col]
        / result.loc[valid_prelim_mask, "street_profile_width_m"]
    )

    result.loc[
        valid_strict_mask,
        "street_profile_height_to_width_ratio_strict",
    ] = (
        result.loc[valid_strict_mask, height_col]
        / result.loc[valid_strict_mask, "street_profile_width_m"]
    )

    result["has_valid_street_profile_ratio_prelim"] = valid_prelim_mask
    result["has_valid_street_profile_ratio_strict"] = valid_strict_mask
    reason = result["street_profile_width_invalid_reason"].astype("string").copy()
    reason.loc[reason.isna() & ~result["street_profile_topology_valid"]] = result.loc[
        reason.isna() & ~result["street_profile_topology_valid"],
        "street_profile_topology_invalid_reason",
    ].fillna("profile_topology_invalid")
    reason.loc[reason.isna() & result[height_col].isna()] = "height_missing"
    reason.loc[reason.isna() & (result[height_col] <= 0)] = "height_non_positive"
    result["street_profile_ratio_prelim_invalid_reason"] = reason
    strict_reason = reason.copy()
    strict_reason.loc[strict_reason.isna() & ~result["has_opposite_profile_evidence"].fillna(False)] = "missing_opposite_profile_evidence"
    result["street_profile_ratio_strict_invalid_reason"] = strict_reason
    result["street_profile_ratio_outlier_flag"] = (
        pd.to_numeric(result["street_profile_height_to_width_ratio_strict"], errors="coerce").gt(outlier_threshold)
    )

    return result


def aggregate_street_profile_ratio_to_units(
    building_street: gpd.GeoDataFrame,
    units: gpd.GeoDataFrame,
    unit_id_col: str = "unit_id",
    building_id_col: str = "building_id",
) -> gpd.GeoDataFrame:
    """
    Aggregate building-level street-profile ratio to spatial units.

    The current aggregation uses building representative points. This avoids
    double-counting buildings crossing cell boundaries, but it should be
    documented as a building-assignment choice.
    """
    if units.empty:
        raise ValueError("Aggregation units are empty.")

    if building_street.empty:
        raise ValueError("Building-street layer is empty.")

    if unit_id_col not in units.columns:
        raise ValueError(f"Unit ID column not found: {unit_id_col}")

    _ensure_projected_crs(building_street, "building_street")
    _ensure_projected_crs(units, "units")
    _same_crs(building_street, units, "building_street", "units")

    building_points = building_street.drop(
        columns=["index_right"],
        errors="ignore",
    ).copy()

    building_points["geometry"] = building_points.geometry.representative_point()

    unit_lookup = units[[unit_id_col, "geometry"]].copy()

    building_grid = gpd.sjoin(
        building_points,
        unit_lookup,
        how="left",
        predicate="within",
    )

    building_grid = building_grid.drop(
        columns=["index_right"],
        errors="ignore",
    )

    grid_ratio_stats = (
        building_grid
        .groupby(unit_id_col)
        .agg(
            street_profile_building_count=(building_id_col, "count"),
            street_profile_ratio_prelim_valid_count=(
                "has_valid_street_profile_ratio_prelim",
                "sum",
            ),
            street_profile_ratio_strict_valid_count=(
                "has_valid_street_profile_ratio_strict",
                "sum",
            ),
            street_profile_width_mean_m=(
                "street_profile_width_m",
                "mean",
            ),
            street_profile_width_median_m=(
                "street_profile_width_m",
                "median",
            ),
            avg_street_profile_height_to_width_ratio_prelim=(
                "street_profile_height_to_width_ratio_prelim",
                "mean",
            ),
            median_street_profile_height_to_width_ratio_prelim=(
                "street_profile_height_to_width_ratio_prelim",
                "median",
            ),
            avg_street_profile_height_to_width_ratio_strict=(
                "street_profile_height_to_width_ratio_strict",
                "mean",
            ),
            median_street_profile_height_to_width_ratio_strict=(
                "street_profile_height_to_width_ratio_strict",
                "median",
            ),
        )
        .reset_index()
    )

    result = units.merge(
        grid_ratio_stats,
        on=unit_id_col,
        how="left",
    )

    count_cols = [
        "street_profile_building_count",
        "street_profile_ratio_prelim_valid_count",
        "street_profile_ratio_strict_valid_count",
    ]

    for col in count_cols:
        if col in result.columns:
            result[col] = result[col].fillna(0).astype(int)

    return result


def summarize_street_profile_quality(
    streets_profile: gpd.GeoDataFrame,
    building_street: gpd.GeoDataFrame,
    grid_street_profile: gpd.GeoDataFrame | None = None,
    join_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Summarize street-profile quality and coverage.

    This summary is intended for JSON and Markdown reports.
    """
    if join_summary is None:
        join_summary = {}

    n_streets = int(len(streets_profile))
    n_buildings = int(len(building_street))

    def count_true(gdf: gpd.GeoDataFrame, col: str) -> int:
        if col not in gdf.columns:
            return 0
        return int(gdf[col].fillna(False).sum())

    def share(count: int, total: int) -> float | None:
        if total == 0:
            return None
        return float(count / total)

    valid_width_count = int(
        streets_profile["street_profile_width_valid"].fillna(False).sum()
        if "street_profile_width_valid" in streets_profile.columns
        else streets_profile.get("street_profile_width_m", pd.Series(dtype=float)).notna().sum()
    )
    invalid_width_count = int(n_streets - valid_width_count)
    topology_valid_count = int(
        streets_profile["street_profile_topology_valid"].fillna(False).sum()
        if "street_profile_topology_valid" in streets_profile.columns
        else n_streets
    )

    capped_width_count = count_true(
        streets_profile,
        "street_profile_width_is_capped",
    )

    opposite_profile_evidence_count = count_true(
        streets_profile,
        "has_opposite_profile_evidence",
    )

    matched_to_street_count = int(
        building_street["street_id"].notna().sum()
        if "street_id" in building_street.columns
        else 0
    )

    valid_height_count = int(
        building_street["height_m"].notna().sum()
        if "height_m" in building_street.columns
        else 0
    )

    valid_ratio_prelim_count = count_true(
        building_street,
        "has_valid_street_profile_ratio_prelim",
    )

    valid_ratio_strict_count = count_true(
        building_street,
        "has_valid_street_profile_ratio_strict",
    )

    summary = {
        "n_street_segments": n_streets,
        "valid_width_count": valid_width_count,
        "valid_width_share": share(valid_width_count, n_streets),
        "invalid_width_count": invalid_width_count,
        "profile_width_geometry_tolerance_m": METRIC_GEOMETRY_TOLERANCE_M,
        "profile_width_invalid_reasons": {
            str(key): int(value) for key, value in streets_profile.get(
                "street_profile_width_invalid_reason", pd.Series(dtype="string")
            ).dropna().value_counts().items()
        },
        "street_profile_topology_rule_version": STREET_PROFILE_TOPOLOGY_RULE_VERSION,
        "topology_valid_count": topology_valid_count,
        "topology_invalid_count": int(n_streets - topology_valid_count),
        "topology_invalid_reasons": {
            str(key): int(value) for key, value in streets_profile.get(
                "street_profile_topology_invalid_reason", pd.Series(dtype="string")
            ).dropna().value_counts().items()
        },
        "width_min_m": _safe_stat(
            streets_profile.get("street_profile_width_m", pd.Series(dtype=float)),
            "min",
        ),
        "width_median_m": _safe_stat(
            streets_profile.get("street_profile_width_m", pd.Series(dtype=float)),
            "median",
        ),
        "width_max_m": _safe_stat(
            streets_profile.get("street_profile_width_m", pd.Series(dtype=float)),
            "max",
        ),
        "capped_width_count": capped_width_count,
        "capped_width_share": share(capped_width_count, n_streets),
        "opposite_profile_evidence_count": opposite_profile_evidence_count,
        "opposite_profile_evidence_share": share(
            opposite_profile_evidence_count,
            n_streets,
        ),
        "n_buildings": n_buildings,
        "raw_join_rows_before_deduplication": join_summary.get(
            "raw_join_rows_before_deduplication"
        ),
        "duplicate_join_rows_removed": join_summary.get(
            "duplicate_join_rows_removed"
        ),
        "matched_to_street_count": matched_to_street_count,
        "matched_to_street_share": share(matched_to_street_count, n_buildings),
        "valid_height_count": valid_height_count,
        "valid_height_share": share(valid_height_count, n_buildings),
        "valid_ratio_prelim_count": valid_ratio_prelim_count,
        "valid_ratio_prelim_share": share(valid_ratio_prelim_count, n_buildings),
        "valid_ratio_strict_count": valid_ratio_strict_count,
        "valid_ratio_strict_share": share(valid_ratio_strict_count, n_buildings),
        "ratio_prelim_min": _safe_stat(
            building_street.get(
                "street_profile_height_to_width_ratio_prelim",
                pd.Series(dtype=float),
            ),
            "min",
        ),
        "ratio_prelim_median": _safe_stat(
            building_street.get(
                "street_profile_height_to_width_ratio_prelim",
                pd.Series(dtype=float),
            ),
            "median",
        ),
        "ratio_prelim_max": _safe_stat(
            building_street.get(
                "street_profile_height_to_width_ratio_prelim",
                pd.Series(dtype=float),
            ),
            "max",
        ),
        "ratio_strict_min": _safe_stat(
            building_street.get(
                "street_profile_height_to_width_ratio_strict",
                pd.Series(dtype=float),
            ),
            "min",
        ),
        "ratio_strict_median": _safe_stat(
            building_street.get(
                "street_profile_height_to_width_ratio_strict",
                pd.Series(dtype=float),
            ),
            "median",
        ),
        "ratio_strict_max": _safe_stat(
            building_street.get(
                "street_profile_height_to_width_ratio_strict",
                pd.Series(dtype=float),
            ),
            "max",
        ),
    }

    if grid_street_profile is not None:
        n_cells = int(len(grid_street_profile))

        prelim_cells = int(
            (grid_street_profile["street_profile_ratio_prelim_valid_count"] > 0).sum()
            if "street_profile_ratio_prelim_valid_count" in grid_street_profile.columns
            else 0
        )

        strict_cells = int(
            (grid_street_profile["street_profile_ratio_strict_valid_count"] > 0).sum()
            if "street_profile_ratio_strict_valid_count" in grid_street_profile.columns
            else 0
        )

        summary.update(
            {
                "n_grid_cells": n_cells,
                "grid_cells_with_prelim_ratio_count": prelim_cells,
                "grid_cells_with_prelim_ratio_share": share(prelim_cells, n_cells),
                "grid_cells_with_strict_ratio_count": strict_cells,
                "grid_cells_with_strict_ratio_share": share(strict_cells, n_cells),
            }
        )

    return summary
