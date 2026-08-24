from __future__ import annotations

import math
from typing import Any

import geopandas as gpd

from segmented_utm import segment_aoi_by_utm_zone


ALLOWED_PROCESSING_MODES = {"single_crs", "segmented_utm", "auto"}


def _utm_zone_for_longitude(lon: float) -> int:
    """
    Return the UTM zone number for a longitude in EPSG:4326 degrees.
    """
    if lon == 180:
        return 60

    zone = math.floor((lon + 180) / 6) + 1
    return max(1, min(60, int(zone)))


def _epsg_for_zone(zone: int, hemisphere: str) -> int:
    if hemisphere == "north":
        return 32600 + zone
    if hemisphere == "south":
        return 32700 + zone
    raise ValueError(f"Unsupported hemisphere: {hemisphere}")


def summarize_crs_strategy(aoi: gpd.GeoDataFrame) -> dict[str, Any]:
    """
    Summarize whether an AOI fits one UTM zone or spans multiple zones.

    This is a diagnostic-only helper. It does not choose or change the CRS used
    by the workflow. The existing auto_utm processing remains authoritative for
    current indicator calculation.
    """
    if aoi is None:
        raise ValueError("AOI GeoDataFrame is None.")

    if aoi.empty:
        raise ValueError("AOI GeoDataFrame is empty.")

    if aoi.crs is None:
        raise ValueError("AOI GeoDataFrame has no CRS.")

    aoi_wgs84 = aoi.to_crs("EPSG:4326")
    min_lon, min_lat, max_lon, max_lat = aoi_wgs84.total_bounds

    if min_lon > max_lon:
        raise ValueError("AOI longitude bounds are invalid.")

    min_zone = _utm_zone_for_longitude(float(min_lon))
    max_zone = _utm_zone_for_longitude(float(max_lon))
    zones = list(range(min_zone, max_zone + 1))

    hemispheres: list[str] = []
    if max_lat >= 0:
        hemispheres.append("north")
    if min_lat < 0:
        hemispheres.append("south")

    epsg_codes = [
        _epsg_for_zone(zone, hemisphere)
        for zone in zones
        for hemisphere in hemispheres
    ]

    is_multi_zone = len(zones) > 1

    return {
        "input_crs": aoi.crs.to_string(),
        "aoi_longitude_min": float(min_lon),
        "aoi_longitude_max": float(max_lon),
        "aoi_longitude_extent_degrees": float(max_lon - min_lon),
        "aoi_latitude_min": float(min_lat),
        "aoi_latitude_max": float(max_lat),
        "intersecting_utm_zones": zones,
        "intersecting_utm_zone_count": len(zones),
        "utm_hemispheres": hemispheres,
        "corresponding_utm_epsg_codes": epsg_codes,
        "is_single_utm_zone": not is_multi_zone,
        "is_multi_utm_zone": is_multi_zone,
        "recommended_crs_strategy": (
            "segmented_utm_recommended" if is_multi_zone else "single_zone_auto_utm"
        ),
    }


def determine_crs_processing_mode(
    config: dict[str, Any],
    aoi: gpd.GeoDataFrame,
) -> dict[str, Any]:
    """
    Decide whether the workflow should use single-CRS or segmented UTM routing.

    This is a routing/diagnostic helper. Segmented UTM currently supports a
    staged core-indicator path only; unsupported branches are rejected by the
    workflow before processing starts.
    """
    requested_mode = (
        config.get("crs_strategy", {}).get("processing_mode", "single_crs")
    )

    if requested_mode not in ALLOWED_PROCESSING_MODES:
        raise ValueError(
            f"Unsupported CRS processing mode: {requested_mode}. "
            f"Allowed values: {sorted(ALLOWED_PROCESSING_MODES)}"
        )

    segments = segment_aoi_by_utm_zone(aoi)
    n_segments = int(len(segments))
    segment_epsg_list = [int(value) for value in segments["epsg"].tolist()]
    segment_utm_zones = [int(value) for value in segments["utm_zone"].tolist()]
    segmented_required = n_segments > 1
    diagnostics: list[str] = []

    for segment_diagnostics in segments.get("diagnostics", []):
        for diagnostic in segment_diagnostics:
            if diagnostic not in diagnostics:
                diagnostics.append(diagnostic)

    if requested_mode == "single_crs":
        resolved_mode = "single_crs"

        if segmented_required:
            diagnostics.append(
                "aoi_intersects_multiple_utm_segments_but_single_crs_requested"
            )

        reason = (
            "single_crs_requested_multi_segment_aoi"
            if segmented_required
            else "single_crs_requested"
        )

    elif requested_mode == "auto":
        if segmented_required:
            resolved_mode = "segmented_utm"
            diagnostics.append("auto_selected_segmented_utm_for_multi_segment_aoi")
            reason = "auto_selected_segmented_utm"
        else:
            resolved_mode = "single_crs"
            reason = "auto_selected_single_crs"

    else:
        resolved_mode = "segmented_utm"

        if segmented_required:
            reason = "segmented_utm_requested_and_required"
        else:
            diagnostics.append("segmented_utm_requested_but_not_required")
            reason = "segmented_utm_requested_single_segment_aoi"

    return {
        "requested_processing_mode": requested_mode,
        "resolved_processing_mode": resolved_mode,
        "segmented_utm_required": bool(segmented_required),
        "n_utm_segments": n_segments,
        "segment_epsg_list": segment_epsg_list,
        "segment_utm_zones": segment_utm_zones,
        "segmented_utm_available": resolved_mode == "segmented_utm",
        "segmented_utm_supported_scope": (
            "core_indicators_height_enrichment_neighbor_distance_and_street_context"
            if resolved_mode == "segmented_utm"
            else "not_used_for_this_run"
        ),
        "segmented_utm_reason": reason,
        "diagnostics": diagnostics,
    }
