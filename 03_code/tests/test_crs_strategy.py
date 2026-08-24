from __future__ import annotations

from pathlib import Path
import sys

import geopandas as gpd
from shapely.geometry import box


TEST_DIR = Path(__file__).resolve().parent
CODE_DIR = TEST_DIR.parent
SRC_DIR = CODE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from crs_strategy import determine_crs_processing_mode, summarize_crs_strategy


def make_aoi(min_lon, min_lat, max_lon, max_lat):
    return gpd.GeoDataFrame(
        {"name": ["test"]},
        geometry=[box(min_lon, min_lat, max_lon, max_lat)],
        crs="EPSG:4326",
    )


def test_crs_strategy_detects_single_utm_zone():
    summary = summarize_crs_strategy(make_aoi(16.2, 48.1, 16.5, 48.3))

    assert summary["input_crs"] == "EPSG:4326"
    assert summary["intersecting_utm_zones"] == [33]
    assert summary["corresponding_utm_epsg_codes"] == [32633]
    assert summary["is_single_utm_zone"] is True
    assert summary["is_multi_utm_zone"] is False
    assert summary["recommended_crs_strategy"] == "single_zone_auto_utm"


def test_crs_strategy_detects_multi_utm_zone_aoi():
    summary = summarize_crs_strategy(make_aoi(5.0, 48.0, 15.0, 49.0))

    assert summary["intersecting_utm_zones"] == [31, 32, 33]
    assert summary["corresponding_utm_epsg_codes"] == [32631, 32632, 32633]
    assert summary["is_single_utm_zone"] is False
    assert summary["is_multi_utm_zone"] is True
    assert summary["recommended_crs_strategy"] == "segmented_utm_recommended"


def test_crs_strategy_uses_southern_hemisphere_epsg_codes():
    summary = summarize_crs_strategy(make_aoi(16.2, -34.2, 16.5, -33.9))

    assert summary["intersecting_utm_zones"] == [33]
    assert summary["utm_hemispheres"] == ["south"]
    assert summary["corresponding_utm_epsg_codes"] == [32733]


def test_crs_strategy_includes_both_hemisphere_epsg_codes_when_aoi_crosses_equator():
    summary = summarize_crs_strategy(make_aoi(16.2, -0.1, 16.5, 0.1))

    assert summary["intersecting_utm_zones"] == [33]
    assert summary["utm_hemispheres"] == ["north", "south"]
    assert summary["corresponding_utm_epsg_codes"] == [32633, 32733]


def test_crs_processing_mode_missing_config_defaults_to_single_crs():
    summary = determine_crs_processing_mode(
        config={},
        aoi=make_aoi(16.2, 48.1, 16.5, 48.3),
    )

    assert summary["requested_processing_mode"] == "single_crs"
    assert summary["resolved_processing_mode"] == "single_crs"
    assert summary["segmented_utm_required"] is False
    assert summary["n_utm_segments"] == 1
    assert summary["segment_epsg_list"] == [32633]


def test_crs_processing_mode_auto_single_zone_resolves_to_single_crs():
    summary = determine_crs_processing_mode(
        config={"crs_strategy": {"processing_mode": "auto"}},
        aoi=make_aoi(16.2, 48.1, 16.5, 48.3),
    )

    assert summary["requested_processing_mode"] == "auto"
    assert summary["resolved_processing_mode"] == "single_crs"
    assert summary["segmented_utm_required"] is False
    assert summary["segmented_utm_reason"] == "auto_selected_single_crs"


def test_crs_processing_mode_auto_multi_zone_resolves_to_segmented_utm():
    summary = determine_crs_processing_mode(
        config={"crs_strategy": {"processing_mode": "auto"}},
        aoi=make_aoi(11.5, 48.0, 12.5, 49.0),
    )

    assert summary["requested_processing_mode"] == "auto"
    assert summary["resolved_processing_mode"] == "segmented_utm"
    assert summary["segmented_utm_required"] is True
    assert summary["n_utm_segments"] == 2
    assert summary["segment_epsg_list"] == [32632, 32633]
    assert "auto_selected_segmented_utm_for_multi_segment_aoi" in summary["diagnostics"]


def test_crs_processing_mode_single_crs_multi_zone_preserves_single_crs_with_warning():
    summary = determine_crs_processing_mode(
        config={"crs_strategy": {"processing_mode": "single_crs"}},
        aoi=make_aoi(11.5, 48.0, 12.5, 49.0),
    )

    assert summary["requested_processing_mode"] == "single_crs"
    assert summary["resolved_processing_mode"] == "single_crs"
    assert summary["segmented_utm_required"] is True
    assert (
        "aoi_intersects_multiple_utm_segments_but_single_crs_requested"
        in summary["diagnostics"]
    )


def test_crs_processing_mode_segmented_requested_single_zone_reports_not_required():
    summary = determine_crs_processing_mode(
        config={"crs_strategy": {"processing_mode": "segmented_utm"}},
        aoi=make_aoi(16.2, 48.1, 16.5, 48.3),
    )

    assert summary["requested_processing_mode"] == "segmented_utm"
    assert summary["resolved_processing_mode"] == "segmented_utm"
    assert summary["segmented_utm_required"] is False
    assert "segmented_utm_requested_but_not_required" in summary["diagnostics"]
