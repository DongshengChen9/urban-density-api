from __future__ import annotations

from pathlib import Path
import sys

import geopandas as gpd
import pytest
from shapely.geometry import box


TEST_DIR = Path(__file__).resolve().parent
CODE_DIR = TEST_DIR.parent
SRC_DIR = CODE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from segmented_utm import (
    build_utm_zone_polygons,
    segment_aoi_by_utm_zone,
    utm_epsg,
    utm_zone_from_longitude,
)


def make_aoi(min_lon, min_lat, max_lon, max_lat, crs="EPSG:4326"):
    return gpd.GeoDataFrame(
        {"name": ["test"]},
        geometry=[box(min_lon, min_lat, max_lon, max_lat)],
        crs=crs,
    )


def test_utm_zone_from_longitude():
    assert utm_zone_from_longitude(-180) == 1
    assert utm_zone_from_longitude(0) == 31
    assert utm_zone_from_longitude(16.3) == 33
    assert utm_zone_from_longitude(180) == 60


def test_utm_epsg_northern_and_southern_hemisphere():
    assert utm_epsg(33, "north") == 32633
    assert utm_epsg(33, "south") == 32733

    with pytest.raises(ValueError, match="Unsupported hemisphere"):
        utm_epsg(33, "east")


def test_single_zone_aoi_returns_one_segment():
    segments = segment_aoi_by_utm_zone(make_aoi(16.2, 48.1, 16.5, 48.3))

    assert len(segments) == 1
    assert segments.loc[0, "segment_id"] == 1
    assert segments.loc[0, "utm_zone"] == 33
    assert segments.loc[0, "hemisphere"] == "north"
    assert segments.loc[0, "epsg"] == 32633
    assert segments.crs.to_string() == "EPSG:4326"


def test_two_zone_aoi_returns_two_segments():
    segments = segment_aoi_by_utm_zone(make_aoi(11.5, 48.0, 12.5, 49.0))

    assert list(segments["utm_zone"]) == [32, 33]
    assert list(segments["hemisphere"]) == ["north", "north"]
    assert list(segments["epsg"]) == [32632, 32633]


def test_southern_hemisphere_segment_uses_southern_epsg():
    segments = segment_aoi_by_utm_zone(make_aoi(16.2, -34.2, 16.5, -33.9))

    assert len(segments) == 1
    assert segments.loc[0, "utm_zone"] == 33
    assert segments.loc[0, "hemisphere"] == "south"
    assert segments.loc[0, "epsg"] == 32733


def test_segment_coverage_and_valid_geometries():
    aoi = make_aoi(11.5, 48.0, 12.5, 49.0)
    segments = segment_aoi_by_utm_zone(aoi)

    original_area = aoi.geometry.union_all().area
    segmented_area = segments.geometry.union_all().area

    assert segments.geometry.notna().all()
    assert (~segments.geometry.is_empty).all()
    assert segments.geometry.is_valid.all()
    assert segmented_area == pytest.approx(original_area)


def test_build_utm_zone_polygons_converts_aoi_to_wgs84():
    aoi = make_aoi(16.2, 48.1, 16.5, 48.3).to_crs("EPSG:3857")
    zone_polygons = build_utm_zone_polygons(aoi)

    assert zone_polygons.crs.to_string() == "EPSG:4326"
    assert zone_polygons.loc[0, "utm_zone"] == 33


def test_outside_utm_latitude_range_adds_diagnostic():
    segments = segment_aoi_by_utm_zone(make_aoi(16.2, 84.5, 16.5, 85.0))

    assert "aoi_outside_standard_utm_latitude_range" in segments.loc[
        0,
        "diagnostics",
    ]


def test_possible_antimeridian_crossing_adds_diagnostic():
    segments = segment_aoi_by_utm_zone(make_aoi(-179.0, 10.0, 179.0, 11.0))

    assert segments["diagnostics"].apply(
        lambda diagnostics: "possible_antimeridian_crossing" in diagnostics
    ).all()
