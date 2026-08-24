from __future__ import annotations

from pathlib import Path
import sys

import geopandas as gpd
import pytest
from shapely.geometry import box
from pyproj import CRS


# Make src importable when tests are run from project root or 03_code
TEST_DIR = Path(__file__).resolve().parent
CODE_DIR = TEST_DIR.parent
SRC_DIR = CODE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from aoi import create_aoi_from_bbox
from preprocessing import (
    estimate_metric_crs,
    reproject_to_metric,
    add_footprint_area,
)
from aggregation import create_grid
from indicators import (
    calculate_gsi,
    calculate_neighbor_distance,
    calculate_built_volume_density,
)


CRS_GEOGRAPHIC = "EPSG:4326"
CRS_METRIC = "EPSG:32633"
CRS_OTHER_METRIC = "EPSG:3857"


def make_buildings(crs=CRS_METRIC):
    """
    Create a simple building GeoDataFrame.

    Geometry is 5 x 5 coordinate units.
    If CRS is projected, area should be 25 m².
    If CRS is EPSG:4326, this must not be used for area calculation.
    """
    return gpd.GeoDataFrame(
        {
            "building_id": ["b_001"],
            "height_m": [10],
            "num_floors": [4],
        },
        geometry=[box(0, 0, 5, 5)],
        crs=crs,
    )


def make_units(crs=CRS_METRIC):
    """
    Create a simple 10 x 10 aggregation unit.
    """
    units = gpd.GeoDataFrame(
        {
            "unit_id": ["cell_00001"],
        },
        geometry=[box(0, 0, 10, 10)],
        crs=crs,
    )

    if not units.crs.is_geographic:
        units["unit_area_m2"] = units.geometry.area

    return units


def test_estimate_metric_crs_for_vienna_returns_projected_crs():
    """
    AOI starts in EPSG:4326, but workflow must estimate a projected CRS
    before area and distance calculations.

    For Vienna, GeoPandas should estimate a projected UTM CRS.
    """
    aoi = create_aoi_from_bbox(
        name="vienna_test",
        bounds={
            "minx": 16.3700,
            "miny": 48.2070,
            "maxx": 16.3740,
            "maxy": 48.2100,
        },
        crs=CRS_GEOGRAPHIC,
    )

    metric_crs = estimate_metric_crs(aoi)
    parsed_crs = CRS.from_user_input(metric_crs)

    assert metric_crs is not None
    assert parsed_crs.is_projected
    assert not parsed_crs.is_geographic


def test_reproject_to_metric_returns_projected_layers():
    """
    Buildings and AOI should both be reprojected to the same metric CRS.
    """
    aoi = create_aoi_from_bbox(
        name="vienna_test",
        bounds={
            "minx": 16.3700,
            "miny": 48.2070,
            "maxx": 16.3740,
            "maxy": 48.2100,
        },
        crs=CRS_GEOGRAPHIC,
    )

    buildings = gpd.GeoDataFrame(
        {
            "building_id": ["b_001"],
            "height_m": [10],
            "num_floors": [4],
        },
        geometry=[box(16.371, 48.208, 16.372, 48.209)],
        crs=CRS_GEOGRAPHIC,
    )

    buildings_metric, aoi_metric = reproject_to_metric(
        buildings=buildings,
        aoi=aoi,
        target_crs=CRS_METRIC,
    )

    assert str(buildings_metric.crs) == CRS_METRIC
    assert str(aoi_metric.crs) == CRS_METRIC
    assert not buildings_metric.crs.is_geographic
    assert not aoi_metric.crs.is_geographic


def test_add_footprint_area_in_metric_crs_returns_square_metres():
    """
    In a projected CRS, a 5 x 5 m building should have area 25 m².
    """
    buildings = make_buildings(crs=CRS_METRIC)

    result = add_footprint_area(buildings)

    assert "footprint_area_m2" in result.columns
    assert result["footprint_area_m2"].iloc[0] == pytest.approx(25.0)


def test_add_footprint_area_rejects_geographic_crs():
    """
    Area calculation in EPSG:4326 would be in square degrees, not square metres.
    The workflow must reject this.
    """
    buildings = make_buildings(crs=CRS_GEOGRAPHIC)

    with pytest.raises(ValueError, match="geographic CRS"):
        add_footprint_area(buildings)


def test_create_grid_rejects_geographic_aoi():
    """
    Grid cell size is defined in metres.
    Therefore, grid generation must not run on EPSG:4326 AOI.
    """
    aoi = create_aoi_from_bbox(
        name="geographic_aoi",
        bounds={
            "minx": 16.3700,
            "miny": 48.2070,
            "maxx": 16.3740,
            "maxy": 48.2100,
        },
        crs=CRS_GEOGRAPHIC,
    )

    with pytest.raises(ValueError, match="geographic CRS"):
        create_grid(aoi=aoi, cell_size_m=100)


def test_create_grid_metric_crs_has_expected_area():
    """
    In projected CRS, a 10 x 10 m AOI with 5 m cells should produce
    four full cells of 25 m² each.
    """
    aoi = gpd.GeoDataFrame(
        {"aoi_id": [1]},
        geometry=[box(0, 0, 10, 10)],
        crs=CRS_METRIC,
    )

    grid = create_grid(aoi=aoi, cell_size_m=5)

    assert len(grid) == 4
    assert grid["unit_area_m2"].sum() == pytest.approx(100.0)
    assert grid["unit_area_m2"].min() == pytest.approx(25.0)
    assert grid["unit_area_m2"].max() == pytest.approx(25.0)
    assert not grid.crs.is_geographic


def test_indicators_reject_geographic_crs_for_gsi():
    """
    GSI requires metric area calculations.
    The indicator must reject EPSG:4326 input.
    """
    buildings = make_buildings(crs=CRS_GEOGRAPHIC)
    units = make_units(crs=CRS_GEOGRAPHIC)

    with pytest.raises(ValueError, match="geographic CRS"):
        calculate_gsi(buildings, units, params={})


def test_indicators_reject_geographic_crs_for_volume():
    """
    Built Volume Density requires footprint area in m².
    Therefore EPSG:4326 input must be rejected.
    """
    buildings = make_buildings(crs=CRS_GEOGRAPHIC)
    units = make_units(crs=CRS_GEOGRAPHIC)

    with pytest.raises(ValueError, match="geographic CRS"):
        calculate_built_volume_density(buildings, units, params={})


def test_indicators_reject_geographic_crs_for_neighbor_distance():
    """
    Neighbour distance must be measured in metres, not degrees.
    Therefore EPSG:4326 input must be rejected.
    """
    buildings = gpd.GeoDataFrame(
        {
            "building_id": ["b_001", "b_002"],
        },
        geometry=[
            box(0, 0, 1, 1),
            box(2, 0, 3, 1),
        ],
        crs=CRS_GEOGRAPHIC,
    )

    units = make_units(crs=CRS_GEOGRAPHIC)

    with pytest.raises(ValueError, match="geographic CRS"):
        calculate_neighbor_distance(buildings, units, params={})


def test_indicators_reject_crs_mismatch():
    """
    Buildings and aggregation units must use the same CRS.
    Otherwise, overlay and distance operations are invalid.
    """
    buildings = make_buildings(crs=CRS_METRIC)
    units = make_units(crs=CRS_OTHER_METRIC)

    with pytest.raises(ValueError, match="CRS mismatch"):
        calculate_gsi(buildings, units, params={})


def test_missing_crs_is_rejected():
    """
    Layers without CRS are unsafe because units are unknown.
    """
    buildings = make_buildings(crs=CRS_METRIC)
    buildings = buildings.set_crs(None, allow_override=True)

    units = make_units(crs=CRS_METRIC)

    with pytest.raises(ValueError, match="has no CRS"):
        calculate_gsi(buildings, units, params={})