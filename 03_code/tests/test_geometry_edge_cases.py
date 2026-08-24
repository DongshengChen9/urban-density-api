from __future__ import annotations

from pathlib import Path
import sys

import geopandas as gpd
import pytest
from shapely.geometry import box, Polygon, LineString


# Make src importable when tests are run from project root or 03_code
TEST_DIR = Path(__file__).resolve().parent
CODE_DIR = TEST_DIR.parent
SRC_DIR = CODE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from aoi import validate_aoi
from preprocessing import (
    clean_building_geometries,
    clip_buildings_to_aoi,
    add_footprint_area,
)
from aggregation import create_grid
from quality import summarize_building_quality, summarize_unit_quality


CRS_METRIC = "EPSG:32633"


def test_validate_aoi_rejects_empty_aoi():
    """
    AOI must not be empty.
    Empty AOI would make data acquisition, clipping, and grid generation invalid.
    """
    aoi = gpd.GeoDataFrame(
        {"aoi_id": []},
        geometry=[],
        crs=CRS_METRIC,
    )

    with pytest.raises(ValueError, match="empty"):
        validate_aoi(aoi)


def test_clean_building_geometries_removes_missing_and_empty_geometries():
    """
    Real building datasets may contain missing or empty geometries.
    The workflow should remove them before spatial operations.
    """
    buildings = gpd.GeoDataFrame(
        {
            "building_id": ["b_valid", "b_missing", "b_empty"],
            "height_m": [10, 12, 8],
            "num_floors": [3, 4, 2],
        },
        geometry=[
            box(0, 0, 5, 5),
            None,
            Polygon(),
        ],
        crs=CRS_METRIC,
    )

    cleaned = clean_building_geometries(buildings)

    assert len(cleaned) == 1
    assert cleaned["building_id"].iloc[0] == "b_valid"
    assert cleaned.geometry.notna().all()
    assert not cleaned.geometry.is_empty.any()


def test_clean_building_geometries_repairs_invalid_polygon():
    """
    Invalid polygons can occur in real-world datasets.
    The cleaning step should attempt to repair them with make_valid().
    """
    invalid_bowtie = Polygon(
        [
            (0, 0),
            (2, 2),
            (0, 2),
            (2, 0),
            (0, 0),
        ]
    )

    assert not invalid_bowtie.is_valid

    buildings = gpd.GeoDataFrame(
        {
            "building_id": ["b_invalid"],
            "height_m": [10],
            "num_floors": [3],
        },
        geometry=[invalid_bowtie],
        crs=CRS_METRIC,
    )

    cleaned = clean_building_geometries(buildings)

    assert len(cleaned) == 1
    assert cleaned.geometry.is_valid.all()
    assert cleaned.geometry.geom_type.iloc[0] in ["Polygon", "MultiPolygon"]


def test_clean_building_geometries_rejects_non_polygon_only_layer():
    """
    Building footprints must be polygonal.
    A layer containing only lines should not pass as building data.
    """
    buildings = gpd.GeoDataFrame(
        {
            "building_id": ["b_line"],
            "height_m": [10],
            "num_floors": [3],
        },
        geometry=[LineString([(0, 0), (1, 1)])],
        crs=CRS_METRIC,
    )

    with pytest.raises(ValueError, match="No polygonal building geometries"):
        clean_building_geometries(buildings)


def test_clip_buildings_to_aoi_keeps_intersection_only():
    """
    If a building crosses the AOI boundary, clipping should keep only the
    portion inside the AOI.
    """
    aoi = gpd.GeoDataFrame(
        {"aoi_id": [1]},
        geometry=[box(0, 0, 10, 10)],
        crs=CRS_METRIC,
    )

    buildings = gpd.GeoDataFrame(
        {
            "building_id": ["b_crossing"],
            "height_m": [10],
            "num_floors": [3],
        },
        geometry=[box(5, 5, 15, 15)],
        crs=CRS_METRIC,
    )

    clipped = clip_buildings_to_aoi(buildings, aoi)
    clipped = add_footprint_area(clipped)

    assert len(clipped) == 1
    assert clipped["footprint_area_m2"].iloc[0] == pytest.approx(25.0)


def test_clip_buildings_to_aoi_rejects_when_no_buildings_remain():
    """
    If no buildings intersect the AOI, the current preprocessing function
    should raise a clear error.
    """
    aoi = gpd.GeoDataFrame(
        {"aoi_id": [1]},
        geometry=[box(0, 0, 10, 10)],
        crs=CRS_METRIC,
    )

    buildings = gpd.GeoDataFrame(
        {
            "building_id": ["b_outside"],
            "height_m": [10],
            "num_floors": [3],
        },
        geometry=[box(20, 20, 30, 30)],
        crs=CRS_METRIC,
    )

    with pytest.raises(ValueError, match="No buildings remain"):
        clip_buildings_to_aoi(buildings, aoi)


def test_create_grid_identifies_partial_cells():
    """
    AOI: 12 x 10 m = 120 m²
    Cell size: 10 m

    Expected:
    - one full 10 x 10 cell = 100 m²
    - one partial 2 x 10 cell = 20 m²
    """
    aoi = gpd.GeoDataFrame(
        {"aoi_id": [1]},
        geometry=[box(0, 0, 12, 10)],
        crs=CRS_METRIC,
    )

    grid = create_grid(aoi=aoi, cell_size_m=10)

    assert len(grid) == 2
    assert grid["unit_area_m2"].sum() == pytest.approx(120.0)
    assert grid["is_partial_cell"].sum() == 1
    assert grid["unit_area_m2"].min() == pytest.approx(20.0)
    assert grid["unit_area_m2"].max() == pytest.approx(100.0)
    assert set(grid["unit_id"]) == {"r00000_c00000", "r00000_c00001"}
    assert set(grid["column_index"]) == {0, 1}


def test_create_grid_rejects_zero_or_negative_cell_size():
    """
    Grid cell size must be positive.
    """
    aoi = gpd.GeoDataFrame(
        {"aoi_id": [1]},
        geometry=[box(0, 0, 10, 10)],
        crs=CRS_METRIC,
    )

    with pytest.raises(ValueError, match="positive"):
        create_grid(aoi=aoi, cell_size_m=0)

    with pytest.raises(ValueError, match="positive"):
        create_grid(aoi=aoi, cell_size_m=-10)


def test_building_quality_detects_duplicate_ids():
    """
    Duplicate building IDs should be reported because they may indicate duplicated
    features or data-source problems.
    """
    buildings = gpd.GeoDataFrame(
        {
            "building_id": ["b_001", "b_001"],
            "height_m": [10, 12],
            "num_floors": [3, 4],
        },
        geometry=[
            box(0, 0, 5, 5),
            box(10, 0, 15, 5),
        ],
        crs=CRS_METRIC,
    )

    buildings = add_footprint_area(buildings)
    quality = summarize_building_quality(buildings)

    assert quality["duplicate_building_id_count"] == 1
    assert quality["invalid_geometry_count"] == 0
    assert quality["empty_geometry_count"] == 0


def test_unit_quality_reports_partial_grid_cells_indirectly_by_area():
    """
    Unit quality should report area ranges.
    Partial cells can be detected because min area is smaller than max area.
    """
    aoi = gpd.GeoDataFrame(
        {"aoi_id": [1]},
        geometry=[box(0, 0, 12, 10)],
        crs=CRS_METRIC,
    )

    grid = create_grid(aoi=aoi, cell_size_m=10)
    quality = summarize_unit_quality(grid)

    assert quality["n_units"] == 2
    assert quality["is_projected"] is True
    assert quality["invalid_geometry_count"] == 0
    assert quality["empty_geometry_count"] == 0
    assert quality["unit_area_min_m2"] == pytest.approx(20.0)
    assert quality["unit_area_max_m2"] == pytest.approx(100.0)


def test_clean_building_geometries_rejects_empty_building_layer():
    """
    Empty building layers should fail during preprocessing, so the user gets a
    clear message instead of silent empty outputs.
    """
    buildings = gpd.GeoDataFrame(
        {
            "building_id": [],
            "height_m": [],
            "num_floors": [],
        },
        geometry=[],
        crs=CRS_METRIC,
    )

    with pytest.raises(ValueError, match="empty"):
        clean_building_geometries(buildings)
