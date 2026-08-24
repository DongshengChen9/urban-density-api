from __future__ import annotations

from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box
from indicators import (
    calculate_gsi,
    calculate_far_fsi,
    calculate_built_volume_density,
    calculate_neighbor_distance,
    calculate_height_to_distance_ratio,
    calculate_building_neighbor_diagnostics,
    run_indicators,
)


# Make src importable when tests are run from project root or 03_code
TEST_DIR = Path(__file__).resolve().parent
CODE_DIR = TEST_DIR.parent
SRC_DIR = CODE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from indicators import (
    calculate_gsi,
    calculate_far_fsi,
    calculate_built_volume_density,
    calculate_neighbor_distance,
    calculate_height_to_distance_ratio,
    run_indicators,
)


CRS_METRIC = "EPSG:32633"


def make_units(geometries):
    return gpd.GeoDataFrame(
        {
            "unit_id": [f"cell_{i+1:05d}" for i in range(len(geometries))],
            "cell_size_m": [10] * len(geometries),
        },
        geometry=geometries,
        crs=CRS_METRIC,
    ).assign(unit_area_m2=lambda df: df.geometry.area)


def make_buildings(geometries, heights=None, floors=None):
    n = len(geometries)

    data = {
        "building_id": [f"b_{i+1:03d}" for i in range(n)],
    }

    if heights is not None:
        data["height_m"] = heights

    if floors is not None:
        data["num_floors"] = floors

    return gpd.GeoDataFrame(
        data,
        geometry=geometries,
        crs=CRS_METRIC,
    )


def value_for_unit(df, unit_id, column):
    return df.loc[df["unit_id"] == unit_id, column].iloc[0]


def test_gsi_simple_square():
    """
    Unit: 10 x 10 m = 100 m²
    Building: 5 x 5 m = 25 m²
    Expected GSI = 25 / 100 = 0.25
    """
    units = make_units([box(0, 0, 10, 10)])
    buildings = make_buildings([box(0, 0, 5, 5)])

    result = calculate_gsi(buildings, units, params={})

    assert result.shape[0] == 1
    assert value_for_unit(result, "cell_00001", "building_footprint_area_m2") == pytest.approx(25.0)
    assert value_for_unit(result, "cell_00001", "gsi") == pytest.approx(0.25)


def test_gsi_building_crosses_two_units():
    """
    A building crosses two 10 x 10 m grid cells.
    Each cell should receive only the intersecting building area.

    Building: x=5..15, y=0..10
    Unit 1: x=0..10, receives 50 m²
    Unit 2: x=10..20, receives 50 m²
    Expected GSI for each = 50 / 100 = 0.5
    """
    units = make_units([
        box(0, 0, 10, 10),
        box(10, 0, 20, 10),
    ])

    buildings = make_buildings([box(5, 0, 15, 10)])

    result = calculate_gsi(buildings, units, params={})

    assert value_for_unit(result, "cell_00001", "building_footprint_area_m2") == pytest.approx(50.0)
    assert value_for_unit(result, "cell_00002", "building_footprint_area_m2") == pytest.approx(50.0)
    assert value_for_unit(result, "cell_00001", "gsi") == pytest.approx(0.5)
    assert value_for_unit(result, "cell_00002", "gsi") == pytest.approx(0.5)


def test_far_fsi_simple_floor_count():
    """
    Unit: 100 m²
    Building footprint: 25 m²
    Floors: 4
    Expected floor area = 25 * 4 = 100 m²
    Expected FAR/FSI = 100 / 100 = 1.0
    """
    units = make_units([box(0, 0, 10, 10)])
    buildings = make_buildings(
        [box(0, 0, 5, 5)],
        floors=[4],
    )

    result = calculate_far_fsi(buildings, units, params={})

    assert value_for_unit(result, "cell_00001", "floor_area_sum_m2") == pytest.approx(100.0)
    assert value_for_unit(result, "cell_00001", "far_fsi") == pytest.approx(1.0)
    assert value_for_unit(result, "cell_00001", "floor_data_valid_area_share") == pytest.approx(1.0)


def test_far_fsi_missing_floors_stays_missing():
    """
    Missing floor count must not be converted to 0 or 1.
    FAR/FSI should be NaN when the only building has missing num_floors.
    """
    units = make_units([box(0, 0, 10, 10)])
    buildings = make_buildings(
        [box(0, 0, 5, 5)],
        floors=[np.nan],
    )

    result = calculate_far_fsi(buildings, units, params={})

    far = value_for_unit(result, "cell_00001", "far_fsi")
    valid_share = value_for_unit(result, "cell_00001", "floor_data_valid_area_share")

    assert pd.isna(far)
    assert valid_share == pytest.approx(0.0)


def test_built_volume_density_simple_height():
    """
    Unit: 100 m²
    Building footprint: 25 m²
    Height: 10 m
    Expected volume = 25 * 10 = 250 m³
    Expected built volume density = 250 / 100 = 2.5
    """
    units = make_units([box(0, 0, 10, 10)])
    buildings = make_buildings(
        [box(0, 0, 5, 5)],
        heights=[10],
    )

    result = calculate_built_volume_density(buildings, units, params={})

    assert value_for_unit(result, "cell_00001", "built_volume_m3") == pytest.approx(250.0)
    assert value_for_unit(result, "cell_00001", "built_volume_density") == pytest.approx(2.5)
    assert value_for_unit(result, "cell_00001", "height_valid_area_share") == pytest.approx(1.0)


def test_built_volume_density_missing_height_stays_missing():
    """
    Missing height must not be converted to zero.
    Built volume density should be NaN if the only building has missing height.
    """
    units = make_units([box(0, 0, 10, 10)])
    buildings = make_buildings(
        [box(0, 0, 5, 5)],
        heights=[np.nan],
    )

    result = calculate_built_volume_density(buildings, units, params={})

    bvd = value_for_unit(result, "cell_00001", "built_volume_density")
    valid_share = value_for_unit(result, "cell_00001", "height_valid_area_share")

    assert pd.isna(bvd)
    assert valid_share == pytest.approx(0.0)


def test_neighbor_distance_between_two_separated_buildings():
    """
    Building A: x=0..10
    Building B: x=15..25
    Edge-to-edge distance = 5 m

    This checks footprint-to-footprint distance, not centroid distance.
    """
    units = make_units([box(-5, -5, 30, 15)])
    buildings = make_buildings([
        box(0, 0, 10, 10),
        box(15, 0, 25, 10),
    ])

    result = calculate_neighbor_distance(buildings, units, params={})

    assert value_for_unit(result, "cell_00001", "avg_neighbor_distance_m") == pytest.approx(5.0)
    assert value_for_unit(result, "cell_00001", "median_neighbor_distance_m") == pytest.approx(5.0)
    assert value_for_unit(result, "cell_00001", "neighbor_distance_valid_count") == 2


def test_height_to_distance_ratio_between_two_buildings():
    """
    Two buildings are 5 m apart.

    Building A: height 10 m -> ratio 10 / 5 = 2
    Building B: height 20 m -> ratio 20 / 5 = 4
    Mean ratio = 3
    Median ratio = 3
    """
    units = make_units([box(-5, -5, 30, 15)])
    buildings = make_buildings(
        [
            box(0, 0, 10, 10),
            box(15, 0, 25, 10),
        ],
        heights=[10, 20],
    )

    result = calculate_height_to_distance_ratio(
        buildings,
        units,
        params={"min_distance_for_ratio_m": 0.5},
    )

    assert value_for_unit(result, "cell_00001", "avg_height_to_distance_ratio") == pytest.approx(3.0)
    assert value_for_unit(result, "cell_00001", "median_height_to_distance_ratio") == pytest.approx(3.0)
    assert value_for_unit(result, "cell_00001", "height_distance_valid_count") == 2


def test_touching_buildings_have_zero_neighbor_distance_but_no_ratio():
    """
    Touching buildings have footprint distance = 0.

    Neighbour distance should be 0.
    Height-to-distance ratio should not be calculated because division by
    zero or near-zero distances would produce unstable values.
    """
    units = make_units([box(-5, -5, 25, 15)])
    buildings = make_buildings(
        [
            box(0, 0, 10, 10),
            box(10, 0, 20, 10),
        ],
        heights=[10, 20],
    )

    distance_result = calculate_neighbor_distance(buildings, units, params={})

    assert value_for_unit(distance_result, "cell_00001", "avg_neighbor_distance_m") == pytest.approx(0.0)
    assert value_for_unit(distance_result, "cell_00001", "median_neighbor_distance_m") == pytest.approx(0.0)
    assert value_for_unit(distance_result, "cell_00001", "neighbor_distance_valid_count") == 2

    ratio_result = calculate_height_to_distance_ratio(
        buildings,
        units,
        params={"min_distance_for_ratio_m": 0.5},
    )

    assert pd.isna(value_for_unit(ratio_result, "cell_00001", "avg_height_to_distance_ratio"))
    assert pd.isna(value_for_unit(ratio_result, "cell_00001", "median_height_to_distance_ratio"))
    assert value_for_unit(ratio_result, "cell_00001", "height_distance_valid_count") == 0


def test_single_building_has_no_neighbor():
    """
    With only one building, nearest-neighbour distance is undefined.
    """
    units = make_units([box(0, 0, 20, 20)])
    buildings = make_buildings(
        [box(0, 0, 10, 10)],
        heights=[10],
    )

    distance_result = calculate_neighbor_distance(buildings, units, params={})
    ratio_result = calculate_height_to_distance_ratio(
        buildings,
        units,
        params={"min_distance_for_ratio_m": 0.5},
    )

    assert pd.isna(value_for_unit(distance_result, "cell_00001", "avg_neighbor_distance_m"))
    assert value_for_unit(distance_result, "cell_00001", "neighbor_distance_valid_count") == 0

    assert pd.isna(value_for_unit(ratio_result, "cell_00001", "avg_height_to_distance_ratio"))
    assert value_for_unit(ratio_result, "cell_00001", "height_distance_valid_count") == 0


def test_run_indicators_returns_all_enabled_outputs():
    """
    Integration-style synthetic test for the indicator runner.
    """
    units = make_units([box(0, 0, 10, 10)])
    buildings = make_buildings(
        [
            box(0, 0, 5, 5),
            box(7, 0, 9, 2),
        ],
        heights=[10, 8],
        floors=[4, 2],
    )

    config = {
        "indicators": {
            "gsi": True,
            "far_fsi": True,
            "built_volume_density": True,
            "neighbor_distance": True,
            "height_to_distance_ratio": True,
        },
        "indicator_parameters": {
            "min_distance_for_ratio_m": 0.5,
        },
    }

    result = run_indicators(buildings, units, config=config)

    expected_columns = {
        "gsi",
        "far_fsi",
        "built_volume_density",
        "avg_neighbor_distance_m",
        "avg_height_to_distance_ratio",
    }

    assert expected_columns.issubset(set(result.columns))
    assert result.shape[0] == 1



def test_building_neighbor_diagnostics_classifies_touching_buildings():
    """
    Touching buildings should be classified as zero-distance touching_boundary,
    and height-to-distance ratio should remain invalid.
    """
    buildings = make_buildings(
        [
            box(0, 0, 10, 10),
            box(10, 0, 20, 10),
        ],
        heights=[10, 20],
    )

    result = calculate_building_neighbor_diagnostics(
        buildings=buildings,
        aoi=None,
        params={"min_distance_for_ratio_m": 0.5},
    )

    assert len(result) == 2
    assert result["is_zero_distance"].sum() == 2
    assert result["zero_distance_relation"].eq("touching_boundary").sum() == 2
    assert result["touches_neighbor"].sum() == 2
    assert result["has_valid_height_to_distance_ratio"].sum() == 0


def test_building_neighbor_diagnostics_classifies_overlapping_buildings():
    """
    Overlapping buildings should be flagged as overlap rather than normal spacing.
    """
    buildings = make_buildings(
        [
            box(0, 0, 10, 10),
            box(5, 0, 15, 10),
        ],
        heights=[10, 20],
    )

    result = calculate_building_neighbor_diagnostics(
        buildings=buildings,
        aoi=None,
        params={"min_distance_for_ratio_m": 0.5},
    )

    assert len(result) == 2
    assert result["is_zero_distance"].sum() == 2
    assert result["zero_distance_relation"].eq("overlap").sum() == 2
    assert result["overlaps_neighbor"].sum() == 2
    assert result["overlap_area_m2"].min() > 0