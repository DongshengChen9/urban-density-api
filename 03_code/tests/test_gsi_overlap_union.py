"""Known-answer tests for union-based GSI coverage."""

from pathlib import Path
import sys

import geopandas as gpd
import pytest
from shapely.geometry import box


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from indicators import calculate_gsi  # noqa: E402


CRS = "EPSG:32633"


def _units(*geometries):
    return gpd.GeoDataFrame(
        {"unit_id": [f"cell_{index}" for index in range(len(geometries))]},
        geometry=list(geometries),
        crs=CRS,
    )


def _buildings(*geometries):
    return gpd.GeoDataFrame(
        {"building_id": [f"building_{index}" for index in range(len(geometries))]},
        geometry=list(geometries),
        crs=CRS,
    )


def test_gsi_uses_union_for_overlapping_and_duplicate_footprints():
    result = calculate_gsi(
        _buildings(box(0, 0, 4, 4), box(2, 0, 6, 4)),
        _units(box(0, 0, 10, 10)),
    ).iloc[0]
    assert result.gsi == pytest.approx(0.24)
    assert result.gsi_raw_sum == pytest.approx(0.32)
    assert result.building_footprint_union_area_m2 == pytest.approx(24.0)
    assert result.footprint_overlap_area_m2 == pytest.approx(8.0)
    assert result.footprint_overlap_flag

    duplicate = calculate_gsi(
        _buildings(box(0, 0, 10, 10), box(0, 0, 10, 10)),
        _units(box(0, 0, 10, 10)),
    ).iloc[0]
    assert duplicate.gsi == pytest.approx(1.0)
    assert duplicate.gsi_raw_sum == pytest.approx(2.0)
    assert duplicate.gsi_overlap_difference == pytest.approx(1.0)


def test_gsi_partitions_crossing_footprints_and_preserves_true_zeros():
    crossing = calculate_gsi(
        _buildings(box(5, 0, 15, 10)),
        _units(box(0, 0, 10, 10), box(10, 0, 20, 10)),
    ).set_index("unit_id")
    assert crossing.loc["cell_0", "gsi"] == pytest.approx(0.5)
    assert crossing.loc["cell_1", "gsi"] == pytest.approx(0.5)

    empty = calculate_gsi(_buildings(), _units(box(0, 0, 10, 10))).iloc[0]
    assert empty.gsi == 0.0
    assert empty.gsi_raw_sum == 0.0
