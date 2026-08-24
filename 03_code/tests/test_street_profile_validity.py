"""Synthetic validity tests for street-profile denominators and topology."""

from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, box


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from street_context import (  # noqa: E402
    add_street_profile_topology_diagnostics,
    add_street_profile_width_diagnostics,
    calculate_building_street_profile_ratio,
)


CRS = "EPSG:32633"


def _profiles(widths, geometry=None):
    return gpd.GeoDataFrame(
        {"street_id": [f"s{index}" for index in range(len(widths))], "street_profile_width_m": widths},
        geometry=geometry or [LineString([(0, index * 5), (20, index * 5)]) for index in range(len(widths))],
        crs=CRS,
    )


def _ratio_input(width, topology=True):
    return gpd.GeoDataFrame(
        {
            "building_id": ["b1"],
            "height_m": [20.0],
            "street_profile_width_m": [width],
            "has_opposite_profile_evidence": [True],
            "street_profile_topology_valid": [topology],
            "street_profile_topology_invalid_reason": [None if topology else "profile_origin_intersects_building"],
        },
        geometry=[box(30, 0, 40, 10)], crs=CRS,
    )


@pytest.mark.parametrize("width,reason", [(0.0, "profile_width_zero"), (-1.0, "profile_width_negative"), (np.nan, "profile_width_missing"), (np.inf, "profile_width_non_finite"), (1e-7, "profile_width_below_metric_tolerance")])
def test_invalid_denominators_are_missing_with_reasons(width, reason):
    result = calculate_building_street_profile_ratio(_ratio_input(width))
    assert not result.loc[0, "has_valid_street_profile_ratio_strict"]
    assert np.isnan(result.loc[0, "street_profile_height_to_width_ratio_strict"])
    assert result.loc[0, "street_profile_ratio_strict_invalid_reason"] == reason


def test_valid_narrow_and_high_ratio_are_retained_not_capped():
    result = calculate_building_street_profile_ratio(_ratio_input(0.01))
    assert result.loc[0, "street_profile_height_to_width_ratio_strict"] == pytest.approx(2000.0)
    assert result.loc[0, "street_profile_ratio_outlier_flag"]


def test_profile_origin_intersection_is_topologically_invalid():
    profiles = _profiles([10.0], [LineString([(0, 0), (20, 0)])])
    buildings = gpd.GeoDataFrame({"building_id": ["b1"]}, geometry=[box(-1, -1, 2, 2)], crs=CRS)
    diagnosed = add_street_profile_topology_diagnostics(profiles, buildings, distance=10.0)
    assert not diagnosed.loc[0, "street_profile_topology_valid"]
    assert diagnosed.loc[0, "street_profile_topology_invalid_reason"] == "profile_origin_intersects_building"
    result = calculate_building_street_profile_ratio(_ratio_input(10.0, topology=False))
    assert np.isnan(result.loc[0, "street_profile_height_to_width_ratio_strict"])
    assert result.loc[0, "street_profile_ratio_strict_invalid_reason"] == "profile_origin_intersects_building"


def test_width_diagnostics_record_zero_length_geometry():
    diagnosed = add_street_profile_width_diagnostics(
        _profiles([5.0], [LineString([(0, 0), (0, 0)])])
    )
    assert not diagnosed.loc[0, "street_profile_width_valid"]
    assert diagnosed.loc[0, "street_profile_width_invalid_reason"] == "profile_geometry_zero_length"
