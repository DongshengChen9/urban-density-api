import math
import sys
from types import SimpleNamespace
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
CODE_DIR = TEST_DIR.parent
SRC_DIR = CODE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if not (SRC_DIR / "street_context.py").exists():
    raise FileNotFoundError(
        f"Cannot find street_context.py at expected path: {SRC_DIR / 'street_context.py'}"
    )

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Polygon

from street_context import (
    _to_string_if_list,
    assign_buildings_to_street_profiles,
    calculate_building_street_profile_ratio,
    calculate_street_profile_segments,
    aggregate_street_profile_ratio_to_units,
    summarize_street_profile_quality,
    fetch_streets_from_osmnx,
)


def test_osm_attributes_normalize_mixed_scalars_and_lists_for_cache_export():
    values = [123, [456, 789], "named", None, np.nan]
    normalized = [_to_string_if_list(value) for value in values]

    assert normalized[:3] == ["123", "456|789", "named"]
    assert normalized[3] is None
    assert np.isnan(normalized[4])


def test_osmnx_acquisition_records_explicit_endpoint_without_fallback(monkeypatch):
    settings = SimpleNamespace(overpass_url="https://default.invalid/api", timeout=180)
    edges = gpd.GeoDataFrame(
        {"osmid": [1], "highway": ["residential"]},
        geometry=[LineString([(0, 0), (10, 0)])],
        crs=CRS_GEOGRAPHIC,
    )
    fake_osmnx = SimpleNamespace(
        __version__="test", settings=settings,
        graph_from_polygon=lambda *_args, **_kwargs: "graph",
        graph_to_gdfs=lambda *_args, **_kwargs: edges,
    )
    monkeypatch.setitem(sys.modules, "osmnx", fake_osmnx)
    aoi = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])], crs=CRS_METRIC)
    streets, provenance = fetch_streets_from_osmnx(
        aoi, acquisition_config={"overpass_endpoint": "https://chosen.invalid/api", "timeout_seconds": 45},
        return_provenance=True,
    )
    assert len(streets) == 1
    assert streets["street_id"].iloc[0] == "street_00000"
    assert provenance["effective_overpass_endpoint"] == "https://chosen.invalid/api"
    assert provenance["endpoint_selection_mode"] == "explicit"
    assert settings.overpass_url == "https://default.invalid/api"


CRS_METRIC = "EPSG:32633"
CRS_OTHER_METRIC = "EPSG:3857"
CRS_GEOGRAPHIC = "EPSG:4326"


def make_buildings(crs=CRS_METRIC):
    """
    Create a simple synthetic building layer.

    Building b1 has valid height.
    Building b2 has missing height.
    """
    buildings = gpd.GeoDataFrame(
        {
            "building_id": ["b1", "b2"],
            "height_m": [20.0, np.nan],
        },
        geometry=[
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            Polygon([(30, 0), (40, 0), (40, 10), (30, 10)]),
        ],
        crs=crs,
    )

    return buildings


def make_street_profiles(crs=CRS_METRIC):
    """
    Create synthetic street-profile segments with already calculated widths.

    street_1 is valid.
    street_2 is valid but farther away.
    """
    streets = gpd.GeoDataFrame(
        {
            "street_id": ["street_1", "street_2"],
            "street_profile_width_m": [10.0, 20.0],
            "street_profile_openness": [0.5, 0.5],
            "street_profile_momepy_height_m": [20.0, 20.0],
            "street_profile_hw_ratio_momepy": [2.0, 1.0],
            "street_profile_width_is_capped": [False, False],
            "has_opposite_profile_evidence": [True, True],
        },
        geometry=[
            LineString([(15, -10), (15, 20)]),
            LineString([(100, -10), (100, 20)]),
        ],
        crs=crs,
    )

    return streets


def make_grid(crs=CRS_METRIC):
    """
    Create two grid cells covering the two synthetic buildings.
    """
    grid = gpd.GeoDataFrame(
        {
            "unit_id": ["cell_1", "cell_2"],
        },
        geometry=[
            Polygon([(-5, -5), (25, -5), (25, 25), (-5, 25)]),
            Polygon([(25, -5), (50, -5), (50, 25), (25, 25)]),
        ],
        crs=crs,
    )

    return grid

def test_assign_rejects_geographic_crs():
    buildings = make_buildings(crs=CRS_GEOGRAPHIC)
    streets = make_street_profiles(crs=CRS_GEOGRAPHIC)

    with pytest.raises(ValueError, match="projected CRS"):
        assign_buildings_to_street_profiles(
            buildings=buildings,
            streets_profile=streets,
            building_id_col="building_id",
            height_col="height_m",
        )


def test_assign_rejects_crs_mismatch():
    buildings = make_buildings(crs=CRS_METRIC)
    streets = make_street_profiles(crs=CRS_OTHER_METRIC)

    with pytest.raises(ValueError, match="same CRS"):
        assign_buildings_to_street_profiles(
            buildings=buildings,
            streets_profile=streets,
            building_id_col="building_id",
            height_col="height_m",
        )

def test_assign_buildings_to_nearest_street_profile_keeps_one_row_per_building():
    buildings = make_buildings()
    streets = make_street_profiles()

    building_street, summary = assign_buildings_to_street_profiles(
        buildings=buildings,
        streets_profile=streets,
        building_id_col="building_id",
        height_col="height_m",
    )

    assert len(building_street) == len(buildings)
    assert summary["n_buildings"] == 2
    assert summary["matched_to_street_count"] == 2
    assert summary["matched_to_street_share"] == 1.0

    assert "building_to_street_centerline_m" in building_street.columns
    assert "street_profile_width_m" in building_street.columns


def test_duplicate_nearest_street_matches_are_deduplicated():
    buildings = gpd.GeoDataFrame(
        {
            "building_id": ["b1"],
            "height_m": [20.0],
        },
        geometry=[
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        ],
        crs=CRS_METRIC,
    )

    # Two identical street geometries create equally nearest matches.
    streets = gpd.GeoDataFrame(
        {
            "street_id": ["street_a", "street_b"],
            "street_profile_width_m": [10.0, 10.0],
            "street_profile_openness": [0.5, 0.5],
            "street_profile_momepy_height_m": [20.0, 20.0],
            "street_profile_hw_ratio_momepy": [2.0, 2.0],
            "street_profile_width_is_capped": [False, False],
            "has_opposite_profile_evidence": [True, True],
        },
        geometry=[
            LineString([(15, -10), (15, 20)]),
            LineString([(15, -10), (15, 20)]),
        ],
        crs=CRS_METRIC,
    )

    building_street, summary = assign_buildings_to_street_profiles(
        buildings=buildings,
        streets_profile=streets,
        building_id_col="building_id",
        height_col="height_m",
    )

    assert summary["raw_join_rows_before_deduplication"] == 2
    assert summary["duplicate_join_rows_removed"] == 1
    assert len(building_street) == 1

    # Deterministic tie-breaking keeps the alphabetically first street_id.
    assert building_street.loc[0, "street_id"] == "street_a"

def test_building_street_profile_ratio_formula():
    building_street = gpd.GeoDataFrame(
        {
            "building_id": ["b1"],
            "height_m": [20.0],
            "street_profile_width_m": [10.0],
            "has_opposite_profile_evidence": [True],
        },
        geometry=[
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        ],
        crs=CRS_METRIC,
    )

    result = calculate_building_street_profile_ratio(
        building_street=building_street,
        height_col="height_m",
    )

    assert bool(result.loc[0, "has_valid_street_profile_ratio_prelim"]) is True
    assert bool(result.loc[0, "has_valid_street_profile_ratio_strict"]) is True
    assert result.loc[0, "street_profile_height_to_width_ratio_prelim"] == pytest.approx(2.0)
    assert result.loc[0, "street_profile_height_to_width_ratio_strict"] == pytest.approx(2.0)


def test_missing_height_produces_missing_ratio_not_zero():
    building_street = gpd.GeoDataFrame(
        {
            "building_id": ["b1"],
            "height_m": [np.nan],
            "street_profile_width_m": [10.0],
            "has_opposite_profile_evidence": [True],
        },
        geometry=[
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        ],
        crs=CRS_METRIC,
    )

    result = calculate_building_street_profile_ratio(
        building_street=building_street,
        height_col="height_m",
    )

    assert bool(result.loc[0, "has_valid_street_profile_ratio_prelim"]) is False
    assert bool(result.loc[0, "has_valid_street_profile_ratio_strict"]) is False
    assert math.isnan(result.loc[0, "street_profile_height_to_width_ratio_prelim"])
    assert math.isnan(result.loc[0, "street_profile_height_to_width_ratio_strict"])


def test_strict_ratio_requires_opposite_profile_evidence():
    building_street = gpd.GeoDataFrame(
        {
            "building_id": ["b1"],
            "height_m": [20.0],
            "street_profile_width_m": [10.0],
            "has_opposite_profile_evidence": [False],
        },
        geometry=[
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        ],
        crs=CRS_METRIC,
    )

    result = calculate_building_street_profile_ratio(
        building_street=building_street,
        height_col="height_m",
    )

    assert bool(result.loc[0, "has_valid_street_profile_ratio_prelim"]) is True
    assert bool(result.loc[0, "has_valid_street_profile_ratio_strict"]) is False
    assert result.loc[0, "street_profile_height_to_width_ratio_prelim"] == pytest.approx(2.0)
    assert math.isnan(result.loc[0, "street_profile_height_to_width_ratio_strict"])


def test_aggregate_street_profile_ratio_to_grid_units():
    buildings = make_buildings()
    streets = make_street_profiles()
    grid = make_grid()

    building_street, _summary = assign_buildings_to_street_profiles(
        buildings=buildings,
        streets_profile=streets,
        building_id_col="building_id",
        height_col="height_m",
    )

    building_street = calculate_building_street_profile_ratio(
        building_street=building_street,
        height_col="height_m",
    )

    grid_result = aggregate_street_profile_ratio_to_units(
        building_street=building_street,
        units=grid,
        unit_id_col="unit_id",
        building_id_col="building_id",
    )

    assert len(grid_result) == 2

    cell_1 = grid_result.loc[grid_result["unit_id"] == "cell_1"].iloc[0]
    cell_2 = grid_result.loc[grid_result["unit_id"] == "cell_2"].iloc[0]

    assert cell_1["street_profile_building_count"] == 1
    assert cell_1["street_profile_ratio_prelim_valid_count"] == 1
    assert cell_1["street_profile_ratio_strict_valid_count"] == 1
    assert cell_1["avg_street_profile_height_to_width_ratio_prelim"] == pytest.approx(2.0)

    assert cell_2["street_profile_building_count"] == 1
    assert cell_2["street_profile_ratio_prelim_valid_count"] == 0
    assert cell_2["street_profile_ratio_strict_valid_count"] == 0
    assert pd.isna(cell_2["avg_street_profile_height_to_width_ratio_prelim"])


def test_summarize_street_profile_quality():
    buildings = make_buildings()
    streets = make_street_profiles()
    grid = make_grid()

    building_street, join_summary = assign_buildings_to_street_profiles(
        buildings=buildings,
        streets_profile=streets,
        building_id_col="building_id",
        height_col="height_m",
    )

    building_street = calculate_building_street_profile_ratio(
        building_street=building_street,
        height_col="height_m",
    )

    grid_result = aggregate_street_profile_ratio_to_units(
        building_street=building_street,
        units=grid,
        unit_id_col="unit_id",
        building_id_col="building_id",
    )

    summary = summarize_street_profile_quality(
        streets_profile=streets,
        building_street=building_street,
        grid_street_profile=grid_result,
        join_summary=join_summary,
    )

    assert summary["n_street_segments"] == 2
    assert summary["valid_width_count"] == 2
    assert summary["valid_width_share"] == 1.0

    assert summary["n_buildings"] == 2
    assert summary["matched_to_street_count"] == 2
    assert summary["valid_height_count"] == 1

    assert summary["valid_ratio_prelim_count"] == 1
    assert summary["valid_ratio_strict_count"] == 1

    assert summary["n_grid_cells"] == 2
    assert summary["grid_cells_with_prelim_ratio_count"] == 1
    assert summary["grid_cells_with_strict_ratio_count"] == 1


def test_calculate_street_profile_segments_standardizes_momepy_output(monkeypatch):
    momepy = pytest.importorskip("momepy")

    streets = gpd.GeoDataFrame(
        {
            "street_id": ["street_1", "street_2"],
        },
        geometry=[
            LineString([(0, 0), (100, 0)]),
            LineString([(0, 50), (100, 50)]),
        ],
        crs=CRS_METRIC,
    )

    buildings = gpd.GeoDataFrame(
        {
            "building_id": ["b1"],
            "height_m": [20.0],
        },
        geometry=[
            Polygon([(0, 10), (10, 10), (10, 20), (0, 20)]),
        ],
        crs=CRS_METRIC,
    )

    def fake_street_profile(streets, buildings, distance, tick_length, height):
        return pd.DataFrame(
            {
                "width": [20.0, 60.0],
                "openness": [0.5, 1.0],
                "width_deviation": [1.0, np.nan],
                "height": [20.0, np.nan],
                "height_deviation": [0.0, np.nan],
                "hw_ratio": [1.0, np.nan],
            }
        )

    monkeypatch.setattr(momepy, "street_profile", fake_street_profile)

    result = calculate_street_profile_segments(
        streets=streets,
        buildings=buildings,
        height_col="height_m",
        distance=10,
        tick_length=60,
    )

    assert "street_profile_width_m" in result.columns
    assert "street_profile_openness" in result.columns
    assert "street_profile_hw_ratio_momepy" in result.columns
    assert "street_profile_width_is_capped" in result.columns
    assert "has_opposite_profile_evidence" in result.columns

    assert result.loc[0, "street_profile_width_m"] == pytest.approx(20.0)
    assert bool(result.loc[0, "street_profile_width_is_capped"]) is False
    assert bool(result.loc[0, "has_opposite_profile_evidence"]) is True

    assert result.loc[1, "street_profile_width_m"] == pytest.approx(60.0)
    assert bool(result.loc[1, "street_profile_width_is_capped"]) is True
    assert bool(result.loc[1, "has_opposite_profile_evidence"]) is False







