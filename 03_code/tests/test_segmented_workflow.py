from __future__ import annotations

from pathlib import Path
import sys

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, box


TEST_DIR = Path(__file__).resolve().parent
CODE_DIR = TEST_DIR.parent
SRC_DIR = CODE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


import segmented_workflow as sw
from segmented_workflow import (
    process_segmented_core_indicators,
    validate_segmented_core_config,
)


def make_cross_zone_aoi() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"name": ["cross_zone"]},
        geometry=[box(11.99, 48.0, 12.01, 48.02)],
        crs="EPSG:4326",
    )


def make_cross_zone_buildings() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "building_id": ["west", "east", "crossing"],
            "height_m": [12.0, 18.0, 15.0],
            "num_floors": [3, 5, 4],
        },
        geometry=[
            box(11.992, 48.006, 11.996, 48.012),
            box(12.004, 48.006, 12.008, 48.012),
            box(11.998, 48.006, 12.002, 48.012),
        ],
        crs="EPSG:4326",
    )


def make_boundary_neighbor_buildings(include_crossing=True) -> gpd.GeoDataFrame:
    building_ids = ["west", "east"]
    geometries = [
        box(11.9990, 48.0060, 11.9993, 48.0063),
        box(12.0001, 48.0060, 12.0004, 48.0063),
    ]

    if include_crossing:
        building_ids.append("crossing")
        geometries.append(box(11.99985, 48.0070, 12.00015, 48.0073))

    return gpd.GeoDataFrame(
        {
            "building_id": building_ids,
            "height_m": [10.0] * len(building_ids),
            "num_floors": [3] * len(building_ids),
        },
        geometry=geometries,
        crs="EPSG:4326",
    )


def make_boundary_street_context_buildings() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "building_id": ["west", "east", "context_only"],
            "height_m": [10.0, 20.0, 30.0],
            "num_floors": [3, 5, 7],
        },
        geometry=[
            box(11.9990, 48.0060, 11.9993, 48.0063),
            box(12.0001, 48.0060, 12.0004, 48.0063),
            box(12.0102, 48.0060, 12.0105, 48.0063),
        ],
        crs="EPSG:4326",
    )


def segmented_config() -> dict:
    return {
        "aggregation": {"cell_size_m": 500},
        "indicators": {
            "gsi": True,
            "far_fsi": True,
            "built_volume_density": True,
            "neighbor_distance": False,
            "height_to_distance_ratio": False,
        },
        "height_enrichment": {"enabled": False},
        "street_context": {"enabled": False},
        "cache": {"enabled": False},
    }


def test_segmented_core_processing_writes_valid_segment_outputs(tmp_path):
    output_dir = tmp_path / "segmented_run"

    result = process_segmented_core_indicators(
        buildings=make_cross_zone_buildings(),
        aoi=make_cross_zone_aoi(),
        config=segmented_config(),
        output_dir=output_dir,
        save_outputs=True,
    )

    merged = result["indicator_grid"]
    summary = result["summary"]

    assert summary["processing_mode"] == "segmented_utm"
    assert summary["n_segments"] == 2
    assert summary["segment_utm_zones"] == [32, 33]
    assert summary["segment_epsg_list"] == [32632, 32633]
    assert merged.crs.to_string() == "EPSG:4326"
    assert merged.geometry.is_valid.all()
    assert (~merged.geometry.is_empty).all()

    required_columns = {
        "cell_id_global",
        "cell_id_local",
        "segment_id",
        "utm_zone",
        "segment_epsg",
        "calculation_crs",
        "cell_area_m2",
        "gsi",
        "far_fsi",
        "built_volume_density",
    }
    assert required_columns.issubset(set(merged.columns))

    assert (
        output_dir
        / "segments"
        / "segment_001"
        / "processed"
        / "aoi_segment_metric.gpkg"
    ).exists()
    assert (
        output_dir
        / "segments"
        / "segment_001"
        / "processed"
        / "buildings_segment_metric.gpkg"
    ).exists()
    assert (
        output_dir
        / "segments"
        / "segment_001"
        / "indicators"
        / "grid_indicators.gpkg"
    ).exists()
    assert (
        output_dir / "indicators" / "grid_indicators_segmented_wgs84.gpkg"
    ).exists()
    assert (output_dir / "reports" / "segmented_crs_summary.json").exists()


def test_cross_boundary_building_is_split_not_counted_whole_in_each_segment(tmp_path):
    output_dir = tmp_path / "segmented_run"

    process_segmented_core_indicators(
        buildings=make_cross_zone_buildings(),
        aoi=make_cross_zone_aoi(),
        config=segmented_config(),
        output_dir=output_dir,
        save_outputs=True,
    )

    segment_pieces = []
    for segment_id in ("segment_001", "segment_002"):
        segment_buildings = gpd.read_file(
            output_dir
            / "segments"
            / segment_id
            / "processed"
            / "buildings_segment_metric.gpkg",
            layer="buildings_segment_metric",
        ).to_crs("EPSG:4326")
        segment_pieces.append(
            segment_buildings[segment_buildings["building_id"] == "crossing"]
        )

    crossing_pieces = gpd.GeoDataFrame(
        geometry=list(segment_pieces[0].geometry) + list(segment_pieces[1].geometry),
        crs="EPSG:4326",
    )

    assert len(crossing_pieces) == 2
    assert crossing_pieces.iloc[0].geometry.bounds[2] <= 12.0 + 1e-9
    assert crossing_pieces.iloc[1].geometry.bounds[0] >= 12.0 - 1e-9

    original_crossing = make_cross_zone_buildings().set_index("building_id").loc[
        "crossing",
        "geometry",
    ]
    split_area = sum(geom.area for geom in crossing_pieces.geometry)

    assert split_area == pytest.approx(original_crossing.area, rel=1e-4)


def test_segmented_core_config_rejects_unsupported_branches():
    config = segmented_config()
    config["indicators"]["height_to_distance_ratio"] = True

    with pytest.raises(ValueError, match="indicators.height_to_distance_ratio"):
        validate_segmented_core_config(config)


def test_segmented_core_config_allows_street_context_enabled():
    config = segmented_config()
    config["street_context"] = {"enabled": True}

    validate_segmented_core_config(config)


def test_segmented_core_config_allows_neighbor_distance_enabled():
    config = segmented_config()
    config["indicators"]["neighbor_distance"] = True

    validate_segmented_core_config(config)


def test_segmented_core_config_allows_height_enrichment_enabled():
    config = segmented_config()
    config["height_enrichment"] = {"enabled": True}

    validate_segmented_core_config(config)


def test_segmented_core_config_rejects_external_cache_reuse():
    config = segmented_config()
    config["cache"] = {
        "enabled": True,
        "use_existing_enriched_buildings": True,
        "source_output_name": "base_run",
    }

    with pytest.raises(ValueError, match="cache.use_existing_enriched_buildings"):
        validate_segmented_core_config(config)


def test_segmented_neighbor_distance_uses_cross_zone_context_buffer(tmp_path):
    config = segmented_config()
    config["indicators"]["neighbor_distance"] = True
    config["crs_strategy"] = {"context_buffer_m": 150}

    result = process_segmented_core_indicators(
        buildings=make_boundary_neighbor_buildings(include_crossing=False),
        aoi=make_cross_zone_aoi(),
        config=config,
        output_dir=tmp_path / "segmented_run",
        save_outputs=True,
    )

    summary = result["summary"]["neighbor_distance_summary"]
    indicator_grid = result["indicator_grid"]

    assert result["summary"]["segmented_neighbor_distance_enabled"] is True
    assert result["summary"]["segmented_context_buffer_m"] == 150.0
    assert summary["valid_neighbor_distance_count"] == 2
    assert summary["valid_neighbor_distance_share"] == 1.0
    assert "avg_neighbor_distance_m" in indicator_grid.columns
    assert "median_neighbor_distance_m" in indicator_grid.columns
    assert "neighbor_distance_valid_count" in indicator_grid.columns
    assert indicator_grid["neighbor_distance_valid_count"].sum() == 2

    west_segment = gpd.read_file(
        tmp_path
        / "segmented_run"
        / "segments"
        / "segment_001"
        / "processed"
        / "buildings_segment_metric.gpkg",
        layer="buildings_segment_metric",
    )
    west = west_segment.set_index("building_id").loc["west"]

    assert west["neighbor_building_id"] == "east"
    assert west["neighbor_distance_m"] > 0


def test_segmented_neighbor_distance_context_buffer_diagnostic_for_too_small_buffer(
    tmp_path,
):
    config = segmented_config()
    config["indicators"]["neighbor_distance"] = True
    config["crs_strategy"] = {"context_buffer_m": 0}

    result = process_segmented_core_indicators(
        buildings=make_boundary_neighbor_buildings(include_crossing=False),
        aoi=make_cross_zone_aoi(),
        config=config,
        output_dir=tmp_path / "segmented_run",
        save_outputs=True,
    )

    summary = result["summary"]["neighbor_distance_summary"]

    assert summary["context_buffer_m"] == 0.0
    assert summary["valid_neighbor_distance_count"] == 0
    assert summary["valid_neighbor_distance_share"] == 0.0
    assert summary["target_buildings_without_neighbor_count"] == 2


def test_segmented_neighbor_distance_excludes_self_for_crossing_building(tmp_path):
    config = segmented_config()
    config["indicators"]["neighbor_distance"] = True
    config["crs_strategy"] = {"context_buffer_m": 150}

    result = process_segmented_core_indicators(
        buildings=make_boundary_neighbor_buildings(include_crossing=True),
        aoi=make_cross_zone_aoi(),
        config=config,
        output_dir=tmp_path / "segmented_run",
        save_outputs=True,
    )

    segment_pieces = []
    for segment_id in ("segment_001", "segment_002"):
        segment_buildings = gpd.read_file(
            tmp_path
            / "segmented_run"
            / "segments"
            / segment_id
            / "processed"
            / "buildings_segment_metric.gpkg",
            layer="buildings_segment_metric",
        )
        segment_pieces.append(
            segment_buildings[segment_buildings["building_id"] == "crossing"]
        )

    crossing_pieces = pd.concat(
        [piece.drop(columns="geometry") for piece in segment_pieces],
        ignore_index=True,
    )

    assert len(crossing_pieces) == 2
    assert crossing_pieces["neighbor_building_id"].notna().all()
    assert set(crossing_pieces["neighbor_building_id"]) != {"crossing"}
    assert (crossing_pieces["neighbor_building_id"] != "crossing").all()
    assert (crossing_pieces["neighbor_distance_m"] > 0).all()
    assert result["indicator_grid"]["neighbor_distance_valid_count"].sum() >= 3


def test_segmented_street_context_uses_context_profiles_without_aggregating_context_only_buildings(
    monkeypatch,
    tmp_path,
):
    profile_context_building_counts = []
    fetch_inputs = []
    profile_street_crs = []

    def fake_fetch_streets(aoi, network_type="drive", target_crs=None):
        fetch_inputs.append(
            {
                "crs": aoi.crs.to_string(),
                "target_crs": target_crs.to_string(),
                "bounds": tuple(float(value) for value in aoi.total_bounds),
            }
        )
        minx, miny, maxx, maxy = aoi.total_bounds
        x = maxx - 5.0
        return gpd.GeoDataFrame(
            {
                "street_id": ["context_street"],
                "osmid": ["mock"],
                "highway": ["residential"],
            },
            geometry=[LineString([(x, miny), (x, maxy)])],
            crs=aoi.crs,
        )

    def fake_calculate_profiles(streets, buildings, **kwargs):
        profile_context_building_counts.append(int(len(buildings)))
        profile_street_crs.append(streets.crs.to_string())
        assert streets.crs.is_projected
        out = streets.copy()
        out["street_profile_width_m"] = 20.0
        out["street_profile_openness"] = 0.5
        out["street_profile_width_deviation_m"] = 0.0
        out["street_profile_momepy_height_m"] = 15.0
        out["street_profile_height_deviation_m"] = 0.0
        out["street_profile_hw_ratio_momepy"] = 0.75
        out["street_profile_width_is_capped"] = False
        out["has_opposite_profile_evidence"] = True
        return out

    monkeypatch.setattr(sw, "fetch_streets_from_osmnx", fake_fetch_streets)
    monkeypatch.setattr(
        sw,
        "calculate_street_profile_segments",
        fake_calculate_profiles,
    )

    config = segmented_config()
    config["street_context"] = {
        "enabled": True,
        "source": "osmnx",
        "network_type": "drive",
        "distance_m": 10,
        "tick_length_m": 60,
    }
    config["crs_strategy"] = {"context_buffer_m": 150}

    result = process_segmented_core_indicators(
        buildings=make_boundary_street_context_buildings(),
        aoi=make_cross_zone_aoi(),
        config=config,
        output_dir=tmp_path / "segmented_run",
        save_outputs=True,
    )

    summary = result["summary"]["street_profile_summary"]
    grid = result["indicator_grid"]

    assert result["summary"]["segmented_street_context_enabled"] is True
    assert result["summary"]["segmented_street_context_buffer_m"] == 150.0
    assert fetch_inputs
    assert {item["crs"] for item in fetch_inputs} == {"EPSG:4326"}
    assert {item["target_crs"] for item in fetch_inputs} == {
        "EPSG:32632",
        "EPSG:32633",
    }
    assert all(-180 <= item["bounds"][0] <= 180 for item in fetch_inputs)
    assert all(-90 <= item["bounds"][1] <= 90 for item in fetch_inputs)
    assert set(profile_street_crs) == {"EPSG:32632", "EPSG:32633"}
    assert max(profile_context_building_counts) > 1
    assert "avg_street_profile_height_to_width_ratio_strict" in grid.columns
    assert "street_profile_ratio_strict_valid_count" in grid.columns
    assert grid["street_profile_building_count"].sum() == 2
    assert grid["street_profile_ratio_strict_valid_count"].sum() == 2

    assert summary["valid_ratio_strict_count"] == 2
    assert summary["valid_ratio_strict_share"] == 1.0
    assert summary["grid_cells_with_strict_ratio_count"] > 0
    assert summary["grid_cells_with_strict_ratio_share"] is not None

    segment_buildings = []
    for segment_id in ("segment_001", "segment_002"):
        segment_buildings.append(
            gpd.read_file(
                tmp_path
                / "segmented_run"
                / "segments"
                / segment_id
                / "processed"
                / "buildings_segment_metric.gpkg",
                layer="buildings_segment_metric",
            ).drop(columns="geometry")
        )

    segment_buildings_all = pd.concat(segment_buildings, ignore_index=True)

    assert "context_only" not in set(segment_buildings_all["building_id"])
    assert segment_buildings_all["street_id"].notna().all()
    assert segment_buildings_all["has_valid_street_profile_ratio_strict"].all()


def test_segmented_street_context_no_graph_segment_records_diagnostic(
    monkeypatch,
    tmp_path,
):
    fetch_calls = {"count": 0}

    def fake_fetch_streets(aoi, network_type="drive", target_crs=None):
        fetch_calls["count"] += 1

        if fetch_calls["count"] == 1:
            raise ValueError("Found no graph nodes within the requested polygon")

        minx, miny, maxx, maxy = aoi.total_bounds
        x = maxx - 5.0
        return gpd.GeoDataFrame(
            {
                "street_id": ["context_street"],
            },
            geometry=[LineString([(x, miny), (x, maxy)])],
            crs=aoi.crs,
        )

    def fake_calculate_profiles(streets, buildings, **kwargs):
        out = streets.copy()
        out["street_profile_width_m"] = 20.0
        out["street_profile_openness"] = 0.5
        out["street_profile_width_deviation_m"] = 0.0
        out["street_profile_momepy_height_m"] = 15.0
        out["street_profile_height_deviation_m"] = 0.0
        out["street_profile_hw_ratio_momepy"] = 0.75
        out["street_profile_width_is_capped"] = False
        out["has_opposite_profile_evidence"] = True
        return out

    monkeypatch.setattr(sw, "fetch_streets_from_osmnx", fake_fetch_streets)
    monkeypatch.setattr(
        sw,
        "calculate_street_profile_segments",
        fake_calculate_profiles,
    )

    config = segmented_config()
    config["street_context"] = {
        "enabled": True,
        "source": "osmnx",
        "network_type": "drive",
        "distance_m": 10,
        "tick_length_m": 60,
    }
    config["crs_strategy"] = {"context_buffer_m": 150}

    result = process_segmented_core_indicators(
        buildings=make_boundary_street_context_buildings(),
        aoi=make_cross_zone_aoi(),
        config=config,
        output_dir=tmp_path / "segmented_run",
        save_outputs=True,
    )

    street_summary = result["summary"]["street_profile_summary"]
    segment_statuses = [
        segment["street_context"]["street_context_status"]
        for segment in result["summary"]["segments"]
    ]

    assert "no_osm_graph" in segment_statuses
    assert street_summary["street_context_status_counts"]["no_osm_graph"] == 1
    assert street_summary["no_graph_segment_count"] == 1
    assert street_summary["grid_cells_with_strict_ratio_count"] >= 0


def test_segmented_street_context_all_no_graph_finishes_with_zero_coverage(
    monkeypatch,
    tmp_path,
):
    def fake_fetch_streets(aoi, network_type="drive", target_crs=None):
        raise ValueError("Found no graph nodes within the requested polygon")

    monkeypatch.setattr(sw, "fetch_streets_from_osmnx", fake_fetch_streets)

    config = segmented_config()
    config["street_context"] = {
        "enabled": True,
        "source": "osmnx",
        "network_type": "drive",
        "distance_m": 10,
        "tick_length_m": 60,
    }
    config["crs_strategy"] = {"context_buffer_m": 150}

    result = process_segmented_core_indicators(
        buildings=make_boundary_street_context_buildings(),
        aoi=make_cross_zone_aoi(),
        config=config,
        output_dir=tmp_path / "segmented_run",
        save_outputs=True,
    )

    street_summary = result["summary"]["street_profile_summary"]
    grid = result["indicator_grid"]

    assert street_summary["street_context_status_counts"]["no_osm_graph"] == 2
    assert street_summary["grid_cells_with_strict_ratio_count"] == 0
    assert street_summary["grid_cells_with_strict_ratio_share"] == 0.0
    assert grid["street_profile_ratio_strict_valid_count"].sum() == 0


def test_segmented_height_enrichment_runs_before_splitting_and_feeds_bvd(
    monkeypatch,
    tmp_path,
):
    calls = []

    def fake_gba_enrichment(buildings, **kwargs):
        calls.append(
            {
                "epsg": buildings.crs.to_epsg(),
                "building_ids": list(buildings["building_id"]),
            }
        )

        out = buildings.copy()
        original = pd.to_numeric(out["height_m"], errors="coerce")
        original_valid = original.notna() & (original > 0)
        fill_values = {
            "east": 30.0,
            "crossing": 21.0,
        }
        fill_mask = (~original_valid) & out["building_id"].isin(fill_values)

        out["height_m_original"] = original
        out["height_gba_m"] = float("nan")
        out.loc[fill_mask, "height_gba_m"] = out.loc[
            fill_mask,
            "building_id",
        ].map(fill_values)
        out["height_m_enriched"] = original
        out.loc[fill_mask, "height_m_enriched"] = out.loc[
            fill_mask,
            "height_gba_m",
        ]
        out["height_m"] = out["height_m_enriched"]
        out["height_source"] = "missing"
        out.loc[original_valid, "height_source"] = "overture"
        out.loc[fill_mask, "height_source"] = "gba_lod1"
        out["height_was_enriched"] = fill_mask

        final_valid = pd.to_numeric(out["height_m"], errors="coerce").notna()
        summary = {
            "valid_height_count_before": int(original_valid.sum()),
            "missing_height_count_before": int((~original_valid).sum()),
            "valid_height_share_before": float(original_valid.mean()),
            "missing_height_share_before": float((~original_valid).mean()),
            "height_enriched_count": int(fill_mask.sum()),
            "valid_height_count_after": int(final_valid.sum()),
            "missing_height_count_after": int((~final_valid).sum()),
            "valid_height_share_after": float(final_valid.mean()),
            "missing_height_share_after": float((~final_valid).mean()),
            "changed_existing_overture_height_count": 0,
        }
        empty = gpd.GeoDataFrame(geometry=[], crs=buildings.crs)
        return out, summary, empty, empty

    monkeypatch.setattr(sw, "enrich_missing_heights_with_gba_lod1", fake_gba_enrichment)

    config = segmented_config()
    config["height_enrichment"] = {
        "enabled": True,
        "save_gba_subset": False,
        "save_matches": False,
    }
    buildings = make_cross_zone_buildings()
    buildings.loc[buildings["building_id"].isin(["east", "crossing"]), "height_m"] = None
    output_dir = tmp_path / "segmented_run"

    result = process_segmented_core_indicators(
        buildings=buildings,
        aoi=make_cross_zone_aoi(),
        config=config,
        output_dir=output_dir,
        save_outputs=True,
        project_root=tmp_path,
    )

    summary = result["summary"]["height_enrichment_summary"]

    assert result["summary"]["height_enrichment_enabled"] is True
    assert [call["epsg"] for call in calls] == [32632, 32633]
    assert calls[1]["building_ids"].count("crossing") == 1
    assert summary["valid_height_count_before"] == 1
    assert summary["valid_height_count_after"] == 3
    assert summary["missing_height_share_before"] == pytest.approx(2 / 3)
    assert summary["missing_height_share_after"] == 0.0
    assert summary["height_enriched_count"] == 2
    assert summary["height_source_overture_count"] == 1
    assert summary["height_source_gba_lod1_count"] == 2
    assert summary["changed_existing_overture_height_count"] == 0

    enriched = gpd.read_file(
        output_dir
        / "processed"
        / "buildings_height_enriched_segmented_wgs84.gpkg",
        layer="buildings_height_enriched_segmented_wgs84",
    )
    enriched_by_id = enriched.set_index("building_id")

    assert enriched_by_id.loc["west", "height_m"] == 12.0
    assert enriched_by_id.loc["west", "height_source"] == "overture"
    assert enriched_by_id.loc["east", "height_m"] == 30.0
    assert enriched_by_id.loc["crossing", "height_m"] == 21.0
    assert len(enriched_by_id.loc[["crossing"]]) == 1

    segment_buildings = []
    for segment_id in ("segment_001", "segment_002"):
        segment_buildings.append(
            gpd.read_file(
                output_dir
                / "segments"
                / segment_id
                / "processed"
                / "buildings_segment_metric.gpkg",
                layer="buildings_segment_metric",
            )
        )

    segment_buildings_all = pd.concat(
        [
            segment_building.drop(columns="geometry")
            for segment_building in segment_buildings
        ],
        ignore_index=True,
    )
    crossing_pieces = segment_buildings_all[
        segment_buildings_all["building_id"] == "crossing"
    ]

    assert len(crossing_pieces) == 2
    assert set(crossing_pieces["height_m"]) == {21.0}

    expected_volume = (
        segment_buildings_all["footprint_area_m2"]
        * pd.to_numeric(segment_buildings_all["height_m"], errors="coerce")
    ).sum()
    actual_volume = result["indicator_grid"]["built_volume_m3"].sum()

    assert actual_volume == pytest.approx(expected_volume, rel=1e-6)
    assert (
        output_dir / "reports" / "height_enrichment_quality_segmented.json"
    ).exists()
