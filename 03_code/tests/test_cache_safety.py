from __future__ import annotations

from pathlib import Path
import sys

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box


TEST_DIR = Path(__file__).resolve().parent
CODE_DIR = TEST_DIR.parent
SRC_DIR = CODE_DIR / "src"
for path in (CODE_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_workflow as workflow_module  # noqa: E402
from run_workflow import (  # noqa: E402
    build_cache_manifest,
    compare_cache_manifests,
    evaluate_raw_building_cache,
    prepare_output_directory,
    validate_raw_buildings_match_aoi,
    verify_cached_building_artifact_hash,
    write_cache_manifest,
)
from cache_contracts import compare_artifact_contracts, contract_is_compatible  # noqa: E402
from performance import legacy_stable_geodataframe_hash, stable_geodataframe_hash  # noqa: E402


def _inputs():
    aoi = gpd.GeoDataFrame(
        {"aoi_name": ["example_area"]},
        geometry=[box(4.47, 51.91, 4.49, 51.93)],
        crs="EPSG:4326",
    )
    buildings = gpd.GeoDataFrame(
        {"building_id": ["b1"], "height_m": [12.0]},
        geometry=[box(4.475, 51.915, 4.476, 51.916)],
        crs="EPSG:4326",
    )
    config = {
        "project": {"run_name": "cache_test", "output_dir": "04_outputs/cache_test"},
        "aoi": {"name": "example_area"},
        "data_source": {
            "type": "overture",
            "release": "2026-05-20.0",
            "provider": "aws",
            "exclude_underground": True,
        },
        "preprocessing": {"target_crs": "auto_utm", "clip_to_aoi": True},
        "height_enrichment": {
            "enabled": True,
            "min_overlap_share": 0.2,
            "min_valid_height_m": 2.0,
            "replace_existing_height": False,
        },
        "street_context": {
            "enabled": True,
            "source": "osmnx",
            "network_type": "drive",
            "distance_m": 10,
            "tick_length_m": 60,
        },
    }
    return config, aoi, buildings


def _manifest(config, aoi, buildings):
    return build_cache_manifest(
        config=config,
        aoi=aoi,
        target_crs="EPSG:32631",
        buildings_clean=buildings,
        buildings_raw=buildings,
        created_at="2026-07-20T00:00:00",
    )


def test_matching_manifest_is_compatible():
    config, aoi, buildings = _inputs()
    manifest = _manifest(config, aoi, buildings)
    result = compare_cache_manifests(manifest, dict(manifest))
    assert result["cache_source_compatibility_status"] == "compatible"
    assert result["cache_source_compatibility_warnings"] == []


def test_changed_scientific_setting_rejects_manifest():
    config, aoi, buildings = _inputs()
    current = _manifest(config, aoi, buildings)
    source = dict(current)
    source["height_enrichment"] = {
        **source["height_enrichment"],
        "min_overlap_share": 0.5,
    }
    result = compare_cache_manifests(current, source)
    assert result["cache_source_compatibility_status"] == "mismatch_detected"


def test_incompatible_area_rejects_raw_cache(tmp_path):
    config, source_aoi, buildings = _inputs()
    requested_aoi = gpd.GeoDataFrame(
        {"aoi_name": ["different_area"]},
        geometry=[box(5.1, 52.0, 5.2, 52.1)],
        crs="EPSG:4326",
    )
    raw_path = tmp_path / "raw" / "buildings_raw_overture.gpkg"
    raw_path.parent.mkdir(parents=True)
    buildings.to_file(raw_path, layer="buildings_raw", driver="GPKG")
    manifest_path = tmp_path / "reports" / "cache_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    write_cache_manifest(manifest_path, _manifest(config, source_aoi, buildings))

    decision = evaluate_raw_building_cache(
        config,
        requested_aoi,
        raw_path,
        manifest_path,
        require_compatible_manifest=True,
    )
    assert decision["use_cache"] is False
    assert decision["cache_compatibility_status"] == "rejected"


def test_non_overlapping_raw_bounds_stop_processing():
    _config, aoi, _buildings = _inputs()
    other = gpd.GeoDataFrame(
        {"building_id": ["other"]},
        geometry=[box(5.1, 52.0, 5.11, 52.01)],
        crs="EPSG:4326",
    )
    ok, message = validate_raw_buildings_match_aoi(other, aoi)
    assert ok is False
    assert "do not spatially match" in message


def test_overwrite_removes_run_but_protects_global_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow_module, "PROJECT_ROOT", tmp_path)
    run_dir = tmp_path / "04_outputs" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "stale.txt").write_text("stale", encoding="utf-8")
    prepare_output_directory(run_dir, overwrite_existing_run=True)
    assert not run_dir.exists()

    global_cache = tmp_path / "04_outputs" / "_cache"
    global_cache.mkdir(parents=True)
    try:
        prepare_output_directory(global_cache, overwrite_existing_run=True)
    except ValueError as exc:
        assert "Refusing to overwrite" in str(exc)
    else:
        raise AssertionError("Global cache deletion should be refused")
    assert global_cache.exists()


def test_artifact_contracts_keep_clean_buildings_reusable_after_height_change():
    config, aoi, buildings = _inputs()
    source = build_cache_manifest(config, aoi, "EPSG:32631", buildings, buildings, buildings.copy())
    changed = dict(config)
    changed["height_enrichment"] = {**config["height_enrichment"], "min_overlap_share": 0.5}
    current = build_cache_manifest(changed, aoi, "EPSG:32631", buildings, buildings, buildings.copy())
    plan = compare_artifact_contracts(current, source)
    assert contract_is_compatible(plan, "building_core")
    assert not contract_is_compatible(plan, "enriched_buildings")


def test_manifest_keeps_distinct_clean_and_enriched_physical_hashes():
    config, aoi, clean = _inputs()
    enriched = clean.copy()
    enriched["height_source"] = ["gba"]
    enriched["height_m"] = [18.0]
    manifest = build_cache_manifest(config, aoi, "EPSG:32631", clean, clean, enriched)
    hashes = manifest["artifact_layer_hashes"]
    assert hashes["buildings_clean"] != hashes["buildings_height_enriched"]


def _clean_buildings_with_nullable_floors():
    return gpd.GeoDataFrame(
        {
            "building_id": ["b1", "b2"],
            "height_m": [12.0, 18.5],
            "num_floors": pd.Series([3, pd.NA], dtype="Int64"),
        },
        geometry=[box(500000, 5300000, 500010, 5300010), box(500020, 5300000, 500030, 5300010)],
        crs="EPSG:32631",
    )


def _building_hash_columns(buildings):
    return [column for column in ["height_m", "num_floors", "height_source"] if column in buildings.columns]


def test_semantic_building_hash_survives_geopackage_nullable_integer_round_trip(tmp_path):
    buildings = _clean_buildings_with_nullable_floors()
    source_hash = stable_geodataframe_hash(buildings, "building_id", _building_hash_columns(buildings))
    path = tmp_path / "buildings_clean.gpkg"
    buildings.to_file(path, layer="buildings_clean", driver="GPKG")

    loaded = gpd.read_file(path, layer="buildings_clean")
    assert str(loaded["num_floors"].dtype) == "float64"
    assert stable_geodataframe_hash(loaded, "building_id", _building_hash_columns(loaded)) == source_hash
    verify_cached_building_artifact_hash(
        loaded,
        {
            "artifact_hash_algorithm": "semantic_geodataframe_v2",
            "artifact_layer_hashes": {"buildings_clean": source_hash},
        },
        "buildings_clean",
    )


def test_semantic_building_hash_rejects_modified_values_or_geometry(tmp_path):
    buildings = _clean_buildings_with_nullable_floors()
    source_hash = stable_geodataframe_hash(buildings, "building_id", _building_hash_columns(buildings))
    path = tmp_path / "buildings_clean.gpkg"
    buildings.to_file(path, layer="buildings_clean", driver="GPKG")
    loaded = gpd.read_file(path, layer="buildings_clean")
    manifest = {
        "artifact_hash_algorithm": "semantic_geodataframe_v2",
        "artifact_layer_hashes": {"buildings_clean": source_hash},
    }

    changed_value = loaded.copy()
    changed_value.loc[0, "height_m"] = 99.0
    with pytest.raises(ValueError, match="hash differs"):
        verify_cached_building_artifact_hash(changed_value, manifest, "buildings_clean")

    changed_geometry = loaded.copy()
    changed_geometry.loc[0, "geometry"] = box(500001, 5300000, 500011, 5300010)
    with pytest.raises(ValueError, match="hash differs"):
        verify_cached_building_artifact_hash(changed_geometry, manifest, "buildings_clean")


def test_legacy_manifest_accepts_only_known_nullable_integer_storage_representation(tmp_path):
    buildings = _clean_buildings_with_nullable_floors()
    legacy_hash = legacy_stable_geodataframe_hash(
        buildings, "building_id", _building_hash_columns(buildings)
    )
    path = tmp_path / "buildings_clean.gpkg"
    buildings.to_file(path, layer="buildings_clean", driver="GPKG")
    loaded = gpd.read_file(path, layer="buildings_clean")
    legacy_manifest = {"artifact_layer_hashes": {"buildings_clean": legacy_hash}}

    verify_cached_building_artifact_hash(loaded, legacy_manifest, "buildings_clean")

    altered = loaded.copy()
    altered.loc[0, "num_floors"] = 4.0
    with pytest.raises(ValueError, match="hash differs"):
        verify_cached_building_artifact_hash(altered, legacy_manifest, "buildings_clean")


def test_semantic_building_hash_is_independent_of_row_order():
    buildings = _clean_buildings_with_nullable_floors()
    columns = _building_hash_columns(buildings)
    assert stable_geodataframe_hash(buildings, "building_id", columns) == stable_geodataframe_hash(
        buildings.iloc[::-1].copy(), "building_id", columns
    )

