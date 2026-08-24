"""Regression tests for deferred dashboard navigation."""

from pathlib import Path
import json
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import (  # noqa: E402
    apply_pending_navigation_before_widgets,
    initialize_dashboard_state,
    schedule_navigation,
    select_completed_run,
    build_grid_size_rerun_config,
    grid_rerun_cache_compatibility,
    grid_rerun_stage_plan,
)


def test_completed_run_selection_schedules_without_late_indicator_mutation():
    state = {}
    initialize_dashboard_state(state)
    state["_active_completed_run"] = "old"
    state["selected_indicator"] = "far"
    assert select_completed_run(state, "new")
    assert state["selected_indicator"] == "far"
    assert state["_pending_selected_run"] == "new"


def test_pending_navigation_is_one_shot_and_resets_selected_cell():
    state = {}
    initialize_dashboard_state(state)
    state["selected_cell_id"] = "old_cell"
    schedule_navigation(state, "new", "gsi")
    assert apply_pending_navigation_before_widgets(state)
    assert state["selected_completed_run"] == "new"
    assert state["selected_indicator"] == "gsi"
    assert state["selected_cell_id"] is None
    assert not apply_pending_navigation_before_widgets(state)


def _source_config():
    return {
        "project": {"run_name": "source", "output_dir": "04_outputs/source"},
        "aoi": {"bounds": {"minx": 16.0, "miny": 48.0, "maxx": 16.1, "maxy": 48.1}},
        "data_source": {"type": "overture", "provider": "aws", "release": "2026-06-17.0", "exclude_underground": True},
        "preprocessing": {"target_crs": "auto_utm", "clip_to_aoi": True},
        "height_enrichment": {"enabled": True, "min_overlap_share": 0.2, "min_valid_height_m": 2.0, "replace_existing_height": False},
        "street_context": {"enabled": True, "source": "osmnx", "network_type": "drive", "distance_m": 10, "tick_length_m": 60},
        "aggregation": {"method": "regular_grid", "cell_size_m": 100, "clip_to_aoi": True},
        "crs_strategy": {"processing_mode": "single_crs"},
    }


def _manifest(config):
    return {
        "aoi_bounds_wgs84": {"min_lon": 16.0, "min_lat": 48.0, "max_lon": 16.1, "max_lat": 48.1},
        "data_source": config["data_source"],
        "preprocessing": config["preprocessing"],
        "height_enrichment": config["height_enrichment"],
        "street_context": {**config["street_context"], "topology_rule_version": 1},
        "processing_mode": "single_crs",
    }


def test_grid_rerun_keeps_upstream_street_context_compatible(tmp_path):
    source = _source_config()
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "cache_manifest.json").write_text(json.dumps(_manifest(source)), encoding="utf-8")
    rerun = build_grid_size_rerun_config(source, "source", "source_50m", 50)
    compatibility = grid_rerun_cache_compatibility(tmp_path, rerun)
    assert compatibility["status"] == "compatible"
    assert rerun["aggregation"]["cell_size_m"] == 50
    assert rerun["project"]["run_name"] == "source_50m"
    assert "grid creation" in grid_rerun_stage_plan(tmp_path)["recalculated"]


def test_grid_rerun_rejects_actual_street_context_change(tmp_path):
    source = _source_config()
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "cache_manifest.json").write_text(json.dumps(_manifest(source)), encoding="utf-8")
    rerun = build_grid_size_rerun_config(source, "source", "renamed", 50)
    rerun["street_context"]["distance_m"] = 15
    compatibility = grid_rerun_cache_compatibility(tmp_path, rerun)
    assert compatibility["status"] == "mismatch_detected"
    assert "Street-context settings differ." in compatibility["reasons"]
