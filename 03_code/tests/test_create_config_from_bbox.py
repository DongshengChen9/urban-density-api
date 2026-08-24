from __future__ import annotations

from pathlib import Path
import sys

import pytest
import yaml


TEST_DIR = Path(__file__).resolve().parent
CODE_DIR = TEST_DIR.parent
SCRIPTS_DIR = CODE_DIR / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


from create_config_from_bbox import (  # noqa: E402
    build_config,
    create_config_from_bbox,
    validate_bbox,
    validate_grid_size,
    validate_run_name,
)


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_valid_bbox_writes_yaml_file(tmp_path):
    output = tmp_path / "configs" / "generated" / "example_area.yaml"

    create_config_from_bbox(
        run_name="example_area_100m",
        min_lon=16.36,
        min_lat=48.20,
        max_lon=16.38,
        max_lat=48.22,
        grid_size=100,
        mode="standard",
        output=output,
    )

    config = load_yaml(output)

    assert output.exists()
    assert config["project"]["run_name"] == "example_area_100m"
    assert config["project"]["output_dir"] == "04_outputs/example_area_100m"
    assert config["project"]["overwrite_existing_run"] is True
    assert config["aoi"]["crs"] == "EPSG:4326"
    assert config["aoi"]["bounds"] == {
        "minx": 16.36,
        "miny": 48.20,
        "maxx": 16.38,
        "maxy": 48.22,
    }
    assert config["aggregation"]["cell_size_m"] == 100
    assert config["preprocessing"]["target_crs"] == "auto_utm"
    assert config["crs_strategy"]["processing_mode"] == "single_crs"
    assert config["data_source"]["overture_release"] == "auto"
    assert config["data_source"]["release"] == "auto"
    assert config["cache"]["require_compatible_manifest"] is True
    assert config["cache"]["use_existing_raw_buildings"] is True


def test_invalid_bbox_raises_clear_error():
    with pytest.raises(ValueError, match="Minimum longitude"):
        validate_bbox(
            min_lon=10,
            min_lat=48,
            max_lon=9,
            max_lat=49,
        )

    with pytest.raises(ValueError, match="Maximum latitude"):
        validate_bbox(
            min_lon=10,
            min_lat=48,
            max_lon=11,
            max_lat=91,
        )


def test_invalid_grid_size_raises_clear_error():
    with pytest.raises(ValueError, match="Grid size must be positive"):
        validate_grid_size(0)

    with pytest.raises(ValueError, match="Grid size must be positive"):
        validate_grid_size(-50)


def test_unsafe_run_name_raises_clear_error():
    with pytest.raises(ValueError, match="Run name must be non-empty"):
        validate_run_name("")

    with pytest.raises(ValueError, match="letters, numbers, underscores, or hyphens"):
        validate_run_name("../bad")

    with pytest.raises(ValueError, match="letters, numbers, underscores, or hyphens"):
        validate_run_name("bad name")


def test_quick_2d_mode_disables_heavy_optional_branches():
    config = build_config(
        run_name="quick_run",
        min_lon=5.0,
        min_lat=50.0,
        max_lon=5.1,
        max_lat=50.1,
        grid_size=100,
        mode="quick_2d",
    )

    assert config["indicators"]["gsi"] is True
    assert config["indicators"]["far_fsi"] is False
    assert config["indicators"]["built_volume_density"] is False
    assert config["indicators"]["neighbor_distance"] is False
    assert config["height_enrichment"]["enabled"] is False
    assert config["street_context"]["enabled"] is False
    assert config["outputs"]["save_neighbor_diagnostics"] is False


def test_standard_mode_enables_core_density_without_context_branches():
    config = build_config(
        run_name="standard_run",
        min_lon=5.0,
        min_lat=50.0,
        max_lon=5.1,
        max_lat=50.1,
        grid_size=100,
        mode="standard",
    )

    assert config["indicators"]["gsi"] is True
    assert config["indicators"]["far_fsi"] is True
    assert config["indicators"]["built_volume_density"] is True
    assert config["indicators"]["neighbor_distance"] is False
    assert config["height_enrichment"]["enabled"] is True
    assert config["height_enrichment"]["mode"] == "fill_missing_only"
    assert config["height_enrichment"]["replace_existing_height"] is False
    assert config["street_context"]["enabled"] is False


def test_full_context_mode_enables_contextual_branches():
    config = build_config(
        run_name="full_context_run",
        min_lon=5.0,
        min_lat=50.0,
        max_lon=5.1,
        max_lat=50.1,
        grid_size=100,
        mode="full_context",
    )

    assert config["indicators"]["gsi"] is True
    assert config["indicators"]["far_fsi"] is True
    assert config["indicators"]["built_volume_density"] is True
    assert config["indicators"]["neighbor_distance"] is True
    assert config["indicators"]["height_to_distance_ratio"] is False
    assert config["height_enrichment"]["enabled"] is True
    assert config["street_context"]["enabled"] is True
    assert config["outputs"]["save_neighbor_diagnostics"] is True


def test_output_parent_folder_is_created_if_missing(tmp_path):
    output = tmp_path / "missing" / "parents" / "generated.yaml"

    create_config_from_bbox(
        run_name="created_parent_run",
        min_lon=16.36,
        min_lat=48.20,
        max_lon=16.38,
        max_lat=48.22,
        grid_size=50,
        mode="quick_2d",
        output=output,
    )

    assert output.exists()
    assert output.parent.exists()
