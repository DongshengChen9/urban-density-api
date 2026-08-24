from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "03_code" / "config"


def test_all_public_yaml_files_parse():
    yaml_files = sorted([*ROOT.glob("*.yaml"), *ROOT.glob("*.yml"), *CONFIG_DIR.glob("*.yaml"), *CONFIG_DIR.glob("*.yml")])
    assert yaml_files
    for path in yaml_files:
        assert yaml.safe_load(path.read_text(encoding="utf-8")) is not None


def test_example_configuration_enables_five_supported_indicators():
    config = yaml.safe_load(
        (CONFIG_DIR / "example_urban_area_100m.yaml").read_text(encoding="utf-8")
    )
    assert config["data_source"]["type"] == "overture"
    assert config["street_context"]["source"] == "osmnx"
    assert config["indicators"] == {
        "gsi": True,
        "far_fsi": True,
        "built_volume_density": True,
        "neighbor_distance": True,
        "height_to_distance_ratio": False,
    }
    assert config["street_context"]["enabled"] is True
    assert config["cache"]["require_compatible_manifest"] is True
    assert config["project"]["output_dir"].startswith("04_outputs/")

