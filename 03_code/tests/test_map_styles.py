"""Shared map-style registry tests."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from map_styles import MISSING_COLOR, ZERO_COLOR, color_for_value, legend_entries, resolved_style, style_for_key  # noqa: E402


def test_registry_covers_all_mapped_indicators_with_distinct_missing_and_zero_colors():
    for key in ("gsi", "far", "built_volume_density", "neighbour_distance", "street_profile_ratio"):
        style = resolved_style(style_for_key(key), [0, 1, 2, 3, 4, 5])
        assert color_for_value(style, None) == MISSING_COLOR
        assert color_for_value(style, 0) == ZERO_COLOR
        assert color_for_value(style, 5) != MISSING_COLOR


def test_shared_legend_is_stable_and_human_readable():
    labels = [label for _key, label, _color in legend_entries(resolved_style("gsi", [0, 1]))]
    assert labels[0] == "Missing / insufficient input"
    assert labels[1] == "Zero"
    assert any("to" in label for label in labels[2:])


def test_dashboard_static_and_web_exports_reference_the_registry():
    code_dir = Path(__file__).resolve().parents[1]
    for path in (code_dir / "app.py", code_dir / "src" / "visualization.py", code_dir / "scripts" / "export_web_map.py"):
        text = path.read_text(encoding="utf-8")
        assert "map_styles" in text
