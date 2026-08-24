from __future__ import annotations

from pathlib import Path
import sys

import geopandas as gpd
import pytest
from shapely.geometry import box


TEST_DIR = Path(__file__).resolve().parent
CODE_DIR = TEST_DIR.parent
SRC_DIR = CODE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from visualization import calculate_shared_value_range, plot_indicator_map  # noqa: E402


def _grid(values: list[float | None]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"gsi": values},
        geometry=[
            box(0, 0, 1, 1),
            box(1, 0, 2, 1),
            box(0, 1, 1, 2),
        ][: len(values)],
        crs="EPSG:3857",
    )


def test_calculate_shared_value_range_uses_all_valid_series_values():
    first = _grid([0.2, 0.4, None])
    second = _grid([0.1, 0.8, None])

    assert calculate_shared_value_range([first, second], "gsi") == (0.1, 0.8)


def test_calculate_shared_value_range_requires_valid_values():
    with pytest.raises(ValueError, match="No valid values"):
        calculate_shared_value_range([_grid([None, None])], "gsi")


def test_plot_indicator_map_accepts_shared_range_and_map_note(tmp_path):
    output_path = tmp_path / "gsi_map.png"

    result = plot_indicator_map(
        indicator_grid=_grid([0.2, None, 0.8]),
        column="gsi",
        output_path=output_path,
        title="GSI / Building Coverage Ratio",
        legend_label="GSI",
        vmin=0.0,
        vmax=1.0,
        map_note="Legend range is matched for comparison.",
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
