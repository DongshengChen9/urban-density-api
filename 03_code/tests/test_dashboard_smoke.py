from __future__ import annotations

from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, box, shape
from streamlit.testing.v1 import AppTest


TEST_DIR = Path(__file__).resolve().parent
CODE_DIR = TEST_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from app import (  # noqa: E402
    create_indicator_overview,
    has_selected_bbox,
    initialize_dashboard_state,
    main,
    query_grid_cell_feature_by_id,
    safe_output_folder_for_run,
)


def test_dashboard_module_imports_and_exposes_main():
    assert callable(main)


def test_fresh_dashboard_state_has_no_selected_area_or_run():
    state = {}
    initialize_dashboard_state(state)
    assert state["setup_run_name"] == ""
    assert state["selected_completed_run"] is None
    assert state["selected_aoi_geometry"] is None
    assert has_selected_bbox(state) is False
    assert safe_output_folder_for_run(state["setup_run_name"]) is None


def test_selected_cell_uses_exact_saved_geometry_and_crs(tmp_path):
    output_folder = tmp_path / "run"
    path = output_folder / "indicators" / "grid_indicators.gpkg"
    path.parent.mkdir(parents=True)
    geometries = [
        box(0, 0, 100, 100),
        Polygon([(100, 0), (200, 0), (200, 100), (150, 70), (100, 100)]),
    ]
    grid = gpd.GeoDataFrame(
        {
            "unit_id": ["full", "partial"],
            "gsi": [0.2, 0.4],
            "is_partial_cell": [False, True],
        },
        geometry=geometries,
        crs="EPSG:3857",
    )
    grid.to_file(path, layer="grid_indicators")
    expected = grid.to_crs("EPSG:4326").set_index("unit_id")
    overview = create_indicator_overview(output_folder, "gsi", max_pixels=200)

    assert overview["raster_crs"] == "EPSG:4326"
    assert np.allclose(overview["bounds_wgs84"], expected.total_bounds, atol=1e-12)
    for unit_id in expected.index:
        feature = query_grid_cell_feature_by_id(output_folder, unit_id)
        assert feature is not None
        assert shape(feature["geometry"]).equals_exact(
            expected.loc[unit_id].geometry, 1e-12
        )


def test_streamlit_application_starts_without_exception():
    app_test = AppTest.from_file(CODE_DIR / "app.py", default_timeout=20).run()
    assert not app_test.exception
    assert next(
        widget for widget in app_test.text_input if widget.label == "Analysis name"
    ).value == ""

