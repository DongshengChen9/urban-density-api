from __future__ import annotations

from pathlib import Path
import sys

import geopandas as gpd
import pytest
from shapely.geometry import box


TEST_DIR = Path(__file__).resolve().parent
CODE_DIR = TEST_DIR.parent
SCRIPTS_DIR = CODE_DIR / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


from export_web_map import (  # noqa: E402
    DEFAULT_BASEMAP,
    INDICATOR_COLUMNS,
    INDICATOR_LABELS,
    INDICATOR_UNITS,
    OSM_LOCAL_FILE_WARNING,
    export_web_map,
    parse_args,
)


def write_gpkg(path: Path, gdf: gpd.GeoDataFrame, layer: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if layer is None:
        gdf.to_file(path)
    else:
        gdf.to_file(path, layer=layer)


def make_aoi() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"name": ["test_aoi"]},
        geometry=[box(16.0, 48.0, 16.02, 48.02)],
        crs="EPSG:4326",
    )


def make_buildings() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"building_id": ["b1", "b2"]},
        geometry=[
            box(16.002, 48.002, 16.006, 48.006),
            box(16.01, 48.01, 16.014, 48.014),
        ],
        crs="EPSG:4326",
    )


def make_grid(include_indicator: bool = True) -> gpd.GeoDataFrame:
    data = {
        "unit_id": ["cell_1", "cell_2"],
        "gsi": [0.2, 0.4],
        "far_fsi": [1.0, 1.5],
        "built_volume_density": [3.0, 4.0],
        "avg_neighbor_distance_m": [8.0, 10.0],
        "avg_street_profile_height_to_width_ratio_strict": [0.6, 0.8],
    }
    if not include_indicator:
        data.pop("gsi")

    return gpd.GeoDataFrame(
        data,
        geometry=[
            box(16.0, 48.0, 16.01, 48.01),
            box(16.01, 48.01, 16.02, 48.02),
        ],
        crs="EPSG:4326",
    )


def make_standard_output(output_folder: Path, include_buildings: bool = True) -> None:
    write_gpkg(output_folder / "processed" / "aoi_metric.gpkg", make_aoi())
    if include_buildings:
        write_gpkg(
            output_folder / "processed" / "buildings_height_enriched.gpkg",
            make_buildings(),
        )
    write_gpkg(
        output_folder / "indicators" / "grid_indicators.gpkg",
        make_grid(),
        layer="grid_indicators",
    )


def test_standard_grid_layer_exports_html_file(tmp_path):
    output_folder = tmp_path / "04_outputs" / "test_run"
    make_standard_output(output_folder)

    html_path = export_web_map(
        output_folder=output_folder,
        indicator="gsi",
    )

    html = html_path.read_text(encoding="utf-8")

    assert html_path == output_folder / "webmap" / "index.html"
    assert html_path.exists()
    assert "AOI boundary" in html
    assert "Building footprints" in html
    assert "Grid indicator: GSI / Building Coverage Ratio" in html
    assert "Opacity controls" in html
    assert "Units: unitless share" in html
    assert "Run: test_run" in html
    assert DEFAULT_BASEMAP == "cartodb_positron"


def test_default_basemap_is_accepted_and_writes_html(tmp_path):
    output_folder = tmp_path / "04_outputs" / "default_basemap_run"
    make_standard_output(output_folder)

    html_path = export_web_map(
        output_folder=output_folder,
        indicator="gsi",
    )
    html = html_path.read_text(encoding="utf-8")

    assert html_path.exists()
    assert "cartodb" in html.lower()
    assert "positron" in html.lower()


def test_openstreetmap_basemap_writes_html_and_emits_warning(tmp_path, capsys):
    output_folder = tmp_path / "04_outputs" / "osm_basemap_run"
    make_standard_output(output_folder)

    html_path = export_web_map(
        output_folder=output_folder,
        indicator="gsi",
        basemap="openstreetmap",
    )
    captured = capsys.readouterr()
    html = html_path.read_text(encoding="utf-8")

    assert html_path.exists()
    assert OSM_LOCAL_FILE_WARNING in captured.out
    assert OSM_LOCAL_FILE_WARNING in html
    assert "openstreetmap" in html.lower()


def test_no_basemap_writes_html_without_failing(tmp_path):
    output_folder = tmp_path / "04_outputs" / "no_basemap_run"
    make_standard_output(output_folder)

    html_path = export_web_map(
        output_folder=output_folder,
        indicator="gsi",
        basemap="none",
    )
    html = html_path.read_text(encoding="utf-8")

    assert html_path.exists()
    assert "Grid indicator: GSI / Building Coverage Ratio" in html


def test_invalid_basemap_value_is_rejected_by_argparse(tmp_path):
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--output-folder",
                str(tmp_path / "run"),
                "--indicator",
                "gsi",
                "--basemap",
                "invalid_basemap",
            ]
        )


def test_segmented_grid_layer_fallback_works(tmp_path):
    output_folder = tmp_path / "04_outputs" / "segmented_run"
    write_gpkg(output_folder / "processed" / "aoi_metric.gpkg", make_aoi())
    write_gpkg(
        output_folder / "indicators" / "grid_indicators_segmented_wgs84.gpkg",
        make_grid(),
        layer="grid_indicators_segmented_wgs84",
    )

    html_path = export_web_map(
        output_folder=output_folder,
        indicator="built_volume_density",
    )

    html = html_path.read_text(encoding="utf-8")

    assert html_path.exists()
    assert "Grid indicator: Built Volume Density" in html
    assert "Units: m3 / m2" in html


def test_missing_optional_building_layer_does_not_fail(tmp_path, capsys):
    output_folder = tmp_path / "04_outputs" / "no_buildings_run"
    make_standard_output(output_folder, include_buildings=False)

    html_path = export_web_map(
        output_folder=output_folder,
        indicator="far",
    )
    captured = capsys.readouterr()
    html = html_path.read_text(encoding="utf-8")

    assert html_path.exists()
    assert "Building footprint layer was not found" in captured.out
    assert "map exported without buildings" in html


def test_missing_selected_indicator_raises_clear_error(tmp_path):
    output_folder = tmp_path / "04_outputs" / "missing_indicator_run"
    write_gpkg(output_folder / "processed" / "aoi_metric.gpkg", make_aoi())
    write_gpkg(
        output_folder / "indicators" / "grid_indicators.gpkg",
        make_grid(include_indicator=False),
        layer="grid_indicators",
    )

    with pytest.raises(ValueError, match="expects column `gsi`"):
        export_web_map(
            output_folder=output_folder,
            indicator="gsi",
        )


def test_output_parent_folder_is_created(tmp_path):
    output_folder = tmp_path / "04_outputs" / "custom_output_run"
    make_standard_output(output_folder)
    output_html = tmp_path / "nested" / "web" / "custom.html"

    export_web_map(
        output_folder=output_folder,
        indicator="neighbour_distance",
        output_html=output_html,
    )

    assert output_html.exists()
    assert output_html.parent.exists()


def test_supported_indicator_column_mapping_is_documented():
    assert INDICATOR_COLUMNS == {
        "gsi": "gsi",
        "far": "far_fsi",
        "built_volume_density": "built_volume_density",
        "neighbour_distance": "avg_neighbor_distance_m",
        "street_profile_ratio": "avg_street_profile_height_to_width_ratio_strict",
    }


def test_user_facing_popup_labels_hide_raw_grid_columns_and_paths(tmp_path):
    output_folder = tmp_path / "04_outputs" / "friendly_labels_run"
    make_standard_output(output_folder)

    html_path = export_web_map(
        output_folder=output_folder,
        indicator="neighbour_distance",
    )
    html = html_path.read_text(encoding="utf-8")

    assert "Average nearest-building distance" in html
    assert "Average nearest-building distance (m)" in html
    assert "Units: metres" in html
    assert "avg_neighbor_distance_m" not in html
    assert str(output_folder) not in html
    assert "Grid source:" not in html


def test_web_map_missing_data_note_is_visible_when_indicator_has_missing_values(tmp_path):
    output_folder = tmp_path / "04_outputs" / "missing_values_run"
    make_standard_output(output_folder)
    grid_path = output_folder / "indicators" / "grid_indicators.gpkg"
    grid = gpd.read_file(grid_path)
    grid.loc[0, "far_fsi"] = None
    grid_path.unlink()
    write_gpkg(grid_path, grid, layer="grid_indicators")

    html_path = export_web_map(
        output_folder=output_folder,
        indicator="far",
    )
    html = html_path.read_text(encoding="utf-8")

    assert (
        "Blank or grey cells indicate missing or insufficient input data, "
        "not zero or low indicator values."
    ) in html
    assert "Legend range is scaled for this map" in html


def test_indicator_names_and_units_are_user_facing():
    assert INDICATOR_LABELS["gsi"] == "GSI / Building Coverage Ratio"
    assert INDICATOR_LABELS["far"] == "FAR/FSI"
    assert INDICATOR_LABELS["built_volume_density"] == "Built Volume Density"
    assert INDICATOR_LABELS["neighbour_distance"] == "Average nearest-building distance"
    assert (
        INDICATOR_LABELS["street_profile_ratio"]
        == "Street-profile height-to-width ratio"
    )
    assert INDICATOR_UNITS["built_volume_density"] == "m3 / m2"
