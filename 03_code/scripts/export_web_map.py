from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path
from typing import Any

import folium
import geopandas as gpd
import pandas as pd
from branca.element import Element

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from map_styles import color_for_value, legend_entries, resolved_style, style_for_column, style_for_key


INDICATOR_COLUMNS = {
    "gsi": "gsi",
    "far": "far_fsi",
    "built_volume_density": "built_volume_density",
    "neighbour_distance": "avg_neighbor_distance_m",
    "street_profile_ratio": "avg_street_profile_height_to_width_ratio_strict",
}

INDICATOR_LABELS = {
    "gsi": "GSI / Building Coverage Ratio",
    "far": "FAR/FSI",
    "built_volume_density": "Built Volume Density",
    "neighbour_distance": "Average nearest-building distance",
    "street_profile_ratio": "Street-profile height-to-width ratio",
}

INDICATOR_UNITS = {
    "gsi": "unitless share",
    "far": "unitless ratio",
    "built_volume_density": "m3 / m2",
    "neighbour_distance": "metres",
    "street_profile_ratio": "unitless ratio",
}

FIELD_LABELS = {
    "gsi": "GSI / Building Coverage Ratio",
    "far_fsi": "FAR/FSI",
    "built_volume_density": "Built Volume Density",
    "avg_neighbor_distance_m": "Average nearest-building distance (m)",
    "avg_street_profile_height_to_width_ratio_strict": (
        "Street-profile height-to-width ratio"
    ),
    "unit_id": "Grid cell ID",
    "cell_id_global": "Global grid cell ID",
    "segment_id": "Segment ID",
}

BASEMAP_TILES = {
    "cartodb_positron": "CartoDB positron",
    "openstreetmap": "OpenStreetMap",
    "cartodb_darkmatter": "CartoDB dark_matter",
    "none": None,
}

DEFAULT_BASEMAP = "cartodb_positron"
OSM_LOCAL_FILE_WARNING = (
    "OpenStreetMap tile servers may block local HTML files without a valid "
    "Referer. If tiles show Access blocked, use --basemap cartodb_positron "
    "or serve the file through a local HTTP server."
)

AOI_CANDIDATES = [
    "processed/aoi_metric.gpkg",
    "processed/aoi.gpkg",
]

BUILDING_CANDIDATES = [
    "processed/buildings_height_enriched.gpkg",
    "processed/buildings_height_enriched_segmented_wgs84.gpkg",
    "processed/buildings_clean.gpkg",
]

STANDARD_GRID = "indicators/grid_indicators.gpkg"
SEGMENTED_GRID = "indicators/grid_indicators_segmented_wgs84.gpkg"


def _read_layer(path: Path) -> gpd.GeoDataFrame:
    return gpd.read_file(path)


def _to_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("Spatial layer has no CRS and cannot be web-mapped safely.")
    return gdf.to_crs("EPSG:4326")


def _prepare_layer(
    gdf: gpd.GeoDataFrame,
    max_features: int | None = None,
    simplify_tolerance: float | None = None,
) -> gpd.GeoDataFrame:
    gdf = _to_wgs84(gdf)

    if simplify_tolerance is not None and simplify_tolerance > 0:
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.simplify(
            simplify_tolerance,
            preserve_topology=True,
        )

    if max_features is not None and max_features > 0 and len(gdf) > max_features:
        return gdf.head(max_features).copy()

    return gdf


def _find_existing(output_folder: Path, candidates: list[str]) -> Path | None:
    for relative_path in candidates:
        path = output_folder / relative_path
        if path.exists():
            return path
    return None


def load_optional_layer(
    output_folder: Path,
    candidates: list[str],
    max_features: int | None,
    simplify_tolerance: float | None,
) -> tuple[gpd.GeoDataFrame | None, str | None]:
    path = _find_existing(output_folder, candidates)
    if path is None:
        return None, None
    return _prepare_layer(
        _read_layer(path),
        max_features=max_features,
        simplify_tolerance=simplify_tolerance,
    ), str(path)


def load_grid_layer(
    output_folder: Path,
    max_features: int | None,
    simplify_tolerance: float | None,
) -> tuple[gpd.GeoDataFrame, str]:
    standard_path = output_folder / STANDARD_GRID
    segmented_path = output_folder / SEGMENTED_GRID

    if standard_path.exists():
        path = standard_path
    elif segmented_path.exists():
        path = segmented_path
    else:
        raise FileNotFoundError(
            "No grid indicator layer found. Expected "
            f"`{STANDARD_GRID}` or `{SEGMENTED_GRID}` under {output_folder}."
        )

    return _prepare_layer(
        _read_layer(path),
        max_features=max_features,
        simplify_tolerance=simplify_tolerance,
    ), str(path)


def _map_center(*layers: gpd.GeoDataFrame | None) -> list[float]:
    for layer in layers:
        if layer is not None and not layer.empty:
            bounds = layer.total_bounds
            return [
                float((bounds[1] + bounds[3]) / 2),
                float((bounds[0] + bounds[2]) / 2),
            ]
    return [0.0, 0.0]


def _safe_numeric_series(gdf: gpd.GeoDataFrame, column: str) -> pd.Series:
    return pd.to_numeric(gdf[column], errors="coerce")


def _color_for_value(value: Any, min_value: float, max_value: float, indicator_column: str) -> str:
    style = style_for_column(indicator_column)
    if style is not None:
        return color_for_value(resolved_style(style, [min_value, max_value]), value)
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "#d9d9d9"
    if max_value <= min_value:
        ratio = 0.5
    else:
        ratio = max(0.0, min(1.0, float((numeric - min_value) / (max_value - min_value))))

    # Simple continuous blue-to-red ramp. This is visual only and does not
    # define scientific classification thresholds.
    red = int(49 + ratio * (215 - 49))
    green = int(130 + ratio * (48 - 130))
    blue = int(189 + ratio * (39 - 189))
    return f"#{red:02x}{green:02x}{blue:02x}"


def _popup_fields(grid: gpd.GeoDataFrame, indicator_column: str) -> list[str]:
    preferred = [
        indicator_column,
        "gsi",
        "far_fsi",
        "built_volume_density",
        "avg_neighbor_distance_m",
        "avg_street_profile_height_to_width_ratio_strict",
        "segment_id",
        "cell_id_global",
        "unit_id",
    ]
    fields = []
    for field in preferred:
        if field in grid.columns and field not in fields:
            fields.append(field)
    return fields[:8]


def _popup_aliases(fields: list[str]) -> list[str]:
    return [FIELD_LABELS.get(field, field.replace("_", " ").title()) for field in fields]


def _grid_layer_name(indicator_label: str) -> str:
    return f"Grid indicator: {indicator_label}"


def _add_aoi_layer(map_object: folium.Map, aoi: gpd.GeoDataFrame | None) -> None:
    if aoi is None or aoi.empty:
        return
    folium.GeoJson(
        aoi,
        name="AOI boundary",
        style_function=lambda _feature: {
            "color": "#111111",
            "weight": 3,
            "fillOpacity": 0.0,
            "className": "aoi-layer",
        },
    ).add_to(map_object)


def _add_building_layer(
    map_object: folium.Map,
    buildings: gpd.GeoDataFrame | None,
) -> None:
    if buildings is None or buildings.empty:
        return
    folium.GeoJson(
        buildings,
        name="Building footprints",
        style_function=lambda _feature: {
            "color": "#4d4d4d",
            "weight": 0.6,
            "fillColor": "#777777",
            "fillOpacity": 0.35,
            "className": "building-layer",
        },
    ).add_to(map_object)


def _add_grid_layer(
    map_object: folium.Map,
    grid: gpd.GeoDataFrame,
    indicator_column: str,
    indicator_label: str,
    indicator_unit: str,
) -> tuple[float, float]:
    values = _safe_numeric_series(grid, indicator_column)
    valid_values = values.dropna()
    min_value = float(valid_values.min())
    max_value = float(valid_values.max())
    popup_fields = _popup_fields(grid, indicator_column)
    popup_aliases = _popup_aliases(popup_fields)
    display_grid = grid[["geometry"]].copy()
    display_fields: list[str] = []
    for field, alias in zip(popup_fields, popup_aliases):
        display_name = alias
        if display_name in display_grid.columns:
            display_name = f"{alias} value"
        display_grid[display_name] = grid[field]
        display_fields.append(display_name)
    indicator_display_field = display_fields[0]

    def style_function(feature: dict[str, Any]) -> dict[str, Any]:
        value = feature.get("properties", {}).get(indicator_display_field)
        return {
            "color": "#222222",
            "weight": 0.7,
            "fillColor": _color_for_value(value, min_value, max_value, indicator_column),
            "fillOpacity": 0.45,
            "className": "grid-indicator-layer",
        }

    folium.GeoJson(
        display_grid,
        name=_grid_layer_name(indicator_label),
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=[indicator_display_field],
            aliases=[f"{indicator_label} ({indicator_unit})"],
        ),
        popup=folium.GeoJsonPopup(fields=display_fields),
    ).add_to(map_object)

    return min_value, max_value


def _warning_html(warnings: list[str]) -> str:
    if not warnings:
        return ""
    items = "".join(f"<li>{html.escape(warning)}</li>" for warning in warnings)
    return (
        '<div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999; '
        'background: white; padding: 10px; border: 1px solid #999; '
        'max-width: 360px; font-size: 13px;">'
        "<strong>Map export warnings</strong><ul>"
        f"{items}</ul></div>"
    )


def _caption_html(
    indicator_label: str,
    indicator_unit: str,
    run_label: str,
    min_value: float,
    max_value: float,
    missing_count: int,
    total_count: int,
) -> str:
    missing_note = ""
    if missing_count > 0:
        missing_note = (
            "<br>Blank or grey cells indicate missing or insufficient input "
            "data, not zero or low indicator values."
        )

    return (
        '<div style="position: fixed; top: 20px; right: 20px; z-index: 9999; '
        'background: white; padding: 10px; border: 1px solid #999; '
        'max-width: 360px; font-size: 13px;">'
        f"<strong>{html.escape(indicator_label)}</strong><br>"
        f"Run: {html.escape(run_label)}<br>"
        f"Units: {html.escape(indicator_unit)}<br>"
        f"Display extent: {min_value:.4g} to {max_value:.4g}<br>"
        "Legend range is scaled for this map when configured or run-derived breaks are used.<br>"
        "Grey is missing or insufficient input; near-white is a true zero."
        f"{missing_note}"
        "</div>"
    )


def _opacity_control_html() -> str:
    return """
<div style="position: fixed; top: 20px; left: 50px; z-index: 9999; background: white; padding: 10px; border: 1px solid #999; font-size: 13px;">
  <strong>Opacity controls</strong><br>
  <label>Grid <input id="grid-opacity-slider" type="range" min="0" max="1" step="0.05" value="0.45"></label><br>
  <label>Buildings <input id="building-opacity-slider" type="range" min="0" max="1" step="0.05" value="0.35"></label>
</div>
<script>
function setLayerOpacity(className, value) {
  document.querySelectorAll('.' + className).forEach(function(path) {
    path.style.fillOpacity = value;
    path.style.opacity = value;
  });
}
document.addEventListener('input', function(event) {
  if (event.target && event.target.id === 'grid-opacity-slider') {
    setLayerOpacity('grid-indicator-layer', event.target.value);
  }
  if (event.target && event.target.id === 'building-opacity-slider') {
    setLayerOpacity('building-layer', event.target.value);
  }
});
</script>
"""


def build_web_map(
    output_folder: Path,
    indicator: str,
    max_features: int | None = None,
    simplify_tolerance: float | None = None,
    basemap: str = DEFAULT_BASEMAP,
) -> folium.Map:
    if indicator not in INDICATOR_COLUMNS:
        raise ValueError(
            f"Unsupported indicator `{indicator}`. Supported indicators: "
            + ", ".join(INDICATOR_COLUMNS)
        )
    if basemap not in BASEMAP_TILES:
        raise ValueError(
            f"Unsupported basemap `{basemap}`. Supported basemaps: "
            + ", ".join(BASEMAP_TILES)
        )

    warnings: list[str] = []
    aoi, aoi_path = load_optional_layer(
        output_folder=output_folder,
        candidates=AOI_CANDIDATES,
        max_features=max_features,
        simplify_tolerance=simplify_tolerance,
    )
    buildings, buildings_path = load_optional_layer(
        output_folder=output_folder,
        candidates=BUILDING_CANDIDATES,
        max_features=max_features,
        simplify_tolerance=simplify_tolerance,
    )
    grid, grid_path = load_grid_layer(
        output_folder=output_folder,
        max_features=max_features,
        simplify_tolerance=simplify_tolerance,
    )

    if aoi_path is None:
        warnings.append("AOI boundary layer was not found; map exported without AOI.")
    if buildings_path is None:
        warnings.append(
            "Building footprint layer was not found; map exported without buildings."
        )
    if basemap == "openstreetmap":
        warnings.append(OSM_LOCAL_FILE_WARNING)

    indicator_column = INDICATOR_COLUMNS[indicator]
    shared_style = style_for_key(indicator)
    indicator_label = shared_style.public_label
    indicator_unit = INDICATOR_UNITS[indicator]

    if indicator_column not in grid.columns:
        raise ValueError(
            f"Selected indicator `{indicator}` expects column "
            f"`{indicator_column}`, but it is missing from the grid layer."
        )

    values = _safe_numeric_series(grid, indicator_column)
    if values.dropna().empty:
        raise ValueError(
            f"Selected indicator column `{indicator_column}` has no numeric values."
        )

    map_object = folium.Map(
        location=_map_center(aoi, grid, buildings),
        zoom_start=14,
        tiles=BASEMAP_TILES[basemap],
        control_scale=True,
    )

    _add_aoi_layer(map_object, aoi)
    _add_building_layer(map_object, buildings)
    min_value, max_value = _add_grid_layer(
        map_object=map_object,
        grid=grid,
        indicator_column=indicator_column,
        indicator_label=indicator_label,
        indicator_unit=indicator_unit,
    )

    folium.LayerControl(collapsed=False).add_to(map_object)
    map_object.get_root().html.add_child(
        Element(
            _caption_html(
                indicator_label=indicator_label,
                indicator_unit=indicator_unit,
                run_label=output_folder.name,
                min_value=min_value,
                max_value=max_value,
                missing_count=int(values.isna().sum()),
                total_count=int(len(values)),
            )
        )
    )
    map_object.get_root().html.add_child(Element(_warning_html(warnings)))
    map_object.get_root().html.add_child(Element(_opacity_control_html()))

    for warning in warnings:
        print(f"WARNING: {warning}")

    return map_object


def default_output_html(output_folder: Path) -> Path:
    return output_folder / "webmap" / "index.html"


def export_web_map(
    output_folder: Path,
    indicator: str,
    output_html: Path | None = None,
    max_features: int | None = None,
    simplify_tolerance: float | None = None,
    basemap: str = DEFAULT_BASEMAP,
) -> Path:
    output_folder = output_folder.resolve()
    if output_html is None:
        output_html = default_output_html(output_folder)

    map_object = build_web_map(
        output_folder=output_folder,
        indicator=indicator,
        max_features=max_features,
        simplify_tolerance=simplify_tolerance,
        basemap=basemap,
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    map_object.save(str(output_html))
    return output_html


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a completed workflow output folder to an HTML web map."
    )
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument("--indicator", choices=sorted(INDICATOR_COLUMNS), required=True)
    parser.add_argument("--output-html", type=Path, default=None)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--simplify-tolerance", type=float, default=None)
    parser.add_argument(
        "--basemap",
        choices=sorted(BASEMAP_TILES),
        default=DEFAULT_BASEMAP,
        help="Basemap provider for the exported HTML map.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    output_html = export_web_map(
        output_folder=args.output_folder,
        indicator=args.indicator,
        output_html=args.output_html,
        max_features=args.max_features,
        simplify_tolerance=args.simplify_tolerance,
        basemap=args.basemap,
    )
    print(f"Wrote web map: {output_html}")


if __name__ == "__main__":
    main()
