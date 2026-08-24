from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import folium
import geopandas as gpd
import pandas as pd
from branca.colormap import linear
from branca.element import MacroElement, Template
from folium.plugins import GroupedLayerControl

from .models import AnalysisRequest


MAP_INDICATOR_SPECS = (
    (
        "GSI / Building Coverage Ratio",
        ("gsi", "building_coverage_ratio", "building_coverage", "site_coverage"),
    ),
    ("FAR / FSI", ("far_fsi", "far", "fsi", "floor_area_ratio")),
    (
        "Built Volume Density",
        ("built_volume_density", "built_volume_density_m3_m2", "volume_density"),
    ),
    (
        "Average nearest-building distance",
        ("avg_neighbor_distance_m", "average_neighbor_distance_m"),
    ),
    (
        "Street-profile height-to-width ratio",
        (
            "avg_street_profile_height_to_width_ratio_strict",
            "street_profile_height_to_width_ratio",
        ),
    ),
)


class _DynamicMetricLegend(MacroElement):
    """One legend whose label/range follows the selected indicator layer."""

    _template = Template(
        """
        {% macro script(this, kwargs) %}
        var {{ this.get_name() }} = L.control({position: 'topright'});
        var {{ this.get_name() }}_items = [
            {% for item in this.items %}
            {
                layer: {{ item.layer_name }},
                label: {{ item.label|tojson }},
                minimum: {{ item.minimum|tojson }},
                maximum: {{ item.maximum|tojson }}
            },
            {% endfor %}
        ];
        {{ this.get_name() }}.onAdd = function (map) {
            this._div = L.DomUtil.create('div', 'udw-indicator-legend');
            this._div.style.cssText = 'background:white;padding:8px 10px;border-radius:4px;box-shadow:0 1px 5px rgba(0,0,0,.35);min-width:220px;font:12px/1.35 Arial,sans-serif';
            return this._div;
        };
        {{ this.get_name() }}.update = function (item) {
            this._div.innerHTML = '<strong>' + item.label + '</strong>' +
                '<div style="height:10px;margin-top:6px;background:linear-gradient(to right,#ffffcc,#ffeda0,#fed976,#feb24c,#fd8d3c,#fc4e2a,#e31a1c,#bd0026,#800026)"></div>' +
                '<div><span>' + item.minimum + '</span><span style="float:right">' + item.maximum + '</span></div>';
        };
        {{ this.get_name() }}.addTo({{ this._parent.get_name() }});
        {{ this.get_name() }}.update({{ this.get_name() }}_items[0]);
        {{ this._parent.get_name() }}.on('overlayadd', function (event) {
            for (var i = 0; i < {{ this.get_name() }}_items.length; i++) {
                if (event.layer === {{ this.get_name() }}_items[i].layer) {
                    {{ this.get_name() }}.update({{ this.get_name() }}_items[i]);
                    break;
                }
            }
        });
        {% endmacro %}
        """
    )

    def __init__(self, items: list[dict[str, Any]]) -> None:
        super().__init__()
        self._name = "DynamicMetricLegend"
        self.items = items


class WorkflowService:
    """Adapter from API models to existing workflow inputs and products."""

    def run_analysis(
        self, analysis_id: str, request: AnalysisRequest, output_directory: Path
    ) -> None:
        # Imports are lazy so /health does not initialize the GIS workflow or
        # perform any source-resolution work.
        from run_workflow import run_workflow
        from scripts.create_config_from_bbox import build_config, write_config

        config = build_config(
            run_name=analysis_id,
            min_lon=request.bbox.min_lon,
            min_lat=request.bbox.min_lat,
            max_lon=request.bbox.max_lon,
            max_lat=request.bbox.max_lat,
            grid_size=request.grid_size,
            mode=request.mode.value,
        )
        # UUID directories make overwriting unnecessary and allow UDW_OUTPUT_DIR
        # to be outside the repository's normal 04_outputs directory.
        config["project"]["output_dir"] = str(output_directory.resolve())
        config["project"]["overwrite_existing_run"] = False
        config["outputs"]["mode"] = "compact"
        config_path = write_config(config, output_directory / "workflow_config.yaml")
        run_workflow(config_path)


def read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path.name)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path.name}")
    return value


def read_analysis_results(output_directory: Path, analysis_id: str) -> dict[str, Any]:
    reports = output_directory / "reports"
    readiness_path = reports / "indicator_readiness.json"
    readiness: list[dict[str, Any]] | dict[str, Any] | None = None
    if readiness_path.is_file():
        readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    return {
        "analysis_id": analysis_id,
        "workflow_summary": read_json_object(reports / "workflow_summary.json"),
        "stage_timings": read_json_object(reports / "stage_timings.json"),
        "indicator_readiness": readiness,
    }


def discover_grid_path(output_directory: Path) -> Path:
    candidates = (
        output_directory / "indicators" / "grid_indicators.gpkg",
        output_directory / "indicators" / "grid_indicators_segmented_wgs84.gpkg",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("No completed grid-indicator layer was found")


def read_grid_wgs84(output_directory: Path) -> gpd.GeoDataFrame:
    """Read the discovered workflow grid and normalize it for web delivery."""
    grid = gpd.read_file(discover_grid_path(output_directory))
    if grid.crs is None:
        raise ValueError("The grid-indicator layer has no CRS")
    if grid.empty:
        raise ValueError("The grid-indicator layer is empty")
    return grid.to_crs("EPSG:4326")


def grid_feature_collection(output_directory: Path) -> dict[str, Any]:
    grid_wgs84 = read_grid_wgs84(output_directory)
    feature_collection = json.loads(grid_wgs84.to_json(drop_id=True))
    if feature_collection.get("type") != "FeatureCollection":
        raise ValueError("The workflow grid did not serialize as a FeatureCollection")
    return feature_collection


def _find_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    """Resolve a known semantic field without assuming one exact schema name."""
    normalized = {
        str(column).strip().lower().replace(" ", "_"): str(column)
        for column in columns
    }
    for candidate in candidates:
        match = normalized.get(candidate)
        if match is not None:
            return match
    return None


def render_grid_map(output_directory: Path) -> str:
    """Render the existing completed grid as a standalone Folium document."""
    grid = read_grid_wgs84(output_directory)
    min_lon, min_lat, max_lon, max_lat = (float(value) for value in grid.total_bounds)
    if not all(math.isfinite(value) for value in (min_lon, min_lat, max_lon, max_lat)):
        raise ValueError("The grid-indicator layer has invalid bounds")

    map_object = folium.Map(
        location=[(min_lat + max_lat) / 2, (min_lon + max_lon) / 2],
        tiles="OpenStreetMap",
        zoom_start=13,
        control_scale=True,
    )

    columns = [str(column) for column in grid.columns if column != grid.geometry.name]
    identifier = _find_column(columns, ("unit_id", "grid_id", "cell_id", "id"))
    available_indicators: list[tuple[str, str, pd.Series]] = []
    for label, candidates in MAP_INDICATOR_SPECS:
        column = _find_column(columns, candidates)
        if column is None:
            continue
        numeric_values = pd.to_numeric(grid[column], errors="coerce")
        if numeric_values.notna().any():
            available_indicators.append((label, column, numeric_values))

    tooltip_fields = [
        column
        for column in (
            identifier,
            *[column for _label, column, _values in available_indicators],
        )
        if column is not None
    ]
    tooltip_aliases = {identifier: "Grid cell"}
    tooltip_aliases.update(
        {column: label for label, column, _values in available_indicators}
    )
    def make_tooltip() -> folium.GeoJsonTooltip | None:
        if not tooltip_fields:
            return None
        return folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=[tooltip_aliases[column] for column in tooltip_fields],
            localize=True,
            sticky=False,
        )

    feature_collection = json.loads(grid.to_json(drop_id=True))
    indicator_layers: list[folium.FeatureGroup] = []
    legend_items: list[dict[str, Any]] = []

    for index, (label, column, numeric_values) in enumerate(available_indicators):
        valid_values = numeric_values.dropna()
        data_minimum = float(valid_values.min())
        data_maximum = float(valid_values.max())
        scale_minimum = data_minimum
        scale_maximum = data_maximum
        if math.isclose(scale_minimum, scale_maximum):
            scale_minimum = min(0.0, scale_minimum)
            scale_maximum = max(1.0, scale_maximum)
        color_scale = linear.YlOrRd_09.scale(scale_minimum, scale_maximum)

        def make_style_function(
            field: str, scale: Any
        ) -> Any:
            def style_function(feature: dict[str, Any]) -> dict[str, Any]:
                value = feature.get("properties", {}).get(field)
                try:
                    numeric_value = float(value)
                    fill_color = (
                        scale(numeric_value)
                        if math.isfinite(numeric_value)
                        else "#bdbdbd"
                    )
                except (TypeError, ValueError):
                    fill_color = "#bdbdbd"
                return {
                    "color": "#4a4a4a",
                    "weight": 0.7,
                    "fillColor": fill_color,
                    "fillOpacity": 0.65,
                }
            return style_function

        layer = folium.FeatureGroup(name=label, overlay=True, show=index == 0)
        folium.GeoJson(
            data=feature_collection,
            name=label,
            style_function=make_style_function(column, color_scale),
            tooltip=make_tooltip(),
        ).add_to(layer)
        layer.add_to(map_object)
        indicator_layers.append(layer)
        legend_items.append(
            {
                "layer_name": layer.get_name(),
                "label": label,
                "minimum": f"{data_minimum:.3g}",
                "maximum": f"{data_maximum:.3g}",
            }
        )

    if indicator_layers:
        GroupedLayerControl(
            groups={"Indicator": indicator_layers},
            exclusive_groups=True,
            collapsed=False,
        ).add_to(map_object)
        _DynamicMetricLegend(legend_items).add_to(map_object)
    else:
        folium.GeoJson(
            data=feature_collection,
            name="Grid cells",
            style_function=lambda _feature: {
                "color": "#4a4a4a",
                "weight": 0.7,
                "fillColor": "#3388ff",
                "fillOpacity": 0.35,
            },
            tooltip=make_tooltip(),
        ).add_to(map_object)
    map_object.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
    return map_object.get_root().render()
