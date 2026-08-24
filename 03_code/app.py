from __future__ import annotations

import json
import hashlib
import math
import copy
import re
import shutil
import subprocess
import sys
from html import escape
from functools import lru_cache
from datetime import datetime
from pathlib import Path

import yaml


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
SCRIPTS_DIR = CODE_DIR / "scripts"
SRC_DIR = CODE_DIR / "src"
GENERATED_CONFIG_DIR = CODE_DIR / "config" / "generated"
OUTPUTS_ROOT = PROJECT_ROOT / "04_outputs"

TABLE_PREVIEW_ROW_LIMIT = 300
BUILDING_MAP_PREVIEW_MAX_FEATURES = 5000
INTERACTIVE_GRID_CELL_THRESHOLD = 20000
LARGE_HTML_EMBED_LIMIT_MB = 50.0
DETAIL_GRID_CELL_LIMIT = 2500
DETAIL_ZOOM_THRESHOLD = 13
OVERVIEW_MAX_PIXELS = 1400
OVERVIEW_FORMAT_VERSION = 2
SETUP_MAP_CENTER = (50.0, 10.0)
SETUP_MAP_ZOOM = 4
PROJECT_ABOUT_TEXT = (
    "This local application calculates, visualizes, and supports "
    "data-quality-aware interpretation of physical urban-density and "
    "contextual morphology indicators."
)
STREAMLIT_CHROME_CSS = """
<style>
/* Streamlit 1.59.1: keep the project About item while hiding local developer chrome. */
[data-testid="stStatusWidget"],
[data-testid="stThemeSwitcher"],
[data-testid="stMainMenuDivider"],
[data-testid="stMainMenuPopover"] *:has(> .stMenuVersionCopyButton) {
    display: none !important;
}
</style>
"""
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from create_config_from_bbox import (  # noqa: E402
    RUN_MODES,
    create_config_from_bbox,
    validate_bbox,
    validate_grid_size,
    validate_run_name,
)
from export_web_map import load_grid_layer  # noqa: E402
from map_styles import (  # noqa: E402
    CARTOGRAPHIC_STYLE_VERSION,
    MISSING_COLOR,
    ZERO_COLOR,
    color_for_value,
    legend_entries,
    resolved_style,
    style_for_key,
)


MODE_HELP = {
    "quick_2d": "Usually produces GSI only and is the fastest first check.",
    "standard": "May produce height-based indicators where height data are available.",
    "full_context": (
        "May produce contextual indicators but is slower and depends on data "
        "coverage."
    ),
}

MISSING_INDICATOR_MESSAGE = (
    "This indicator is not available for this run. Quick 2D mode usually "
    "provides GSI only. Use Standard or Full Context mode for height/contextual "
    "indicators where data are available."
)

RUN_STATES = ("not_started", "config_ready", "running", "completed", "failed")
INPUTS_CHANGED_MESSAGE = "Inputs changed. Run analysis again to update results."
NO_RESULTS_MESSAGE = "No results yet. Run analysis to create a map and reports."

USER_COMPLETION_MESSAGES = {
    "analysis_completed": "Analysis completed.",
    "map_created": "Interactive map created.",
    "results_saved": "Results are saved in the project outputs folder.",
}

APP_INDICATORS = {
    "gsi": {
        "label": "GSI / Building Coverage Ratio",
        "column": "gsi",
        "unit": "ratio",
        "role": "Primary physical density indicator",
        "measures": "The share of land covered by building footprints.",
        "higher_lower": (
            "Higher values mean more of the cell is covered by buildings; lower "
            "values mean less footprint coverage."
        ),
        "requirements": "Valid building footprints and grid-cell areas.",
        "limitation": "The official value unions overlapping footprint coverage; the raw sum remains a diagnostic.",
        "valid_share_column": None,
    },
    "far": {
        "label": "FAR/FSI",
        "column": "far_fsi",
        "unit": "ratio",
        "role": "Conditional physical density indicator",
        "measures": "Estimated floor area relative to land area.",
        "higher_lower": (
            "Higher values mean more estimated floor area per unit of land; "
            "lower values mean less."
        ),
        "requirements": "Valid building footprints and reliable floor counts or floor area.",
        "limitation": "Incomplete floor data can strongly understate or limit the result.",
        "valid_share_column": "floor_data_valid_area_share",
    },
    "built_volume_density": {
        "label": "Built Volume Density",
        "column": "built_volume_density",
        "unit": "m3/m2",
        "role": "Conditional physical density indicator",
        "measures": "Estimated built volume relative to land area.",
        "higher_lower": (
            "Higher values mean more estimated building volume per unit of land; "
            "lower values mean less."
        ),
        "requirements": "Valid building footprints and reliable building heights.",
        "limitation": "Interpretation depends directly on height-data completeness and quality.",
        "valid_share_column": "height_valid_area_share",
    },
    "neighbour_distance": {
        "label": "Average nearest-building distance",
        "column": "avg_neighbor_distance_m",
        "unit": "m",
        "role": "Contextual morphology indicator",
        "measures": "For each building, the nearest other footprint-to-footprint distance, aggregated to grid cells.",
        "higher_lower": (
            "Higher values indicate more spacing between buildings; lower values "
            "indicate more compact or attached fabric."
        ),
        "requirements": "Valid building geometries and stable building identifiers.",
        "limitation": "Attached, touching, or overlapping buildings can legitimately produce zero distance.",
        "valid_share_column": None,
    },
    "street_profile_ratio": {
        "label": "Street-profile height-to-width ratio",
        "column": "avg_street_profile_height_to_width_ratio_strict",
        "unit": "ratio",
        "role": "Contextual morphology indicator",
        "measures": "Building height relative to the width of its street profile.",
        "higher_lower": (
            "Higher values indicate a taller, narrower street profile; lower "
            "values indicate a lower or wider profile."
        ),
        "requirements": "Building height, street-profile width, and a valid building-to-street match.",
        "limitation": (
            "Values exist only where both height and street-profile information "
            "are sufficient; blank cells are not zero values."
        ),
        "valid_share_column": None,
    },
}

APP_INDICATOR_ALIASES = {
    "gsi": "gsi",
    "gsi building coverage ratio": "gsi",
    "building coverage ratio": "gsi",
    "far": "far",
    "far fsi": "far",
    "far_fsi": "far",
    "built volume density": "built_volume_density",
    "built_volume_density": "built_volume_density",
    "neighbour distance": "neighbour_distance",
    "neighbor distance": "neighbour_distance",
    "average neighbour distance": "neighbour_distance",
    "average neighbor distance": "neighbour_distance",
    "average neighbour distance m": "neighbour_distance",
    "average neighbor distance m": "neighbour_distance",
    "avg neighbor distance m": "neighbour_distance",
    "avg neighbour distance m": "neighbour_distance",
    "avg_neighbor_distance_m": "neighbour_distance",
    "neighbor_distance": "neighbour_distance",
    "neighbour_distance": "neighbour_distance",
    "average_neighbor_distance": "neighbour_distance",
    "average_neighbour_distance": "neighbour_distance",
    "street profile height to width ratio": "street_profile_ratio",
    "street_profile_ratio": "street_profile_ratio",
    "avg_street_profile_height_to_width_ratio_strict": "street_profile_ratio",
}

GRID_RERUN_CACHE_SETTINGS = {
    "enabled": True,
    "use_existing_raw_buildings": True,
    "use_existing_cleaned_buildings": True,
    "use_existing_enriched_buildings": True,
    "use_existing_street_network": True,
    "use_existing_street_profiles": True,
    "use_existing_street_context": True,
    "use_existing_neighbor_context": True,
    "force_refresh": False,
    "require_compatible_manifest": True,
}

APP_SAFE_CACHE_SETTINGS = {
    "use_existing_raw_buildings": True,
    "use_existing_enriched_buildings": False,
    "use_existing_street_context": False,
    "force_refresh": False,
    "require_compatible_manifest": True,
}

BUILDING_BOUNDS_CANDIDATES = [
    ("raw/buildings_raw_overture.gpkg", "buildings_raw"),
    ("processed/buildings_clean.gpkg", None),
    ("processed/buildings_height_enriched.gpkg", None),
    ("processed/buildings_height_enriched_segmented_wgs84.gpkg", None),
]


def bbox_preview_properties(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
) -> dict[str, object]:
    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2
    return {
        "center": [center_lat, center_lon],
        "bounds": [[min_lat, min_lon], [max_lat, max_lon]],
        "polygon": [
            [min_lat, min_lon],
            [min_lat, max_lon],
            [max_lat, max_lon],
            [max_lat, min_lon],
            [min_lat, min_lon],
        ],
    }


def estimate_bbox_size_km(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
) -> dict[str, float]:
    mean_lat_rad = math.radians((min_lat + max_lat) / 2)
    width_km = abs(max_lon - min_lon) * 111.32 * max(math.cos(mean_lat_rad), 0)
    height_km = abs(max_lat - min_lat) * 110.574
    return {
        "width_km": width_km,
        "height_km": height_km,
        "area_km2": width_km * height_km,
    }


def build_aoi_preview_html(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
) -> str:
    import folium

    preview = bbox_preview_properties(
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
    )
    map_object = folium.Map(
        location=preview["center"],
        zoom_start=15,
        tiles="CartoDB positron",
    )
    folium.Rectangle(
        bounds=preview["bounds"],
        color="#1d4ed8",
        fill=True,
        fill_color="#60a5fa",
        fill_opacity=0.18,
        weight=3,
        tooltip="Selected analysis area",
    ).add_to(map_object)
    map_object.fit_bounds(preview["bounds"], padding=(20, 20))
    return map_object.get_root().render()


def build_aoi_draw_map(
    min_lon: float | None = None,
    min_lat: float | None = None,
    max_lon: float | None = None,
    max_lat: float | None = None,
) -> object:
    import folium
    from folium.plugins import Draw

    has_selection = all(
        value is not None for value in (min_lon, min_lat, max_lon, max_lat)
    )
    preview = (
        bbox_preview_properties(
            min_lon=float(min_lon),
            min_lat=float(min_lat),
            max_lon=float(max_lon),
            max_lat=float(max_lat),
        )
        if has_selection
        else None
    )
    map_object = folium.Map(
        location=preview["center"] if preview else SETUP_MAP_CENTER,
        zoom_start=14 if preview else SETUP_MAP_ZOOM,
        tiles="CartoDB positron",
    )
    if preview:
        folium.Rectangle(
            bounds=preview["bounds"],
            color="#1d4ed8",
            fill=True,
            fill_color="#60a5fa",
            fill_opacity=0.12,
            weight=2,
            tooltip="Current selected analysis area",
        ).add_to(map_object)
    Draw(
        export=False,
        draw_options={
            "polyline": False,
            "polygon": False,
            "circle": False,
            "circlemarker": False,
            "marker": False,
            "rectangle": True,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(map_object)
    if preview:
        map_object.fit_bounds(preview["bounds"], padding=(20, 20))
    return map_object


def latest_drawn_feature(draw_result: dict[str, object] | None) -> dict[str, object] | None:
    if not draw_result:
        return None
    all_drawings = draw_result.get("all_drawings")
    if isinstance(all_drawings, list) and all_drawings:
        return all_drawings[-1]
    last_active = draw_result.get("last_active_drawing")
    if isinstance(last_active, dict):
        return last_active
    return None


def _iter_geojson_positions(coordinates: object):
    if (
        isinstance(coordinates, list)
        and len(coordinates) >= 2
        and all(isinstance(value, (int, float)) for value in coordinates[:2])
    ):
        yield float(coordinates[0]), float(coordinates[1])
        return
    if isinstance(coordinates, list):
        for item in coordinates:
            yield from _iter_geojson_positions(item)


def extract_bbox_from_drawn_geojson(feature: dict[str, object]) -> tuple[float, float, float, float]:
    if not isinstance(feature, dict):
        raise ValueError("The drawn analysis area is not a valid map feature.")

    if feature.get("type") == "FeatureCollection":
        features = feature.get("features")
        if not isinstance(features, list) or not features:
            raise ValueError("No drawn analysis area was found.")
        return extract_bbox_from_drawn_geojson(features[-1])

    geometry = feature.get("geometry", feature)
    if not isinstance(geometry, dict):
        raise ValueError("The drawn analysis area has no valid geometry.")

    geometry_type = geometry.get("type")
    if geometry_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError("The drawn analysis area must be a rectangle or polygon.")

    positions = list(_iter_geojson_positions(geometry.get("coordinates")))
    if not positions:
        raise ValueError("The drawn analysis area has no valid coordinates.")

    longitudes = [position[0] for position in positions]
    latitudes = [position[1] for position in positions]
    min_lon = min(longitudes)
    max_lon = max(longitudes)
    min_lat = min(latitudes)
    max_lat = max(latitudes)
    validate_bbox(
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
    )
    return min_lon, min_lat, max_lon, max_lat


def update_session_bbox(
    session_state: object,
    bbox: tuple[float, float, float, float],
) -> None:
    min_lon, min_lat, max_lon, max_lat = bbox
    session_state["bbox_min_lon"] = min_lon
    session_state["bbox_min_lat"] = min_lat
    session_state["bbox_max_lon"] = max_lon
    session_state["bbox_max_lat"] = max_lat


def has_selected_bbox(session_state: object) -> bool:
    return all(
        session_state.get(key) is not None
        for key in ("bbox_min_lon", "bbox_min_lat", "bbox_max_lon", "bbox_max_lat")
    )


def apply_drawn_area(session_state: object, feature: dict[str, object]) -> bool:
    """Store the newest drawn AOI and replace any previous drawing."""
    serialized = json.dumps(feature, sort_keys=True)
    if serialized == session_state.get("selected_aoi_geometry"):
        return False
    bbox = extract_bbox_from_drawn_geojson(feature)
    update_session_bbox(session_state, bbox)
    size = estimate_bbox_size_km(*bbox)
    session_state["selected_aoi_geometry"] = serialized
    session_state["aoi_drawing_payload"] = feature
    session_state["aoi_area_km2"] = float(size["area_km2"])
    session_state["aoi_validation_message"] = None
    session_state["show_aoi_preview"] = True
    session_state["aoi_draw_reset_token"] = int(
        session_state.get("aoi_draw_reset_token") or 0
    ) + 1
    return True


def clear_selected_area(session_state: object) -> int:
    """Clear setup AOI state and force the drawing component to remount."""
    for key in ("bbox_min_lon", "bbox_min_lat", "bbox_max_lon", "bbox_max_lat"):
        session_state[key] = None
    session_state["selected_aoi_geometry"] = None
    session_state["aoi_drawing_payload"] = None
    session_state["aoi_area_km2"] = None
    session_state["aoi_validation_message"] = None
    session_state["show_aoi_preview"] = False
    session_state["setup_signature"] = None
    session_state["setup_input_changed"] = True
    token = int(session_state.get("aoi_draw_reset_token") or 0) + 1
    session_state["aoi_draw_reset_token"] = token
    return token


def apply_app_safe_cache_settings(config_path: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    project_config = dict(config.get("project") or {})
    project_config["overwrite_existing_run"] = True
    config["project"] = project_config
    cache_config = dict(config.get("cache") or {})
    cache_config.update(APP_SAFE_CACHE_SETTINGS)
    config["cache"] = cache_config
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    return cache_config


def input_signature_changed(
    previous_signature: tuple[object, ...] | None,
    current_signature: tuple[object, ...],
) -> bool:
    return previous_signature is not None and previous_signature != current_signature


def output_folder_requires_overwrite(output_folder: Path) -> bool:
    return output_folder.exists() and any(output_folder.iterdir())


def isolate_existing_output_folder(output_folder: Path) -> Path | None:
    if not output_folder.exists():
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_backup = output_folder.with_name(f"{output_folder.name}_stale_{timestamp}")
    backup = base_backup
    counter = 1
    while backup.exists():
        backup = output_folder.with_name(f"{base_backup.name}_{counter}")
        counter += 1
    shutil.move(str(output_folder), str(backup))
    return backup


def bboxes_overlap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    first_min_lon, first_min_lat, first_max_lon, first_max_lat = first
    second_min_lon, second_min_lat, second_max_lon, second_max_lat = second
    return not (
        first_max_lon < second_min_lon
        or second_max_lon < first_min_lon
        or first_max_lat < second_min_lat
        or second_max_lat < first_min_lat
    )


def read_building_bounds_wgs84(output_folder: Path) -> tuple[float, float, float, float] | None:
    import geopandas as gpd

    for relative_path, layer in BUILDING_BOUNDS_CANDIDATES:
        path = output_folder / relative_path
        if not path.exists():
            continue
        try:
            if layer is None:
                buildings = gpd.read_file(path)
            else:
                buildings = gpd.read_file(path, layer=layer)
        except Exception:
            continue
        if buildings.empty or buildings.crs is None:
            continue
        bounds = buildings.to_crs("EPSG:4326").total_bounds
        return (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))
    return None


def building_bounds_overlap_selected_bbox(
    output_folder: Path,
    selected_bbox: tuple[float, float, float, float],
) -> tuple[bool, str | None]:
    building_bounds = read_building_bounds_wgs84(output_folder)
    if building_bounds is None:
        return True, None
    if bboxes_overlap(selected_bbox, building_bounds):
        return True, None
    return (
        False,
        (
            "The saved building data do not overlap the selected analysis area. "
            "Results are hidden to avoid showing stale or incompatible outputs. "
            "Use a new analysis name or rerun with overwrite enabled."
        ),
    )


def parse_decimal_input(value: object, label: str) -> float:
    if isinstance(value, str):
        value = value.strip().replace(",", ".")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} must be a valid WGS84 decimal degree value."
        ) from exc


def validate_analysis_inputs(
    min_lon: object,
    min_lat: object,
    max_lon: object,
    max_lat: object,
    grid_size: object,
) -> tuple[float, float, float, float, int | float]:
    parsed_min_lon = parse_decimal_input(min_lon, "Minimum longitude")
    parsed_min_lat = parse_decimal_input(min_lat, "Minimum latitude")
    parsed_max_lon = parse_decimal_input(max_lon, "Maximum longitude")
    parsed_max_lat = parse_decimal_input(max_lat, "Maximum latitude")
    parsed_grid_size = parse_decimal_input(grid_size, "Grid cell size")

    validate_bbox(
        min_lon=parsed_min_lon,
        min_lat=parsed_min_lat,
        max_lon=parsed_max_lon,
        max_lat=parsed_max_lat,
    )
    validated_grid_size = validate_grid_size(parsed_grid_size)

    return (
        parsed_min_lon,
        parsed_min_lat,
        parsed_max_lon,
        parsed_max_lat,
        validated_grid_size,
    )


def build_run_signature(
    run_name: str,
    min_lon: float | None,
    min_lat: float | None,
    max_lon: float | None,
    max_lat: float | None,
    grid_size: float,
    mode: str,
    indicator: str | None = None,
    basemap: str | None = None,
) -> tuple[object, ...]:
    def coordinate(value: float | None) -> float | None:
        return None if value is None else round(float(value), 8)

    return (
        run_name,
        coordinate(min_lon),
        coordinate(min_lat),
        coordinate(max_lon),
        coordinate(max_lat),
        round(float(grid_size), 4),
        mode,
    )


def build_map_signature(
    run_signature: tuple[object, ...],
    indicator: str,
    basemap: str,
) -> tuple[object, ...]:
    return (
        *run_signature,
        indicator,
        basemap,
    )


def should_show_results(
    run_state: str,
    completed_signature: tuple[object, ...] | None,
    current_signature: tuple[object, ...],
) -> bool:
    return run_state == "completed" and completed_signature == current_signature


def should_show_map(
    run_state: str,
    completed_signature: tuple[object, ...] | None,
    map_signature: tuple[object, ...] | None,
    current_signature: tuple[object, ...],
) -> bool:
    return (
        should_show_results(run_state, completed_signature, current_signature)
        and map_signature is not None
        and map_signature[: len(current_signature)] == current_signature
    )


def friendly_error_summary(
    returncode: int,
    stderr: str,
    stdout: str = "",
) -> dict[str, str]:
    text = f"{stdout}\n{stderr}".lower()

    if "expects column" in text or "missing from the grid layer" in text:
        return {
            "reason": (
                "The selected indicator is not available in this run's grid output."
            ),
            "next_action": (
                "Choose GSI, or rerun with Standard or Full Context mode for "
                "height and contextual indicators where data are available."
            ),
        }

    if (
        "no buildings remain after clipping to aoi" in text
        or "no buildings remain after clipping" in text
    ):
        return {
            "reason": (
                "No buildings were found inside the selected analysis area using the "
                "workflow building source. Buildings visible on the basemap may "
                "come from a different source. Check the coordinates, enlarge "
                "the analysis area, or choose a more urban area."
            ),
            "next_action": (
                "Check longitude/latitude order, confirm that the preview "
                "rectangle covers the intended location, enlarge the analysis "
                "area, try Quick 2D first, or try a known urban area."
            ),
        }

    if (
        "no buildings found" in text
        or "empty building" in text
        or "building layer is empty" in text
        or "no usable buildings" in text
    ):
        return {
            "reason": "No usable building footprints were found for this analysis area.",
            "next_action": (
                "Check that the coordinates cover an urban area, enlarge the bbox, "
                "or try Quick 2D first."
            ),
        }

    if (
        "minimum longitude" in text
        or "maximum longitude" in text
        or "minimum latitude" in text
        or "maximum latitude" in text
        or "invalid bbox" in text
        or "coordinate" in text
    ):
        return {
            "reason": "The analysis-area coordinates look invalid.",
            "next_action": (
                "Use WGS84 decimal degrees and make sure min values are smaller "
                "than max values."
            ),
        }

    if (
        "download" in text
        or "connection" in text
        or "timeout" in text
        or "http" in text
        or "network" in text
        or "overture" in text
        or "s3" in text
    ):
        return {
            "reason": "External building data could not be downloaded or reached.",
            "next_action": (
                "Check the internet connection, try again later, or use a smaller "
                "analysis area first."
            ),
        }

    if (
        "found no graph nodes" in text
        or "osmnx" in text
        or "street_context" in text
        or "street-profile" in text
        or "street profile" in text
    ):
        return {
            "reason": "The street-profile/context branch could not be completed.",
            "next_action": (
                "Try Standard mode, reduce the analysis area, or run Quick 2D before using "
                "Full Context."
            ),
        }

    if (
        "gba" in text
        or "globalbuildingatlas" in text
        or "height_enrichment" in text
        or "height enrichment" in text
    ):
        return {
            "reason": "The height-enrichment branch could not be completed.",
            "next_action": (
                "Try Quick 2D, reduce the analysis area, or rerun later if external height "
                "tiles are unavailable."
            ),
        }

    return {
        "reason": f"The workflow stopped before completion (exit code {returncode}).",
        "next_action": (
            "Try a smaller analysis area or Quick 2D mode. Technical logs are available in "
            "Advanced / Developer details."
        ),
    }


def generated_config_path(
    run_name: str,
    config_dir: Path = GENERATED_CONFIG_DIR,
) -> Path:
    validated = validate_run_name(run_name)
    return config_dir / f"{validated}.yaml"


def read_completed_run_config(output_folder: Path) -> dict[str, object]:
    path = output_folder / "reports" / "config_used.yaml"
    if not path.exists():
        raise FileNotFoundError("The completed analysis configuration is unavailable.")
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("The completed analysis configuration is invalid.")
    return config


def build_grid_size_rerun_config(
    source_config: dict[str, object],
    source_run_name: str,
    new_run_name: str,
    grid_size_m: float,
) -> dict[str, object]:
    """Copy a completed run and change only grid-dependent output settings."""
    source_name = validate_run_name(source_run_name)
    target_name = validate_run_name(new_run_name)
    grid_size = validate_grid_size(grid_size_m)
    config = copy.deepcopy(source_config)
    if not isinstance(config.get("aoi"), dict):
        raise ValueError("The completed analysis has no reusable analysis-area definition.")

    project = config.setdefault("project", {})
    project.update(
        {
            "run_name": target_name,
            "output_dir": f"04_outputs/{target_name}",
            "overwrite_existing_run": True,
        }
    )
    aggregation = config.setdefault("aggregation", {})
    aggregation["cell_size_m"] = grid_size

    cache = config.setdefault("cache", {})
    cache.pop("source_output_dir", None)
    cache.update(GRID_RERUN_CACHE_SETTINGS)
    cache["source_output_name"] = source_name

    # Compact mode avoids copying large AOI-level artifacts into every grid-size
    # result. The compatible source run remains the authoritative reusable cache.
    outputs = config.setdefault("outputs", {})
    outputs.update(
        {
            "mode": "compact",
            "save_raw_buildings": False,
            "save_processed_buildings": False,
            "save_neighbor_diagnostics": False,
            "save_grid": True,
            "save_indicator_grid": True,
            "save_tables": True,
            "save_reports": True,
        }
    )
    visualization = config.setdefault("visualization", {})
    visualization["save_static_maps"] = False
    return config


def _manifest_bounds_tuple(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, dict):
        return None
    try:
        return (
            float(value["min_lon"]),
            float(value["min_lat"]),
            float(value["max_lon"]),
            float(value["max_lat"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _config_bbox(config: dict[str, object]) -> tuple[float, float, float, float] | None:
    aoi = config.get("aoi") or {}
    bounds = aoi.get("bounds") if isinstance(aoi, dict) else None
    if not isinstance(bounds, dict):
        return None
    try:
        return (
            float(bounds["minx"]),
            float(bounds["miny"]),
            float(bounds["maxx"]),
            float(bounds["maxy"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def grid_rerun_cache_compatibility(
    source_output_folder: Path,
    rerun_config: dict[str, object],
) -> dict[str, object]:
    """Perform a readable preflight; the workflow remains the final cache gate."""
    manifest_path = source_output_folder / "reports" / "cache_manifest.json"
    if not manifest_path.exists():
        return {
            "status": "manifest_missing",
            "reasons": ["The selected analysis has no cache compatibility manifest."],
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "manifest_invalid", "reasons": ["The cache manifest is invalid."]}

    reasons: list[str] = []
    config_bounds = _config_bbox(rerun_config)
    manifest_bounds = _manifest_bounds_tuple(manifest.get("aoi_bounds_wgs84"))
    if config_bounds is None or manifest_bounds is None:
        reasons.append("Analysis-area bounds are unavailable for compatibility checking.")
    elif any(abs(left - right) > 1e-9 for left, right in zip(config_bounds, manifest_bounds)):
        reasons.append("The selected analysis area does not match the cached analysis area.")

    data_source = rerun_config.get("data_source") or {}
    expected_data_source = {
        "type": data_source.get("type"),
        "provider": data_source.get("provider"),
        "release": data_source.get("release"),
        "exclude_underground": bool(data_source.get("exclude_underground", True)),
    }
    if expected_data_source != manifest.get("data_source"):
        reasons.append("Building source, release, provider, or underground setting differs.")

    preprocessing = rerun_config.get("preprocessing") or {}
    expected_preprocessing = {
        "target_crs": preprocessing.get("target_crs", "auto_utm"),
        "clip_to_aoi": preprocessing.get("clip_to_aoi", True),
    }
    if expected_preprocessing != manifest.get("preprocessing"):
        reasons.append("Preprocessing or CRS settings differ.")

    height = rerun_config.get("height_enrichment") or {}
    expected_height = {
        "enabled": bool(height.get("enabled", False)),
        "min_overlap_share": height.get("min_overlap_share"),
        "min_valid_height_m": height.get("min_valid_height_m"),
        "replace_existing_height": bool(height.get("replace_existing_height", False)),
    }
    if expected_height != manifest.get("height_enrichment"):
        reasons.append("Height-enrichment settings differ.")

    street = rerun_config.get("street_context") or {}
    expected_street = {
        "enabled": bool(street.get("enabled", False)),
        "source": street.get("source", "osmnx"),
        "network_type": street.get("network_type"),
        "distance_m": street.get("distance_m"),
        "tick_length_m": street.get("tick_length_m"),
        "topology_rule_version": street.get("topology_rule_version", 1),
    }
    if expected_street != manifest.get("street_context"):
        reasons.append("Street-context settings differ.")

    processing_mode = (rerun_config.get("crs_strategy") or {}).get(
        "processing_mode", "single_crs"
    )
    if processing_mode == "segmented_utm":
        reasons.append("External cache reuse is not supported for segmented UTM runs.")
    elif processing_mode != manifest.get("processing_mode", "single_crs"):
        reasons.append("CRS processing mode differs.")

    return {
        "status": "mismatch_detected" if reasons else "compatible",
        "reasons": reasons,
        "source_manifest": manifest,
    }


def grid_rerun_stage_plan(source_output_folder: Path) -> dict[str, object]:
    def available(relative_stem: str) -> bool:
        path = source_output_folder / relative_stem
        return path.exists() or path.with_suffix(".parquet").exists()

    return {
        "prepared_buildings": available("processed/buildings_height_enriched.gpkg")
        or available("processed/buildings_clean.gpkg"),
        "neighbour_context": available("processed/building_neighbor_diagnostics.gpkg"),
        "street_context": available("processed/building_street_profile_ratio.gpkg"),
        "street_network": available("processed/streets_osmnx.gpkg"),
        "street_profiles": available("processed/street_profile_segments.gpkg"),
        "recalculated": [
            "grid creation",
            "building-grid intersections",
            "grid aggregation and diagnostics",
            "statistics and dashboard overview",
        ],
    }


def grid_rerun_cache_has_required_stages(
    source_output_folder: Path,
    config: dict[str, object],
) -> bool:
    plan = grid_rerun_stage_plan(source_output_folder)
    indicators = config.get("indicators") or {}
    street = config.get("street_context") or {}
    if not plan["prepared_buildings"]:
        return False
    if indicators.get("neighbor_distance") and not plan["neighbour_context"]:
        return False
    if street.get("enabled") and not plan["street_context"]:
        return False
    return True


def resolve_grid_rerun_cache_source(
    selected_output_folder: Path,
    source_config: dict[str, object],
    outputs_root: Path = OUTPUTS_ROOT,
) -> Path:
    """Use the selected run or follow its recorded compatible cache source."""
    if grid_rerun_cache_has_required_stages(selected_output_folder, source_config):
        return selected_output_folder

    cache = source_config.get("cache") or {}
    candidates: list[Path] = []
    source_dir = cache.get("source_output_dir")
    if source_dir:
        path = Path(str(source_dir))
        candidates.append(path if path.is_absolute() else PROJECT_ROOT / path)
    source_name = cache.get("source_output_name")
    if source_name:
        candidates.append(outputs_root / str(source_name))
    for candidate in candidates:
        if grid_rerun_cache_has_required_stages(candidate, source_config):
            return candidate
    return selected_output_folder


def set_grid_rerun_cache_source(
    rerun_config: dict[str, object],
    cache_source_folder: Path,
    outputs_root: Path = OUTPUTS_ROOT,
) -> None:
    cache = rerun_config.setdefault("cache", {})
    cache.pop("source_output_name", None)
    cache.pop("source_output_dir", None)
    try:
        relative = cache_source_folder.resolve().relative_to(outputs_root.resolve())
    except ValueError:
        cache["source_output_dir"] = str(cache_source_folder)
    else:
        if len(relative.parts) == 1:
            cache["source_output_name"] = relative.name
        else:
            cache["source_output_dir"] = str(cache_source_folder)


def write_grid_size_rerun_config(
    config: dict[str, object],
    path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def output_folder_path(
    run_name: str,
    outputs_root: Path = OUTPUTS_ROOT,
) -> Path:
    validated = validate_run_name(run_name)
    return outputs_root / validated


def web_map_path(run_name: str, outputs_root: Path = OUTPUTS_ROOT) -> Path:
    return output_folder_path(run_name, outputs_root=outputs_root) / "webmap" / "index.html"


def available_indicators_for_output(output_folder: Path) -> list[str]:
    try:
        attributes = read_grid_attributes(output_folder)
    except (FileNotFoundError, ValueError):
        return []
    return available_indicators_from_columns(attributes.columns, attributes)


def indicator_has_mappable_values(grid: object, indicator: str, column: str) -> bool:
    if column not in grid.columns:
        return False
    if indicator == "street_profile_ratio":
        return bool(grid[column].notna().any())
    return True


def normalize_indicator_key(indicator: str) -> str:
    raw = str(indicator or "").strip()
    token = re.sub(r"[^a-z0-9_]+", " ", raw.lower()).strip()
    token = re.sub(r"\s+", " ", token)
    return APP_INDICATOR_ALIASES.get(raw, APP_INDICATOR_ALIASES.get(token, raw))


def grid_column_for_indicator(indicator: str) -> str:
    normalized = normalize_indicator_key(indicator)
    if normalized not in APP_INDICATORS:
        raise KeyError(f"Unsupported indicator: {indicator}")
    return str(APP_INDICATORS[normalized]["column"])


def expected_indicator_keys() -> list[str]:
    return list(APP_INDICATORS)


def indicator_columns_for_output(output_folder: Path) -> dict[str, str]:
    try:
        attributes = read_grid_attributes(output_folder)
    except (FileNotFoundError, ValueError):
        return {}

    return {
        indicator: str(definition["column"])
        for indicator, definition in APP_INDICATORS.items()
        if indicator in available_indicators_from_columns(attributes.columns, attributes)
    }


def sorted_indicators_with_gsi_first(indicators: list[str]) -> list[str]:
    unique_indicators = sorted(set(indicators))
    if "gsi" in unique_indicators:
        return ["gsi", *[item for item in unique_indicators if item != "gsi"]]
    return unique_indicators


def indicator_label(indicator: str) -> str:
    normalized = normalize_indicator_key(indicator)
    if normalized in APP_INDICATORS:
        return str(APP_INDICATORS[normalized]["label"])
    return indicator.replace("_", " ").title()


def available_indicator_labels_for_output(output_folder: Path) -> dict[str, str]:
    return {
        indicator: indicator_label(indicator)
        for indicator in sorted_indicators_with_gsi_first(
            available_indicators_for_output(output_folder)
        )
    }


def unavailable_indicators(available: list[str]) -> list[str]:
    return [
        indicator
        for indicator in sorted_indicators_with_gsi_first(expected_indicator_keys())
        if indicator not in set(available)
    ]


def indicator_options_for_output(output_folder: Path) -> list[str]:
    available = available_indicators_for_output(output_folder)
    if available:
        return sorted_indicators_with_gsi_first(available)
    return ["gsi"]


def indicator_options_for_mode_and_output(mode: str, output_folder: Path) -> list[str]:
    available = available_indicators_for_output(output_folder)
    if available:
        return sorted_indicators_with_gsi_first(available)
    if mode == "quick_2d":
        return ["gsi"]
    return sorted_indicators_with_gsi_first(expected_indicator_keys())


def default_indicator_for_mode(mode: str, available: list[str] | None = None) -> str:
    available = available or []
    if "gsi" in available or not available:
        return "gsi"
    return sorted_indicators_with_gsi_first(available)[0]


def mode_help_text(mode: str) -> str:
    return MODE_HELP[mode]


def read_indicator_readiness_records(output_folder: Path) -> list[dict[str, object]]:
    path = output_folder / "reports" / "indicator_readiness.json"
    if not path.exists():
        return []
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, dict)]


def read_building_source_summary(output_folder: Path) -> dict[str, object]:
    path = output_folder / "reports" / "building_source_summary.json"
    if not path.exists():
        return {}
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return summary if isinstance(summary, dict) else {}


def building_source_user_message(summary: dict[str, object]) -> str | None:
    actual_source = summary.get("actual_building_source_used") or summary.get(
        "actual_source_used"
    )
    requested_source = summary.get("requested_data_source") or {}
    if isinstance(requested_source, dict):
        source_name = str(requested_source.get("type") or "building source").title()
    else:
        source_name = "Building source"

    if actual_source == "new_download":
        return (
            f"Building data source: {source_name} buildings, newly loaded for "
            "this analysis area."
        )
    if actual_source == "compatible_raw_cache":
        return "Building data source: compatible cached buildings."
    if actual_source == "external_cache":
        return f"Building data source: {source_name}, compatible external cache."
    if actual_source == "compatible_enriched_cache":
        return f"Building data source: {source_name}, compatible processed cache."
    if actual_source == "failed":
        return "Building data source check failed."
    return None


def building_source_summary_allows_results(summary: dict[str, object]) -> bool:
    if not summary:
        return True
    actual_source = summary.get("actual_building_source_used") or summary.get(
        "actual_source_used"
    )
    if actual_source == "failed":
        return False
    overlap = summary.get("raw_bounds_overlap_requested_aoi")
    if overlap is None:
        overlap = summary.get("raw_bounds_overlap_selected_aoi")
    return overlap is not False


def readiness_records_by_indicator(
    records: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    lookup = {}
    for record in records:
        raw_indicator = str(record.get("indicator", ""))
        indicator_key = normalize_indicator_key(raw_indicator)
        if indicator_key in APP_INDICATORS:
            lookup[indicator_key] = record
    return lookup


def sanitize_main_ui_text(text: object) -> str:
    sanitized = str(text or "")
    replacements = {
        "Use strict street-profile ratios for interpretation; preliminary ratios are diagnostic only.": (
            "Street-profile height-to-width ratio is available only where both "
            "building height and street-profile width are available. Missing "
            "areas indicate insufficient input data, not low values."
        ),
        "Preliminary ratios are diagnostic only; strict ratios should guide interpretation.": (
            "Street-profile height-to-width ratio is available only where both "
            "building height and street-profile width are available."
        ),
        "Strict street-profile grid-cell coverage": "Reliable street-profile grid-cell coverage",
        "strict street-profile grid-cell coverage": "reliable street-profile grid-cell coverage",
        "strict street-profile ratio coverage": "reliable street-profile coverage",
        "strict-ratio coverage": "reliable coverage",
        "strict ratios": "reliable values",
        "Strict ratios": "Reliable values",
        "preliminary ratios": "supporting values",
        "Preliminary ratios": "Supporting values",
        "diagnostic only": "not used for main interpretation",
    }
    for old, new in replacements.items():
        sanitized = sanitized.replace(old, new)
    return sanitized


def street_profile_summary_line(record: dict[str, object]) -> str:
    status = str(record.get("status") or "Status not reported")
    status_reason = sanitize_main_ui_text(record.get("status_reason", ""))
    coverage_explanation = sanitize_main_ui_text(
        record.get("coverage_explanation")
        or (
            "Street-profile height-to-width ratio is available only where both "
            "building height and street-profile width are available. Missing "
            "areas indicate insufficient input data, not low values."
        )
    )
    if status in {"OK", "LIMITED"}:
        availability = "Available" if status == "OK" else "Available with limitations"
        parts = [
            f"**Street-profile height-to-width ratio:** {availability}",
            "Use: contextual height-width morphology indicator.",
            coverage_explanation,
        ]
        if status_reason:
            parts.append(f"Data completeness: {status_reason}")
        return " ".join(parts)

    if status == "WEAK":
        return (
            "**Street-profile height-to-width ratio:** Limited - Reliable "
            "street-profile coverage is too low for confident interpretation in "
            f"this run. {coverage_explanation}"
        )

    if status == "DO_NOT_INTERPRET":
        return (
            "**Street-profile height-to-width ratio:** Not available for "
            "interpretation - Reliable street-profile coverage is too low for "
            f"this run. Do not use this indicator for conclusions. {coverage_explanation}"
        )

    if status == "NOT_AVAILABLE":
        return (
            "**Street-profile height-to-width ratio:** Not calculated in this run."
        )

    return (
        "**Street-profile height-to-width ratio:** Status not reported - Review "
        "the full report."
    )


def main_readiness_summary_lines(
    records: list[dict[str, object]],
    available: list[str],
) -> list[str]:
    lookup = readiness_records_by_indicator(records)
    lines = []
    for indicator in sorted_indicators_with_gsi_first(available):
        record = lookup.get(indicator, {})
        if indicator == "street_profile_ratio":
            lines.append(street_profile_summary_line(record))
            continue
        status = record.get("status", "Status not reported")
        recommended_use = sanitize_main_ui_text(
            record.get("recommended_use") or "Review the full report."
        )
        lines.append(f"**{indicator_label(indicator)}:** {status} - {recommended_use}")
    return lines


def main_warning_lines(
    records: list[dict[str, object]],
    available: list[str],
) -> list[str]:
    lookup = readiness_records_by_indicator(records)
    warnings = []
    for indicator in sorted_indicators_with_gsi_first(available):
        record = lookup.get(indicator, {})
        status = record.get("status")
        key_warnings = record.get("key_warnings") or []
        if status == "DO_NOT_INTERPRET":
            do_not_interpret = record.get("do_not_interpret_reason")
            if do_not_interpret:
                warnings.append(
                    f"{indicator_label(indicator)}: {sanitize_main_ui_text(do_not_interpret)}"
                )
        if isinstance(key_warnings, list):
            warnings.extend(
                f"{indicator_label(indicator)}: {sanitize_main_ui_text(warning)}"
                for warning in key_warnings
            )
    return warnings


def main_summary_contains_technical_sections(summary_text: str) -> bool:
    technical_terms = [
        "Aggregation unit quality",
        "Cache diagnostics",
        "workflow_summary",
        "Generated YAML",
        "technical logs",
    ]
    return any(term.lower() in summary_text.lower() for term in technical_terms)


def indicator_unavailable_message(indicator: str, mode: str) -> str:
    if mode == "quick_2d" or indicator != "gsi":
        return MISSING_INDICATOR_MESSAGE
    return (
        f"This indicator is not available for the selected run: `{indicator}`. "
        "Choose one of the available indicators or rerun with a mode that "
        "calculates it."
    )


def can_export_indicator(indicator: str, available: list[str], mode: str) -> tuple[bool, str | None]:
    if indicator in available:
        return True, None
    return False, indicator_unavailable_message(indicator, mode)


def user_result_messages(
    analysis_completed: bool,
    map_created: bool,
) -> list[str]:
    messages = []
    if analysis_completed:
        messages.append(USER_COMPLETION_MESSAGES["analysis_completed"])
        messages.append(USER_COMPLETION_MESSAGES["results_saved"])
    if map_created:
        messages.append(USER_COMPLETION_MESSAGES["map_created"])
    return messages


def build_workflow_command(config_path: Path) -> list[str]:
    return [
        sys.executable,
        str(CODE_DIR / "run_workflow.py"),
        str(config_path),
    ]


def build_export_map_command(
    output_folder: Path,
    indicator: str,
    basemap: str,
) -> list[str]:
    return [
        sys.executable,
        str(SCRIPTS_DIR / "export_web_map.py"),
        "--output-folder",
        str(output_folder),
        "--indicator",
        indicator,
        "--basemap",
        basemap,
    ]


def preflight_warnings(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    grid_size: float,
    mode: str,
) -> list[str]:
    warnings = []
    size = estimate_bbox_size_km(
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
    )

    if size["width_km"] < 0.05 or size["height_km"] < 0.05:
        warnings.append(
            "The selected analysis area is very small. It may contain no buildings after clipping; "
            "preview the analysis area and enlarge the selected area if needed."
        )

    if grid_size < 50:
        warnings.append(
            "Grid size is small; this can create many grid cells and slow down "
            "processing and web-map export."
        )

    if mode == "full_context":
        warnings.append(
            "full_context mode can be slow because street-profile and contextual "
            "branches may require additional processing."
        )

    return warnings


def run_command(command: list[str], cwd: Path = PROJECT_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def read_text_if_exists(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def read_bytes_if_exists(path: Path) -> bytes | None:
    if not path.exists():
        return None
    return path.read_bytes()


def dataframe_preview(
    dataframe: object,
    max_rows: int = TABLE_PREVIEW_ROW_LIMIT,
) -> tuple[object, int, bool]:
    """Return a row-limited, non-geometry table preview."""
    geometry_name = getattr(dataframe, "geometry", None)
    geometry_col = getattr(geometry_name, "name", None)
    drop_cols = []
    if geometry_col is not None and geometry_col in dataframe.columns:
        drop_cols.append(geometry_col)
    elif "geometry" in dataframe.columns:
        drop_cols.append("geometry")
    preview = dataframe.drop(columns=drop_cols, errors="ignore").head(max_rows).copy()
    total_rows = int(len(dataframe))
    return preview, total_rows, total_rows > max_rows


def deterministic_feature_preview(
    geodataframe: object,
    max_features: int = BUILDING_MAP_PREVIEW_MAX_FEATURES,
    id_column: str = "building_id",
) -> object:
    """Return a deterministic display-only feature sample without mutating input."""
    if len(geodataframe) <= max_features:
        return geodataframe.copy()
    key = geodataframe[id_column].astype(str) if id_column in geodataframe.columns else geodataframe.index.astype(str)
    hashed = key.apply(
        lambda value: int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:16], 16)
    )
    selected_index = hashed.sort_values(kind="mergesort").head(max_features).index
    return geodataframe.loc[selected_index].copy()


def map_display_mode(
    n_grid_cells: int | None,
    threshold: int = INTERACTIVE_GRID_CELL_THRESHOLD,
) -> str:
    if n_grid_cells is None:
        return "interactive"
    return "interactive" if int(n_grid_cells) <= threshold else "static_preview"


def should_embed_html_file(
    html_path: Path,
    max_mb: float = LARGE_HTML_EMBED_LIMIT_MB,
) -> bool:
    return html_path.exists() and (html_path.stat().st_size / 1024 / 1024) <= max_mb


def html_size_mb(html_path: Path) -> float | None:
    if not html_path.exists():
        return None
    return html_path.stat().st_size / 1024 / 1024


def read_workflow_summary(output_folder: Path) -> dict[str, object]:
    path = output_folder / "reports" / "workflow_summary.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def read_gsi_sanity_summary(output_folder: Path) -> dict[str, object]:
    path = output_folder / "reports" / "gsi_sanity_summary.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def grid_cell_count_for_output(output_folder: Path) -> int | None:
    summary = read_workflow_summary(output_folder)
    value = summary.get("n_grid_cells")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def format_percent(value: object) -> str:
    if value is None:
        return "not reported"
    try:
        if math.isnan(float(value)):
            return "not reported"
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "not reported"


def large_run_warning_needed(summary: dict[str, object]) -> bool:
    n_buildings = int(summary.get("n_buildings") or 0)
    n_grid_cells = int(summary.get("n_grid_cells") or 0)
    return n_buildings > BUILDING_MAP_PREVIEW_MAX_FEATURES or n_grid_cells > INTERACTIVE_GRID_CELL_THRESHOLD


def large_run_quality_lines(
    summary: dict[str, object],
    readiness_records: list[dict[str, object]],
    gsi_sanity: dict[str, object],
) -> list[str]:
    readiness_lookup = {
        str(record.get("indicator")): str(record.get("status"))
        for record in readiness_records
        if isinstance(record, dict)
    }
    lines = [
        f"Processed buildings: {int(summary.get('n_buildings') or 0):,}",
        f"Grid cells: {int(summary.get('n_grid_cells') or 0):,}",
        f"Street segments: {int(summary.get('street_profile_n_street_segments') or 0):,}",
        (
            "Height coverage after enrichment: "
            f"{format_percent(summary.get('height_valid_share_after_enrichment'))} "
            "of buildings and "
            f"{format_percent(summary.get('height_valid_area_share_after_enrichment'))} "
            "of footprint area."
        ),
        (
            "Floor-count missing share: "
            f"{format_percent(summary.get('missing_num_floors_share'))}; "
            "floor-valid footprint-area share: "
            f"{format_percent(summary.get('floor_valid_area_share'))}."
        ),
        (
            "FAR/FSI readiness: "
            f"{readiness_lookup.get('FAR/FSI', 'not reported')}."
        ),
        (
            "Street-profile valid grid-cell share: "
            f"{format_percent(summary.get('street_profile_valid_grid_cell_share'))}."
        ),
        (
            "Zero building-level neighbour-distance share: "
            f"{format_percent(summary.get('zero_neighbor_distance_share_building_level'))}."
        ),
        (
            "Raw GSI > 1 cells: "
            f"{int(summary.get('cells_with_gsi_over_1') or gsi_sanity.get('cells_with_gsi_over_1') or 0)} "
            "(see `reports/gsi_sanity_summary.json`)."
        ),
    ]
    return lines


def concise_output_files(output_folder: Path) -> list[str]:
    candidates = [
        "reports/indicator_readiness.md",
        "reports/quality_report.md",
        "reports/workflow_summary.json",
        "reports/gsi_sanity_summary.json",
        "indicators/grid_indicators.gpkg",
        "processed/buildings_height_enriched.gpkg",
    ]
    return [item for item in candidates if (output_folder / item).exists()]


def static_preview_path(
    output_folder: Path,
    indicator: str,
) -> Path:
    return output_folder / "webmap" / f"static_preview_{indicator}.png"


def create_static_grid_preview(
    output_folder: Path,
    indicator: str,
    output_path: Path | None = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    column = grid_column_for_indicator(indicator)
    grid, _grid_path = load_grid_layer(
        output_folder=output_folder,
        max_features=None,
        simplify_tolerance=None,
    )
    if column not in grid.columns:
        raise ValueError(f"Selected indicator column is missing: {column}")
    if output_path is None:
        output_path = static_preview_path(output_folder, indicator)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_grid = grid[[column, "geometry"]].copy()
    fig, ax = plt.subplots(figsize=(9, 8), dpi=130)
    style = resolved_style(style_for_key(indicator), plot_grid[column])
    valid = plot_grid[plot_grid[column].notna()]
    missing = plot_grid[plot_grid[column].isna()]
    if not missing.empty:
        missing.plot(ax=ax, color=style.missing_color, edgecolor="none")
    if not valid.empty:
        valid["_style_color"] = valid[column].map(lambda value: color_for_value(style, value))
        valid.plot(ax=ax, color=valid["_style_color"], linewidth=0)
    ax.set_axis_off()
    ax.set_title(f"{indicator_label(indicator)} preview")
    fig.text(
        0.02,
        0.02,
        (
            "Cartographic display preview only. Full-resolution indicator "
            "outputs are preserved in the run output directory."
        ),
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def grid_indicator_path(output_folder: Path) -> Path:
    standard = output_folder / "indicators" / "grid_indicators.gpkg"
    segmented = output_folder / "indicators" / "grid_indicators_segmented_wgs84.gpkg"
    if standard.exists():
        return standard
    if segmented.exists():
        return segmented
    raise FileNotFoundError("No completed grid-indicator layer was found for this analysis.")


@lru_cache(maxsize=8)
def _grid_info_cached(path_text: str, modified_ns: int) -> dict[str, object]:
    import pyogrio

    info = pyogrio.read_info(path_text)
    return {
        "crs": str(info.get("crs") or ""),
        "feature_count": int(info.get("features") or 0),
        "fields": [str(field) for field in info.get("fields", [])],
    }


def grid_layer_info(output_folder: Path) -> dict[str, object]:
    path = grid_indicator_path(output_folder)
    info = _grid_info_cached(str(path), path.stat().st_mtime_ns)
    return {**info, "path": path}


@lru_cache(maxsize=8)
def _read_grid_attributes_cached(path_text: str, modified_ns: int):
    import pyogrio

    return pyogrio.read_dataframe(path_text, read_geometry=False)


def read_grid_attributes(output_folder: Path):
    path = grid_indicator_path(output_folder)
    return _read_grid_attributes_cached(str(path), path.stat().st_mtime_ns)


def available_indicators_from_columns(columns: object, attributes: object | None = None) -> list[str]:
    column_names = set(str(column) for column in columns)
    available = []
    for indicator, definition in APP_INDICATORS.items():
        column = str(definition["column"])
        if column not in column_names:
            continue
        if attributes is not None and indicator == "street_profile_ratio":
            if not bool(attributes[column].notna().any()):
                continue
        available.append(indicator)
    return sorted_indicators_with_gsi_first(available)


def dashboard_available_indicators(output_folder: Path) -> list[str]:
    attributes = read_grid_attributes(output_folder)
    return available_indicators_from_columns(attributes.columns, attributes)


def indicator_definition(indicator: str) -> dict[str, object]:
    normalized = normalize_indicator_key(indicator)
    if normalized not in APP_INDICATORS:
        raise KeyError(f"Unsupported indicator: {indicator}")
    return APP_INDICATORS[normalized]


def indicator_unit(indicator: str) -> str:
    return str(indicator_definition(indicator)["unit"])


def readiness_display_status(
    record: dict[str, object] | None,
    indicator: str | None = None,
) -> str:
    """Return one of the five user-facing readiness labels."""
    record = record or {}
    status = str(record.get("status") or "").strip().upper().replace(" ", "_")
    if status in {"OK", "LIMITED", "WEAK"}:
        return status
    if status in {"NOT_AVAILABLE", "DO_NOT_INTERPRET", "UNAVAILABLE"}:
        return "UNAVAILABLE"
    return "NOT REPORTED"


def readiness_interpretability(status: str) -> str:
    messages = {
        "OK": "This indicator can be interpreted for this run.",
        "LIMITED": "Interpret this indicator with caution.",
        "WEAK": "Use this indicator only as supporting information.",
        "UNAVAILABLE": "Do not interpret this indicator for this run.",
        "NOT REPORTED": "No formal readiness category is currently available.",
    }
    return messages[status]


def analysis_run_display_status(run_state: object) -> str:
    """Normalize internal run state to a concise user-facing status."""
    normalized = str(run_state or "").strip().lower()
    return {
        "running": "PROCESSING",
        "completed": "COMPLETED",
        "failed": "FAILED",
    }.get(normalized, "UNAVAILABLE")


def _finite_numeric_series(values: object):
    import numpy as np
    import pandas as pd

    numeric = pd.to_numeric(values, errors="coerce")
    return numeric[np.isfinite(numeric)]


def weighted_aoi_indicator_value(attributes: object, indicator: str) -> float | None:
    import numpy as np
    import pandas as pd

    denominator = pd.to_numeric(attributes.get("unit_area_m2"), errors="coerce")
    valid_denominator = denominator[np.isfinite(denominator) & (denominator > 0)]
    if valid_denominator.empty:
        return None

    if indicator == "gsi":
        numerator = pd.to_numeric(
            attributes.get("building_footprint_area_m2"), errors="coerce"
        ).fillna(0)
        return float(numerator.sum() / valid_denominator.sum())
    if indicator == "far":
        numerator = pd.to_numeric(attributes.get("floor_area_sum_m2"), errors="coerce")
        return float(numerator.fillna(0).sum() / valid_denominator.sum())
    if indicator == "built_volume_density":
        numerator = pd.to_numeric(attributes.get("built_volume_m3"), errors="coerce")
        return float(numerator.fillna(0).sum() / valid_denominator.sum())

    definition = indicator_definition(indicator)
    values = pd.to_numeric(attributes.get(str(definition["column"])), errors="coerce")
    if indicator == "neighbour_distance":
        weights = pd.to_numeric(
            attributes.get("neighbor_distance_valid_count"), errors="coerce"
        ).fillna(0)
    else:
        weights = pd.to_numeric(
            attributes.get("street_profile_ratio_strict_valid_count"), errors="coerce"
        ).fillna(0)
    valid = np.isfinite(values) & (weights > 0)
    if not valid.any():
        return None
    return float((values[valid] * weights[valid]).sum() / weights[valid].sum())


def aoi_indicator_summary(
    output_folder: Path,
    indicator: str,
    readiness_records: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    attributes = read_grid_attributes(output_folder)
    definition = indicator_definition(indicator)
    column = str(definition["column"])
    if column not in attributes.columns:
        raise ValueError(f"{definition['label']} was not calculated for this run.")
    valid = _finite_numeric_series(attributes[column])
    total = int(len(attributes))
    valid_count = int(len(valid))
    missing_count = total - valid_count
    readiness = readiness_records_by_indicator(readiness_records or []).get(indicator, {})
    return {
        "indicator": indicator,
        "label": definition["label"],
        "unit": definition["unit"],
        "weighted_value": weighted_aoi_indicator_value(attributes, indicator),
        "median": float(valid.median()) if valid_count else None,
        "p10": float(valid.quantile(0.10)) if valid_count else None,
        "p90": float(valid.quantile(0.90)) if valid_count else None,
        "valid_count": valid_count,
        "valid_share": valid_count / total if total else 0.0,
        "missing_count": missing_count,
        "missing_share": missing_count / total if total else 0.0,
        "readiness": str(readiness.get("status") or "Not reported"),
        "readiness_reason": sanitize_main_ui_text(readiness.get("status_reason") or ""),
    }


def relative_aoi_interpretation(value: object, valid_values: object) -> dict[str, object] | None:
    import numpy as np

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric_value):
        return None
    valid = _finite_numeric_series(valid_values)
    if valid.empty:
        return None
    percentile = float((valid <= numeric_value).mean() * 100)
    if percentile <= 10:
        phrase = "Low relative to this analysis area"
    elif percentile < 45:
        phrase = "Below the analysis-area median"
    elif percentile <= 55:
        phrase = "Around the analysis-area median"
    elif percentile < 90:
        phrase = "Above the analysis-area median"
    else:
        phrase = "High relative to this analysis area"
    return {
        "phrase": phrase,
        "percentile": percentile,
        "median": float(valid.median()),
    }


def format_indicator_value(value: object, indicator: str) -> str:
    import numpy as np

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "Missing"
    if not np.isfinite(numeric):
        return "Missing"
    unit = indicator_unit(indicator)
    if unit == "m":
        return f"{numeric:.2f} m"
    if unit == "m3/m2":
        return f"{numeric:.3f} m3/m2"
    return f"{numeric:.3f}"


def cell_indicator_state(cell: dict[str, object], indicator: str) -> dict[str, str]:
    import numpy as np

    definition = indicator_definition(indicator)
    column = str(definition["column"])
    if column not in cell:
        return {"state": "unavailable", "display": "Indicator unavailable"}
    value = cell.get(column)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = math.nan
    if np.isfinite(numeric):
        if numeric == 0:
            return {
                "state": "zero",
                "display": f"{format_indicator_value(numeric, indicator)} (measured zero)",
            }
        return {"state": "value", "display": format_indicator_value(numeric, indicator)}

    building_area = cell.get("building_footprint_area_m2")
    try:
        has_buildings = float(building_area) > 0
    except (TypeError, ValueError):
        has_buildings = False
    if indicator in {"far", "built_volume_density", "street_profile_ratio"} and has_buildings:
        return {
            "state": "insufficient_input",
            "display": "Insufficient input data (not zero)",
        }
    return {"state": "missing", "display": "Missing value (not zero)"}


def selected_cell_popup_payload(
    cell: dict[str, object] | None,
    indicator: str,
) -> dict[str, str] | None:
    if cell is None:
        return None
    state = cell_indicator_state(cell, indicator)
    state_labels = {
        "value": "Valid result",
        "zero": "Zero value",
        "missing": "Missing value",
        "insufficient_input": "Insufficient input data",
        "unavailable": "Indicator unavailable",
    }
    column = str(indicator_definition(indicator)["column"])
    if state["state"] in {"value", "zero"}:
        value = format_indicator_value(cell.get(column), indicator)
        if indicator_unit(indicator) == "ratio":
            value = f"{value} ratio"
    else:
        value = "Not available"
    return {
        "indicator": indicator_label(indicator),
        "value": value,
        "state": state_labels[state["state"]],
    }


def selected_cell_card(
    cell: dict[str, object],
    readiness_records: list[dict[str, object]],
) -> dict[str, object]:
    readiness = readiness_records_by_indicator(readiness_records)
    values = {
        indicator: cell_indicator_state(cell, indicator)
        for indicator in expected_indicator_keys()
    }
    warnings = []
    try:
        if float(cell.get("gsi")) > 1:
            warnings.append(
                "GSI is above 1 in this cell. Review overlapping-footprint and geometry diagnostics before interpretation."
            )
    except (TypeError, ValueError):
        pass
    if readiness.get("far", {}).get("status") in {"WEAK", "DO_NOT_INTERPRET"}:
        warnings.append(
            "FAR/FSI has weak floor-data support for this run; avoid strong conclusions from this value."
        )
    if values["street_profile_ratio"]["state"] != "value":
        warnings.append(
            "A blank street-profile value means insufficient height or street-profile input data, not a low ratio."
        )
    return {
        "unit_id": str(cell.get("unit_id") or "Not reported"),
        "partial_cell": bool(cell.get("is_partial_cell")),
        "values": values,
        "floor_valid_area_share": cell.get("floor_data_valid_area_share"),
        "height_valid_area_share": cell.get("height_valid_area_share"),
        "neighbor_valid_count": cell.get("neighbor_distance_valid_count"),
        "street_profile_valid_count": cell.get("street_profile_ratio_strict_valid_count"),
        "warnings": warnings,
    }


def _transform_wgs84_bounds_to_crs(
    bounds: tuple[float, float, float, float], target_crs: str
) -> tuple[float, float, float, float]:
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    return tuple(float(value) for value in transformer.transform_bounds(*bounds))


def normalize_map_bounds(bounds: object) -> tuple[float, float, float, float] | None:
    if not isinstance(bounds, dict):
        return None
    southwest = bounds.get("_southWest") or bounds.get("southWest")
    northeast = bounds.get("_northEast") or bounds.get("northEast")
    if not isinstance(southwest, dict) or not isinstance(northeast, dict):
        return None
    try:
        return (
            float(southwest["lng"]),
            float(southwest["lat"]),
            float(northeast["lng"]),
            float(northeast["lat"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def load_viewport_grid(
    output_folder: Path,
    viewport_bounds_wgs84: tuple[float, float, float, float],
    indicator: str,
    max_features: int = DETAIL_GRID_CELL_LIMIT,
) -> tuple[object, bool]:
    import geopandas as gpd
    import pyogrio

    info = grid_layer_info(output_folder)
    path = Path(info["path"])
    crs = str(info["crs"])
    source_bbox = _transform_wgs84_bounds_to_crs(viewport_bounds_wgs84, crs)
    column = grid_column_for_indicator(indicator)
    columns = [
        "unit_id",
        column,
        "is_partial_cell",
        "floor_data_valid_area_share",
        "height_valid_area_share",
    ]
    available_columns = [column_name for column_name in columns if column_name in set(info["fields"])]
    subset = pyogrio.read_dataframe(
        path,
        bbox=source_bbox,
        columns=available_columns,
        max_features=max_features + 1,
    )
    too_many = len(subset) > max_features
    if too_many:
        return gpd.GeoDataFrame(columns=[*available_columns, "geometry"], crs=crs), True
    return subset.to_crs("EPSG:4326"), False


def query_grid_cell_at_location(
    output_folder: Path,
    longitude: float,
    latitude: float,
) -> dict[str, object] | None:
    import pyogrio
    from pyproj import Transformer
    from shapely.geometry import Point

    info = grid_layer_info(output_folder)
    path = Path(info["path"])
    crs = str(info["crs"])
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = transformer.transform(float(longitude), float(latitude))
    point = Point(x, y)
    candidates = pyogrio.read_dataframe(
        path,
        bbox=(x - 0.05, y - 0.05, x + 0.05, y + 0.05),
    )
    if candidates.empty:
        return None
    containing = candidates[candidates.geometry.apply(lambda geometry: geometry.covers(point))]
    if containing.empty:
        return None
    containing = containing.assign(_area=containing.geometry.area).sort_values(
        ["_area", "unit_id"], kind="stable"
    )
    row = containing.iloc[0].drop(labels=["geometry", "_area"], errors="ignore")
    return row.to_dict()


@lru_cache(maxsize=512)
def _query_grid_cell_by_id_cached(
    path_text: str,
    modified_ns: int,
    unit_id: str,
) -> dict[str, object] | None:
    import pyogrio

    safe_unit_id = unit_id.replace("'", "''")
    cells = pyogrio.read_dataframe(
        path_text,
        where=f"unit_id = '{safe_unit_id}'",
        max_features=1,
    )
    if cells.empty:
        return None
    row = cells.iloc[0].drop(labels=["geometry"], errors="ignore")
    return row.to_dict()


def query_grid_cell_by_id(
    output_folder: Path,
    unit_id: str | None,
) -> dict[str, object] | None:
    if not unit_id:
        return None
    path = grid_indicator_path(output_folder)
    value = _query_grid_cell_by_id_cached(
        str(path), path.stat().st_mtime_ns, str(unit_id)
    )
    return dict(value) if value is not None else None


@lru_cache(maxsize=512)
def _query_grid_cell_feature_by_id_cached(
    path_text: str,
    modified_ns: int,
    unit_id: str,
) -> dict[str, object] | None:
    import geopandas as gpd
    import pyogrio
    from shapely.geometry import mapping

    safe_unit_id = unit_id.replace("'", "''")
    cells = pyogrio.read_dataframe(
        path_text,
        where=f"unit_id = '{safe_unit_id}'",
        columns=["unit_id"],
        max_features=1,
    )
    if cells.empty:
        return None
    if cells.crs is None:
        raise ValueError("The selected grid cell has no CRS.")
    geometry = gpd.GeoSeries([cells.geometry.iloc[0]], crs=cells.crs).to_crs("EPSG:4326").iloc[0]
    point = geometry.representative_point()
    return {
        "unit_id": str(cells.iloc[0]["unit_id"]),
        "geometry": mapping(geometry),
        "location": [float(point.y), float(point.x)],
        "bounds_wgs84": [float(value) for value in geometry.bounds],
        "source_crs": str(cells.crs),
    }


def query_grid_cell_feature_by_id(
    output_folder: Path,
    unit_id: str | None,
) -> dict[str, object] | None:
    """Load one selected cell geometry for the browser map."""
    if not unit_id:
        return None
    path = grid_indicator_path(output_folder)
    value = _query_grid_cell_feature_by_id_cached(
        str(path), path.stat().st_mtime_ns, str(unit_id)
    )
    return dict(value) if value is not None else None


def overview_asset_paths(output_folder: Path, indicator: str) -> tuple[Path, Path]:
    folder = output_folder / "webmap" / "dashboard"
    return folder / f"overview_{indicator}.png", folder / f"overview_{indicator}.json"


def create_indicator_overview(
    output_folder: Path,
    indicator: str,
    max_pixels: int = OVERVIEW_MAX_PIXELS,
) -> dict[str, object]:
    import numpy as np
    import pyogrio
    from PIL import Image
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    info = grid_layer_info(output_folder)
    path = Path(info["path"])
    column = grid_column_for_indicator(indicator)
    grid = pyogrio.read_dataframe(path, columns=[column])
    if grid.empty:
        raise ValueError("The grid layer is empty.")
    if grid.crs is None:
        raise ValueError("The grid layer has no CRS.")
    source_crs = str(grid.crs)
    # Leaflet places image overlays in longitude/latitude bounds. Rasterizing
    # projected UTM cells and only transforming the outer bounds distorts the
    # image relative to exact vector cells, especially across a large extent.
    # Reproject the temporary display copy before rasterization instead.
    grid = grid.to_crs("EPSG:4326")
    style = resolved_style(style_for_key(indicator), grid[column])
    values = _finite_numeric_series(grid[column])
    if values.empty:
        raise ValueError(f"{indicator_label(indicator)} has no valid values.")
    minx, miny, maxx, maxy = (float(value) for value in grid.total_bounds)
    width_units = max(maxx - minx, 1.0)
    height_units = max(maxy - miny, 1.0)
    if width_units >= height_units:
        width = max_pixels
        height = max(240, int(round(max_pixels * height_units / width_units)))
    else:
        height = max_pixels
        width = max(240, int(round(max_pixels * width_units / height_units)))
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    vmin = float(values.min())
    vmax = float(values.max())
    span = vmax - vmin

    category_codes = {"missing": 0, "zero": 1}
    category_codes.update({f"valid_{index}": index + 2 for index in range(len(style.valid_colors))})

    def cell_code(raw_value: object) -> int:
        from map_styles import classify_value
        return category_codes[classify_value(style, raw_value)]

    shapes = (
        (geometry, cell_code(value))
        for geometry, value in zip(grid.geometry, grid[column], strict=False)
        if geometry is not None and not geometry.is_empty
    )
    raster = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=-1,
        dtype="int16",
    )
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    outside = raster < 0
    missing = raster == 0
    zero = raster == 1
    rgba[missing] = np.array([217, 217, 217, 205], dtype=np.uint8)
    rgba[zero] = np.array([251, 251, 247, 255], dtype=np.uint8)
    for index, color in enumerate(style.valid_colors, start=2):
        valid_pixels = raster == index
        if valid_pixels.any():
            rgba[valid_pixels] = np.array([*bytes.fromhex(color[1:]), 255], dtype=np.uint8)
    rgba[outside] = np.array([0, 0, 0, 0], dtype=np.uint8)

    png_path, metadata_path = overview_asset_paths(output_folder, indicator)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(png_path, optimize=True)
    metadata = {
        "overview_format_version": OVERVIEW_FORMAT_VERSION,
        "indicator": indicator,
        "label": indicator_label(indicator),
        "unit": indicator_unit(indicator),
        "value_min": vmin,
        "value_max": vmax,
        "valid_count": int(len(values)),
        "missing_count": int(len(grid) - len(values)),
        "bounds_wgs84": [minx, miny, maxx, maxy],
        "source_crs": source_crs,
        "raster_crs": "EPSG:4326",
        "source_modified_ns": path.stat().st_mtime_ns,
        "display_only": True,
        "cartographic_style_version": CARTOGRAPHIC_STYLE_VERSION,
        "style_breaks": list(style.fixed_breaks or ()),
        "style_missing_color": style.missing_color,
        "style_zero_color": style.zero_color,
        "style_valid_colors": list(style.valid_colors),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {**metadata, "image_path": png_path}


def ensure_indicator_overview(output_folder: Path, indicator: str) -> dict[str, object]:
    png_path, metadata_path = overview_asset_paths(output_folder, indicator)
    source_path = grid_indicator_path(output_folder)
    if png_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {}
        if (
            metadata.get("source_modified_ns") == source_path.stat().st_mtime_ns
            and metadata.get("overview_format_version") == OVERVIEW_FORMAT_VERSION
            and metadata.get("raster_crs") == "EPSG:4326"
        ):
            return {**metadata, "image_path": png_path}
    return create_indicator_overview(output_folder, indicator)


def _value_color(value: object, indicator: str, values: object) -> str:
    """Return the registry color without altering the scientific value."""
    return color_for_value(resolved_style(style_for_key(indicator), values), value)


def build_results_map(
    output_folder: Path,
    indicator: str,
    zoom: int = 10,
    viewport_bounds_wgs84: tuple[float, float, float, float] | None = None,
    center_wgs84: tuple[float, float] | None = None,
    max_vector_cells: int = DETAIL_GRID_CELL_LIMIT,
    selected_cell_feature: dict[str, object] | None = None,
    selected_cell: dict[str, object] | None = None,
) -> tuple[object, dict[str, object]]:
    import folium

    overview = ensure_indicator_overview(output_folder, indicator)
    min_lon, min_lat, max_lon, max_lat = overview["bounds_wgs84"]
    if center_wgs84 is None:
        center = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]
    else:
        center = [float(center_wgs84[1]), float(center_wgs84[0])]
    map_object = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB positron")
    folium.raster_layers.ImageOverlay(
        image=str(overview["image_path"]),
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        opacity=0.72,
        name=indicator_label(indicator),
        interactive=False,
        cross_origin=False,
        zindex=2,
    ).add_to(map_object)
    folium.Rectangle(
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        color="#4b5563",
        weight=1,
        fill=False,
        tooltip="Analysis extent",
    ).add_to(map_object)

    detail_count = 0
    detail_status = "overview"
    if zoom >= DETAIL_ZOOM_THRESHOLD and viewport_bounds_wgs84 is not None:
        detail, too_many = load_viewport_grid(
            output_folder,
            viewport_bounds_wgs84,
            indicator,
            max_features=max_vector_cells,
        )
        if too_many:
            detail_status = "zoom_in"
        elif not detail.empty:
            definition = indicator_definition(indicator)
            column = str(definition["column"])
            label = str(definition["label"])
            display = detail[["unit_id", column, "geometry"]].copy()
            display = display.rename(columns={"unit_id": "Grid cell", column: label})
            minimum = float(overview["value_min"])
            maximum = float(overview["value_max"])

            def style_function(feature: dict[str, object]) -> dict[str, object]:
                value = feature.get("properties", {}).get(label)
                missing = value is None
                return {
                    "fillColor": _value_color(value, indicator, detail[column]),
                    "color": "#374151" if not missing else "#6b7280",
                    "weight": 0.7,
                    "fillOpacity": 0.72 if not missing else 0.45,
                    "dashArray": "4 3" if missing else None,
                }

            folium.GeoJson(
                display,
                name="Detailed grid cells",
                style_function=style_function,
                tooltip=folium.GeoJsonTooltip(
                    fields=["Grid cell", label],
                    aliases=["Grid cell", f"{label} ({indicator_unit(indicator)})"],
                    localize=True,
                ),
            ).add_to(map_object)
            detail_count = int(len(display))
            detail_status = "detail"
    popup_payload = selected_cell_popup_payload(selected_cell, indicator)
    if selected_cell_feature is not None and popup_payload is not None:
        folium.GeoJson(
            selected_cell_feature["geometry"],
            name="Selected cell",
            style_function=lambda _feature: {
                "fillColor": "#000000",
                "color": "#111827",
                "weight": 3,
                "fillOpacity": 0.06,
            },
        ).add_to(map_object)
        popup_html = (
            f"<strong>{escape(popup_payload['indicator'])}</strong><br>"
            f"{escape(popup_payload['value'])}<br>"
            f"{escape(popup_payload['state'])}"
        )
        folium.CircleMarker(
            location=selected_cell_feature["location"],
            radius=4,
            color="#111827",
            weight=2,
            fill=True,
            fill_color="#ffffff",
            fill_opacity=1,
            popup=folium.Popup(popup_html, max_width=280, show=True),
        ).add_to(map_object)
    folium.LayerControl(collapsed=True).add_to(map_object)
    return map_object, {
        "overview": overview,
        "detail_status": detail_status,
        "detail_count": detail_count,
        "browser_vector_limit": max_vector_cells,
        "selected_cell_displayed": selected_cell_feature is not None,
        "popup": popup_payload,
    }


def quality_issue_cards(
    summary: dict[str, object],
    readiness_records: list[dict[str, object]],
    gsi_summary: dict[str, object],
) -> list[dict[str, str]]:
    readiness = readiness_records_by_indicator(readiness_records)
    far_status = str(readiness.get("far", {}).get("status") or "Not reported")
    gsi_over_one = int(
        summary.get("cells_with_gsi_over_1")
        or gsi_summary.get("cells_with_gsi_over_1")
        or 0
    )
    return [
        {
            "title": "Building height coverage",
            "observed": (
                f"{format_percent(summary.get('height_valid_share_after_enrichment'))} of buildings and "
                f"{format_percent(summary.get('height_valid_area_share_after_enrichment'))} of footprint area have valid height."
            ),
            "affected": "Built Volume Density and Street-profile height-to-width ratio.",
            "guidance": "Blank height-dependent values represent missing input, not zero building height.",
            "unknown": "This coverage measure does not independently verify real-world height accuracy.",
        },
        {
            "title": "Floor-count coverage",
            "observed": (
                f"{format_percent(summary.get('missing_num_floors_share'))} of buildings lack floor counts; "
                f"{format_percent(summary.get('floor_valid_area_share'))} of footprint area has valid floor data."
            ),
            "affected": f"FAR/FSI readiness is {far_status}.",
            "guidance": "Use FAR/FSI only as weak supporting evidence for this run.",
            "unknown": "Missing floors are data gaps; their real values are not inferred as zero.",
        },
        {
            "title": "Street-profile coverage",
            "observed": (
                f"Valid values occur in {format_percent(summary.get('street_profile_valid_grid_cell_share'))} of grid cells."
            ),
            "affected": "Street-profile height-to-width ratio.",
            "guidance": "Blank cells indicate insufficient height or street-profile data, not low ratios.",
            "unknown": "Coarser mapped coverage would not improve the underlying input data.",
        },
        {
            "title": "Zero neighbour distances",
            "observed": (
                f"{format_percent(summary.get('zero_neighbor_distance_share_building_level'))} of building-level distances are zero."
            ),
            "affected": "Average neighbour distance.",
            "guidance": "Interpret zero as attached, touching, or overlapping urban fabric, not as missing data.",
            "unknown": "The diagnostic does not by itself separate every real attachment from every geometry overlap.",
        },
        {
            "title": "GSI geometry diagnostic",
            "observed": f"{gsi_over_one} grid cells have raw GSI above 1.",
            "affected": "GSI / Building Coverage Ratio.",
            "guidance": "Treat these cells as overlap or geometry diagnostics, not normal high coverage values.",
            "unknown": "The workflow reports the issue rather than hiding or clipping the raw result.",
        },
    ]


def indicator_quality_cards(
    summary: dict[str, object],
    readiness_records: list[dict[str, object]],
    gsi_summary: dict[str, object],
) -> list[dict[str, str]]:
    """Build one concise, user-facing quality card per indicator."""
    readiness = readiness_records_by_indicator(readiness_records)
    gsi_over_one = int(
        summary.get("cells_with_gsi_over_1")
        or gsi_summary.get("cells_with_gsi_over_1")
        or 0
    )
    limitations = {
        "gsi": (
            f"{gsi_over_one} grid cells have GSI above 1 and should be treated as geometry or overlap warnings."
            if gsi_over_one
            else str(APP_INDICATORS["gsi"]["limitation"])
        ),
        "far": (
            "Floor-count coverage is insufficient for confident interpretation. "
            "Use this indicator only as supporting information."
        ),
        "built_volume_density": (
            f"Valid height data cover {format_percent(summary.get('height_valid_area_share_after_enrichment'))} "
            "of building footprint area; missing height data remain unavailable rather than zero."
        ),
        "neighbour_distance": (
            "Interpret the result together with missing-cell coverage and the high share of zero "
            "building distances."
        ),
        "street_profile_ratio": (
            f"Valid values occur in {format_percent(summary.get('street_profile_valid_grid_cell_share'))} "
            "of grid cells. Blank cells indicate insufficient input data, not low values."
        ),
    }
    cards = []
    for indicator in expected_indicator_keys():
        record = readiness.get(indicator, {})
        status = readiness_display_status(record, indicator)
        cards.append(
            {
                "indicator": indicator,
                "label": indicator_label(indicator),
                "status": status,
                "interpretability": readiness_interpretability(status),
                "limitation": limitations[indicator],
            }
        )
    return cards


def main_interface_exposes_internal_paths(text: str) -> bool:
    internal_terms = [
        str(PROJECT_ROOT),
        "grid_indicators.gpkg",
        "workflow_summary.json",
        "cache_manifest.json",
        "04_outputs/",
        "04_outputs\\",
    ]
    lowered = text.lower()
    return any(term.lower() in lowered for term in internal_terms)


def initialize_dashboard_state(session_state: object) -> None:
    """Initialize UI state without assuming that a run already exists."""
    defaults = {
        "setup_run_name": "",
        "selected_completed_run": None,
        "_active_completed_run": None,
        "selected_indicator": "gsi",
        "selected_cell_id": None,
        "selected_click_coordinates": None,
        "active_page": "Analysis setup",
        "run_state": "not_started",
        "setup_input_changed": False,
        "show_aoi_preview": False,
        "bbox_min_lon": None,
        "bbox_min_lat": None,
        "bbox_max_lon": None,
        "bbox_max_lat": None,
        "selected_aoi_geometry": None,
        "aoi_drawing_payload": None,
        "aoi_area_km2": None,
        "aoi_validation_message": None,
        "aoi_draw_reset_token": 0,
        "setup_selection_mode": "Draw rectangle on map",
        "dashboard_grid_size": 100.0,
        "dashboard_mode": "quick_2d",
        "_pending_selected_run": None,
        "_pending_selected_indicator": None,
        "_pending_page": None,
        "_pending_reset_selected_cell": False,
        "_pending_navigation_token": None,
    }
    for key, value in defaults.items():
        session_state.setdefault(key, value)


def schedule_navigation(
    session_state: object,
    run_name: str | None,
    indicator: str | None = "gsi",
    page: str | None = "Results map",
) -> None:
    """Store one navigation request without touching widget-bound keys late."""
    session_state["_pending_selected_run"] = run_name
    session_state["_pending_selected_indicator"] = indicator
    session_state["_pending_page"] = page
    session_state["_pending_reset_selected_cell"] = True
    session_state["_pending_navigation_token"] = f"{run_name}:{indicator}:{page}"


def apply_pending_navigation_before_widgets(session_state: object) -> bool:
    """Apply and clear a one-shot request before Streamlit instantiates widgets."""
    token = session_state.get("_pending_navigation_token")
    if not token:
        return False
    pending_run = session_state.get("_pending_selected_run")
    pending_indicator = session_state.get("_pending_selected_indicator")
    pending_page = session_state.get("_pending_page")
    if pending_run is not None:
        session_state["selected_completed_run"] = pending_run
        session_state["_active_completed_run"] = pending_run
    if pending_indicator is not None:
        session_state["selected_indicator"] = pending_indicator
    if pending_page is not None:
        session_state["active_page"] = pending_page
    if session_state.get("_pending_reset_selected_cell"):
        session_state["selected_cell_id"] = None
        session_state["selected_click_coordinates"] = None
    for key in (
        "_pending_selected_run",
        "_pending_selected_indicator",
        "_pending_page",
        "_pending_reset_selected_cell",
        "_pending_navigation_token",
    ):
        session_state[key] = False if key == "_pending_reset_selected_cell" else None
    return True


def safe_output_folder_for_run(
    run_name: object,
    outputs_root: Path = OUTPUTS_ROOT,
) -> Path | None:
    text = str(run_name or "").strip()
    if not text:
        return None
    try:
        return output_folder_path(text, outputs_root=outputs_root)
    except ValueError:
        return None


def completed_run_names(outputs_root: Path = OUTPUTS_ROOT) -> list[str]:
    if not outputs_root.exists():
        return []
    names = []
    for folder in outputs_root.iterdir():
        if not folder.is_dir() or folder.name.startswith("_"):
            continue
        if not (folder / "reports" / "workflow_summary.json").exists():
            continue
        try:
            grid_indicator_path(folder)
        except FileNotFoundError:
            continue
        names.append(folder.name)
    return sorted(names)


def select_completed_run(session_state: object, run_name: str | None) -> bool:
    normalized = str(run_name).strip() if run_name else None
    changed = session_state.get("_active_completed_run") != normalized
    if changed:
        schedule_navigation(session_state, normalized)
    return changed


def persist_map_click_selection(
    session_state: object,
    click: dict[str, object] | None,
    cell: dict[str, object] | None,
) -> bool:
    """Persist a valid click; a missing event never clears the selection."""
    if not click or cell is None:
        return False
    if "lng" not in click or "lat" not in click or not cell.get("unit_id"):
        return False
    session_state["selected_cell_id"] = str(cell["unit_id"])
    session_state["selected_click_coordinates"] = (
        float(click["lng"]),
        float(click["lat"]),
    )
    return True


def clear_cell_selection(session_state: object) -> None:
    session_state["selected_cell_id"] = None
    session_state["selected_click_coordinates"] = None


def _initialize_dashboard_state(st: object) -> None:
    initialize_dashboard_state(st.session_state)


def _current_dashboard_signature(st: object) -> tuple[object, ...]:
    return build_run_signature(
        str(st.session_state.get("setup_run_name") or ""),
        st.session_state.get("bbox_min_lon"),
        st.session_state.get("bbox_min_lat"),
        st.session_state.get("bbox_max_lon"),
        st.session_state.get("bbox_max_lat"),
        float(st.session_state["dashboard_grid_size"]),
        str(st.session_state["dashboard_mode"]),
    )


def _invalidate_results_after_input_change(st: object) -> None:
    current_signature = _current_dashboard_signature(st)
    previous = st.session_state.get("setup_signature")
    if input_signature_changed(previous, current_signature):
        st.session_state["setup_input_changed"] = True
    st.session_state["setup_signature"] = current_signature


def _render_analysis_setup(st: object, components: object, st_folium: object | None) -> None:
    st.header("Analysis setup")
    st.write(
        "Choose an analysis area and analysis level. Coordinates use WGS84 decimal degrees. "
        "Large areas, small cells, and Full Context mode may take longer."
    )
    top = st.columns([1.2, 1, 1])
    with top[0]:
        st.text_input(
            "Analysis name",
            key="setup_run_name",
            placeholder="Enter a name when ready to run",
        )
    with top[1]:
        st.number_input("Grid cell size (m)", min_value=1.0, key="dashboard_grid_size")
    with top[2]:
        st.selectbox(
            "Analysis mode",
            RUN_MODES,
            key="dashboard_mode",
            help="; ".join(f"{key}: {value}" for key, value in MODE_HELP.items()),
        )
    st.caption(MODE_HELP[str(st.session_state["dashboard_mode"])])

    selection_mode = st.radio(
        "Analysis-area selection",
        ["Draw rectangle on map", "Enter coordinates manually"],
        key="setup_selection_mode",
        horizontal=True,
    )
    if selection_mode == "Enter coordinates manually":
        coordinates = st.columns(4)
        labels = ["Min longitude", "Min latitude", "Max longitude", "Max latitude"]
        keys = ["bbox_min_lon", "bbox_min_lat", "bbox_max_lon", "bbox_max_lat"]
        for column, label, key in zip(coordinates, labels, keys, strict=True):
            with column:
                st.number_input(label, key=key, format="%.6f")
    elif st_folium is None:
        st.warning("Rectangle drawing is unavailable. Enter coordinates manually instead.")
    else:
        instructions, clear_action = st.columns([4, 1])
        with instructions:
            st.caption("Draw a rectangle on the map to select the analysis area.")
        with clear_action:
            if st.button(
                "Clear selected area",
                disabled=not has_selected_bbox(st.session_state),
                use_container_width=True,
            ):
                clear_selected_area(st.session_state)
                st.rerun()
        draw_result = st_folium(
            build_aoi_draw_map(
                st.session_state.get("bbox_min_lon"),
                st.session_state.get("bbox_min_lat"),
                st.session_state.get("bbox_max_lon"),
                st.session_state.get("bbox_max_lat"),
            ),
            height=430,
            use_container_width=True,
            key=f"dashboard_aoi_draw_{st.session_state['aoi_draw_reset_token']}",
            returned_objects=["all_drawings", "last_active_drawing"],
        )
        drawn = latest_drawn_feature(draw_result)
        if drawn is not None:
            try:
                changed = apply_drawn_area(st.session_state, drawn)
            except ValueError as exc:
                st.warning(str(exc))
            else:
                if changed:
                    st.rerun()

    _invalidate_results_after_input_change(st)
    values = None
    input_error = None
    if has_selected_bbox(st.session_state):
        try:
            values = validate_analysis_inputs(
                st.session_state["bbox_min_lon"],
                st.session_state["bbox_min_lat"],
                st.session_state["bbox_max_lon"],
                st.session_state["bbox_max_lat"],
                st.session_state["dashboard_grid_size"],
            )
        except ValueError as exc:
            input_error = str(exc)
            st.error(input_error)

    if values is not None:
        min_lon, min_lat, max_lon, max_lat, grid_size = values
        size = estimate_bbox_size_km(min_lon, min_lat, max_lon, max_lat)
        st.caption(
            f"Selected area: {size['width_km']:.1f} km x {size['height_km']:.1f} km "
            f"({size['area_km2']:.1f} km2). Longitude is east/west; latitude is north/south."
        )
        with st.expander("Preview analysis area", expanded=False):
            components.html(
                build_aoi_preview_html(min_lon, min_lat, max_lon, max_lat),
                height=420,
                scrolling=False,
            )
        for warning in preflight_warnings(
            min_lon, min_lat, max_lon, max_lat, grid_size, str(st.session_state["dashboard_mode"])
        ):
            st.warning(warning)

    setup_name = str(st.session_state.get("setup_run_name") or "").strip()
    output_folder = safe_output_folder_for_run(setup_name)
    overwrite = False
    if output_folder is not None and output_folder_requires_overwrite(output_folder):
        st.info("This analysis name already has saved results.")
        overwrite = st.checkbox("Replace the saved results when running again")

    if st.button("Run analysis", type="primary"):
        try:
            validated_name = validate_run_name(setup_name)
        except ValueError as exc:
            st.error(str(exc))
            return
        if values is None:
            st.error("Draw a rectangle or enter coordinates before running the analysis.")
            return
        if input_error:
            st.session_state["run_state"] = "failed"
            st.error(input_error)
            return
        output_folder = output_folder_path(validated_name)
        if output_folder_requires_overwrite(output_folder) and not overwrite:
            st.error("Choose a new analysis name or confirm replacement of the saved results.")
            return
        if overwrite:
            isolate_existing_output_folder(output_folder)
        config_path = generated_config_path(validated_name)
        create_config_from_bbox(
            run_name=validated_name,
            min_lon=values[0],
            min_lat=values[1],
            max_lon=values[2],
            max_lat=values[3],
            grid_size=values[4],
            mode=str(st.session_state["dashboard_mode"]),
            output=config_path,
        )
        apply_app_safe_cache_settings(config_path)
        st.session_state["run_state"] = "running"
        with st.spinner("PROCESSING — Running the analysis..."):
            result = run_command(build_workflow_command(config_path))
        if result.returncode != 0:
            st.session_state["run_state"] = "failed"
            friendly = friendly_error_summary(result.returncode, result.stderr, result.stdout)
            st.error(f"FAILED — {friendly['reason']}")
            st.info(friendly["next_action"])
            return
        bounds_ok, bounds_message = building_bounds_overlap_selected_bbox(
            output_folder, (values[0], values[1], values[2], values[3])
        )
        if not bounds_ok:
            st.session_state["run_state"] = "failed"
            st.error(f"FAILED — {bounds_message}")
            return
        st.session_state["run_state"] = "completed"
        select_completed_run(st.session_state, validated_name)
        st.session_state["setup_input_changed"] = False
        st.success("COMPLETED — Results are ready in the dashboard.")


def _render_indicator_legend(st: object, overview: dict[str, object], indicator: str) -> None:
    style = resolved_style(style_for_key(indicator), [overview.get("value_min"), overview.get("value_max")])
    chips = " ".join(
        f"<span style='display:inline-block;margin-right:8px'><span style='display:inline-block;width:12px;height:12px;border:1px solid #777;background:{color};vertical-align:middle'></span> {escape(label)}</span>"
        for _key, label, color in legend_entries(style)
    )
    st.markdown(chips, unsafe_allow_html=True)


def _render_aoi_summary(st: object, summary: dict[str, object]) -> None:
    columns = st.columns(2)
    median = summary.get("median")
    columns[0].metric(
        "Median",
        format_indicator_value(median, str(summary["indicator"]))
        if median is not None
        else "Not available",
    )
    columns[1].metric("Missing cells", f"{float(summary['missing_share']):.1%}")


def _render_grid_size_rerun(st: object, source_run: str, source_output: Path) -> None:
    source_summary = read_workflow_summary(source_output)
    current_size = float(source_summary.get("cell_size_m") or 100.0)
    with st.expander("Recalculate with another grid size", expanded=False):
        st.write(
            "Reuse the same analysis area and prepared building/context information, "
            "then recalculate the grid-dependent results."
        )
        columns = st.columns(2)
        with columns[0]:
            new_size = st.number_input(
                "New grid cell size (m)",
                min_value=1.0,
                value=current_size,
                key=f"grid_rerun_size_{source_run}",
            )
        suggested_name = f"{source_run}_{int(new_size)}m"
        with columns[1]:
            new_name = st.text_input(
                "New analysis name",
                value=suggested_name,
                key=f"grid_rerun_name_{source_run}",
            )
        target_folder = safe_output_folder_for_run(new_name)
        replace = False
        if target_folder is not None and output_folder_requires_overwrite(target_folder):
            st.info("This analysis name already has saved results.")
            replace = st.checkbox(
                "Replace the saved results",
                key=f"grid_rerun_replace_{source_run}",
            )

        if not st.button(
            "Recalculate grid",
            type="primary",
            key=f"grid_rerun_submit_{source_run}",
        ):
            return
        try:
            source_config = read_completed_run_config(source_output)
            rerun_config = build_grid_size_rerun_config(
                source_config,
                source_run_name=source_run,
                new_run_name=new_name,
                grid_size_m=new_size,
            )
        except (FileNotFoundError, ValueError) as exc:
            st.error(str(exc))
            return

        cache_source = resolve_grid_rerun_cache_source(source_output, source_config)
        set_grid_rerun_cache_source(rerun_config, cache_source)
        compatibility = grid_rerun_cache_compatibility(cache_source, rerun_config)
        if compatibility["status"] != "compatible":
            st.error(
                "The prepared data are not compatible with this recalculation. "
                "No cached data were reused."
            )
            for reason in compatibility["reasons"]:
                st.write(f"- {reason}")
            return

        plan = grid_rerun_stage_plan(cache_source)
        indicators = rerun_config.get("indicators") or {}
        street = rerun_config.get("street_context") or {}
        missing_cache = []
        if not plan["prepared_buildings"]:
            missing_cache.append("prepared building data")
        if indicators.get("neighbor_distance") and not plan["neighbour_context"]:
            missing_cache.append("building-level neighbour results")
        if street.get("enabled") and not plan["street_context"]:
            missing_cache.append("building-level street-profile results")
        if missing_cache:
            st.error(
                "This completed analysis does not contain all reusable prepared "
                "data needed for a grid-only recalculation: " + ", ".join(missing_cache) + "."
            )
            return

        target_folder = output_folder_path(str(new_name).strip())
        if output_folder_requires_overwrite(target_folder) and not replace:
            st.error("Choose a new analysis name or confirm replacement of saved results.")
            return
        config_path = generated_config_path(str(new_name).strip())
        write_grid_size_rerun_config(rerun_config, config_path)

        with st.status("Recalculating the grid...", expanded=True) as status_box:
            status_box.write("Reusing prepared building data")
            if indicators.get("neighbor_distance"):
                status_box.write("Reusing building-level neighbour context")
            if street.get("enabled"):
                status_box.write("Reusing building-level street-profile context")
            status_box.write(f"Creating {new_size:g} m grid")
            status_box.write("Aggregating indicators")
            result = run_command(build_workflow_command(config_path))
            if result.returncode != 0:
                status_box.update(label="Grid recalculation failed", state="error")
                friendly = friendly_error_summary(result.returncode, result.stderr, result.stdout)
                st.error(friendly["reason"])
                st.info(friendly["next_action"])
                return
            status_box.write("Preparing map")
            status_box.update(label="Grid recalculation completed", state="complete")

        completed_summary = read_workflow_summary(target_folder)
        cache_status = completed_summary.get("cache_source_compatibility_status")
        if cache_status != "compatible":
            st.error(
                "The workflow did not confirm compatible cache reuse. Results are not "
                "shown as a successful grid-only recalculation."
            )
            return
        select_completed_run(st.session_state, str(new_name).strip())
        st.success("The new grid results are ready.")
        st.rerun()


def _render_results_map(st: object, st_folium: object | None) -> None:
    st.header("Results map")
    runs = completed_run_names()
    if not runs:
        st.info("No completed analyses are available yet. Use Analysis setup to create one.")
        return
    current_run = st.session_state.get("selected_completed_run")
    selected_run = st.selectbox(
        "Completed analysis",
        runs,
        index=runs.index(current_run) if current_run in runs else None,
        placeholder="Choose a completed analysis",
        key="selected_completed_run",
    )
    if select_completed_run(st.session_state, selected_run):
        st.rerun()
        return
    if selected_run is None:
        st.info("Choose a completed analysis to view its map and results.")
        return
    output_folder = safe_output_folder_for_run(selected_run)
    if output_folder is None:
        st.error("The selected analysis name is invalid.")
        return
    source_summary = read_building_source_summary(output_folder)
    if not building_source_summary_allows_results(source_summary):
        st.error("The saved building data do not spatially match this analysis area. Results are hidden.")
        return

    readiness_records = read_indicator_readiness_records(output_folder)
    available = dashboard_available_indicators(output_folder)
    if not available:
        st.error("No mapped indicators are available for this completed analysis.")
        return
    if st.session_state.get("selected_indicator") not in available:
        schedule_navigation(
            st.session_state,
            selected_run,
            indicator="gsi" if "gsi" in available else available[0],
        )
        st.rerun()
        return
    indicator = st.selectbox(
        "Indicator",
        available,
        format_func=indicator_label,
        key="selected_indicator",
    )
    definition = indicator_definition(indicator)
    st.write(f"**{definition['role']}** | Unit: **{definition['unit']}**")
    st.write(str(definition["measures"]))
    st.caption(str(definition["higher_lower"]))
    readiness_lookup = readiness_records_by_indicator(readiness_records)
    status = readiness_display_status(readiness_lookup.get(indicator), indicator)
    st.caption(f"Readiness: {status}")
    aoi_summary = aoi_indicator_summary(output_folder, indicator, readiness_records)
    _render_aoi_summary(st, aoi_summary)
    selected_cell_id = st.session_state.get("selected_cell_id")
    selected_cell = query_grid_cell_by_id(output_folder, selected_cell_id)
    selected_feature = query_grid_cell_feature_by_id(output_folder, selected_cell_id)
    with st.spinner("Preparing the map display..."):
        map_object, map_meta = build_results_map(
            output_folder,
            indicator,
            zoom=10,
            center_wgs84=st.session_state.get("selected_click_coordinates"),
            selected_cell_feature=selected_feature,
            selected_cell=selected_cell,
        )
    _render_indicator_legend(st, map_meta["overview"], indicator)
    if st_folium is None:
        st.image(str(map_meta["overview"]["image_path"]), use_container_width=True)
        st.warning("Interactive cell selection requires streamlit-folium.")
    else:
        map_result = st_folium(
            map_object,
            height=680,
            use_container_width=True,
            key=f"results_map_{selected_run}",
            returned_objects=["last_clicked"],
        )
        clicked = (map_result or {}).get("last_clicked")
        if isinstance(clicked, dict) and "lng" in clicked and "lat" in clicked:
            clicked_cell = query_grid_cell_at_location(
                output_folder, float(clicked["lng"]), float(clicked["lat"])
            )
            previous_id = st.session_state.get("selected_cell_id")
            if persist_map_click_selection(st.session_state, clicked, clicked_cell):
                if st.session_state.get("selected_cell_id") != previous_id:
                    st.rerun()
    st.caption("Grey or dashed cells indicate missing or insufficient input data, not zero or low values.")
    if selected_cell_id and st.button("Clear selected cell"):
        clear_cell_selection(st.session_state)
        st.rerun()
    _render_grid_size_rerun(st, selected_run, output_folder)


def _render_data_quality(st: object) -> None:
    st.header("Data quality and limitations")
    selected_run = st.session_state.get("selected_completed_run")
    output_folder = safe_output_folder_for_run(selected_run)
    if output_folder is None:
        st.info("Choose a completed analysis on the Results map page first.")
        return
    summary = read_workflow_summary(output_folder)
    if not summary:
        st.info("Quality information is not available for the selected analysis.")
        return
    readiness = read_indicator_readiness_records(output_folder)
    gsi = read_gsi_sanity_summary(output_folder)
    for card in indicator_quality_cards(summary, readiness, gsi):
        with st.container(border=True):
            st.subheader(f"{card['label']}: {card['status']}")
            st.write(card["interpretability"])
            st.write(card["limitation"])


def _render_indicator_guide(st: object) -> None:
    st.header("Indicator guide")
    for indicator in expected_indicator_keys():
        definition = indicator_definition(indicator)
        with st.expander(str(definition["label"]), expanded=indicator == "gsi"):
            st.write(f"**Unit:** {definition['unit']}")
            st.write(f"**What it measures:** {definition['measures']}")
            st.write(f"**Higher and lower values:** {definition['higher_lower']}")
            st.warning(f"Main limitation: {definition['limitation']}")


def main() -> None:
    try:
        import streamlit as st
        import streamlit.components.v1 as components
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Streamlit is required for the local dashboard. Install it with "
            "`python -m pip install streamlit streamlit-folium`."
        ) from exc
    try:
        from streamlit_folium import st_folium
    except ModuleNotFoundError:
        st_folium = None

    st.set_page_config(
        page_title="Urban Density Analysis",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={
            "Get Help": None,
            "Report a bug": None,
            "About": PROJECT_ABOUT_TEXT,
        },
    )
    st.markdown(STREAMLIT_CHROME_CSS, unsafe_allow_html=True)
    _initialize_dashboard_state(st)
    apply_pending_navigation_before_widgets(st.session_state)
    st.title("Urban Density Analysis")
    st.caption(
        "Local research dashboard for physical urban density and contextual morphology. "
        "Results require data-quality-aware interpretation."
    )
    views = [
        "Analysis setup",
        "Results map",
        "Data quality and limitations",
        "Indicator guide",
    ]
    selected_view = st.radio(
        "Dashboard view",
        views,
        key="active_page",
        horizontal=True,
        label_visibility="collapsed",
    )
    if selected_view == "Analysis setup":
        _render_analysis_setup(st, components, st_folium)
    elif selected_view == "Results map":
        _render_results_map(st, st_folium)
    elif selected_view == "Data quality and limitations":
        _render_data_quality(st)
    else:
        _render_indicator_guide(st)


if __name__ == "__main__":
    main()
