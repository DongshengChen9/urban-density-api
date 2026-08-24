from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
import geopandas as gpd


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
CODE_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = CODE_DIR.parent
SRC_DIR = CODE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from street_context import (
    fetch_streets_from_osmnx,
    calculate_street_profile_segments,
    assign_buildings_to_street_profiles,
    calculate_building_street_profile_ratio,
    aggregate_street_profile_ratio_to_units,
    summarize_street_profile_quality,
)

from height_enrichment import (
    enrich_missing_heights_with_gba_lod1,
    write_height_enrichment_outputs,
)

# ---------------------------------------------------------------------
# Local workflow imports
# ---------------------------------------------------------------------

from aoi import create_aoi_from_bbox, load_aoi, validate_aoi
from data_io import OvertureAdapter
from overture_releases import ResolvedOvertureRelease, is_dated_release, resolve_overture_release
from preprocessing import (
    estimate_metric_crs,
    reproject_to_metric,
    clean_building_geometries,
    clip_buildings_to_aoi,
    add_footprint_area,
)
from quality import (
    summarize_building_quality,
    check_indicator_readiness,
    summarize_unit_quality,
)
from aggregation import create_grid
from cache_contracts import (
    artifact_contract_signature,
    compare_artifact_contracts,
    contract_is_compatible,
    refresh_artifact_contracts,
)
from spatial_identity import (
    CANONICAL_GRID_ID_CONVENTION,
    acquisition_query_wgs84_identity,
    canonical_grid_identity,
    canonical_metric_aoi_identity,
)
from indicators import (
    run_indicators,
    calculate_building_neighbor_diagnostics,
)
from visualization import save_default_workflow_maps
from crs_strategy import determine_crs_processing_mode, summarize_crs_strategy
from segmented_workflow import (
    process_segmented_core_indicators,
    validate_segmented_core_config,
)
from interpretation import (
    build_indicator_readiness_records,
    write_indicator_readiness_outputs,
)
from performance import (
    ARTIFACT_HASH_ALGORITHM,
    CACHE_MANIFEST_VERSION,
    INDICATOR_DEFINITION_VERSION,
    PerformanceRecorder,
    StageStateTracker,
    compact_grid_for_dashboard,
    file_bytes,
    geodataframe_memory_bytes,
    process_rss_bytes,
    read_geodata_layer,
    restore_singlepart_polygon_types,
    resolve_geodata_cache_path,
    stable_geodataframe_hash,
    legacy_storage_compatible_hashes,
    write_geodata_cache,
)


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def load_config(config_path: Path) -> dict[str, Any]:
    """
    Load workflow configuration from a YAML file.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    return config


def resolve_configured_overture_release(config: dict[str, Any]) -> ResolvedOvertureRelease | None:
    """Resolve ``auto`` once and normalize final source provenance."""
    source = config.get("data_source", {})
    if source.get("type") != "overture":
        return None
    requested = str(source.get("overture_release", source.get("release", "auto"))).strip()
    provider = str(source.get("provider", "aws"))
    if requested.lower() == "latest":
        raise ValueError("Use overture_release: auto instead of latest.")
    if requested.lower() == "auto":
        resolution = resolve_overture_release("auto", provider=provider)
    elif is_dated_release(requested):
        # Pinned data may be validly supplied by a compatible local cache after
        # upstream retention expires, so availability is checked only on fetch.
        resolution = ResolvedOvertureRelease(requested, requested, "pinned", provider, "official AWS Buildings prefix verification at acquisition")
    else:
        raise ValueError("overture_release must be auto or a dated identifier.")
    source.update({
        "overture_release": resolution.resolved_release,
        "release": resolution.resolved_release,
        "requested_overture_release": resolution.requested_release,
        "resolved_overture_release": resolution.resolved_release,
        "release_mode": resolution.mode,
        "release_discovery_url": resolution.discovery_url,
    })
    config["data_source"] = source
    return resolution


def setup_output_folders(output_dir: Path) -> dict[str, Path]:
    """
    Create standard output folders for one workflow run.
    """
    folders = {
        "raw": output_dir / "raw",
        "processed": output_dir / "processed",
        "indicators": output_dir / "indicators",
        "tables": output_dir / "tables",
        "reports": output_dir / "reports",
        "logs": output_dir / "logs",
        "maps": output_dir / "maps",
    }

    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    return folders


def prepare_output_directory(
    output_dir: Path,
    overwrite_existing_run: bool = False,
) -> None:
    """
    Prepare a run output directory before folder creation.

    When overwrite is enabled, only the concrete run directory is removed.
    Shared cache directories such as 04_outputs/_cache are protected.
    """
    if not overwrite_existing_run or not output_dir.exists():
        return

    resolved_output = output_dir.resolve()
    outputs_root = (PROJECT_ROOT / "04_outputs").resolve()
    protected_paths = {
        outputs_root,
        (outputs_root / "_cache").resolve(),
    }

    if resolved_output in protected_paths or resolved_output.parent != outputs_root:
        raise ValueError(
            f"Refusing to overwrite protected or non-run output directory: {output_dir}"
        )

    shutil.rmtree(output_dir)


def setup_logging(log_path: Path) -> None:
    """
    Configure logging to both file and console.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def make_json_serializable(value: Any) -> Any:
    """
    Convert values to JSON-serializable Python objects.
    """
    if isinstance(value, dict):
        return {str(k): make_json_serializable(v) for k, v in value.items()}

    if isinstance(value, list):
        return [make_json_serializable(v) for v in value]

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if isinstance(value, Path):
        return str(value)

    return value


def should_use_cached_enriched_buildings(
    cache_config: dict[str, Any],
    cached_enriched_buildings_path: Path,
) -> bool:
    """
    Return whether the workflow should reuse processed GBA-enriched buildings.

    This is a conservative cache gate: it only enables reuse when explicitly
    requested, global cache is enabled, force_refresh is false, and the expected
    processed artifact already exists.
    """
    return (
        bool(cache_config.get("enabled", False))
        and bool(cache_config.get("use_existing_enriched_buildings", False))
        and not bool(cache_config.get("force_refresh", False))
        and cached_enriched_buildings_path.exists()
    )


def should_use_cached_cleaned_buildings(
    cache_config: dict[str, Any],
    cached_cleaned_buildings_path: Path,
) -> bool:
    """Return whether the cleaned/preprocessed AOI-level cache was requested."""
    return (
        bool(cache_config.get("enabled", False))
        and bool(cache_config.get("use_existing_cleaned_buildings", False))
        and not bool(cache_config.get("force_refresh", False))
        and cached_cleaned_buildings_path.exists()
    )


def should_use_cached_street_context(
    cache_config: dict[str, Any],
    cached_street_context_path: Path,
) -> bool:
    """
    Return whether the workflow should reuse building-level street-context output.
    """
    return (
        bool(cache_config.get("enabled", False))
        and bool(cache_config.get("use_existing_street_context", False))
        and not bool(cache_config.get("force_refresh", False))
        and cached_street_context_path.exists()
    )


def resolve_cache_source_output_dir(
    cache_config: dict[str, Any],
    current_output_dir: Path,
    project_root: Path,
) -> tuple[Path, str | None, bool]:
    """
    Resolve the output directory used as the read source for AOI-level caches.

    If no explicit cache source is configured, the current output directory is
    used, preserving the original same-config rerun behavior.
    """
    source_output_name = cache_config.get("source_output_name")
    source_output_dir = cache_config.get("source_output_dir")

    if source_output_dir:
        resolved = Path(source_output_dir)

        if not resolved.is_absolute():
            resolved = project_root / resolved

        return resolved, source_output_name, resolved != current_output_dir

    if source_output_name:
        resolved = project_root / "04_outputs" / str(source_output_name)
        return resolved, str(source_output_name), resolved != current_output_dir

    return current_output_dir, None, False


def load_height_enrichment_summary_if_available(
    height_enrichment_quality_path: Path,
) -> tuple[dict[str, Any] | None, bool]:
    """
    Load cached height-enrichment metadata when it exists.
    """
    if not height_enrichment_quality_path.exists():
        return None, False

    with height_enrichment_quality_path.open("r", encoding="utf-8") as f:
        return json.load(f), True


def load_json_if_available(path: Path) -> tuple[dict[str, Any] | None, bool]:
    """
    Load a JSON metadata file when it exists.
    """
    if not path.exists():
        return None, False

    with path.open("r", encoding="utf-8") as f:
        return json.load(f), True


def load_cache_manifest_with_legacy_metadata(
    source_output_dir: Path,
) -> tuple[dict[str, Any] | None, bool, list[str]]:
    """Load a cache manifest and normalize two documented version-1 fields.

    Version-1 manifests predate processing-mode and indicator-definition
    metadata. They can only be normalized when the source run's saved config is
    available. Existing manifest values are never replaced.
    """
    manifest_path = source_output_dir / "reports" / "cache_manifest.json"
    manifest, found = load_json_if_available(manifest_path)
    if manifest is None or not found or manifest.get("manifest_version") != 1:
        return manifest, found, []

    config_path = source_output_dir / "reports" / "config_used.yaml"
    if not config_path.exists():
        return manifest, found, []
    try:
        source_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return manifest, found, []
    if not isinstance(source_config, dict):
        return manifest, found, []

    normalized = deepcopy(manifest)
    normalizations: list[str] = []
    if "processing_mode" not in normalized:
        normalized["processing_mode"] = source_config.get("crs_strategy", {}).get(
            "processing_mode", "single_crs"
        )
        normalizations.append(
            "processing_mode restored from the source run configuration"
        )
    if "indicator_definition_version" not in normalized:
        # Manifest v1 was produced under the still-current version-1 indicator
        # definitions; this migration records that historical schema fact.
        normalized["indicator_definition_version"] = "1"
        normalizations.append(
            "indicator_definition_version restored as version 1 for a legacy manifest"
        )
    return normalized, found, normalizations


def _rounded_bounds(bounds, precision: int = 8) -> list[float]:
    return [round(float(value), precision) for value in bounds]


def _bounds_dict_from_sequence(bounds, precision: int = 8) -> dict[str, float]:
    rounded = _rounded_bounds(bounds, precision=precision)
    return {
        "min_lon": rounded[0],
        "min_lat": rounded[1],
        "max_lon": rounded[2],
        "max_lat": rounded[3],
    }


def _bounds_tuple_from_dict(bounds: dict[str, Any] | None) -> tuple[float, float, float, float] | None:
    if not bounds:
        return None
    try:
        return (
            float(bounds["min_lon"]),
            float(bounds["min_lat"]),
            float(bounds["max_lon"]),
            float(bounds["max_lat"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _gdf_bounds_wgs84(gdf: gpd.GeoDataFrame) -> dict[str, float] | None:
    if gdf.empty or gdf.crs is None:
        return None
    bounds = gdf.to_crs("EPSG:4326").total_bounds
    if any(pd.isna(value) for value in bounds):
        return None
    return _bounds_dict_from_sequence(bounds)


def _bounds_overlap(
    first: dict[str, Any] | None,
    second: dict[str, Any] | None,
) -> bool:
    first_tuple = _bounds_tuple_from_dict(first)
    second_tuple = _bounds_tuple_from_dict(second)
    if first_tuple is None or second_tuple is None:
        return False
    first_min_lon, first_min_lat, first_max_lon, first_max_lat = first_tuple
    second_min_lon, second_min_lat, second_max_lon, second_max_lat = second_tuple
    return not (
        first_max_lon < second_min_lon
        or second_max_lon < first_min_lon
        or first_max_lat < second_min_lat
        or second_max_lat < first_min_lat
    )


def _bounds_contain(
    outer: dict[str, Any] | None,
    inner: dict[str, Any] | None,
) -> bool:
    outer_tuple = _bounds_tuple_from_dict(outer)
    inner_tuple = _bounds_tuple_from_dict(inner)
    if outer_tuple is None or inner_tuple is None:
        return False
    outer_min_lon, outer_min_lat, outer_max_lon, outer_max_lat = outer_tuple
    inner_min_lon, inner_min_lat, inner_max_lon, inner_max_lat = inner_tuple
    return (
        outer_min_lon <= inner_min_lon
        and outer_min_lat <= inner_min_lat
        and outer_max_lon >= inner_max_lon
        and outer_max_lat >= inner_max_lat
    )


def _aoi_cache_identity(aoi: gpd.GeoDataFrame) -> dict[str, Any]:
    """
    Build stable AOI identity fields for cache compatibility diagnostics.
    """
    aoi_wgs84 = aoi.to_crs("EPSG:4326")

    try:
        geometry = aoi_wgs84.geometry.union_all()
    except AttributeError:
        geometry = aoi_wgs84.unary_union

    bounds = _rounded_bounds(aoi_wgs84.total_bounds)
    hash_payload = {
        "bounds_wgs84": bounds,
        "geometry_wkb_hex": geometry.wkb_hex,
    }
    aoi_hash = hashlib.sha256(
        json.dumps(hash_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    return {
        "input_aoi_crs": aoi.crs.to_string(),
        "aoi_bounds_wgs84": {
            "min_lon": bounds[0],
            "min_lat": bounds[1],
            "max_lon": bounds[2],
            "max_lat": bounds[3],
        },
        "aoi_geometry_hash": aoi_hash,
    }


def _cache_relevant_settings(config: dict[str, Any]) -> dict[str, Any]:
    """
    Extract config settings that affect AOI-level cached artifacts.
    """
    data_source_cfg = config.get("data_source", {})
    preprocessing_cfg = config.get("preprocessing", {})
    height_cfg = config.get("height_enrichment", {})
    street_cfg = config.get("street_context", {})
    crs_cfg = config.get("crs_strategy", {})

    return {
        "data_source": {
            "type": data_source_cfg.get("type"),
            "provider": data_source_cfg.get("provider"),
            "release": data_source_cfg.get("release"),
            "exclude_underground": bool(
                data_source_cfg.get("exclude_underground", True)
            ),
        },
        "preprocessing": {
            "target_crs": preprocessing_cfg.get("target_crs", "auto_utm"),
            "clip_to_aoi": preprocessing_cfg.get("clip_to_aoi", True),
        },
        "height_enrichment": {
            "enabled": bool(height_cfg.get("enabled", False)),
            "min_overlap_share": height_cfg.get("min_overlap_share"),
            "min_valid_height_m": height_cfg.get("min_valid_height_m"),
            "replace_existing_height": bool(
                height_cfg.get("replace_existing_height", False)
            ),
        },
        "street_context": {
            "enabled": bool(street_cfg.get("enabled", False)),
            "source": street_cfg.get("source", "osmnx"),
            "network_type": street_cfg.get("network_type"),
            "distance_m": street_cfg.get("distance_m"),
            "tick_length_m": street_cfg.get("tick_length_m"),
            "topology_rule_version": street_cfg.get("topology_rule_version", 1),
        },
        "aggregation": {
            "method": config.get("aggregation", {}).get("method", "regular_grid"),
            "cell_size_m": config.get("aggregation", {}).get("cell_size_m"),
            "clip_to_aoi": bool(config.get("aggregation", {}).get("clip_to_aoi", True)),
            "grid_id_convention": config.get("aggregation", {}).get("grid_id_convention", CANONICAL_GRID_ID_CONVENTION),
        },
        "processing_mode": crs_cfg.get("processing_mode", "single_crs"),
        "indicator_definition_version": INDICATOR_DEFINITION_VERSION,
    }


def build_cache_manifest(
    config: dict[str, Any],
    aoi: gpd.GeoDataFrame,
    target_crs: Any,
    buildings_clean: gpd.GeoDataFrame | None,
    buildings_raw: gpd.GeoDataFrame | None = None,
    buildings_height_enriched: gpd.GeoDataFrame | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """
    Build a diagnostic manifest describing AOI-level cached artifacts.
    """
    aoi_identity = _aoi_cache_identity(aoi)
    project_cfg = config.get("project", {})
    aoi_cfg = config.get("aoi", {})

    manifest = {
        "manifest_version": CACHE_MANIFEST_VERSION,
        "created_at": created_at or datetime.now().isoformat(timespec="seconds"),
        "run_name": project_cfg.get("run_name"),
        "output_dir": project_cfg.get("output_dir"),
        "config_name": project_cfg.get("config_name"),
        "aoi_name": aoi_cfg.get("name"),
        "aoi_identifier": aoi_cfg.get("id", aoi_cfg.get("name")),
        "target_metric_crs": str(target_crs),
        "building_count_after_preprocessing_enrichment": int(
            len(buildings_height_enriched)
            if buildings_height_enriched is not None
            else (len(buildings_clean) if buildings_clean is not None else 0)
        ),
        "raw_building_count": (
            int(len(buildings_raw)) if buildings_raw is not None else None
        ),
        "raw_building_bounds_wgs84": (
            _gdf_bounds_wgs84(buildings_raw)
            if buildings_raw is not None
            else None
        ),
    }
    manifest.update(aoi_identity)
    manifest.update(canonical_metric_aoi_identity(aoi, target_crs))
    manifest.update(acquisition_query_wgs84_identity(aoi))
    manifest.update(_cache_relevant_settings(config))

    def building_hash(frame: gpd.GeoDataFrame | None) -> str | None:
        if frame is None:
            return None
        return stable_geodataframe_hash(
            frame,
            id_column="building_id" if "building_id" in frame.columns else None,
            attribute_columns=[column for column in ["height_m", "num_floors", "height_source"] if column in frame.columns],
        )
    manifest["input_layer_hashes"] = {
        "buildings_clean": building_hash(buildings_clean),
        "buildings_height_enriched": building_hash(buildings_height_enriched),
        "buildings_raw": (
            stable_geodataframe_hash(
                buildings_raw,
                id_column="building_id" if "building_id" in buildings_raw.columns else None,
                attribute_columns=[
                    column
                    for column in ["height_m", "num_floors"]
                    if column in buildings_raw.columns
                ],
            )
            if buildings_raw is not None
            else None
        ),
    }
    manifest["artifact_layer_hashes"] = dict(manifest["input_layer_hashes"])
    manifest["artifact_hash_algorithm"] = ARTIFACT_HASH_ALGORITHM
    refresh_artifact_contracts(manifest)

    return manifest


def compare_cache_manifests(
    current_manifest: dict[str, Any],
    source_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Compare cache-relevant manifest fields without changing workflow behavior.
    """
    if source_manifest is None:
        return {
            "cache_source_manifest_found": False,
            "cache_source_compatibility_status": "manifest_missing",
            "cache_source_compatibility_warnings": [
                "Source cache manifest was not found."
            ],
            "cache_source_aoi_hash": None,
            "current_aoi_hash": current_manifest.get("aoi_geometry_hash"),
        }

    fields_to_compare = [
        "aoi_geometry_hash",
        "aoi_bounds_wgs84",
        "target_metric_crs",
        "data_source",
        "preprocessing",
        "height_enrichment",
        "street_context",
        "processing_mode",
        "indicator_definition_version",
    ]

    if (
        source_manifest.get("manifest_version", 1) >= CACHE_MANIFEST_VERSION
        and current_manifest.get("input_layer_hashes") is not None
    ):
        fields_to_compare.append("input_layer_hashes")

    warnings = []

    for field in fields_to_compare:
        current_value = current_manifest.get(field)
        source_value = source_manifest.get(field)
        if field == "input_layer_hashes" and isinstance(current_value, dict):
            current_value = {
                key: value for key, value in current_value.items() if value is not None
            }
            source_value = {
                key: (source_value or {}).get(key) for key in current_value
            }
        if current_value != source_value:
            warnings.append(
                f"Cache manifest mismatch for `{field}`: "
                f"current={current_value!r}; "
                f"source={source_value!r}"
            )

    return {
        "cache_source_manifest_found": True,
        "cache_source_compatibility_status": (
            "mismatch_detected" if warnings else "compatible"
        ),
        "cache_source_compatibility_warnings": warnings,
        "cache_source_aoi_hash": source_manifest.get("aoi_geometry_hash"),
        "current_aoi_hash": current_manifest.get("aoi_geometry_hash"),
    }


def write_cache_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """
    Write cache manifest JSON.
    """
    path.write_text(
        json.dumps(make_json_serializable(manifest), indent=2),
        encoding="utf-8",
    )


def verify_cached_building_artifact_hash(
    buildings: gpd.GeoDataFrame,
    source_manifest: dict[str, Any] | None,
    artifact_name: str,
) -> None:
    """Prove that a loaded physical building layer matches its manifest."""
    expected = ((source_manifest or {}).get("artifact_layer_hashes") or {}).get(artifact_name)
    if not expected:
        raise ValueError(f"Artifact-aware `{artifact_name}` cache is missing its stable layer hash.")
    actual = stable_geodataframe_hash(
        buildings,
        id_column="building_id" if "building_id" in buildings.columns else None,
        attribute_columns=[column for column in ["height_m", "num_floors", "height_source"] if column in buildings.columns],
    )
    if actual == expected:
        return
    # v0.3.0 manifests predate semantic artifact hashing. Accept only a
    # documented GeoPackage nullable-integer representation of their recorded
    # legacy digest; modified scientific values or geometry still fail.
    if (source_manifest or {}).get("artifact_hash_algorithm") is None and expected in legacy_storage_compatible_hashes(
        buildings,
        id_column="building_id" if "building_id" in buildings.columns else None,
        attribute_columns=[column for column in ["height_m", "num_floors", "height_source"] if column in buildings.columns],
    ):
        return
    if actual != expected:
        raise ValueError(f"Cached `{artifact_name}` artifact hash differs from its compatibility manifest.")


def evaluate_raw_building_cache(
    config: dict[str, Any],
    aoi: gpd.GeoDataFrame,
    raw_buildings_path: Path,
    cache_manifest_path: Path,
    require_compatible_manifest: bool = False,
) -> dict[str, Any]:
    """
    Decide whether an existing raw building cache is safe to reuse.
    """
    requested_identity = _aoi_cache_identity(aoi)
    requested_bounds = requested_identity["aoi_bounds_wgs84"]
    requested_settings = _cache_relevant_settings(config)
    reasons: list[str] = []

    if not raw_buildings_path.exists():
        return {
            "use_cache": False,
            "cache_compatibility_status": "not_available",
            "cache_compatibility_reasons": ["Raw building cache file is missing."],
            "cache_path": str(raw_buildings_path),
            "raw_building_bounds_wgs84": None,
            "raw_building_count": None,
        }

    source_manifest, manifest_found = load_json_if_available(cache_manifest_path)
    if require_compatible_manifest and not manifest_found:
        reasons.append("Cache manifest is required but was not found.")

    if manifest_found and source_manifest is not None:
        source_data = source_manifest.get("data_source")
        if source_data != requested_settings.get("data_source"):
            reasons.append(
                "Data source settings differ between requested run and cache."
            )

        if source_manifest.get("aoi_geometry_hash") != requested_identity.get(
            "aoi_geometry_hash"
        ):
            reasons.append("AOI geometry hash differs between requested run and cache.")

        source_aoi_bounds = source_manifest.get("aoi_bounds_wgs84")
        if source_aoi_bounds != requested_bounds and not _bounds_contain(
            source_aoi_bounds,
            requested_bounds,
        ):
            reasons.append("Cached AOI bounds do not match or contain requested AOI.")

        raw_bounds = source_manifest.get("raw_building_bounds_wgs84")
        raw_count = source_manifest.get("raw_building_count")
    else:
        raw_bounds = None
        raw_count = None
        if require_compatible_manifest:
            raw_bounds = None
        else:
            reasons.append(
                "Cache manifest was not found; only spatial bounds can be checked."
            )

    if raw_bounds is None or raw_count is None:
        try:
            raw_buildings = read_geodata_layer(
                raw_buildings_path,
                layer="buildings_raw",
            )
            raw_bounds = _gdf_bounds_wgs84(raw_buildings)
            raw_count = int(len(raw_buildings))
        except Exception as exc:
            reasons.append(f"Could not inspect raw building cache: {exc}")

    if raw_count == 0:
        reasons.append("Raw building cache contains no features.")

    if not _bounds_overlap(raw_bounds, requested_bounds):
        reasons.append("Raw building bounds do not overlap requested AOI.")

    manifest_problem_only = (
        not manifest_found
        and not require_compatible_manifest
        and len(reasons) == 1
        and "only spatial bounds can be checked" in reasons[0]
    )
    use_cache = not reasons or manifest_problem_only
    if use_cache and manifest_problem_only:
        status = "reused"
    else:
        status = "reused" if use_cache else "rejected"

    return {
        "use_cache": use_cache,
        "cache_compatibility_status": status,
        "cache_compatibility_reasons": reasons,
        "cache_path": str(raw_buildings_path),
        "raw_building_bounds_wgs84": raw_bounds,
        "raw_building_count": raw_count,
        "cache_manifest_found": manifest_found,
        "cache_manifest_path": str(cache_manifest_path),
    }


def build_building_source_summary(
    config: dict[str, Any],
    aoi: gpd.GeoDataFrame,
    actual_building_source_used: str,
    cache_decision: dict[str, Any] | None = None,
    buildings_raw: gpd.GeoDataFrame | None = None,
) -> dict[str, Any]:
    aoi_identity = _aoi_cache_identity(aoi)
    raw_bounds = _gdf_bounds_wgs84(buildings_raw) if buildings_raw is not None else None
    requested_bounds = aoi_identity["aoi_bounds_wgs84"]
    return {
        "run_name": config.get("project", {}).get("run_name"),
        "requested_aoi_bounds_wgs84": requested_bounds,
        "requested_data_source": _cache_relevant_settings(config)["data_source"],
        "actual_building_source_used": actual_building_source_used,
        "cache_path": (cache_decision or {}).get("cache_path"),
        "cache_compatibility_status": (cache_decision or {}).get(
            "cache_compatibility_status"
        ),
        "cache_compatibility_reasons": (cache_decision or {}).get(
            "cache_compatibility_reasons",
            [],
        ),
        "raw_building_count": (
            int(len(buildings_raw)) if buildings_raw is not None else None
        ),
        "raw_building_bounds_wgs84": raw_bounds,
        "raw_bounds_overlap_requested_aoi": _bounds_overlap(
            raw_bounds,
            requested_bounds,
        )
        if raw_bounds is not None
        else None,
        "raw_bounds_contain_requested_aoi": _bounds_contain(
            raw_bounds,
            requested_bounds,
        )
        if raw_bounds is not None
        else None,
    }


def write_building_source_summary(
    reports_dir: Path,
    summary: dict[str, Any],
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "building_source_summary.json").write_text(
        json.dumps(make_json_serializable(summary), indent=2),
        encoding="utf-8",
    )


def write_failure_summary(
    reports_dir: Path,
    config: dict[str, Any],
    aoi: gpd.GeoDataFrame | None,
    failure_stage: str,
    technical_error: str,
    friendly_error_category: str,
    cache_decision: dict[str, Any] | None = None,
    buildings_raw: gpd.GeoDataFrame | None = None,
) -> None:
    selected_bounds = (
        _aoi_cache_identity(aoi)["aoi_bounds_wgs84"] if aoi is not None else None
    )
    raw_bounds = _gdf_bounds_wgs84(buildings_raw) if buildings_raw is not None else None
    summary = {
        "run_name": config.get("project", {}).get("run_name"),
        "selected_aoi_bounds_wgs84": selected_bounds,
        "failure_stage": failure_stage,
        "technical_error": technical_error,
        "friendly_error_category": friendly_error_category,
        "cache_decision": cache_decision,
        "raw_building_count": (
            int(len(buildings_raw)) if buildings_raw is not None else None
        ),
        "raw_building_bounds_wgs84": raw_bounds,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "failure_summary.json").write_text(
        json.dumps(make_json_serializable(summary), indent=2),
        encoding="utf-8",
    )


def validate_raw_buildings_match_aoi(
    buildings_raw: gpd.GeoDataFrame,
    aoi: gpd.GeoDataFrame,
) -> tuple[bool, str | None]:
    requested_bounds = _aoi_cache_identity(aoi)["aoi_bounds_wgs84"]
    raw_bounds = _gdf_bounds_wgs84(buildings_raw)
    if raw_bounds is None or not _bounds_overlap(raw_bounds, requested_bounds):
        return (
            False,
            "Loaded building data do not spatially match the selected AOI. "
            "Existing cache or previous outputs were rejected because they appear stale.",
        )
    return True, None



def record_stage(
    stage_name: str,
    start_time: float,
    timings: dict[str, float],
    performance_recorder: PerformanceRecorder | None = None,
) -> None:
    """
    Record elapsed time for one workflow stage.

    Stage timings are used for workflow performance diagnostics and
    software-engineering evaluation.
    """
    elapsed = time.perf_counter() - start_time
    timings[stage_name] = float(elapsed)
    if performance_recorder is not None:
        performance_recorder.add_record(
            stage_name.removesuffix("_seconds"),
            status="completed",
            wall_clock_seconds=float(elapsed),
            peak_process_memory_bytes_approx=process_rss_bytes(),
            cache_status="computed",
        )
    logging.info("Stage completed: %s in %.2f seconds", stage_name, elapsed)


def create_or_load_aoi(config: dict[str, Any]):
    """
    Create or load AOI according to config.
    """
    aoi_config = config["aoi"]
    mode = aoi_config.get("mode", "bbox")

    if mode == "bbox":
        aoi = create_aoi_from_bbox(
            name=aoi_config["name"],
            bounds=aoi_config["bounds"],
            crs=aoi_config.get("crs", "EPSG:4326"),
        )
    elif mode == "file":
        path = PROJECT_ROOT / aoi_config["path"]
        layer = aoi_config.get("layer")
        aoi = load_aoi(path, layer=layer)
    else:
        raise ValueError(f"Unsupported AOI mode: {mode}")

    return validate_aoi(aoi)


def acquire_buildings(config: dict[str, Any], aoi):
    """
    Acquire building data from the configured source.
    """
    source_config = config["data_source"]
    source_type = source_config["type"]

    if source_type == "overture":
        adapter = OvertureAdapter(
            release=source_config["release"],
            provider=source_config.get("provider", "aws"),
            exclude_underground=source_config.get("exclude_underground", True),
            clip_to_aoi=False,
        )

        return adapter.fetch_buildings(aoi)

    raise NotImplementedError(
        f"Data source '{source_type}' is not implemented in workflow v0.1."
    )


def compute_indicator_diagnostics(indicator_grid) -> dict[str, Any]:
    """
    Compute simple diagnostics for indicator outputs.
    """
    diagnostics = {
        "total_cells": int(len(indicator_grid)),
    }

    if "is_partial_cell" in indicator_grid.columns:
        diagnostics["partial_cells_count"] = int(
            indicator_grid["is_partial_cell"].sum()
        )

    if "unit_area_m2" in indicator_grid.columns:
        diagnostics["very_small_cells_under_1000m2"] = int(
            (indicator_grid["unit_area_m2"] < 1000).sum()
        )

    if "avg_neighbor_distance_m" in indicator_grid.columns:
        diagnostics["zero_avg_neighbor_distance_cells"] = int(
            (indicator_grid["avg_neighbor_distance_m"] == 0).sum()
        )

    if "median_neighbor_distance_m" in indicator_grid.columns:
        diagnostics["zero_median_neighbor_distance_cells"] = int(
            (indicator_grid["median_neighbor_distance_m"] == 0).sum()
        )

    if "height_distance_valid_count" in indicator_grid.columns:
        diagnostics["no_valid_height_distance_ratio_cells"] = int(
            (indicator_grid["height_distance_valid_count"] == 0).sum()
        )

    # Sanity checks for GSI / Building Coverage Ratio.
    # Theoretically, GSI should be between 0 and 1.
    # Values > 1 are not silently corrected because they can indicate
    # overlapping building footprints, very small edge cells, or source
    # geometry artefacts.
    if "gsi" in indicator_grid.columns:
        gsi_values = pd.to_numeric(indicator_grid["gsi"], errors="coerce")
        valid_gsi = gsi_values.dropna()

        diagnostics["gsi_missing_count"] = int(gsi_values.isna().sum())
        diagnostics["gsi_min"] = (
            float(valid_gsi.min()) if not valid_gsi.empty else None
        )
        diagnostics["gsi_max"] = (
            float(valid_gsi.max()) if not valid_gsi.empty else None
        )
        diagnostics["cells_with_gsi_below_0"] = int((gsi_values < 0).sum())
        diagnostics["cells_with_gsi_over_1"] = int((gsi_values > 1).sum())

    return diagnostics


def diagnose_gsi_over_1_cells(indicator_grid, buildings) -> pd.DataFrame:
    """
    Diagnose cells where GSI is greater than 1.

    This diagnostic compares raw summed building intersection area with
    dissolved / unioned building coverage area. If raw GSI is above 1 but
    dissolved GSI is below or equal to 1, the issue is likely caused by
    overlapping footprints / double counting rather than by CRS, unit-area,
    or formula errors.
    """
    raw_column = "gsi_raw_sum" if "gsi_raw_sum" in indicator_grid.columns else "gsi"
    if raw_column not in indicator_grid.columns:
        return pd.DataFrame()

    if "unit_area_m2" not in indicator_grid.columns:
        return pd.DataFrame()

    gsi_values = pd.to_numeric(indicator_grid[raw_column], errors="coerce")
    problem_cells = indicator_grid[gsi_values > 1].copy()

    if problem_cells.empty:
        return pd.DataFrame()

    rows = []

    for _, cell in problem_cells.iterrows():
        unit_id = cell["unit_id"]
        cell_geom = cell.geometry
        cell_area = float(cell["unit_area_m2"])

        candidates = buildings[buildings.intersects(cell_geom)].copy()

        intersections = candidates.geometry.intersection(cell_geom)
        intersections = intersections[~intersections.is_empty]

        raw_intersection_area = float(intersections.area.sum())

        if len(intersections) > 0:
            try:
                dissolved_geom = intersections.union_all()
            except AttributeError:
                dissolved_geom = intersections.unary_union
            dissolved_area = float(dissolved_geom.area)
        else:
            dissolved_area = 0.0

        raw_gsi = raw_intersection_area / cell_area if cell_area > 0 else None
        dissolved_gsi = dissolved_area / cell_area if cell_area > 0 else None
        overlap_excess_area = raw_intersection_area - dissolved_area

        rows.append(
            {
                "unit_id": unit_id,
                "cell_area_m2": cell_area,
                "n_intersecting_buildings": int(len(candidates)),
                "raw_intersection_area_m2": raw_intersection_area,
                "dissolved_intersection_area_m2": dissolved_area,
                "overlap_excess_area_m2": float(overlap_excess_area),
                "raw_gsi": raw_gsi,
                "dissolved_gsi": dissolved_gsi,
            }
        )

    return pd.DataFrame(rows)


def summarize_gsi_sanity(
    indicator_grid,
    gsi_over_1_diagnostics: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Summarize GSI sanity checks for the Markdown quality report.
    """
    summary = {
        "gsi_available": False,
        "gsi_missing_count": None,
        "gsi_min": None,
        "gsi_max": None,
        "cells_with_gsi_below_0": None,
        "cells_with_gsi_over_1": None,
        "gsi_over_1_diagnostics_available": False,
        "max_raw_gsi": None,
        "max_dissolved_gsi": None,
        "total_overlap_excess_area_m2": None,
        "all_dissolved_gsi_within_expected_range": None,
        "interpretation": None,
    }

    if "gsi" not in indicator_grid.columns:
        summary["interpretation"] = "GSI was not calculated in this workflow run."
        return summary

    gsi_values = pd.to_numeric(indicator_grid["gsi"], errors="coerce")
    valid_gsi = gsi_values.dropna()

    summary["gsi_available"] = True
    summary["gsi_missing_count"] = int(gsi_values.isna().sum())
    summary["gsi_min"] = float(valid_gsi.min()) if not valid_gsi.empty else None
    summary["gsi_max"] = float(valid_gsi.max()) if not valid_gsi.empty else None
    summary["cells_with_gsi_below_0"] = int((gsi_values < 0).sum())
    summary["cells_with_gsi_over_1"] = int((gsi_values > 1).sum())

    if gsi_over_1_diagnostics is not None and not gsi_over_1_diagnostics.empty:
        summary["gsi_over_1_diagnostics_available"] = True
        summary["max_raw_gsi"] = float(gsi_over_1_diagnostics["raw_gsi"].max())
        summary["max_dissolved_gsi"] = float(
            gsi_over_1_diagnostics["dissolved_gsi"].max()
        )
        summary["total_overlap_excess_area_m2"] = float(
            gsi_over_1_diagnostics["overlap_excess_area_m2"].sum()
        )

        dissolved_values = pd.to_numeric(
            gsi_over_1_diagnostics["dissolved_gsi"],
            errors="coerce",
        ).dropna()

        if dissolved_values.empty:
            summary["all_dissolved_gsi_within_expected_range"] = None
        else:
            summary["all_dissolved_gsi_within_expected_range"] = bool(
                (dissolved_values <= 1).all()
            )

    if summary["cells_with_gsi_over_1"] == 0:
        summary["interpretation"] = (
            "No cells exceeded the theoretical GSI range. "
            "GSI values are within the expected 0-1 interval."
        )
    elif summary["gsi_over_1_diagnostics_available"]:
        if summary["all_dissolved_gsi_within_expected_range"] is True:
            summary["interpretation"] = (
                "Some cells exceeded the theoretical GSI range in the raw "
                "calculation, but dissolved GSI values were within the expected "
                "range. This suggests that the issue is likely caused by "
                "overlapping building footprints / double counting rather than "
                "by CRS, unit-area, or formula errors."
            )
        else:
            summary["interpretation"] = (
                "Some cells exceeded the theoretical GSI range. Diagnostic "
                "outputs were created, but dissolved GSI did not fully resolve "
                "the issue. These cells require further spatial inspection."
            )
    else:
        summary["interpretation"] = (
            "Some cells exceeded the theoretical GSI range, but detailed "
            "raw-vs-dissolved diagnostics were not available for this run."
        )

    return summary


def summarize_neighbor_diagnostics(
    neighbor_diagnostics,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Summarize building-level neighbour diagnostics for reports.
    """
    if params is None:
        params = {}

    high_ratio_threshold = float(params.get("high_ratio_threshold", 10.0))
    min_distance_for_ratio_m = float(params.get("min_distance_for_ratio_m", 0.5))
    boundary_distance_threshold_m = float(
        params.get("boundary_distance_threshold_m", 50.0)
    )

    n_buildings = int(len(neighbor_diagnostics))

    def count_true(column: str) -> int:
        if column not in neighbor_diagnostics.columns:
            return 0
        return int(neighbor_diagnostics[column].fillna(False).sum())

    def share(count: int) -> float | None:
        if n_buildings == 0:
            return None
        return float(count / n_buildings)

    def numeric_stat(column: str, stat: str):
        if column not in neighbor_diagnostics.columns:
            return None

        values = pd.to_numeric(
            neighbor_diagnostics[column],
            errors="coerce",
        ).dropna()

        if values.empty:
            return None

        if stat == "mean":
            return float(values.mean())
        if stat == "median":
            return float(values.median())
        if stat == "min":
            return float(values.min())
        if stat == "max":
            return float(values.max())

        raise ValueError(f"Unsupported statistic: {stat}")

    relation_counts = {}

    if "zero_distance_relation" in neighbor_diagnostics.columns:
        relation_counts = (
            neighbor_diagnostics["zero_distance_relation"]
            .fillna("not_zero_or_no_relation")
            .value_counts()
            .astype(int)
            .to_dict()
        )

        relation_counts = {
            str(key): int(value) for key, value in relation_counts.items()
        }

    zero_distance_count = count_true("is_zero_distance")
    near_zero_distance_count = count_true("is_near_zero_distance")
    valid_ratio_count = count_true("has_valid_height_to_distance_ratio")
    near_boundary_count = count_true("is_near_aoi_boundary")
    missing_height_count = n_buildings - count_true("has_valid_height")

    high_ratio_mask = pd.Series(False, index=neighbor_diagnostics.index)

    if "height_to_distance_ratio" in neighbor_diagnostics.columns:
        ratio_values = pd.to_numeric(
            neighbor_diagnostics["height_to_distance_ratio"],
            errors="coerce",
        )
        high_ratio_mask = ratio_values > high_ratio_threshold

    high_ratio_count = int(high_ratio_mask.sum())
    high_ratio = neighbor_diagnostics.loc[high_ratio_mask].copy()

    if high_ratio.empty:
        high_ratio_height_median = None
        high_ratio_distance_median = None
        high_ratio_height_max = None
        high_ratio_distance_min = None
    else:
        high_ratio_height_median = float(
            pd.to_numeric(high_ratio["height_m"], errors="coerce").median()
        )
        high_ratio_distance_median = float(
            pd.to_numeric(high_ratio["neighbor_distance_m"], errors="coerce").median()
        )
        high_ratio_height_max = float(
            pd.to_numeric(high_ratio["height_m"], errors="coerce").max()
        )
        high_ratio_distance_min = float(
            pd.to_numeric(high_ratio["neighbor_distance_m"], errors="coerce").min()
        )

    summary = {
        "n_buildings": n_buildings,
        "height_to_distance_ratio_role": "diagnostic_only",
        "min_distance_for_ratio_m": min_distance_for_ratio_m,
        "boundary_distance_threshold_m": boundary_distance_threshold_m,
        "high_ratio_threshold": high_ratio_threshold,
        "zero_distance_count": zero_distance_count,
        "zero_distance_share": share(zero_distance_count),
        "near_zero_distance_count": near_zero_distance_count,
        "near_zero_distance_share": share(near_zero_distance_count),
        "missing_height_count": int(missing_height_count),
        "missing_height_share": share(int(missing_height_count)),
        "valid_height_to_distance_ratio_count": valid_ratio_count,
        "valid_height_to_distance_ratio_share": share(valid_ratio_count),
        "near_aoi_boundary_count": near_boundary_count,
        "near_aoi_boundary_share": share(near_boundary_count),
        "neighbor_distance_mean_m": numeric_stat("neighbor_distance_m", "mean"),
        "neighbor_distance_median_m": numeric_stat("neighbor_distance_m", "median"),
        "neighbor_distance_min_m": numeric_stat("neighbor_distance_m", "min"),
        "neighbor_distance_max_m": numeric_stat("neighbor_distance_m", "max"),
        "height_to_distance_ratio_mean": numeric_stat(
            "height_to_distance_ratio", "mean"
        ),
        "height_to_distance_ratio_median": numeric_stat(
            "height_to_distance_ratio", "median"
        ),
        "height_to_distance_ratio_max": numeric_stat(
            "height_to_distance_ratio", "max"
        ),
        "high_ratio_count": high_ratio_count,
        "high_ratio_share": share(high_ratio_count),
        "high_ratio_height_median_m": high_ratio_height_median,
        "high_ratio_height_max_m": high_ratio_height_max,
        "high_ratio_neighbor_distance_median_m": high_ratio_distance_median,
        "high_ratio_neighbor_distance_min_m": high_ratio_distance_min,
        "zero_distance_relation_counts": relation_counts,
    }

    return summary

def build_workflow_summary(
    config: dict[str, Any],
    building_quality: dict[str, Any],
    unit_quality: dict[str, Any],
    diagnostics: dict[str, Any],
    indicator_grid,
    neighbor_diagnostics_summary: dict[str, Any] | None = None,
    street_profile_quality: dict[str, Any] | None = None,
    gsi_sanity_summary: dict[str, Any] | None = None,
    height_enrichment_summary=None,
    crs_strategy_summary: dict[str, Any] | None = None,
    crs_processing_summary: dict[str, Any] | None = None,
    cache_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a compact one-row summary for comparing workflow runs.

    This summary is intended for thesis evaluation and cross-AOI comparison.
    It is more interpretable than the generic indicator_descriptive_statistics.csv.
    """
    def get_value(source: dict[str, Any] | None, key: str):
        if source is None:
            return None
        return source.get(key)

    def numeric_stat(column: str, stat: str):
        if column not in indicator_grid.columns:
            return None

        values = pd.to_numeric(indicator_grid[column], errors="coerce").dropna()

        if values.empty:
            return None

        if stat == "mean":
            return float(values.mean())
        if stat == "median":
            return float(values.median())
        if stat == "min":
            return float(values.min())
        if stat == "max":
            return float(values.max())

        raise ValueError(f"Unsupported statistic: {stat}")

    def valid_count_from_column(column: str):
        if column not in indicator_grid.columns:
            return None
        values = pd.to_numeric(indicator_grid[column], errors="coerce")
        return int((values > 0).sum())

    aggregation_cfg = config.get("aggregation", {})
    data_source_cfg = config.get("data_source", {})
    street_context_cfg = config.get("street_context", {})

    old_valid_cells = valid_count_from_column("height_distance_valid_count")

    new_valid_cells = get_value(
        street_profile_quality,
        "grid_cells_with_strict_ratio_count",
    )

    if new_valid_cells is None:
        new_valid_cells = valid_count_from_column(
            "street_profile_ratio_strict_valid_count"
        )

    if old_valid_cells is not None and new_valid_cells is not None:
        street_profile_cell_gain_over_old = int(new_valid_cells - old_valid_cells)
    else:
        street_profile_cell_gain_over_old = None

    street_context_enabled = bool(street_context_cfg.get("enabled", False))

    if not street_context_enabled:
        official_height_width_availability = "disabled_not_calculated"
    elif new_valid_cells is None:
        official_height_width_availability = "enabled_not_available"
    elif new_valid_cells > 0:
        official_height_width_availability = "available"
    else:
        official_height_width_availability = "calculated_no_valid_cells"

    summary = {
        # Run identity
        "run_name": config.get("project", {}).get("run_name"),
        "output_dir": config.get("project", {}).get("output_dir"),
        "aoi_name": config.get("aoi", {}).get("name"),
        "data_source_type": data_source_cfg.get("type"),
        "data_source_release": data_source_cfg.get("release"),
        "aggregation_method": aggregation_cfg.get("method"),
        "cell_size_m": aggregation_cfg.get("cell_size_m"),

        # Cache diagnostics
        "used_cached_enriched_buildings": get_value(
            cache_summary,
            "used_cached_enriched_buildings",
        ),
        "cached_enriched_buildings_path": get_value(
            cache_summary,
            "cached_enriched_buildings_path",
        ),
        "cached_height_enrichment_metadata_loaded": get_value(
            cache_summary,
            "cached_height_enrichment_metadata_loaded",
        ),
        "cache_source_output_name": get_value(
            cache_summary,
            "cache_source_output_name",
        ),
        "cache_source_output_dir": get_value(
            cache_summary,
            "cache_source_output_dir",
        ),
        "used_external_cache_source": get_value(
            cache_summary,
            "used_external_cache_source",
        ),
        "used_cached_street_context": get_value(
            cache_summary,
            "used_cached_street_context",
        ),
        "cached_street_context_path": get_value(
            cache_summary,
            "cached_street_context_path",
        ),
        "cached_street_profile_quality_loaded": get_value(
            cache_summary,
            "cached_street_profile_quality_loaded",
        ),
        "cache_manifest_written": get_value(
            cache_summary,
            "cache_manifest_written",
        ),
        "cache_source_manifest_found": get_value(
            cache_summary,
            "cache_source_manifest_found",
        ),
        "cache_source_compatibility_status": get_value(
            cache_summary,
            "cache_source_compatibility_status",
        ),
        "cache_source_compatibility_warnings": get_value(
            cache_summary,
            "cache_source_compatibility_warnings",
        ),
        "cache_manifest_metadata_normalizations": get_value(
            cache_summary,
            "cache_manifest_metadata_normalizations",
        ),
        "cache_source_aoi_hash": get_value(
            cache_summary,
            "cache_source_aoi_hash",
        ),
        "current_aoi_hash": get_value(
            cache_summary,
            "current_aoi_hash",
        ),

        # CRS strategy diagnostics. These do not change current processing CRS.
        "crs_strategy_input_crs": get_value(crs_strategy_summary, "input_crs"),
        "crs_strategy_aoi_longitude_min": get_value(
            crs_strategy_summary,
            "aoi_longitude_min",
        ),
        "crs_strategy_aoi_longitude_max": get_value(
            crs_strategy_summary,
            "aoi_longitude_max",
        ),
        "crs_strategy_aoi_longitude_extent_degrees": get_value(
            crs_strategy_summary,
            "aoi_longitude_extent_degrees",
        ),
        "crs_strategy_intersecting_utm_zones": get_value(
            crs_strategy_summary,
            "intersecting_utm_zones",
        ),
        "crs_strategy_corresponding_utm_epsg_codes": get_value(
            crs_strategy_summary,
            "corresponding_utm_epsg_codes",
        ),
        "crs_strategy_is_multi_utm_zone": get_value(
            crs_strategy_summary,
            "is_multi_utm_zone",
        ),
        "crs_strategy_recommended_strategy": get_value(
            crs_strategy_summary,
            "recommended_crs_strategy",
        ),
        "crs_requested_processing_mode": get_value(
            crs_processing_summary,
            "requested_processing_mode",
        ),
        "crs_resolved_processing_mode": get_value(
            crs_processing_summary,
            "resolved_processing_mode",
        ),
        "crs_segmented_utm_required": get_value(
            crs_processing_summary,
            "segmented_utm_required",
        ),
        "crs_n_utm_segments": get_value(
            crs_processing_summary,
            "n_utm_segments",
        ),
        "crs_segment_epsg_list": get_value(
            crs_processing_summary,
            "segment_epsg_list",
        ),
        "crs_segment_utm_zones": get_value(
            crs_processing_summary,
            "segment_utm_zones",
        ),
        "crs_segmented_utm_available": get_value(
            crs_processing_summary,
            "segmented_utm_available",
        ),
        "crs_segmented_utm_supported_scope": get_value(
            crs_processing_summary,
            "segmented_utm_supported_scope",
        ),
        "crs_segmented_utm_reason": get_value(
            crs_processing_summary,
            "segmented_utm_reason",
        ),
        "crs_processing_diagnostics": get_value(
            crs_processing_summary,
            "diagnostics",
        ),

        # Basic size
        "n_buildings": get_value(building_quality, "n_buildings"),
        "n_grid_cells": int(len(indicator_grid)),
        "partial_cells_count": get_value(diagnostics, "partial_cells_count"),
        "very_small_cells_under_1000m2": get_value(
            diagnostics,
            "very_small_cells_under_1000m2",
        ),

        # Building data quality
        "missing_height_share": get_value(
            building_quality,
            "missing_height_share",
        ),
        "height_valid_area_share": get_value(
            building_quality,
            "height_valid_area_share",
        ),
        "missing_num_floors_share": get_value(
            building_quality,
            "missing_num_floors_share",
        ),
        "floor_valid_area_share": get_value(
            building_quality,
            "floor_valid_area_share",
        ),

        # Core density indicators
        "gsi_mean": numeric_stat("gsi", "mean"),
        "gsi_median": numeric_stat("gsi", "median"),
        "gsi_min": numeric_stat("gsi", "min"),
        "gsi_max": numeric_stat("gsi", "max"),
        "far_fsi_mean": numeric_stat("far_fsi", "mean"),
        "far_fsi_median": numeric_stat("far_fsi", "median"),
        "built_volume_density_mean": numeric_stat(
            "built_volume_density",
            "mean",
        ),
        "built_volume_density_median": numeric_stat(
            "built_volume_density",
            "median",
        ),

        # Neighbour diagnostics
        "avg_neighbor_distance_mean_m": numeric_stat(
            "avg_neighbor_distance_m",
            "mean",
        ),
        "avg_neighbor_distance_median_m": numeric_stat(
            "avg_neighbor_distance_m",
            "median",
        ),
        "zero_avg_neighbor_distance_cells": get_value(
            diagnostics,
            "zero_avg_neighbor_distance_cells",
        ),
        "zero_neighbor_distance_share_building_level": get_value(
            neighbor_diagnostics_summary,
            "zero_distance_share",
        ),
        "old_height_distance_valid_cells": old_valid_cells,
        "nearest_neighbor_height_distance_ratio_role": "diagnostic_only",
        "diagnostic_nearest_neighbor_height_distance_valid_cells": old_valid_cells,

        # Street-profile branch
        "street_context_enabled": street_context_enabled,
        "street_context_network_type": street_context_cfg.get("network_type"),
        "street_context_distance_m": street_context_cfg.get("distance_m"),
        "street_context_tick_length_m": street_context_cfg.get("tick_length_m"),
        "official_contextual_height_width_indicator": (
            "street_profile_height_to_width_ratio"
        ),
        "official_contextual_height_width_basis": (
            "street_profile_width_from_street_context_branch"
        ),
        "official_contextual_height_width_availability": (
            official_height_width_availability
        ),

        "street_profile_n_street_segments": get_value(
            street_profile_quality,
            "n_street_segments",
        ),
        "street_profile_valid_width_share": get_value(
            street_profile_quality,
            "valid_width_share",
        ),
        "street_profile_matched_to_street_share": get_value(
            street_profile_quality,
            "matched_to_street_share",
        ),
        "street_profile_valid_height_share": get_value(
            street_profile_quality,
            "valid_height_share",
        ),
        "street_profile_valid_ratio_strict_share": get_value(
            street_profile_quality,
            "valid_ratio_strict_share",
        ),
        "street_profile_opposite_profile_evidence_share": get_value(
            street_profile_quality,
            "opposite_profile_evidence_share",
        ),
        "street_profile_width_median_m": get_value(
            street_profile_quality,
            "width_median_m",
        ),
        "street_profile_capped_width_share": get_value(
            street_profile_quality,
            "capped_width_share",
        ),
        "street_profile_duplicate_nearest_matches_removed": get_value(
            street_profile_quality,
            "duplicate_join_rows_removed",
        ),
        "street_profile_valid_building_share": get_value(
            street_profile_quality,
            "valid_ratio_strict_share",
        ),
        "street_profile_valid_grid_cell_count": new_valid_cells,
        "street_profile_valid_grid_cell_share": get_value(
            street_profile_quality,
            "grid_cells_with_strict_ratio_share",
        ),
        "street_profile_cell_gain_over_old_ratio": street_profile_cell_gain_over_old,

        # Street-profile ratio statistics at grid level
        "avg_street_profile_height_to_width_ratio_strict_mean": numeric_stat(
            "avg_street_profile_height_to_width_ratio_strict",
            "mean",
        ),
        "avg_street_profile_height_to_width_ratio_strict_median": numeric_stat(
            "avg_street_profile_height_to_width_ratio_strict",
            "median",
        ),
        "median_street_profile_height_to_width_ratio_strict_median": numeric_stat(
            "median_street_profile_height_to_width_ratio_strict",
            "median",
        ),

        # GSI sanity
        "cells_with_gsi_over_1": get_value(
            gsi_sanity_summary,
            "cells_with_gsi_over_1",
        ),
        "gsi_sanity_interpretation": get_value(
            gsi_sanity_summary,
            "interpretation",
        ),
    }
    summary.update({
        "height_enrichment_enabled": height_enrichment_summary is not None,
    })
    
    if height_enrichment_summary is not None:
        summary.update({
            "height_valid_share_before_enrichment": height_enrichment_summary.get("valid_height_share_before"),
            "height_valid_share_after_enrichment": height_enrichment_summary.get("valid_height_share_after"),
            "missing_height_share_before_enrichment": height_enrichment_summary.get("missing_height_share_before"),
            "missing_height_share_after_enrichment": height_enrichment_summary.get("missing_height_share_after"),
            "height_valid_area_share_before_enrichment": height_enrichment_summary.get("height_valid_area_share_before"),
            "height_valid_area_share_after_enrichment": height_enrichment_summary.get("height_valid_area_share_after"),
            "height_enriched_count": height_enrichment_summary.get("height_enriched_count"),
            "height_enriched_share_of_missing_before": height_enrichment_summary.get("height_enriched_share_of_missing_before"),
            "height_source_overture_count": height_enrichment_summary.get("height_source_overture_count"),
            "height_source_gba_lod1_count": height_enrichment_summary.get("height_source_gba_lod1_count"),
            "height_source_missing_count": height_enrichment_summary.get("height_source_missing_count"),
            "changed_existing_overture_height_count": height_enrichment_summary.get("changed_existing_overture_height_count"),
            "gba_min_overlap_share_for_enrichment": height_enrichment_summary.get("min_overlap_share_for_enrichment"),
            "gba_min_valid_height_m": height_enrichment_summary.get("min_valid_height_m"),
        })
    else:
        summary.update({
            "height_valid_share_before_enrichment": None,
            "height_valid_share_after_enrichment": None,
            "missing_height_share_before_enrichment": None,
            "missing_height_share_after_enrichment": None,
            "height_valid_area_share_before_enrichment": None,
            "height_valid_area_share_after_enrichment": None,
            "height_enriched_count": None,
            "height_enriched_share_of_missing_before": None,
            "height_source_overture_count": None,
            "height_source_gba_lod1_count": None,
            "height_source_missing_count": None,
            "changed_existing_overture_height_count": None,
            "gba_min_overlap_share_for_enrichment": None,
            "gba_min_valid_height_m": None,
        })
    return summary


def add_segmented_workflow_summary_metadata(
    workflow_summary: dict[str, Any],
    segmented_crs_summary: dict[str, Any],
) -> dict[str, Any]:
    """
    Add compact segmented UTM metadata to the workflow summary.
    """
    neighbor_summary = segmented_crs_summary.get("neighbor_distance_summary", {})
    street_summary = segmented_crs_summary.get("street_profile_summary", {})
    grid_coverage_share = neighbor_summary.get(
        "grid_cell_neighbor_distance_coverage_share"
    )

    workflow_summary.update(
        {
            "segmented_processing_enabled": True,
            "segmented_processing_supported_scope": (
                "core_indicators_height_enrichment_neighbor_distance_and_street_context"
            ),
            "segmented_n_segments": segmented_crs_summary["n_segments"],
            "segmented_segment_epsg_list": segmented_crs_summary[
                "segment_epsg_list"
            ],
            "segmented_segment_utm_zones": segmented_crs_summary[
                "segment_utm_zones"
            ],
            "segmented_neighbor_distance_enabled": segmented_crs_summary.get(
                "segmented_neighbor_distance_enabled"
            ),
            "segmented_context_buffer_m": segmented_crs_summary.get(
                "segmented_context_buffer_m"
            ),
            "segmented_neighbor_distance_valid_building_share": neighbor_summary.get(
                "valid_neighbor_distance_share"
            ),
            "segmented_neighbor_distance_valid_building_count": neighbor_summary.get(
                "valid_neighbor_distance_count"
            ),
            "segmented_neighbor_distance_grid_cell_coverage_share": (
                grid_coverage_share
            ),
            "segmented_neighbor_distance_valid_grid_cell_share": (
                grid_coverage_share
            ),
            "segmented_neighbor_distance_grid_cells_with_values": (
                neighbor_summary.get("grid_cells_with_neighbor_distance")
            ),
            "segmented_neighbor_distance_n_grid_cells": neighbor_summary.get(
                "n_grid_cells"
            ),
            "segmented_street_context_enabled": segmented_crs_summary.get(
                "segmented_street_context_enabled"
            ),
            "segmented_street_context_buffer_m": segmented_crs_summary.get(
                "segmented_street_context_buffer_m"
            ),
            "segmented_street_profile_valid_building_count": street_summary.get(
                "valid_ratio_strict_count"
            ),
            "segmented_street_profile_valid_building_share": street_summary.get(
                "valid_ratio_strict_share"
            ),
            "segmented_street_profile_grid_cells_with_values": street_summary.get(
                "grid_cells_with_strict_ratio_count"
            ),
            "segmented_street_profile_n_grid_cells": street_summary.get(
                "n_grid_cells"
            ),
            "segmented_street_profile_grid_cell_coverage_share": street_summary.get(
                "grid_cells_with_strict_ratio_share"
            ),
            "segmented_street_context_status_counts": street_summary.get(
                "street_context_status_counts"
            ),
            "segmented_street_context_no_graph_segment_count": street_summary.get(
                "no_graph_segment_count"
            ),
            "segmented_street_context_no_street_segment_count": street_summary.get(
                "no_street_segment_count"
            ),
        }
    )

    return workflow_summary


def write_markdown_quality_report(
    path: Path,
    building_quality: dict[str, Any],
    unit_quality: dict[str, Any],
    indicator_readiness: dict[str, Any],
    diagnostics: dict[str, Any],
    neighbor_diagnostics_summary: dict[str, Any] | None = None,
    street_profile_quality: dict[str, Any] | None = None,
    gsi_sanity_summary: dict[str, Any] | None = None,
    saved_maps: list[Any] | None = None,
    height_enrichment_summary: dict[str, Any] | None = None,
    workflow_summary: dict[str, Any] | None = None,
    crs_strategy_summary: dict[str, Any] | None = None,
    crs_processing_summary: dict[str, Any] | None = None,
    cache_summary: dict[str, Any] | None = None,
    indicator_interpretation_records: list[dict[str, Any]] | None = None,
) -> None:
    """
    Write a simple Markdown quality and diagnostics report.
    """
    lines = [
        "# Pilot workflow quality and diagnostics report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Building data quality",
        "",
    ]

    for key, value in building_quality.items():
        lines.append(f"- **{key}**: {value}")

    lines.append("## Height enrichment diagnostics")
    lines.append("")
    
    if height_enrichment_summary is None:
        if cache_summary and cache_summary.get("used_cached_enriched_buildings"):
            lines.append(
                "Cached GBA-enriched buildings were used for this run, but "
                "cached height enrichment quality metadata was not available."
            )
        else:
            lines.append("Height enrichment was not enabled for this run.")
    
    else:
        lines.append(
            "GlobalBuildingAtlas LoD1 was used as an external height-enrichment source "
            "only for buildings with missing Overture height values. Existing valid "
            "Overture height values were not overwritten."
        )
    
        lines.append("")
    
        for key in [
            "valid_height_count_before",
            "missing_height_count_before",
            "valid_height_share_before",
            "missing_height_share_before",
            "height_enriched_count",
            "height_enriched_share_of_missing_before",
            "valid_height_count_after",
            "missing_height_count_after",
            "valid_height_share_after",
            "missing_height_share_after",
            "height_valid_area_share_before",
            "height_valid_area_share_after",
            "height_source_overture_count",
            "height_source_gba_lod1_count",
            "height_source_missing_count",
            "changed_existing_overture_height_count",
            "min_overlap_share_for_enrichment",
            "min_valid_height_m",
        ]:
            if key in height_enrichment_summary:
                lines.append(
                    f"- `{key}`: {height_enrichment_summary[key]}"
                )
    
        lines.append("")
    
        lines.append(
            "Interpretation: GBA-derived heights reduce missing height values "
            "but introduce source heterogeneity. Height-based indicators should "
            "therefore still be interpreted together with height-source and "
            "enrichment-quality diagnostics."
        )
    
    lines.append("")

    lines.extend([
        "",
        "## Aggregation unit quality",
        "",
    ])

    for key, value in unit_quality.items():
        lines.append(f"- **{key}**: {value}")

    lines.extend([
        "",
        "## Indicator readiness",
        "",
    ])

    for key, value in indicator_readiness.items():
        lines.append(f"### {key}")

        if isinstance(value, dict):
            lines.append(f"- **status**: {value.get('status')}")
            lines.append(f"- **reason**: {value.get('reason')}")
            lines.append(f"- **available**: {value.get('available')}")
        else:
            lines.append(f"- {value}")

        lines.append("")

    if indicator_interpretation_records:
        lines.extend([
            "",
            "## Indicator interpretation readiness",
            "",
            "These user-facing statuses summarize whether each indicator is "
            "interpretable in this run. The thresholds are pragmatic workflow "
            "reporting thresholds, not universal scientific thresholds.",
            "",
        ])

        for record in indicator_interpretation_records:
            lines.append(
                "- **{indicator}**: {status} - {recommended_use}".format(
                    indicator=record.get("indicator"),
                    status=record.get("status"),
                    recommended_use=record.get("recommended_use"),
                )
            )

        lines.append("")

    lines.extend([
        "",
        "## Indicator diagnostics",
        "",
    ])

    for key, value in diagnostics.items():
        lines.append(f"- **{key}**: {value}")

    if cache_summary is not None:
        lines.extend([
            "",
            "## Cache diagnostics",
            "",
            "- **used_cached_enriched_buildings**: "
            f"{cache_summary.get('used_cached_enriched_buildings')}",
            "- **cached_enriched_buildings_path**: "
            f"{cache_summary.get('cached_enriched_buildings_path')}",
            "- **cached_height_enrichment_metadata_loaded**: "
            f"{cache_summary.get('cached_height_enrichment_metadata_loaded')}",
            "- **cache_source_output_name**: "
            f"{cache_summary.get('cache_source_output_name')}",
            "- **cache_source_output_dir**: "
            f"{cache_summary.get('cache_source_output_dir')}",
            "- **used_external_cache_source**: "
            f"{cache_summary.get('used_external_cache_source')}",
            "- **used_cached_street_context**: "
            f"{cache_summary.get('used_cached_street_context')}",
            "- **cached_street_context_path**: "
            f"{cache_summary.get('cached_street_context_path')}",
            "- **cached_street_profile_quality_loaded**: "
            f"{cache_summary.get('cached_street_profile_quality_loaded')}",
            "- **cache_manifest_written**: "
            f"{cache_summary.get('cache_manifest_written')}",
            "- **cache_source_manifest_found**: "
            f"{cache_summary.get('cache_source_manifest_found')}",
            "- **cache_source_compatibility_status**: "
            f"{cache_summary.get('cache_source_compatibility_status')}",
            "- **cache_source_aoi_hash**: "
            f"{cache_summary.get('cache_source_aoi_hash')}",
            "- **current_aoi_hash**: "
            f"{cache_summary.get('current_aoi_hash')}",
        ])

        if cache_summary.get("used_cached_enriched_buildings"):
            lines.append(
                "Cached GBA-enriched processed buildings were reused for this "
                "run. Raw Overture acquisition, building CRS preprocessing, and "
                "GBA height enrichment were skipped; grid creation, indicators, "
                "street context, maps, tables, and reports were recalculated."
            )

        if cache_summary.get("used_cached_street_context"):
            lines.append(
                "Cached building-level street-profile ratio output was reused "
                "for this run. Street fetching, street-profile calculation, and "
                "building-to-street assignment were skipped; the building-level "
                "ratios were aggregated to the current grid."
            )

        compatibility_warnings = cache_summary.get(
            "cache_source_compatibility_warnings"
        ) or []

        if compatibility_warnings:
            lines.extend([
                "",
                "### Cache compatibility warnings",
                "",
            ])

            for warning in compatibility_warnings:
                lines.append(f"- {warning}")

    if crs_strategy_summary is not None:
        lines.extend([
            "",
            "## CRS strategy diagnostics",
            "",
            "This diagnostic checks whether the AOI falls within one UTM zone or "
            "spans multiple UTM zones. It does not change the current metric "
            "processing CRS for this run.",
            "",
        ])

        key_order = [
            "input_crs",
            "aoi_longitude_min",
            "aoi_longitude_max",
            "aoi_longitude_extent_degrees",
            "intersecting_utm_zones",
            "corresponding_utm_epsg_codes",
            "is_single_utm_zone",
            "is_multi_utm_zone",
            "recommended_crs_strategy",
        ]

        for key in key_order:
            if key in crs_strategy_summary:
                lines.append(f"- **{key}**: {crs_strategy_summary[key]}")

        if crs_strategy_summary.get("is_multi_utm_zone"):
            if (
                crs_processing_summary is not None
                and crs_processing_summary.get("resolved_processing_mode")
                == "segmented_utm"
            ):
                lines.extend([
                    "",
                    "### CRS strategy note",
                    "",
                    "This AOI intersects multiple UTM zones. This run uses the "
                    "staged segmented UTM path for core grid indicators, with "
                    "metric calculations performed in each segment's corresponding "
                    "UTM CRS.",
                ])
            else:
                lines.extend([
                    "",
                    "### CRS strategy warning",
                    "",
                    "This AOI intersects multiple UTM zones. The recommended future "
                    "strategy is segmented UTM processing: split the AOI by UTM zone, "
                    "calculate indicators in each segment's corresponding CRS, and "
                    "aggregate the results for visualization. This run still uses the "
                    "existing single-CRS processing path.",
                ])

    if crs_processing_summary is not None:
        lines.extend([
            "",
            "## CRS processing mode",
            "",
            "- **requested_processing_mode**: "
            f"{crs_processing_summary.get('requested_processing_mode')}",
            "- **resolved_processing_mode**: "
            f"{crs_processing_summary.get('resolved_processing_mode')}",
            "- **segmented_utm_required**: "
            f"{crs_processing_summary.get('segmented_utm_required')}",
            "- **n_utm_segments**: "
            f"{crs_processing_summary.get('n_utm_segments')}",
            "- **segment_epsg_list**: "
            f"{crs_processing_summary.get('segment_epsg_list')}",
            "- **segment_utm_zones**: "
            f"{crs_processing_summary.get('segment_utm_zones')}",
            "- **segmented_utm_available**: "
            f"{crs_processing_summary.get('segmented_utm_available')}",
            "- **segmented_utm_supported_scope**: "
            f"{crs_processing_summary.get('segmented_utm_supported_scope')}",
            "- **segmented_utm_reason**: "
            f"{crs_processing_summary.get('segmented_utm_reason')}",
        ])

        if crs_processing_summary.get("resolved_processing_mode") == "segmented_utm":
            lines.extend([
                "",
                "Segmented UTM processing currently supports core grid "
                "indicators, optional GBA height enrichment, and nearest-neighbour "
                "distance with a segment context buffer.",
            ])

        diagnostics = crs_processing_summary.get("diagnostics") or []

        if diagnostics:
            lines.extend([
                "",
                "Diagnostics:",
                "",
            ])
            lines.extend(f"- {diagnostic}" for diagnostic in diagnostics)

    if workflow_summary is not None:
        official_indicator = workflow_summary.get(
            "official_contextual_height_width_indicator",
            "street_profile_height_to_width_ratio",
        )
        official_basis = workflow_summary.get(
            "official_contextual_height_width_basis",
            "street_profile_width_from_street_context_branch",
        )
        official_availability = workflow_summary.get(
            "official_contextual_height_width_availability",
            "unknown",
        )
        nearest_neighbor_role = workflow_summary.get(
            "nearest_neighbor_height_distance_ratio_role",
            "diagnostic_only",
        )

        lines.extend([
            "",
            "## Contextual height-width indicator",
            "",
            "- **official_contextual_height_width_indicator**: "
            f"{official_indicator}",
            "- **official_contextual_height_width_basis**: "
            f"{official_basis}",
            "- **official_contextual_height_width_availability**: "
            f"{official_availability}",
            "- **nearest_neighbor_height_distance_ratio_role**: "
            f"{nearest_neighbor_role}",
            "",
            "Interpretation: The official contextual height-width indicator is "
            "the street-profile-based height-to-width ratio, calculated as "
            "building height divided by street-profile width. The legacy "
            "nearest-neighbour `height_to_distance_ratio` is retained as a "
            "diagnostic-only output because nearest-neighbour distances may be "
            "zero or near-zero in attached, overlapping, or very compact urban "
            "fabric.",
        ])

        if workflow_summary.get("segmented_processing_enabled"):
            lines.extend([
                "",
                "## Segmented UTM contextual distance",
                "",
                "- **segmented_neighbor_distance_enabled**: "
                f"{workflow_summary.get('segmented_neighbor_distance_enabled')}",
                "- **segmented_context_buffer_m**: "
                f"{workflow_summary.get('segmented_context_buffer_m')}",
                "- **segmented_neighbor_distance_valid_building_share**: "
                f"{workflow_summary.get('segmented_neighbor_distance_valid_building_share')}",
                "- **segmented_neighbor_distance_grid_cell_coverage_share**: "
                f"{workflow_summary.get('segmented_neighbor_distance_grid_cell_coverage_share')}",
                "- **segmented_neighbor_distance_grid_cells_with_values**: "
                f"{workflow_summary.get('segmented_neighbor_distance_grid_cells_with_values')}",
                "- **segmented_neighbor_distance_n_grid_cells**: "
                f"{workflow_summary.get('segmented_neighbor_distance_n_grid_cells')}",
                "",
                "Segmented nearest-neighbour distance is calculated from full "
                "building geometries within each segment's context buffer, then "
                "attached back to clipped segment pieces for grid aggregation. "
                "This avoids treating UTM-boundary split pieces as independent "
                "neighbour candidates.",
                "",
                "## Segmented UTM street-profile height-width context",
                "",
                "- **segmented_street_context_enabled**: "
                f"{workflow_summary.get('segmented_street_context_enabled')}",
                "- **segmented_street_context_buffer_m**: "
                f"{workflow_summary.get('segmented_street_context_buffer_m')}",
                "- **segmented_street_profile_valid_building_count**: "
                f"{workflow_summary.get('segmented_street_profile_valid_building_count')}",
                "- **segmented_street_profile_valid_building_share**: "
                f"{workflow_summary.get('segmented_street_profile_valid_building_share')}",
                "- **segmented_street_profile_grid_cells_with_values**: "
                f"{workflow_summary.get('segmented_street_profile_grid_cells_with_values')}",
                "- **segmented_street_profile_n_grid_cells**: "
                f"{workflow_summary.get('segmented_street_profile_n_grid_cells')}",
                "- **segmented_street_profile_grid_cell_coverage_share**: "
                f"{workflow_summary.get('segmented_street_profile_grid_cell_coverage_share')}",
                "- **segmented_street_context_status_counts**: "
                f"{workflow_summary.get('segmented_street_context_status_counts')}",
                "- **segmented_street_context_no_graph_segment_count**: "
                f"{workflow_summary.get('segmented_street_context_no_graph_segment_count')}",
                "- **segmented_street_context_no_street_segment_count**: "
                f"{workflow_summary.get('segmented_street_context_no_street_segment_count')}",
                "",
                "Segmented street-profile height-width context is calculated "
                "with context streets and context buildings around each segment, "
                "then attached only to target segment building pieces for grid "
                "aggregation.",
            ])

    if street_profile_quality is not None:
        lines.extend([
            "",
            "## Street-profile context diagnostics",
            "",
            "This section summarizes the optional street-context branch used to calculate "
            "the street-profile-based height-to-width ratio. This is the official "
            "contextual height-width indicator for the thesis. It is a contextual "
            "3D morphology indicator, not a primary density indicator.",
            "",
        ])

        key_order = [
            "n_street_segments",
            "valid_width_count",
            "valid_width_share",
            "width_min_m",
            "width_median_m",
            "width_max_m",
            "capped_width_count",
            "capped_width_share",
            "opposite_profile_evidence_count",
            "opposite_profile_evidence_share",
            "n_buildings",
            "raw_join_rows_before_deduplication",
            "duplicate_join_rows_removed",
            "matched_to_street_count",
            "matched_to_street_share",
            "valid_height_count",
            "valid_height_share",
            "valid_ratio_prelim_count",
            "valid_ratio_prelim_share",
            "valid_ratio_strict_count",
            "valid_ratio_strict_share",
            "ratio_strict_min",
            "ratio_strict_median",
            "ratio_strict_max",
            "n_grid_cells",
            "grid_cells_with_prelim_ratio_count",
            "grid_cells_with_prelim_ratio_share",
            "grid_cells_with_strict_ratio_count",
            "grid_cells_with_strict_ratio_share",
        ]

        for key in key_order:
            if key in street_profile_quality:
                lines.append(f"- **{key}**: {street_profile_quality[key]}")

        lines.extend([
            "",
            "### Interpretation of street-profile diagnostics",
            "",
            "- Street-profile width is used as an approximation of building-to-opposite-building distance.",
            "- The resulting height-to-width ratio is the official contextual height-width indicator for the thesis.",
            "- Missing ratio values are not interpreted as zero density; they may result from missing building heights or invalid street-profile information.",
            "- Capped street-profile widths indicate cases where the profile may have reached the configured search limit.",
            "- Duplicate nearest-street matches are resolved deterministically but remain a matching uncertainty.",
            "- The indicator should be interpreted together with building height completeness and street-profile quality diagnostics.",
        ])


    if gsi_sanity_summary is not None:
        lines.extend([
            "",
            "## GSI sanity checks",
            "",
            "Official union-based GSI / Building Coverage Ratio is bounded between 0 and 1. "
            "The separately retained raw-sum diagnostic can exceed 1 where input footprints overlap.",
            "",
        ])

        for key, value in gsi_sanity_summary.items():
            lines.append(f"- **{key}**: {value}")

        if gsi_sanity_summary.get("cells_with_gsi_over_1", 0) > 0:
            lines.extend([
                "",
                "### GSI warning",
                "",
                "One or more aggregation cells have raw-sum GSI above 1. This can indicate "
                "overlapping building footprints, source geometry artefacts, or "
                "double counting of overlapping footprint areas.",
                "",
                "The workflow exports suspicious cells and raw-vs-dissolved diagnostics "
                "for spatial inspection when `save_gsi_sanity_diagnostics` is enabled.",
                "",
                "Diagnostic outputs:",
                "",
                "- `indicators/gsi_over_1_cells.gpkg`",
                "- `tables/gsi_over_1_cells.csv`",
                "- `tables/gsi_over_1_diagnostics.csv`",
            ])
        else:
            lines.extend([
                "",
                "No cells with raw-sum GSI above 1 were detected in this run.",
            ])

    if saved_maps is not None:
        lines.extend([
            "",
            "## Static maps",
            "",
        ])

        if len(saved_maps) == 0:
            lines.append("No static maps were generated for this run.")
        else:
            lines.append(
                "The workflow generated the following reproducible static maps:"
            )
            lines.append("")

            for path_item in saved_maps:
                lines.append(f"- `{path_item}`")

    if neighbor_diagnostics_summary is not None:
        lines.extend([
            "",
            "## Building-level neighbour diagnostics",
            "",
            "This section summarizes building-level diagnostics used to interpret "
            "nearest-neighbour distance and diagnostic-only height-to-distance "
            "ratio outputs.",
            "",
        ])

        relation_counts = neighbor_diagnostics_summary.get(
            "zero_distance_relation_counts",
            {},
        )

        for key, value in neighbor_diagnostics_summary.items():
            if key == "zero_distance_relation_counts":
                continue
            lines.append(f"- **{key}**: {value}")

        lines.extend([
            "",
            "### Zero-distance relation counts",
            "",
        ])

        for key, value in relation_counts.items():
            lines.append(f"- **{key}**: {value}")

        lines.extend([
            "",
            "### Interpretation of neighbour diagnostics",
            "",
            "- Zero nearest-neighbour distances are not automatically errors.",
            "- If most zero-distance cases are classified as `touching_boundary`, "
            "they likely represent attached or contiguous urban fabric.",
            "- Cases classified as `overlap`, `within_neighbor`, or "
            "`contains_neighbor` should be treated as possible geometry or "
            "source-data quality issues.",
            "- Height-to-distance ratio is only meaningful where positive spacing "
            "exists between neighbouring buildings.",
            "- High height-to-distance ratio values should be interpreted by checking "
            "both building height and neighbour distance. In this workflow, high values "
            "are treated as diagnostic cases because they may be driven by very small "
            "positive spacing rather than unusually tall buildings.",
            "- The nearest-neighbour height-to-distance ratio is retained as a "
            "diagnostic only; the official contextual height-width indicator is the "
            "street-profile-based height-to-width ratio from the street_context branch.",
            "- The diagnostic ratio should therefore not be interpreted without the accompanying "
            "`neighbor_distance_m`, `height_m`, and valid-count diagnostics.",
        ])

    lines.extend([
        "",
        "## Interpretation notes",
        "",
        "- GSI is calculated from valid building footprints after reprojection to a metric CRS.",
        "- GSI values should theoretically fall between 0 and 1.",
        "- Cells with `GSI > 1` are flagged as suspicious rather than corrected automatically.",
        "- Such cases may indicate overlapping building footprints, very small edge cells, or source geometry artefacts and should be inspected spatially.",
        "- FAR/FSI is conditional because it depends on available floor-count or floor-area information.",
        "- Built Volume Density is conditional because it depends on available height information.",
        "- Neighbour distance is a contextual morphology indicator, not a primary density indicator.",
        "- The street-profile-based height-to-width ratio is the official contextual height-width indicator.",
        "- The nearest-neighbour height-to-distance ratio is retained as a diagnostic-only output because nearest-neighbour distance can be zero in attached or overlapping urban fabric.",
        "- Missing height and floor-count values are treated as data gaps, not as zero.",
        "- Partial edge cells and very small grid cells should be interpreted cautiously.",
        "- Zero neighbour distances may indicate attached buildings, overlapping footprints, or highly contiguous urban fabric.",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------

def run_workflow(config_path: Path) -> None:
    """
    Run the complete minimal workflow for one pilot AOI.
    """
    config = load_config(config_path)
    resolve_configured_overture_release(config)
    output_mode = config.get("outputs", {}).get("mode", "research")
    if output_mode not in {"compact", "research"}:
        raise ValueError("outputs.mode must be `compact` or `research`.")
    research_mode = output_mode == "research"

    output_dir = PROJECT_ROOT / config["project"]["output_dir"]
    overwrite_existing_run = bool(
        config.get("project", {}).get(
            "overwrite_existing_run",
            config.get("outputs", {}).get("overwrite_existing", False),
        )
    )
    prepare_output_directory(
        output_dir=output_dir,
        overwrite_existing_run=overwrite_existing_run,
    )
    folders = setup_output_folders(output_dir)

    log_path = folders["logs"] / "workflow.log"
    setup_logging(log_path)

    stage_timings: dict[str, float] = {}
    performance_recorder = PerformanceRecorder()
    stage_tracker = StageStateTracker.load(
        folders["reports"] / "workflow_stages.json",
        config["project"]["run_name"],
    )
    for workflow_stage in [
        "footprint_core",
        "vertical_density",
        "contextual_morphology",
        "presentation",
    ]:
        if workflow_stage not in stage_tracker.stages:
            stage_tracker.mark(workflow_stage, "pending")
    workflow_start = time.perf_counter()

    logging.info("Starting workflow.")
    logging.info("Project root: %s", PROJECT_ROOT)
    logging.info("Config path: %s", config_path)
    logging.info("Output directory: %s", output_dir)

    # Save the resolved exact release, never the moving ``auto`` request.
    (folders["reports"] / "config_used.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )


    # 1. AOI
    stage_start = time.perf_counter()
    logging.info("Creating/loading AOI.")
    aoi = create_or_load_aoi(config)
    logging.info("AOI created. CRS=%s, features=%s", aoi.crs, len(aoi))
    crs_strategy_summary = summarize_crs_strategy(aoi)
    logging.info(
        "CRS strategy diagnostic: zones=%s, recommended=%s",
        crs_strategy_summary["intersecting_utm_zones"],
        crs_strategy_summary["recommended_crs_strategy"],
    )
    crs_processing_summary = determine_crs_processing_mode(
        config=config,
        aoi=aoi,
    )
    logging.info(
        "CRS processing mode: requested=%s, resolved=%s, segments=%s",
        crs_processing_summary["requested_processing_mode"],
        crs_processing_summary["resolved_processing_mode"],
        crs_processing_summary["n_utm_segments"],
    )

    record_stage("aoi_creation_seconds", stage_start, stage_timings)

    # 2. Building data acquisition
    stage_start = time.perf_counter()

    cache_config = config.get("cache", {})
    cache_enabled = bool(cache_config.get("enabled", False))
    use_existing_raw_buildings = bool(
        cache_config.get("use_existing_raw_buildings", False)
    )
    force_refresh = bool(cache_config.get("force_refresh", False))
    (
        cache_source_output_dir,
        cache_source_output_name,
        used_external_cache_source,
    ) = resolve_cache_source_output_dir(
        cache_config=cache_config,
        current_output_dir=output_dir,
        project_root=PROJECT_ROOT,
    )

    raw_buildings_output_path = folders["raw"] / "buildings_raw_overture.gpkg"
    raw_buildings_path = resolve_geodata_cache_path(
        cache_source_output_dir / "raw" / "buildings_raw_overture.gpkg"
    )
    raw_cache_manifest_path = (
        cache_source_output_dir / "reports" / "cache_manifest.json"
    )
    require_compatible_manifest = bool(
        cache_config.get("require_compatible_manifest", False)
    )
    building_source_summary: dict[str, Any] | None = None
    raw_cache_decision: dict[str, Any] | None = None
    cached_enriched_buildings_path = resolve_geodata_cache_path(
        cache_source_output_dir / "processed" / "buildings_height_enriched.gpkg"
    )
    cached_cleaned_buildings_path = resolve_geodata_cache_path(
        cache_source_output_dir / "processed" / "buildings_clean.gpkg"
    )
    height_enrichment_quality_path = (
        cache_source_output_dir / "reports" / "height_enrichment_quality.json"
    )
    requested_target_crs = config.get("preprocessing", {}).get(
        "target_crs", "auto_utm"
    )
    if requested_target_crs in {"auto", "auto_utm"}:
        requested_target_crs = estimate_metric_crs(aoi)
    cache_request_manifest = {
        **_aoi_cache_identity(aoi),
        **canonical_metric_aoi_identity(aoi, requested_target_crs),
        **acquisition_query_wgs84_identity(aoi),
        **_cache_relevant_settings(config),
        "target_metric_crs": str(requested_target_crs),
    }
    refresh_artifact_contracts(cache_request_manifest)
    (
        source_cache_manifest,
        source_cache_manifest_found,
        cache_manifest_metadata_normalizations,
    ) = load_cache_manifest_with_legacy_metadata(
        cache_source_output_dir
    )
    preflight_cache_compatibility = compare_cache_manifests(
        current_manifest=cache_request_manifest,
        source_manifest=source_cache_manifest,
    )
    artifact_cache_plan = compare_artifact_contracts(
        current_manifest=cache_request_manifest,
        source_manifest=source_cache_manifest,
    )
    compatible_building_core_cache = contract_is_compatible(artifact_cache_plan, "building_core")
    compatible_enriched_building_cache = contract_is_compatible(artifact_cache_plan, "enriched_buildings")
    compatible_neighbor_cache = contract_is_compatible(artifact_cache_plan, "neighbor_context")
    compatible_grid_cache = contract_is_compatible(artifact_cache_plan, "canonical_grid")
    compatible_street_network_cache = contract_is_compatible(artifact_cache_plan, "street_network")
    compatible_street_profiles_cache = contract_is_compatible(artifact_cache_plan, "street_profiles")
    compatible_street_assignment_cache = contract_is_compatible(artifact_cache_plan, "street_assignments")
    stage_signature_hash = hashlib.sha256(
        json.dumps(
            cache_request_manifest,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    stage_tracker.mark(
        "footprint_core",
        "processing",
        signature_hash=stage_signature_hash,
    )
    stage_tracker.mark(
        "vertical_density",
        "processing",
        signature_hash=stage_signature_hash,
    )
    if crs_processing_summary["resolved_processing_mode"] == "segmented_utm":
        validate_segmented_core_config(config)

        stage_start = time.perf_counter()
        logging.info("Acquiring building data for segmented UTM core processing.")
        buildings_raw = acquire_buildings(config, aoi)
        record_stage("building_acquisition_seconds", stage_start, stage_timings)

        stage_start = time.perf_counter()
        segmented_result = process_segmented_core_indicators(
            buildings=buildings_raw,
            aoi=aoi,
            config=config,
            output_dir=output_dir,
            save_outputs=True,
            project_root=PROJECT_ROOT,
        )
        indicator_grid = segmented_result["indicator_grid"]
        segmented_crs_summary = segmented_result["summary"]
        height_enrichment_summary = segmented_crs_summary.get(
            "height_enrichment_summary"
        )
        street_profile_quality = segmented_crs_summary.get(
            "street_profile_summary"
        )
        record_stage("segmented_core_processing_seconds", stage_start, stage_timings)

        stage_start = time.perf_counter()
        diagnostics = compute_indicator_diagnostics(indicator_grid)
        gsi_sanity_summary = summarize_gsi_sanity(
            indicator_grid=indicator_grid,
            gsi_over_1_diagnostics=None,
        )
        record_stage("indicator_diagnostics_seconds", stage_start, stage_timings)

        building_quality = {
            "n_buildings": int(
                sum(segment["n_buildings"] for segment in segmented_crs_summary["segments"])
            ),
            "status": "segmented_utm_core_processing",
            "crs": "per_segment_utm",
            "is_projected": True,
        }
        unit_quality = {
            "n_units": int(len(indicator_grid)),
            "status": "segmented_utm_core_processing",
            "crs": "EPSG:4326_output_with_per_segment_metric_fields",
            "is_projected": False,
        }
        indicator_readiness = {
            "gsi": {
                "status": "ready",
                "reason": "Calculated per segment in the segment UTM CRS.",
            },
            "far_fsi": {
                "status": (
                    "conditional_ready"
                    if "far_fsi" in indicator_grid.columns
                    else "not_calculated_missing_floor_data_or_disabled"
                ),
                "reason": "Calculated only when floor attributes are available.",
            },
            "built_volume_density": {
                "status": (
                    "conditional_ready"
                    if "built_volume_density" in indicator_grid.columns
                    else "not_calculated_missing_height_data_or_disabled"
                ),
                "reason": "Calculated only when height attributes are available.",
            },
        }

        if config.get("indicators", {}).get("neighbor_distance", True):
            indicator_readiness["neighbor_distance"] = {
                "status": "ready",
                "reason": (
                    "Calculated from full building geometries using each "
                    "segment's context buffer, then aggregated to the segment grid."
                ),
            }

        if config.get("street_context", {}).get("enabled", False):
            indicator_readiness["street_context"] = {
                "status": "ready",
                "reason": (
                    "Calculated from context streets and context buildings using "
                    "each segment's context buffer, then aggregated only from "
                    "target segment buildings."
                ),
            }

        cache_summary = {
            "used_cached_enriched_buildings": False,
            "cached_enriched_buildings_path": None,
            "cached_height_enrichment_metadata_loaded": False,
            "cache_source_output_name": cache_source_output_name,
            "cache_source_output_dir": str(cache_source_output_dir),
            "used_external_cache_source": bool(used_external_cache_source),
            "used_cached_street_context": False,
            "cached_street_context_path": None,
            "cached_street_profile_quality_loaded": False,
            "cache_manifest_written": False,
            "cache_manifest_metadata_normalizations": cache_manifest_metadata_normalizations,
            "cache_source_manifest_found": False,
            "cache_source_compatibility_status": "not_applicable",
            "cache_source_compatibility_warnings": [],
            "cache_source_aoi_hash": None,
            "current_aoi_hash": None,
        }

        workflow_summary = build_workflow_summary(
            config=config,
            building_quality=building_quality,
            unit_quality=unit_quality,
            diagnostics=diagnostics,
            indicator_grid=indicator_grid,
            neighbor_diagnostics_summary=None,
            street_profile_quality=street_profile_quality,
            gsi_sanity_summary=gsi_sanity_summary,
            height_enrichment_summary=height_enrichment_summary,
            crs_strategy_summary=crs_strategy_summary,
            crs_processing_summary=crs_processing_summary,
            cache_summary=cache_summary,
        )
        workflow_summary = add_segmented_workflow_summary_metadata(
            workflow_summary=workflow_summary,
            segmented_crs_summary=segmented_crs_summary,
        )
        indicator_interpretation_records = build_indicator_readiness_records(
            workflow_summary
        )
        write_indicator_readiness_outputs(
            records=indicator_interpretation_records,
            reports_dir=folders["reports"],
            tables_dir=folders["tables"],
            save_reports=config.get("outputs", {}).get("save_reports", True),
            save_tables=config.get("outputs", {}).get("save_tables", True),
            workflow_summary=workflow_summary,
        )

        if config.get("outputs", {}).get("save_tables", True):
            pd.DataFrame([workflow_summary]).to_csv(
                folders["tables"] / "workflow_summary.csv",
                index=False,
            )
            indicator_grid.drop(columns="geometry").to_csv(
                folders["tables"] / "grid_indicators_segmented_wgs84.csv",
                index=False,
            )

        if config.get("outputs", {}).get("save_reports", True):
            (folders["reports"] / "workflow_summary.json").write_text(
                json.dumps(make_json_serializable(workflow_summary), indent=2),
                encoding="utf-8",
            )
            (folders["reports"] / "indicator_diagnostics.json").write_text(
                json.dumps(make_json_serializable(diagnostics), indent=2),
                encoding="utf-8",
            )
            (folders["reports"] / "gsi_sanity_summary.json").write_text(
                json.dumps(make_json_serializable(gsi_sanity_summary), indent=2),
                encoding="utf-8",
            )
            (folders["reports"] / "crs_strategy.json").write_text(
                json.dumps(make_json_serializable(crs_strategy_summary), indent=2),
                encoding="utf-8",
            )
            metadata = {
                "run_name": config["project"]["run_name"],
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "config_path": str(config_path),
                "output_dir": str(output_dir),
                "target_crs": "segmented_utm",
                "building_quality": building_quality,
                "unit_quality": unit_quality,
                "indicator_readiness": indicator_readiness,
                "indicator_interpretation_readiness": (
                    indicator_interpretation_records
                ),
                "diagnostics": diagnostics,
                "gsi_sanity_summary": gsi_sanity_summary,
                "workflow_summary": workflow_summary,
                "crs_strategy_summary": crs_strategy_summary,
                "crs_processing_summary": crs_processing_summary,
                "segmented_crs_summary": segmented_crs_summary,
                "height_enrichment_summary": height_enrichment_summary,
                "street_profile_quality": street_profile_quality,
                "cache_summary": cache_summary,
            }
            (folders["reports"] / "run_metadata.json").write_text(
                json.dumps(make_json_serializable(metadata), indent=2),
                encoding="utf-8",
            )
            write_markdown_quality_report(
                path=folders["reports"] / "quality_report.md",
                building_quality=building_quality,
                unit_quality=unit_quality,
                indicator_readiness=indicator_readiness,
                diagnostics=diagnostics,
                neighbor_diagnostics_summary=None,
                street_profile_quality=street_profile_quality,
                gsi_sanity_summary=gsi_sanity_summary,
                saved_maps=[],
                height_enrichment_summary=height_enrichment_summary,
                workflow_summary=workflow_summary,
                crs_strategy_summary=crs_strategy_summary,
                crs_processing_summary=crs_processing_summary,
                cache_summary=cache_summary,
                indicator_interpretation_records=indicator_interpretation_records,
            )

        record_stage("output_writing_seconds", stage_start, stage_timings)
        stage_timings["total_runtime_seconds"] = float(
            time.perf_counter() - workflow_start
        )
        (folders["reports"] / "stage_timings.json").write_text(
            json.dumps(make_json_serializable(stage_timings), indent=2),
            encoding="utf-8",
        )
        pd.DataFrame([stage_timings]).to_csv(
            folders["tables"] / "stage_timings.csv",
            index=False,
        )
        logging.info("Segmented UTM core workflow completed successfully.")
        logging.info("Outputs saved to: %s", output_dir)
        return

    used_cached_enriched_buildings = should_use_cached_enriched_buildings(
        cache_config=cache_config,
        cached_enriched_buildings_path=cached_enriched_buildings_path,
    ) and compatible_enriched_building_cache
    used_cached_cleaned_buildings = (
        not used_cached_enriched_buildings
        and should_use_cached_cleaned_buildings(
            cache_config=cache_config,
            cached_cleaned_buildings_path=cached_cleaned_buildings_path,
        )
        and compatible_building_core_cache
    )
    if (
        should_use_cached_enriched_buildings(
            cache_config=cache_config,
            cached_enriched_buildings_path=cached_enriched_buildings_path,
        )
        and not compatible_enriched_building_cache
    ):
        logging.warning(
            "Enriched-building cache rejected: %s",
            preflight_cache_compatibility["cache_source_compatibility_warnings"],
        )
    if (
        should_use_cached_cleaned_buildings(
            cache_config=cache_config,
            cached_cleaned_buildings_path=cached_cleaned_buildings_path,
        )
        and not compatible_building_core_cache
    ):
        logging.warning(
            "Cleaned-building cache rejected: %s",
            preflight_cache_compatibility["cache_source_compatibility_warnings"],
        )
    height_enrichment_summary = None
    cached_height_enrichment_metadata_loaded = False
    buildings_clean_pre_enrichment: gpd.GeoDataFrame | None = None

    if used_cached_enriched_buildings:
        logging.info(
            "Loading cached processed/enriched buildings: %s",
            cached_enriched_buildings_path,
        )

        buildings_clean = read_geodata_layer(
            cached_enriched_buildings_path,
            layer="buildings_height_enriched",
        )
        buildings_clean = restore_singlepart_polygon_types(buildings_clean)
        verify_cached_building_artifact_hash(
            buildings_clean, source_cache_manifest, "buildings_height_enriched"
        )

        if buildings_clean.crs is None:
            raise ValueError(
                "Cached enriched buildings have no CRS: "
                f"{cached_enriched_buildings_path}"
            )

        target_crs = buildings_clean.crs.to_string()
        aoi_metric_path = resolve_geodata_cache_path(
            cache_source_output_dir / "processed" / "aoi_metric.gpkg"
        )

        if aoi_metric_path.exists():
            logging.info("Loading cached metric AOI: %s", aoi_metric_path)
            aoi_metric = read_geodata_layer(aoi_metric_path, layer="aoi")

            if aoi_metric.crs != buildings_clean.crs:
                aoi_metric = aoi_metric.to_crs(buildings_clean.crs)
        else:
            logging.info(
                "Cached metric AOI not found; projecting AOI to cached building CRS."
            )
            aoi_metric = aoi.to_crs(buildings_clean.crs)

        (
            height_enrichment_summary,
            cached_height_enrichment_metadata_loaded,
        ) = load_height_enrichment_summary_if_available(
            height_enrichment_quality_path
        )

        if cached_height_enrichment_metadata_loaded:
            logging.info(
                "Loaded cached height enrichment metadata: %s",
                height_enrichment_quality_path,
            )
        else:
            logging.info(
                "Cached height enrichment metadata not available: %s",
                height_enrichment_quality_path,
            )

        logging.info(
            "Cached processed buildings ready. CRS=%s, features=%s",
            buildings_clean.crs,
            len(buildings_clean),
        )

        record_stage("cached_enriched_buildings_load_seconds", stage_start, stage_timings)
        stage_timings["building_acquisition_seconds"] = 0.0
        stage_timings["crs_preprocessing_seconds"] = 0.0
        stage_timings["height_enrichment_seconds"] = 0.0
        building_source_summary = build_building_source_summary(
            config=config,
            aoi=aoi,
            actual_building_source_used=(
                "external_cache" if used_external_cache_source else "compatible_enriched_cache"
            ),
            cache_decision={
                "cache_path": str(cached_enriched_buildings_path),
                "cache_compatibility_status": "reused",
                "cache_compatibility_reasons": [],
            },
            buildings_raw=None,
        )
        if source_cache_manifest is not None:
            cached_raw_bounds = source_cache_manifest.get(
                "raw_building_bounds_wgs84"
            )
            requested_bounds = _aoi_cache_identity(aoi)["aoi_bounds_wgs84"]
            building_source_summary.update(
                {
                    "raw_building_count": source_cache_manifest.get(
                        "raw_building_count"
                    ),
                    "raw_building_bounds_wgs84": cached_raw_bounds,
                    "raw_bounds_overlap_requested_aoi": (
                        _bounds_overlap(cached_raw_bounds, requested_bounds)
                        if cached_raw_bounds is not None
                        else None
                    ),
                    "raw_bounds_contain_requested_aoi": (
                        _bounds_contain(cached_raw_bounds, requested_bounds)
                        if cached_raw_bounds is not None
                        else None
                    ),
                }
            )
        write_building_source_summary(
            reports_dir=folders["reports"],
            summary=building_source_summary,
        )

    else:
        if used_cached_cleaned_buildings:
            logging.info(
                "Loading compatible cleaned/preprocessed buildings: %s",
                cached_cleaned_buildings_path,
            )
            buildings_clean = read_geodata_layer(
                cached_cleaned_buildings_path,
                layer="buildings_clean",
            )
            buildings_clean = restore_singlepart_polygon_types(buildings_clean)
            verify_cached_building_artifact_hash(
                buildings_clean, source_cache_manifest, "buildings_clean"
            )
            buildings_clean_pre_enrichment = buildings_clean.copy()
            if buildings_clean.crs is None:
                raise ValueError(
                    "Cached cleaned buildings have no CRS: "
                    f"{cached_cleaned_buildings_path}"
                )
            target_crs = buildings_clean.crs.to_string()
            aoi_metric_path = resolve_geodata_cache_path(
                cache_source_output_dir / "processed" / "aoi_metric.gpkg"
            )
            aoi_metric = (
                read_geodata_layer(aoi_metric_path, layer="aoi")
                if aoi_metric_path.exists()
                else aoi.to_crs(buildings_clean.crs)
            )
            if aoi_metric.crs != buildings_clean.crs:
                aoi_metric = aoi_metric.to_crs(buildings_clean.crs)

            record_stage(
                "cached_cleaned_buildings_load_seconds",
                stage_start,
                stage_timings,
            )
            stage_timings["building_acquisition_seconds"] = 0.0
            stage_timings["crs_preprocessing_seconds"] = 0.0
            building_source_summary = build_building_source_summary(
                config=config,
                aoi=aoi,
                actual_building_source_used=(
                    "external_cache"
                    if used_external_cache_source
                    else "compatible_cleaned_cache"
                ),
                cache_decision={
                    "cache_path": str(cached_cleaned_buildings_path),
                    "cache_compatibility_status": "reused",
                    "cache_compatibility_reasons": [],
                },
                buildings_raw=None,
            )
            if source_cache_manifest is not None:
                building_source_summary.update(
                    {
                        "raw_building_count": source_cache_manifest.get(
                            "raw_building_count"
                        ),
                        "raw_building_bounds_wgs84": source_cache_manifest.get(
                            "raw_building_bounds_wgs84"
                        ),
                    }
                )
            write_building_source_summary(
                reports_dir=folders["reports"],
                summary=building_source_summary,
            )
        else:
            raw_cache_decision = evaluate_raw_building_cache(
                config=config,
                aoi=aoi,
                raw_buildings_path=raw_buildings_path,
                cache_manifest_path=raw_cache_manifest_path,
                require_compatible_manifest=require_compatible_manifest,
            )
            if (
                cache_enabled
                and use_existing_raw_buildings
                and not force_refresh
                and raw_cache_decision["use_cache"]
            ):
                logging.info("Loading cached raw building data: %s", raw_buildings_path)
                buildings_raw = read_geodata_layer(
                    raw_buildings_path,
                    layer="buildings_raw",
                )
                actual_building_source = "compatible_raw_cache"
            else:
                if cache_enabled and use_existing_raw_buildings and not force_refresh:
                    logging.info(
                        "Raw building cache rejected/refreshed. status=%s reasons=%s",
                        raw_cache_decision.get("cache_compatibility_status"),
                        raw_cache_decision.get("cache_compatibility_reasons"),
                    )
                logging.info("Acquiring building data.")
                buildings_raw = acquire_buildings(config, aoi)
                actual_building_source = "new_download"

                if config.get("outputs", {}).get("save_raw_buildings", True):
                    if research_mode:
                        buildings_raw.to_file(
                            raw_buildings_output_path,
                            layer="buildings_raw",
                            driver="GPKG",
                        )
                    else:
                        write_geodata_cache(
                            buildings_raw,
                            raw_buildings_output_path.with_suffix(".parquet"),
                        )

            raw_bounds_ok, raw_bounds_error = validate_raw_buildings_match_aoi(
                buildings_raw,
                aoi,
            )
            building_source_summary = build_building_source_summary(
                config=config,
                aoi=aoi,
                actual_building_source_used=actual_building_source,
                cache_decision=raw_cache_decision,
                buildings_raw=buildings_raw,
            )
            if not raw_bounds_ok:
                building_source_summary["actual_building_source_used"] = "failed"
                building_source_summary["cache_compatibility_status"] = "failed"
                write_building_source_summary(
                    reports_dir=folders["reports"],
                    summary=building_source_summary,
                )
                write_failure_summary(
                    reports_dir=folders["reports"],
                    config=config,
                    aoi=aoi,
                    failure_stage="raw_building_spatial_validation",
                    technical_error=raw_bounds_error or "",
                    friendly_error_category="building_source_aoi_mismatch",
                    cache_decision=raw_cache_decision,
                    buildings_raw=buildings_raw,
                )
                raise ValueError(raw_bounds_error)

            write_building_source_summary(
                reports_dir=folders["reports"],
                summary=building_source_summary,
            )
            logging.info(
                "Building data ready. CRS=%s, features=%s",
                buildings_raw.crs,
                len(buildings_raw),
            )
            record_stage("building_acquisition_seconds", stage_start, stage_timings)

            stage_start = time.perf_counter()
            logging.info("Estimating metric CRS.")
            preprocessing_config = config.get("preprocessing", {})
            target_crs = preprocessing_config.get("target_crs", "auto_utm")
            if target_crs in ["auto", "auto_utm"]:
                target_crs = estimate_metric_crs(aoi)
            logging.info("Target metric CRS: %s", target_crs)
            buildings_metric, aoi_metric = reproject_to_metric(
                buildings=buildings_raw,
                aoi=aoi,
                target_crs=target_crs,
            )
            logging.info("Cleaning building geometries.")
            buildings_clean = clean_building_geometries(buildings_metric)
            if preprocessing_config.get("clip_to_aoi", True):
                logging.info("Clipping buildings to AOI.")
                buildings_clean = clip_buildings_to_aoi(buildings_clean, aoi_metric)
            logging.info("Adding footprint area.")
            buildings_clean = add_footprint_area(buildings_clean)
            buildings_clean_pre_enrichment = buildings_clean.copy()
            logging.info(
                "Processed buildings ready. CRS=%s, features=%s",
                buildings_clean.crs,
                len(buildings_clean),
            )
            record_stage("crs_preprocessing_seconds", stage_start, stage_timings)

        stage_start = time.perf_counter()
        height_cfg = config.get("height_enrichment", {})
        
        if height_cfg.get("enabled", False):
            logging.info(
                "Height enrichment enabled: GlobalBuildingAtlas LoD1, fill missing only."
            )
        
            cache_dir = Path(
                height_cfg.get(
                    "cache_dir",
                    PROJECT_ROOT / "04_outputs" / "_cache" / "gba_lod1_parquet",
                )
            )
        
            if not cache_dir.is_absolute():
                cache_dir = PROJECT_ROOT / cache_dir
        
            buildings_clean, height_enrichment_summary, gba_subset, gba_best_matches = (
                enrich_missing_heights_with_gba_lod1(
                    buildings=buildings_clean,
                    cache_dir=cache_dir,
                    base_url=height_cfg.get(
                        "base_url",
                        "https://data.source.coop/tge-labs/globalbuildingatlas-lod1",
                    ),
                    height_col="height_m",
                    min_overlap_share=float(
                        height_cfg.get("min_overlap_share", 0.2)
                    ),
                    min_valid_height_m=float(
                        height_cfg.get("min_valid_height_m", 2.0)
                    ),
                    bbox_buffer_deg=float(
                        height_cfg.get("bbox_buffer_deg", 0.002)
                    ),
                    max_download_size_mb=float(
                        height_cfg.get("max_download_size_mb", 2000)
                    ),
                    replace_existing_height=bool(
                        height_cfg.get("replace_existing_height", False)
                    ),
                )
            )
        
            write_height_enrichment_outputs(
                buildings_enriched=buildings_clean,
                summary=height_enrichment_summary,
                gba_subset=gba_subset,
                best_matches=gba_best_matches,
                output_dirs=folders,
                save_enriched_buildings=bool(
                    height_cfg.get("save_enriched_buildings", True)
                    and research_mode
                ),
                save_gba_subset=bool(
                    height_cfg.get("save_gba_subset", True)
                    and research_mode
                ),
                save_matches=bool(
                    height_cfg.get("save_matches", True)
                    and research_mode
                ),
            )
            if (
                not research_mode
                and height_cfg.get("save_enriched_buildings", True)
            ):
                write_geodata_cache(
                    buildings_clean,
                    folders["processed"] / "buildings_height_enriched.parquet",
                )
        
            logging.info(
                (
                    "Height enrichment completed: valid height share %.3f -> %.3f; "
                    "enriched %s buildings."
                ),
                height_enrichment_summary.get(
                    "valid_height_share_before",
                    float("nan"),
                ),
                height_enrichment_summary.get(
                    "valid_height_share_after",
                    float("nan"),
                ),
                height_enrichment_summary.get("height_enriched_count"),
            )
        
        else:
            logging.info("Height enrichment disabled.")
        
        record_stage("height_enrichment_seconds", stage_start, stage_timings)

    cache_summary = {
        "used_cached_enriched_buildings": bool(used_cached_enriched_buildings),
        "cached_enriched_buildings_path": (
            str(cached_enriched_buildings_path)
            if used_cached_enriched_buildings
            else None
        ),
        "cached_height_enrichment_metadata_loaded": bool(
            cached_height_enrichment_metadata_loaded
        ),
        "used_cached_cleaned_buildings": bool(used_cached_cleaned_buildings),
        "cached_cleaned_buildings_path": (
            str(cached_cleaned_buildings_path)
            if used_cached_cleaned_buildings
            else None
        ),
        "cache_source_output_name": cache_source_output_name,
        "cache_source_output_dir": str(cache_source_output_dir),
        "used_external_cache_source": bool(used_external_cache_source),
        "used_cached_street_context": False,
        "cached_street_context_path": None,
        "cached_street_profile_quality_loaded": False,
        "cache_manifest_metadata_normalizations": cache_manifest_metadata_normalizations,
    }

    current_cache_manifest = build_cache_manifest(
        config=config,
        aoi=aoi,
        target_crs=target_crs,
        buildings_clean=buildings_clean_pre_enrichment,
        buildings_raw=buildings_raw if "buildings_raw" in locals() else None,
        buildings_height_enriched=(
            buildings_clean if config.get("height_enrichment", {}).get("enabled", False) else None
        ),
    )

    source_cache_manifest = None

    if used_external_cache_source:
        (
            source_cache_manifest,
            _source_cache_manifest_found,
            cache_manifest_metadata_normalizations,
        ) = load_cache_manifest_with_legacy_metadata(
            cache_source_output_dir
        )
        artifact_cache_plan = compare_artifact_contracts(
            current_manifest=current_cache_manifest,
            source_manifest=source_cache_manifest,
        )
        used_artifacts = (["enriched_buildings"] if used_cached_enriched_buildings else
                          ["building_core"] if used_cached_cleaned_buildings else [])
        cache_compatibility = {
            "cache_source_manifest_found": bool(source_cache_manifest),
            "cache_source_compatibility_status": "compatible" if all(
                contract_is_compatible(artifact_cache_plan, name) for name in used_artifacts
            ) else "mismatch_detected",
            "cache_source_compatibility_warnings": [reason for name in used_artifacts for reason in artifact_cache_plan[name]["reasons"]],
            "cache_source_aoi_hash": (source_cache_manifest or {}).get("canonical_aoi_metric_hash"),
            "current_aoi_hash": current_cache_manifest.get("canonical_aoi_metric_hash"),
            "artifact_cache_plan": artifact_cache_plan,
        }
    else:
        cache_compatibility = {
            "cache_source_manifest_found": False,
            "cache_source_compatibility_status": "not_applicable",
            "cache_source_compatibility_warnings": [],
            "cache_source_aoi_hash": None,
            "current_aoi_hash": current_cache_manifest.get("aoi_geometry_hash"),
        }

    cache_summary.update(
        {
            "cache_manifest_written": bool(
                config.get("outputs", {}).get("save_reports", True)
            ),
            **cache_compatibility,
        }
    )

    if (
        bool(cache_config.get("require_compatible_manifest", False))
        and used_external_cache_source
        and cache_summary["cache_source_compatibility_status"] != "compatible"
    ):
        raise ValueError(
            "External cache source is not compatible with the current run: "
            f"{cache_summary['cache_source_compatibility_status']}"
        )

    if config.get("outputs", {}).get("save_processed_buildings", True):
        if research_mode and buildings_clean_pre_enrichment is not None:
            buildings_clean_pre_enrichment.to_file(
                folders["processed"] / "buildings_clean.gpkg",
                layer="buildings_clean",
                driver="GPKG",
            )
        elif buildings_clean_pre_enrichment is not None:
            write_geodata_cache(
                buildings_clean_pre_enrichment,
                folders["processed"] / "buildings_clean.parquet",
            )

        if config.get("height_enrichment", {}).get("enabled", False):
            enriched_path = folders["processed"] / ("buildings_height_enriched.gpkg" if research_mode else "buildings_height_enriched.parquet")
            if research_mode:
                buildings_clean.to_file(enriched_path, layer="buildings_height_enriched", driver="GPKG")
            else:
                write_geodata_cache(buildings_clean, enriched_path)

        aoi_metric.to_file(
            folders["processed"] / "aoi_metric.gpkg",
            layer="aoi",
            driver="GPKG",
        )

    # 5. Quality checks
    stage_start = time.perf_counter()
    logging.info("Running building quality checks.")
    building_quality = summarize_building_quality(buildings_clean)
    indicator_readiness = check_indicator_readiness(buildings_clean)
    record_stage("quality_checks_seconds", stage_start, stage_timings)

    # 6. Aggregation grid
    stage_start = time.perf_counter()
    aggregation_config = config["aggregation"]

    if aggregation_config.get("method", "regular_grid") != "regular_grid":
        raise NotImplementedError(
            "Only regular_grid aggregation is implemented in workflow v0.1."
        )

    logging.info("Creating aggregation grid.")
    grid = create_grid(
        aoi=aoi_metric,
        cell_size_m=int(aggregation_config["cell_size_m"]),
        grid_id_convention=aggregation_config.get("grid_id_convention", CANONICAL_GRID_ID_CONVENTION),
    )
    current_cache_manifest.update(canonical_grid_identity(
        grid,
        origin_x=float(aoi_metric.total_bounds[0]),
        origin_y=float(aoi_metric.total_bounds[1]),
        cell_size_m=float(aggregation_config["cell_size_m"]),
    ))
    refresh_artifact_contracts(current_cache_manifest)

    unit_quality = summarize_unit_quality(grid)

    logging.info("Grid created. CRS=%s, cells=%s", grid.crs, len(grid))

    if config.get("outputs", {}).get("save_grid", True):
        grid.to_file(
            folders["processed"] / "aggregation_grid.gpkg",
            layer="grid",
            driver="GPKG",
        )

    record_stage("grid_creation_seconds", stage_start, stage_timings)

    neighbor_diagnostics_summary = None
    neighbor_diagnostics = None
    gsi_sanity_summary = None
    street_profile_quality = None
    workflow_summary = None
    saved_maps: list[Path] = []

    # Reuse or compute building-level nearest-neighbour values once. The same
    # table is used for grid aggregation and optional diagnostics.
    neighbor_enabled = bool(config.get("indicators", {}).get("neighbor_distance", True))
    save_neighbor_diagnostics = bool(
        config.get("outputs", {}).get("save_neighbor_diagnostics", False)
    )
    cached_neighbor_path = resolve_geodata_cache_path(
        cache_source_output_dir / "processed" / "building_neighbor_diagnostics.gpkg"
    )
    use_cached_neighbor = bool(
        cache_enabled
        and cache_config.get("use_existing_neighbor_context", False)
        and not force_refresh
        and compatible_neighbor_cache
        and cached_neighbor_path.exists()
    )
    if neighbor_enabled and (save_neighbor_diagnostics or use_cached_neighbor):
        neighbor_stage_start = time.perf_counter()
        if use_cached_neighbor:
            logging.info(
                "Loading compatible building-level neighbour cache: %s",
                cached_neighbor_path,
            )
            neighbor_diagnostics = read_geodata_layer(
                cached_neighbor_path,
                layer="building_neighbor_diagnostics",
            )
            neighbor_diagnostics = restore_singlepart_polygon_types(
                neighbor_diagnostics
            )
            stage_timings["neighbor_context_cache_hit"] = True
        else:
            logging.info("Calculating building-level neighbour diagnostics once.")
            neighbor_diagnostics = calculate_building_neighbor_diagnostics(
                buildings=buildings_clean,
                aoi=aoi_metric,
                params=config.get("indicator_parameters", {}),
            )
            stage_timings["neighbor_context_cache_hit"] = False
        if neighbor_diagnostics.crs != grid.crs:
            raise ValueError(
                "Cached neighbour-context CRS does not match current grid CRS. "
                f"cached={neighbor_diagnostics.crs}; grid={grid.crs}"
            )
        neighbor_diagnostics_summary = summarize_neighbor_diagnostics(
            neighbor_diagnostics=neighbor_diagnostics,
            params=config.get("indicator_parameters", {}),
        )
        stage_timings["neighbor_diagnostics_seconds"] = float(
            time.perf_counter() - neighbor_stage_start
        )

    # 7. Indicator calculation
    stage_start = time.perf_counter()
    logging.info("Running indicators.")
    area_performance_metrics: dict[str, Any] = {}
    indicator_grid = run_indicators(
        buildings=buildings_clean,
        units=grid,
        config=config,
        precomputed_neighbor_table=neighbor_diagnostics,
        performance_metrics=area_performance_metrics,
    )
    stage_timings.update(area_performance_metrics)

    logging.info(
        "Indicators calculated. Cells=%s, columns=%s",
        len(indicator_grid),
        len(indicator_grid.columns),
    )
    record_stage("indicator_calculation_seconds", stage_start, stage_timings)
    stage_tracker.mark(
        "footprint_core",
        "completed",
        signature_hash=stage_signature_hash,
        cache_used=bool(used_cached_enriched_buildings),
    )
    stage_tracker.mark(
        "vertical_density",
        "completed",
        signature_hash=stage_signature_hash,
        cache_used=bool(used_cached_enriched_buildings),
    )

    stage_start = time.perf_counter()
    stage_tracker.mark(
        "contextual_morphology",
        "processing",
        signature_hash=stage_signature_hash,
    )

    # -------------------------------------------------------------------------
    # Optional street-context branch
    # -------------------------------------------------------------------------
    street_context_cfg = config.get("street_context", {})
    street_context_enabled = bool(street_context_cfg.get("enabled", False))

    if street_context_enabled:
        logging.info("Street-context branch enabled.")

        cached_street_context_path = resolve_geodata_cache_path(
            cache_source_output_dir
            / "processed"
            / "building_street_profile_ratio.gpkg"
        )
        cached_street_profile_quality_path = (
            cache_source_output_dir / "reports" / "street_profile_quality.json"
        )
        output_street_profile_quality_path = (
            folders["reports"] / "street_profile_quality.json"
        )
        used_cached_street_context = should_use_cached_street_context(
            cache_config=cache_config,
            cached_street_context_path=cached_street_context_path,
        ) and compatible_street_assignment_cache

        if used_cached_street_context:
            logging.info(
                "Loading cached building-level street context: %s",
                cached_street_context_path,
            )

            building_street = read_geodata_layer(
                cached_street_context_path,
                layer="building_street_profile_ratio",
            )
            building_street = restore_singlepart_polygon_types(building_street)

            if building_street.crs != grid.crs:
                raise ValueError(
                    "Cached street-context CRS does not match current grid CRS. "
                    f"cached={building_street.crs}; grid={grid.crs}"
                )

            (
                street_profile_quality,
                cached_street_profile_quality_loaded,
            ) = load_json_if_available(cached_street_profile_quality_path)

            if street_profile_quality is None:
                street_profile_quality = {}

            cache_summary.update(
                {
                    "used_cached_street_context": True,
                    "cached_street_context_path": str(cached_street_context_path),
                    "cached_street_profile_quality_loaded": bool(
                        cached_street_profile_quality_loaded
                    ),
                }
            )
            stage_timings["osm_street_acquisition_seconds"] = 0.0
            stage_timings["osm_street_acquisition_cache_hit"] = True
            stage_timings["street_profile_calculation_seconds"] = 0.0
            stage_timings["street_profile_calculation_cache_hit"] = True
            stage_timings["building_to_street_matching_seconds"] = 0.0
            stage_timings["building_to_street_candidate_pairs"] = int(
                (street_profile_quality or {}).get(
                    "raw_join_rows_before_deduplication", 0
                )
            )
            stage_timings["building_to_street_duplicates_removed"] = int(
                (street_profile_quality or {}).get(
                    "duplicate_join_rows_removed", 0
                )
            )

        else:
            source = street_context_cfg.get("source", "osmnx")
            network_type = street_context_cfg.get("network_type", "drive")
            distance_m = float(street_context_cfg.get("distance_m", 10))
            tick_length_m = float(street_context_cfg.get("tick_length_m", 60))
            use_existing_if_available = bool(
                street_context_cfg.get("use_existing_if_available", True)
            )

            save_streets = bool(street_context_cfg.get("save_streets", True))
            save_street_profiles = bool(
                street_context_cfg.get("save_street_profiles", True)
            )
            save_building_street_assignment = bool(
                street_context_cfg.get("save_building_street_assignment", True)
            )

            if source != "osmnx":
                raise ValueError(
                    f"Unsupported street_context source: {source}. "
                    "Currently only 'osmnx' is implemented."
                )

            streets_output_path = folders["processed"] / "streets_osmnx.gpkg"
            cached_streets_path = resolve_geodata_cache_path(
                cache_source_output_dir / "processed" / "streets_osmnx.gpkg"
            )
            use_cached_streets = bool(
                cache_config.get(
                    "use_existing_street_network",
                    cache_config.get("use_existing_street_context", False),
                )
            )
            street_fetch_start = time.perf_counter()

            if (
                use_existing_if_available
                and use_cached_streets
                and compatible_street_network_cache
                and cached_streets_path.exists()
            ):
                logging.info("Loading compatible street network: %s", cached_streets_path)
                streets = read_geodata_layer(cached_streets_path, layer="streets_osmnx")
                stage_timings["osm_street_acquisition_seconds"] = float(
                    time.perf_counter() - street_fetch_start
                )
                stage_timings["osm_street_acquisition_cache_hit"] = True
            else:
                logging.info("Fetching street network from OSMnx.")
                try:
                    streets, street_acquisition = fetch_streets_from_osmnx(
                        aoi=aoi_metric,
                        network_type=network_type,
                        acquisition_config=street_context_cfg.get("acquisition"),
                        query_context={"scope": "analysis_aoi"},
                        return_provenance=True,
                    )
                except Exception as exc:
                    provenance = getattr(exc, "provenance", None)
                    if provenance:
                        (folders["reports"] / "street_acquisition_failure.json").write_text(
                            json.dumps(make_json_serializable(provenance), indent=2),
                            encoding="utf-8",
                        )
                    raise
                current_cache_manifest["street_acquisition"] = street_acquisition
                cache_summary["street_acquisition"] = street_acquisition

                if save_streets:
                    if research_mode:
                        streets.to_file(
                            streets_output_path,
                            layer="streets_osmnx",
                            driver="GPKG",
                        )
                    else:
                        write_geodata_cache(
                            streets,
                            streets_output_path.with_suffix(".parquet"),
                        )
                stage_timings["osm_street_acquisition_seconds"] = float(
                    time.perf_counter() - street_fetch_start
                )
                stage_timings["osm_street_acquisition_cache_hit"] = False

            cached_street_profiles_path = resolve_geodata_cache_path(
                cache_source_output_dir / "processed" / "street_profile_segments.gpkg"
            )
            use_cached_profiles = bool(
                cache_config.get("use_existing_street_profiles", False)
            )
            street_profile_start = time.perf_counter()
            if (
                use_cached_profiles
                and compatible_street_profiles_cache
                and cached_street_profiles_path.exists()
            ):
                logging.info(
                    "Loading compatible street-profile segments: %s",
                    cached_street_profiles_path,
                )
                streets_profile = read_geodata_layer(
                    cached_street_profiles_path,
                    layer="street_profile_segments",
                )
                stage_timings["street_profile_calculation_cache_hit"] = True
            else:
                streets_profile = calculate_street_profile_segments(
                    streets=streets,
                    buildings=buildings_clean,
                    height_col="height_m",
                    distance=distance_m,
                    tick_length=tick_length_m,
                )
                stage_timings["street_profile_calculation_cache_hit"] = False
            stage_timings["street_profile_calculation_seconds"] = float(
                time.perf_counter() - street_profile_start
            )

            if save_street_profiles:
                if research_mode:
                    streets_profile.to_file(
                        folders["processed"] / "street_profile_segments.gpkg",
                        layer="street_profile_segments",
                        driver="GPKG",
                    )
                    streets_profile.drop(columns="geometry").to_csv(
                        folders["tables"] / "street_profile_segments.csv",
                        index=False,
                    )
                else:
                    write_geodata_cache(
                        streets_profile,
                        folders["processed"] / "street_profile_segments.parquet",
                    )

            street_match_start = time.perf_counter()
            building_street, street_join_summary = assign_buildings_to_street_profiles(
                buildings=buildings_clean,
                streets_profile=streets_profile,
                building_id_col="building_id",
                height_col="height_m",
            )

            building_street = calculate_building_street_profile_ratio(
                building_street=building_street,
                height_col="height_m",
            )
            stage_timings["building_to_street_matching_seconds"] = float(
                time.perf_counter() - street_match_start
            )
            stage_timings["building_to_street_candidate_pairs"] = int(
                street_join_summary.get("raw_join_rows_before_deduplication", 0)
            )
            stage_timings["building_to_street_duplicates_removed"] = int(
                street_join_summary.get("duplicate_join_rows_removed", 0)
            )

            if save_building_street_assignment:
                if research_mode:
                    building_street.to_file(
                        folders["processed"] / "building_street_profile_ratio.gpkg",
                        layer="building_street_profile_ratio",
                        driver="GPKG",
                    )
                    building_street.drop(columns="geometry").to_csv(
                        folders["tables"] / "building_street_profile_ratio.csv",
                        index=False,
                    )
                else:
                    write_geodata_cache(
                        building_street,
                        folders["processed"] / "building_street_profile_ratio.parquet",
                    )

        street_grid_start = time.perf_counter()
        grid_street_profile = aggregate_street_profile_ratio_to_units(
            building_street=building_street,
            units=grid,
            unit_id_col="unit_id",
            building_id_col="building_id",
        )
        stage_timings["street_profile_grid_aggregation_seconds"] = float(
            time.perf_counter() - street_grid_start
        )

        if research_mode:
            grid_street_profile.to_file(
                folders["indicators"] / "grid_street_profile_ratio.gpkg",
                layer="grid_street_profile_ratio",
                driver="GPKG",
            )
            grid_street_profile.drop(columns="geometry").to_csv(
                folders["tables"] / "grid_street_profile_ratio.csv",
                index=False,
            )

        if used_cached_street_context:
            n_cells = int(len(grid_street_profile))
            prelim_cells = int(
                (
                    grid_street_profile[
                        "street_profile_ratio_prelim_valid_count"
                    ] > 0
                ).sum()
            )
            strict_cells = int(
                (
                    grid_street_profile[
                        "street_profile_ratio_strict_valid_count"
                    ] > 0
                ).sum()
            )
            street_profile_quality.update(
                {
                    "n_grid_cells": n_cells,
                    "grid_cells_with_prelim_ratio_count": prelim_cells,
                    "grid_cells_with_prelim_ratio_share": (
                        float(prelim_cells / n_cells) if n_cells > 0 else None
                    ),
                    "grid_cells_with_strict_ratio_count": strict_cells,
                    "grid_cells_with_strict_ratio_share": (
                        float(strict_cells / n_cells) if n_cells > 0 else None
                    ),
                    "used_cached_street_context": True,
                    "cached_street_context_path": str(cached_street_context_path),
                }
            )
        else:
            street_profile_quality = summarize_street_profile_quality(
                streets_profile=streets_profile,
                building_street=building_street,
                grid_street_profile=grid_street_profile,
                join_summary=street_join_summary,
            )

        output_street_profile_quality_path.write_text(
            json.dumps(
                make_json_serializable(street_profile_quality),
                indent=2,
            ),
            encoding="utf-8",
        )

        street_profile_cols = [
            "street_profile_building_count",
            "street_profile_ratio_prelim_valid_count",
            "street_profile_ratio_strict_valid_count",
            "street_profile_width_mean_m",
            "street_profile_width_median_m",
            "avg_street_profile_height_to_width_ratio_prelim",
            "median_street_profile_height_to_width_ratio_prelim",
            "avg_street_profile_height_to_width_ratio_strict",
            "median_street_profile_height_to_width_ratio_strict",
        ]

        existing_street_profile_cols = [
            col for col in street_profile_cols
            if col in indicator_grid.columns
        ]

        if existing_street_profile_cols:
            indicator_grid = indicator_grid.drop(
                columns=existing_street_profile_cols,
            )

        indicator_grid = indicator_grid.merge(
            grid_street_profile[["unit_id"] + street_profile_cols],
            on="unit_id",
            how="left",
        )

        logging.info(
            "Street-context branch complete. Valid strict ratio cells: %s/%s",
            int(
                (
                    indicator_grid[
                        "street_profile_ratio_strict_valid_count"
                    ] > 0
                ).sum()
            ),
            len(indicator_grid),
        )

    else:
        logging.info("Street-context branch disabled.")
    
    record_stage("street_context_seconds", stage_start, stage_timings)

    if config.get("outputs", {}).get("save_indicator_grid", True):
        indicator_grid.to_file(
            folders["indicators"] / "grid_indicators.gpkg",
            layer="grid_indicators",
            driver="GPKG",
        )
        if not research_mode:
            write_geodata_cache(
                compact_grid_for_dashboard(indicator_grid),
                folders["indicators"] / "grid_dashboard.parquet",
            )

    stage_start = time.perf_counter()

    # Optional static map export.
    # Maps are created after indicator calculation and use the same grid-level
    # indicator outputs that are saved to GeoPackage/CSV.
    if config.get("visualization", {}).get("save_static_maps", False):
        logging.info("Creating static workflow maps.")

        try:
            saved_maps = save_default_workflow_maps(
                indicator_grid=indicator_grid,
                output_dir=folders["maps"],
                config=config,
                aoi=aoi_metric,
            )

            logging.info(
                "Static maps saved: %s",
                [path.name for path in saved_maps],
            )

        except Exception as exc:
            logging.exception("Static map generation failed: %s", exc)
            saved_maps = []

    record_stage("static_maps_seconds", stage_start, stage_timings)

    # 8. Optional building-level neighbour diagnostics
    stage_start = time.perf_counter()
    if save_neighbor_diagnostics:
        if neighbor_diagnostics is None:
            logging.info("Calculating building-level neighbour diagnostics.")
            neighbor_diagnostics = calculate_building_neighbor_diagnostics(
                buildings=buildings_clean,
                aoi=aoi_metric,
                params=config.get("indicator_parameters", {}),
            )
            neighbor_diagnostics_summary = summarize_neighbor_diagnostics(
                neighbor_diagnostics=neighbor_diagnostics,
                params=config.get("indicator_parameters", {}),
            )

        if research_mode:
            neighbor_diagnostics.to_file(
                folders["processed"] / "building_neighbor_diagnostics.gpkg",
                layer="building_neighbor_diagnostics",
                driver="GPKG",
            )
            neighbor_diagnostics.drop(columns="geometry").to_csv(
                folders["tables"] / "building_neighbor_diagnostics.csv",
                index=False,
            )
        else:
            write_geodata_cache(
                neighbor_diagnostics,
                folders["processed"] / "building_neighbor_diagnostics.parquet",
            )

        neighbor_summary_path = folders["reports"] / "neighbor_diagnostics_summary.json"
        neighbor_summary_path.write_text(
            json.dumps(
                make_json_serializable(neighbor_diagnostics_summary),
                indent=2,
            ),
            encoding="utf-8",
        )

        logging.info(
            "Neighbour diagnostics saved. Buildings=%s, zero distances=%s, valid ratios=%s",
            len(neighbor_diagnostics),
            int(neighbor_diagnostics["is_zero_distance"].sum()),
            int(neighbor_diagnostics["has_valid_height_to_distance_ratio"].sum()),
        )

    neighbor_write_seconds = float(time.perf_counter() - stage_start)
    stage_timings["neighbor_diagnostics_writing_seconds"] = neighbor_write_seconds
    if "neighbor_diagnostics_seconds" not in stage_timings:
        stage_timings["neighbor_diagnostics_seconds"] = neighbor_write_seconds
    stage_tracker.mark(
        "contextual_morphology",
        "completed",
        signature_hash=stage_signature_hash,
        neighbor_cache_used=bool(use_cached_neighbor),
        street_cache_used=bool(
            cache_summary.get("used_cached_street_context", False)
        ),
    )

    # 9. Indicator diagnostics and GSI sanity diagnostics
    stage_start = time.perf_counter()
    diagnostics = compute_indicator_diagnostics(indicator_grid)

    gsi_over_1_diagnostics = pd.DataFrame()

    if "gsi" in indicator_grid.columns:
        raw_gsi_values = pd.to_numeric(
            indicator_grid.get("gsi_raw_sum", indicator_grid["gsi"]),
            errors="coerce",
        )
        gsi_over_1_cells = indicator_grid[raw_gsi_values > 1].copy()

        if not gsi_over_1_cells.empty:
            logging.warning(
                "Found %s cells with raw summed GSI > 1; official union GSI is retained.",
                len(gsi_over_1_cells),
            )

            if config.get("outputs", {}).get("save_gsi_sanity_diagnostics", False):
                logging.warning(
                    "Exporting raw-sum overlap diagnostics for inspection."
                )

                gsi_over_1_cells.to_file(
                    folders["indicators"] / "gsi_over_1_cells.gpkg",
                    layer="gsi_over_1_cells",
                    driver="GPKG",
                )

                gsi_over_1_cells.drop(columns="geometry").to_csv(
                    folders["tables"] / "gsi_over_1_cells.csv",
                    index=False,
                )

                gsi_over_1_diagnostics = diagnose_gsi_over_1_cells(
                    indicator_grid=indicator_grid,
                    buildings=buildings_clean,
                )

                gsi_over_1_diagnostics.to_csv(
                    folders["tables"] / "gsi_over_1_diagnostics.csv",
                    index=False,
                )

                logging.warning(
                    "GSI > 1 diagnostics saved. Max raw GSI=%s, max dissolved GSI=%s",
                    gsi_over_1_diagnostics["raw_gsi"].max(),
                    gsi_over_1_diagnostics["dissolved_gsi"].max(),
                )
        else:
            logging.info("No cells with raw-sum GSI > 1 found.")

    gsi_sanity_summary = summarize_gsi_sanity(
        indicator_grid=indicator_grid,
        gsi_over_1_diagnostics=gsi_over_1_diagnostics,
    )
    record_stage("indicator_diagnostics_seconds", stage_start, stage_timings)

    stage_start = time.perf_counter()
    workflow_summary = build_workflow_summary(
        config=config,
        building_quality=building_quality,
        unit_quality=unit_quality,
        diagnostics=diagnostics,
        indicator_grid=indicator_grid,
        neighbor_diagnostics_summary=neighbor_diagnostics_summary,
        street_profile_quality=street_profile_quality,
        gsi_sanity_summary=gsi_sanity_summary,
        height_enrichment_summary=height_enrichment_summary,
        crs_strategy_summary=crs_strategy_summary,
        crs_processing_summary=crs_processing_summary,
        cache_summary=cache_summary,
        
    )
    if building_source_summary is not None:
        workflow_summary.update(
            {
                "building_source_actual_source_used": building_source_summary.get(
                    "actual_building_source_used"
                ),
                "building_source_cache_compatibility_status": building_source_summary.get(
                    "cache_compatibility_status"
                ),
                "building_source_cache_compatibility_reasons": building_source_summary.get(
                    "cache_compatibility_reasons"
                ),
                "raw_building_count": building_source_summary.get(
                    "raw_building_count"
                ),
                "raw_building_bounds_wgs84": building_source_summary.get(
                    "raw_building_bounds_wgs84"
                ),
                "raw_bounds_overlap_requested_aoi": building_source_summary.get(
                    "raw_bounds_overlap_requested_aoi"
                ),
                "raw_bounds_contain_requested_aoi": building_source_summary.get(
                    "raw_bounds_contain_requested_aoi"
                ),
            }
        )
    indicator_interpretation_records = build_indicator_readiness_records(
        workflow_summary
    )
    write_indicator_readiness_outputs(
        records=indicator_interpretation_records,
        reports_dir=folders["reports"],
        tables_dir=folders["tables"],
        save_reports=config.get("outputs", {}).get("save_reports", True),
        save_tables=config.get("outputs", {}).get("save_tables", True),
        workflow_summary=workflow_summary,
    )
    record_stage("workflow_summary_seconds", stage_start, stage_timings)
    

    stage_tracker.mark(
        "presentation",
        "processing",
        signature_hash=stage_signature_hash,
    )

    # 10. Tables
    stage_start = time.perf_counter()
    if config.get("outputs", {}).get("save_tables", True):
        pd.DataFrame([building_quality]).to_csv(
            folders["tables"] / "building_quality_summary.csv",
            index=False,
        )
        if workflow_summary is not None:
            pd.DataFrame([workflow_summary]).to_csv(
                folders["tables"] / "workflow_summary.csv",
                index=False,
        )

        pd.DataFrame([unit_quality]).to_csv(
            folders["tables"] / "unit_quality_summary.csv",
            index=False,
        )

        if research_mode:
            indicator_grid.drop(columns="geometry").to_csv(
                folders["tables"] / "grid_indicators.csv",
                index=False,
            )

        indicator_cols = [
            "gsi",
            "far_fsi",
            "built_volume_density",
            "avg_neighbor_distance_m",
            "median_neighbor_distance_m",
            "avg_height_to_distance_ratio",
            "median_height_to_distance_ratio",
            "street_profile_width_mean_m",
            "street_profile_width_median_m",
            "avg_street_profile_height_to_width_ratio_strict",
            "median_street_profile_height_to_width_ratio_strict",
        ]

        existing_indicator_cols = [
            col for col in indicator_cols if col in indicator_grid.columns
        ]

        if existing_indicator_cols and research_mode:
            indicator_grid[existing_indicator_cols].describe().to_csv(
                folders["tables"] / "indicator_descriptive_statistics.csv"
            )

    # 11. Reports and metadata
    if config.get("outputs", {}).get("save_reports", True):
        diagnostics_path = folders["reports"] / "indicator_diagnostics.json"
        diagnostics_path.write_text(
            json.dumps(make_json_serializable(diagnostics), indent=2),
            encoding="utf-8",
        )

        gsi_sanity_summary_path = folders["reports"] / "gsi_sanity_summary.json"
        gsi_sanity_summary_path.write_text(
            json.dumps(
                make_json_serializable(gsi_sanity_summary),
                indent=2,
            ),
            encoding="utf-8",
        )
        crs_strategy_path = folders["reports"] / "crs_strategy.json"
        crs_strategy_path.write_text(
            json.dumps(
                make_json_serializable(crs_strategy_summary),
                indent=2,
            ),
            encoding="utf-8",
        )
        cache_manifest_path = folders["reports"] / "cache_manifest.json"
        write_cache_manifest(cache_manifest_path, current_cache_manifest)
        if workflow_summary is not None:
            workflow_summary_path = folders["reports"] / "workflow_summary.json"
            workflow_summary_path.write_text(
                json.dumps(
                    make_json_serializable(workflow_summary),
                    indent=2,
                ),
                encoding="utf-8",
        )

        relative_saved_maps = [path.relative_to(output_dir) for path in saved_maps]

        metadata = {
            "run_name": config["project"]["run_name"],
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "config_path": str(config_path),
            "output_dir": str(output_dir),
            "target_crs": str(target_crs),
            "building_quality": building_quality,
            "unit_quality": unit_quality,
            "indicator_readiness": indicator_readiness,
            "indicator_interpretation_readiness": indicator_interpretation_records,
            "diagnostics": diagnostics,
            "neighbor_diagnostics_summary": neighbor_diagnostics_summary,
            "gsi_sanity_summary": gsi_sanity_summary,
            "saved_maps": [str(path) for path in relative_saved_maps],
            "street_profile_quality": street_profile_quality,
            "workflow_summary": workflow_summary,
            "height_enrichment_summary": height_enrichment_summary,
            "crs_strategy_summary": crs_strategy_summary,
            "crs_processing_summary": crs_processing_summary,
            "cache_manifest": current_cache_manifest,
            "cache_summary": cache_summary,
            "building_source_summary": building_source_summary,
            
        }

        metadata_path = folders["reports"] / "run_metadata.json"
        metadata_path.write_text(
            json.dumps(make_json_serializable(metadata), indent=2),
            encoding="utf-8",
        )

        maps_inventory_path = folders["reports"] / "maps_inventory.json"
        maps_inventory_path.write_text(
            json.dumps(
                make_json_serializable(
                    {
                        "n_maps": len(relative_saved_maps),
                        "maps": [str(path) for path in relative_saved_maps],
                    }
                ),
                indent=2,
            ),
            encoding="utf-8",
        )

        write_markdown_quality_report(
            path=folders["reports"] / "quality_report.md",
            building_quality=building_quality,
            unit_quality=unit_quality,
            indicator_readiness=indicator_readiness,
            diagnostics=diagnostics,
            neighbor_diagnostics_summary=neighbor_diagnostics_summary,
            street_profile_quality=street_profile_quality,
            gsi_sanity_summary=gsi_sanity_summary,
            saved_maps=relative_saved_maps,
            height_enrichment_summary=height_enrichment_summary,
            workflow_summary=workflow_summary,
            crs_strategy_summary=crs_strategy_summary,
            crs_processing_summary=crs_processing_summary,
            cache_summary=cache_summary,
            indicator_interpretation_records=indicator_interpretation_records,
        )

    record_stage("output_writing_seconds", stage_start, stage_timings)
    stage_timings["total_runtime_seconds"] = float(
        time.perf_counter() - workflow_start
    )

    stage_rows = {
        "building_acquisition": (len(aoi), int(building_source_summary.get("raw_building_count") or len(buildings_clean)) if building_source_summary else len(buildings_clean)),
        "crs_preprocessing": (int(building_source_summary.get("raw_building_count") or len(buildings_clean)) if building_source_summary else len(buildings_clean), len(buildings_clean)),
        "height_enrichment": (len(buildings_clean), len(buildings_clean)),
        "grid_creation": (len(aoi_metric), len(grid)),
        "indicator_calculation": (len(buildings_clean), len(indicator_grid)),
        "street_context": (len(buildings_clean), len(indicator_grid)),
        "neighbor_diagnostics": (len(buildings_clean), len(neighbor_diagnostics) if neighbor_diagnostics is not None else 0),
    }
    cache_status_by_stage = {
        "building_acquisition": bool(
            building_source_summary
            and building_source_summary.get("actual_building_source_used")
            in {"compatible_raw_cache", "compatible_enriched_cache", "external_cache"}
        ),
        "height_enrichment": bool(used_cached_enriched_buildings),
        "street_context": bool(cache_summary.get("used_cached_street_context")),
        "neighbor_diagnostics": bool(use_cached_neighbor),
    }
    for timing_name, timing_value in stage_timings.items():
        if not timing_name.endswith("_seconds"):
            continue
        stage_name = timing_name.removesuffix("_seconds")
        input_rows, output_rows = stage_rows.get(stage_name, (None, None))
        performance_recorder.add_record(
            stage_name,
            status="completed",
            wall_clock_seconds=float(timing_value),
            input_rows=input_rows,
            output_rows=output_rows,
            candidate_pair_count=(
                stage_timings.get("building_grid_candidate_pairs")
                if stage_name == "indicator_calculation"
                else stage_timings.get("building_to_street_candidate_pairs")
                if stage_name == "building_to_street_matching"
                else None
            ),
            peak_process_memory_bytes_approx=process_rss_bytes(),
            dataframe_memory_bytes=(
                geodataframe_memory_bytes(buildings_clean)
                if stage_name in {"crs_preprocessing", "height_enrichment", "neighbor_diagnostics"}
                else geodataframe_memory_bytes(indicator_grid)
                if stage_name in {"indicator_calculation", "street_context"}
                else None
            ),
            bytes_read=(
                raw_buildings_path.stat().st_size
                if stage_name == "building_acquisition" and raw_buildings_path.exists()
                else cached_enriched_buildings_path.stat().st_size
                if stage_name == "height_enrichment" and used_cached_enriched_buildings
                else None
            ),
            bytes_written=None,
            cache_status=(
                "loaded_from_cache"
                if cache_status_by_stage.get(stage_name, False)
                else "computed"
            ),
        )
    performance_recorder.add_record(
        "workflow_total",
        status="completed",
        wall_clock_seconds=stage_timings["total_runtime_seconds"],
        input_rows=len(buildings_clean),
        output_rows=len(indicator_grid),
        peak_process_memory_bytes_approx=process_rss_bytes(),
        bytes_written=file_bytes(
            [path for path in output_dir.rglob("*") if path.is_file()]
        ),
        cache_status="mixed",
        output_mode=output_mode,
    )
    performance_recorder.write(folders["reports"], folders["tables"])
    stage_tracker.mark(
        "presentation",
        "completed",
        signature_hash=stage_signature_hash,
        output_mode=output_mode,
    )

    stage_timings_path = folders["reports"] / "stage_timings.json"
    stage_timings_path.write_text(
        json.dumps(make_json_serializable(stage_timings), indent=2),
        encoding="utf-8",
    )

    pd.DataFrame([stage_timings]).to_csv(
        folders["tables"] / "stage_timings.csv",
        index=False,
    )

    logging.info("Stage timings saved: %s", stage_timings_path)
    logging.info("Workflow completed successfully.")
    logging.info("Outputs saved to: %s", output_dir)


if __name__ == "__main__":
    default_config = CODE_DIR / "config" / "example_urban_area_100m.yaml"

    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1]).resolve()
    else:
        config_path = default_config

    run_workflow(config_path)
