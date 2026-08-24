from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import yaml


SAFE_RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
RUN_MODES = ("quick_2d", "standard", "full_context")


def validate_run_name(run_name: str) -> str:
    if not run_name or not run_name.strip():
        raise ValueError("Run name must be non-empty.")

    if not SAFE_RUN_NAME_PATTERN.match(run_name):
        raise ValueError(
            "Run name must start with a letter or number and contain only "
            "letters, numbers, underscores, or hyphens."
        )

    return run_name


def validate_bbox(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
) -> None:
    if not -180 <= min_lon <= 180:
        raise ValueError("Minimum longitude must be between -180 and 180.")
    if not -180 <= max_lon <= 180:
        raise ValueError("Maximum longitude must be between -180 and 180.")
    if not -90 <= min_lat <= 90:
        raise ValueError("Minimum latitude must be between -90 and 90.")
    if not -90 <= max_lat <= 90:
        raise ValueError("Maximum latitude must be between -90 and 90.")
    if min_lon >= max_lon:
        raise ValueError("Minimum longitude must be smaller than maximum longitude.")
    if min_lat >= max_lat:
        raise ValueError("Minimum latitude must be smaller than maximum latitude.")


def validate_grid_size(grid_size: float) -> int | float:
    if grid_size <= 0:
        raise ValueError("Grid size must be positive.")

    if float(grid_size).is_integer():
        return int(grid_size)

    return grid_size


def mode_settings(mode: str) -> dict[str, Any]:
    """
    Map interface run modes to the existing workflow config schema.

    The mapping is intentionally conservative. It only toggles existing
    workflow branches and does not introduce a new workflow schema.
    """
    if mode == "quick_2d":
        return {
            "indicators": {
                "gsi": True,
                "far_fsi": False,
                "built_volume_density": False,
                "neighbor_distance": False,
                "height_to_distance_ratio": False,
            },
            "height_enrichment_enabled": False,
            "street_context_enabled": False,
            "save_neighbor_diagnostics": False,
        }

    if mode == "standard":
        return {
            "indicators": {
                "gsi": True,
                "far_fsi": True,
                "built_volume_density": True,
                "neighbor_distance": False,
                "height_to_distance_ratio": False,
            },
            "height_enrichment_enabled": True,
            "street_context_enabled": False,
            "save_neighbor_diagnostics": False,
        }

    if mode == "full_context":
        return {
            "indicators": {
                "gsi": True,
                "far_fsi": True,
                "built_volume_density": True,
                "neighbor_distance": True,
                "height_to_distance_ratio": False,
            },
            "height_enrichment_enabled": True,
            "street_context_enabled": True,
            "save_neighbor_diagnostics": True,
        }

    raise ValueError(f"Unsupported run mode: {mode}")


def build_config(
    run_name: str,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    grid_size: float,
    mode: str,
) -> dict[str, Any]:
    run_name = validate_run_name(run_name)
    validate_bbox(
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
    )
    grid_size = validate_grid_size(grid_size)
    settings = mode_settings(mode)

    return {
        "project": {
            "run_name": run_name,
            "output_dir": f"04_outputs/{run_name}",
            "overwrite_existing_run": True,
        },
        "aoi": {
            "mode": "bbox",
            "name": run_name,
            "crs": "EPSG:4326",
            "bounds": {
                "minx": min_lon,
                "miny": min_lat,
                "maxx": max_lon,
                "maxy": max_lat,
            },
        },
        "crs_strategy": {
            "processing_mode": "single_crs",
            "context_buffer_m": 100,
        },
        "data_source": {
            "type": "overture",
            "overture_release": "auto",
            "release": "auto",
            "provider": "aws",
            "exclude_underground": True,
        },
        "preprocessing": {
            "target_crs": "auto_utm",
            "clean_geometries": True,
            "clip_to_aoi": True,
        },
        "aggregation": {
            "method": "regular_grid",
            "cell_size_m": grid_size,
        },
        "indicators": settings["indicators"],
        "indicator_parameters": {
            "min_distance_for_ratio_m": 0.5,
            "boundary_distance_threshold_m": 50.0,
            "high_ratio_threshold": 10.0,
        },
        "height_enrichment": {
            "enabled": settings["height_enrichment_enabled"],
            "source": "gba_lod1_parquet",
            "mode": "fill_missing_only",
            "replace_existing_height": False,
            "cache_dir": "04_outputs/_cache/gba_lod1_parquet",
            "base_url": "https://data.source.coop/tge-labs/globalbuildingatlas-lod1",
            "min_overlap_share": 0.2,
            "min_valid_height_m": 2.0,
            "bbox_buffer_deg": 0.002,
            "max_download_size_mb": 2000,
            "save_enriched_buildings": True,
            "save_gba_subset": True,
            "save_matches": True,
        },
        "street_context": {
            "enabled": settings["street_context_enabled"],
            "source": "osmnx",
            "network_type": "drive",
            "distance_m": 10,
            "tick_length_m": 60,
            "use_existing_if_available": True,
            "save_streets": True,
            "save_street_profiles": True,
            "save_building_street_assignment": True,
        },
        "outputs": {
            "save_raw_buildings": True,
            "save_processed_buildings": True,
            "save_grid": True,
            "save_indicator_grid": True,
            "save_neighbor_diagnostics": settings["save_neighbor_diagnostics"],
            "save_tables": True,
            "save_reports": True,
            "save_gsi_sanity_diagnostics": True,
        },
        "visualization": {
            "save_static_maps": False,
            "figure_dpi": 300,
            "figure_format": "png",
        },
        "cache": {
            "enabled": True,
            "use_existing_raw_buildings": True,
            "use_existing_enriched_buildings": False,
            "use_existing_street_context": False,
            "force_refresh": False,
            "require_compatible_manifest": True,
        },
    }


def write_config(config: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    return output_path


def create_config_from_bbox(
    run_name: str,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    grid_size: float,
    mode: str,
    output: Path,
) -> Path:
    config = build_config(
        run_name=run_name,
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
        grid_size=grid_size,
        mode=mode,
    )
    return write_config(config=config, output_path=output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a workflow YAML config from bbox coordinates."
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--min-lon", type=float, required=True)
    parser.add_argument("--min-lat", type=float, required=True)
    parser.add_argument("--max-lon", type=float, required=True)
    parser.add_argument("--max-lat", type=float, required=True)
    parser.add_argument("--grid-size", type=float, required=True)
    parser.add_argument("--mode", choices=RUN_MODES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = create_config_from_bbox(
        run_name=args.run_name,
        min_lon=args.min_lon,
        min_lat=args.min_lat,
        max_lon=args.max_lon,
        max_lat=args.max_lat,
        grid_size=args.grid_size,
        mode=args.mode,
        output=args.output,
    )
    print(f"Wrote config: {output_path}")


if __name__ == "__main__":
    main()
