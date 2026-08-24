from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from aggregation import create_grid
from height_enrichment import enrich_missing_heights_with_gba_lod1
from indicators import (
    calculate_built_volume_density,
    calculate_far_fsi,
    calculate_gsi,
)
from preprocessing import add_footprint_area, clean_building_geometries
from segmented_utm import segment_aoi_by_utm_zone, utm_epsg, utm_zone_from_longitude
from street_context import (
    aggregate_street_profile_ratio_to_units,
    assign_buildings_to_street_profiles,
    calculate_building_street_profile_ratio,
    calculate_street_profile_segments,
    fetch_streets_from_osmnx,
    summarize_street_profile_quality,
)


CORE_INDICATORS = {
    "gsi",
    "far_fsi",
    "built_volume_density",
}

UNSUPPORTED_SEGMENTED_INDICATORS = {
    "height_to_distance_ratio",
}


def validate_segmented_core_config(config: dict[str, Any]) -> None:
    """
    Reject staged segmented mode requests that need unsupported branches.
    """
    errors = []
    indicators_cfg = config.get("indicators", {})

    if config.get("visualization", {}).get("save_static_maps", False):
        errors.append("visualization.save_static_maps")

    for indicator in sorted(UNSUPPORTED_SEGMENTED_INDICATORS):
        if indicators_cfg.get(indicator, True):
            errors.append(f"indicators.{indicator}")

    cache_cfg = config.get("cache", {})

    if cache_cfg.get("use_existing_enriched_buildings", False):
        errors.append("cache.use_existing_enriched_buildings")

    if cache_cfg.get("use_existing_street_context", False):
        errors.append("cache.use_existing_street_context")

    if cache_cfg.get("source_output_name") or cache_cfg.get("source_output_dir"):
        errors.append("cache.source_output_name/source_output_dir")

    if errors:
        raise ValueError(
            "Segmented UTM processing currently supports core indicator "
            "calculation, optional GBA fill-missing-only height enrichment, "
            "nearest-neighbour distance, and street-profile context. "
            "Disable the following options for this staged implementation: "
            + ", ".join(errors)
        )


def _indicator_enabled(config: dict[str, Any], indicator: str) -> bool:
    return bool(config.get("indicators", {}).get(indicator, True))


def _segmented_context_buffer_m(config: dict[str, Any]) -> float:
    crs_strategy_cfg = config.get("crs_strategy", {})
    segmented_cfg = config.get("segmented_utm", {})
    context_buffer_m = crs_strategy_cfg.get(
        "context_buffer_m",
        segmented_cfg.get("context_buffer_m", 100),
    )
    context_buffer_m = float(context_buffer_m)

    if context_buffer_m < 0:
        raise ValueError("Segmented UTM context buffer must be >= 0 metres.")

    return context_buffer_m


def _valid_height_mask(series: pd.Series, min_height_m: float = 0.0) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.notna() & (values > min_height_m)


def _height_source_counts(buildings: gpd.GeoDataFrame) -> dict[str, int]:
    if "height_source" not in buildings.columns:
        return {
            "height_source_overture_count": int(
                _valid_height_mask(buildings["height_m"]).sum()
                if "height_m" in buildings.columns
                else 0
            ),
            "height_source_gba_lod1_count": 0,
            "height_source_missing_count": int(
                (~_valid_height_mask(buildings["height_m"])).sum()
                if "height_m" in buildings.columns
                else len(buildings)
            ),
        }

    return {
        "height_source_overture_count": int(
            (buildings["height_source"] == "overture").sum()
        ),
        "height_source_gba_lod1_count": int(
            (buildings["height_source"] == "gba_lod1").sum()
        ),
        "height_source_missing_count": int(
            (buildings["height_source"] == "missing").sum()
        ),
    }


def _assign_height_enrichment_zone(buildings: gpd.GeoDataFrame) -> pd.DataFrame:
    buildings_wgs84 = buildings.to_crs("EPSG:4326")
    records = []

    for idx, geom in buildings_wgs84.geometry.items():
        centroid = geom.centroid
        lon = float(centroid.x)
        lat = float(centroid.y)
        hemisphere = "north" if lat >= 0 else "south"
        zone = utm_zone_from_longitude(lon)
        records.append(
            {
                "index": idx,
                "height_enrichment_centroid_lon": lon,
                "height_enrichment_centroid_lat": lat,
                "height_enrichment_utm_zone": zone,
                "height_enrichment_hemisphere": hemisphere,
                "height_enrichment_epsg": utm_epsg(zone, hemisphere),
            }
        )

    return pd.DataFrame.from_records(records).set_index("index")


def _resolve_gba_cache_dir(
    height_cfg: dict[str, Any],
    output_dir: Path | None,
    project_root: Path | None,
) -> Path:
    cache_dir = Path(
        height_cfg.get(
            "cache_dir",
            Path("04_outputs") / "_cache" / "gba_lod1_parquet",
        )
    )

    if cache_dir.is_absolute():
        return cache_dir

    if project_root is not None:
        return Path(project_root) / cache_dir

    if output_dir is not None:
        return Path(output_dir).parent.parent / cache_dir

    return Path.cwd() / cache_dir


def _aggregate_segmented_height_summary(
    buildings_before: gpd.GeoDataFrame,
    buildings_after: gpd.GeoDataFrame,
    zone_summaries: list[dict[str, Any]],
    assignment_table: pd.DataFrame,
) -> dict[str, Any]:
    before_valid = _valid_height_mask(buildings_before["height_m"])
    after_valid = _valid_height_mask(buildings_after["height_m"])
    changed_existing = (
        before_valid
        & (
            pd.to_numeric(buildings_before["height_m"], errors="coerce")
            != pd.to_numeric(buildings_after["height_m"], errors="coerce")
        )
    )

    if int(changed_existing.sum()) != 0:
        raise AssertionError(
            "Some valid Overture heights were changed during segmented "
            "height enrichment."
        )

    source_counts = _height_source_counts(buildings_after)
    enriched_mask = buildings_after.get(
        "height_was_enriched",
        pd.Series(False, index=buildings_after.index),
    ).fillna(False)

    area = pd.to_numeric(
        buildings_after.get(
            "footprint_area_m2",
            pd.Series(0.0, index=buildings_after.index),
        ),
        errors="coerce",
    ).fillna(0.0)
    total_area = float(area.sum())
    first_zone_summary = zone_summaries[0] if zone_summaries else {}

    return {
        "source": "global_building_atlas_lod1_parquet",
        "method": "segmented_centroid_zone_gba_lod1_largest_overlap_matching",
        "policy": "fill_missing_only",
        "replace_existing_overture_height": False,
        "segmented_mode": True,
        "height_enrichment_zone_assignment_rule": "building_centroid_wgs84",
        "n_height_enrichment_zones": int(
            assignment_table["height_enrichment_epsg"].nunique()
        ),
        "height_enrichment_epsg_list": [
            int(value)
            for value in sorted(assignment_table["height_enrichment_epsg"].unique())
        ],
        "n_buildings": int(len(buildings_after)),
        "valid_height_count_before": int(before_valid.sum()),
        "missing_height_count_before": int((~before_valid).sum()),
        "valid_height_share_before": float(before_valid.mean()),
        "missing_height_share_before": float((~before_valid).mean()),
        "height_enriched_count": int(
            buildings_after.get(
                "height_was_enriched",
                pd.Series(False, index=buildings_after.index),
            ).fillna(False).sum()
        ),
        "valid_height_count_after": int(after_valid.sum()),
        "missing_height_count_after": int((~after_valid).sum()),
        "valid_height_share_after": float(after_valid.mean()),
        "missing_height_share_after": float((~after_valid).mean()),
        "height_valid_area_share_before": (
            float(area[before_valid].sum()) / total_area
            if total_area > 0
            else None
        ),
        "height_valid_area_share_after": (
            float(area[after_valid].sum()) / total_area if total_area > 0 else None
        ),
        "height_enriched_area_share": (
            float(area[enriched_mask].sum()) / total_area if total_area > 0 else None
        ),
        "height_enriched_share_of_all_buildings": float(enriched_mask.mean()),
        "height_enriched_share_of_missing_before": (
            float(enriched_mask.sum() / (~before_valid).sum())
            if int((~before_valid).sum()) > 0
            else None
        ),
        "min_overlap_share_for_enrichment": first_zone_summary.get(
            "min_overlap_share_for_enrichment"
        ),
        "min_valid_height_m": first_zone_summary.get("min_valid_height_m"),
        "changed_existing_overture_height_count": int(changed_existing.sum()),
        "per_zone_enrichment_summaries": zone_summaries,
        **source_counts,
    }


def enrich_segmented_building_heights(
    buildings: gpd.GeoDataFrame,
    config: dict[str, Any],
    output_dir: Path | None = None,
    save_outputs: bool = True,
    project_root: Path | None = None,
) -> tuple[gpd.GeoDataFrame, dict[str, Any] | None]:
    """
    Enrich full buildings by centroid-assigned UTM zone before segmentation.
    """
    height_cfg = config.get("height_enrichment", {})

    if not height_cfg.get("enabled", False):
        return buildings, None

    if buildings.empty:
        raise ValueError("Buildings GeoDataFrame is empty.")

    if buildings.crs is None:
        raise ValueError("Buildings GeoDataFrame has no CRS.")

    if "building_id" not in buildings.columns:
        raise ValueError("Buildings are missing required column: building_id")

    if "height_m" not in buildings.columns:
        raise ValueError("Buildings are missing required column: height_m")

    if bool(height_cfg.get("replace_existing_height", False)):
        raise ValueError(
            "replace_existing_height=True is not allowed for segmented "
            "height enrichment. Use fill_missing_only."
        )

    if buildings["building_id"].duplicated().any():
        raise ValueError(
            "Segmented height enrichment requires unique building_id values."
        )

    buildings_before = buildings.copy()
    assignment_table = _assign_height_enrichment_zone(buildings_before)
    buildings_assigned = buildings_before.join(assignment_table)
    cache_dir = _resolve_gba_cache_dir(height_cfg, output_dir, project_root)

    enriched_groups: list[gpd.GeoDataFrame] = []
    zone_summaries: list[dict[str, Any]] = []
    gba_subsets: list[gpd.GeoDataFrame] = []
    best_matches: list[gpd.GeoDataFrame] = []

    for epsg, group in buildings_assigned.groupby("height_enrichment_epsg"):
        zone_buildings = gpd.GeoDataFrame(
            group.drop(columns="geometry"),
            geometry=group.geometry,
            crs=buildings_assigned.crs,
        ).to_crs(f"EPSG:{int(epsg)}")
        zone_buildings = add_footprint_area(zone_buildings)

        enriched_zone, zone_summary, gba_subset, zone_best_matches = (
            enrich_missing_heights_with_gba_lod1(
                buildings=zone_buildings,
                cache_dir=cache_dir,
                base_url=height_cfg.get(
                    "base_url",
                    "https://data.source.coop/tge-labs/globalbuildingatlas-lod1",
                ),
                height_col="height_m",
                min_overlap_share=float(height_cfg.get("min_overlap_share", 0.2)),
                min_valid_height_m=float(height_cfg.get("min_valid_height_m", 2.0)),
                bbox_buffer_deg=float(height_cfg.get("bbox_buffer_deg", 0.002)),
                max_download_size_mb=float(
                    height_cfg.get("max_download_size_mb", 2000)
                ),
                replace_existing_height=False,
            )
        )

        zone_summary = {
            **zone_summary,
            "height_enrichment_epsg": int(epsg),
            "height_enrichment_utm_zone": int(
                group["height_enrichment_utm_zone"].iloc[0]
            ),
            "height_enrichment_hemisphere": group[
                "height_enrichment_hemisphere"
            ].iloc[0],
        }
        zone_summaries.append(zone_summary)

        enriched_groups.append(enriched_zone.to_crs("EPSG:4326"))

        if not gba_subset.empty:
            gba_subsets.append(gba_subset.to_crs("EPSG:4326"))

        if not zone_best_matches.empty:
            matches = zone_best_matches.copy()
            matches["height_enrichment_epsg"] = int(epsg)
            best_matches.append(matches)

    enriched_zone_attributes = gpd.GeoDataFrame(
        pd.concat(enriched_groups, ignore_index=True),
        geometry="geometry",
        crs="EPSG:4326",
    ).drop(columns="geometry").set_index("building_id")

    enriched = buildings_before.set_index("building_id").copy()

    for column in enriched_zone_attributes.columns:
        enriched[column] = enriched_zone_attributes.loc[enriched.index, column]

    enriched = enriched.reset_index()
    enriched = gpd.GeoDataFrame(enriched, geometry="geometry", crs=buildings.crs)

    summary = _aggregate_segmented_height_summary(
        buildings_before=buildings_before.reset_index(drop=True),
        buildings_after=enriched.reset_index(drop=True),
        zone_summaries=zone_summaries,
        assignment_table=assignment_table,
    )

    if save_outputs and output_dir is not None:
        processed_dir = Path(output_dir) / "processed"
        tables_dir = Path(output_dir) / "tables"
        reports_dir = Path(output_dir) / "reports"
        processed_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        enriched.to_crs("EPSG:4326").to_file(
            processed_dir / "buildings_height_enriched_segmented_wgs84.gpkg",
            layer="buildings_height_enriched_segmented_wgs84",
            driver="GPKG",
        )

        pd.DataFrame([summary]).to_csv(
            tables_dir / "height_enrichment_summary_segmented.csv",
            index=False,
        )

        (reports_dir / "height_enrichment_quality_segmented.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        if gba_subsets and bool(height_cfg.get("save_gba_subset", True)):
            gpd.GeoDataFrame(
                pd.concat(gba_subsets, ignore_index=True),
                geometry="geometry",
                crs="EPSG:4326",
            ).to_file(
                processed_dir / "gba_lod1_subset_segmented.gpkg",
                layer="gba_lod1_subset_segmented",
                driver="GPKG",
            )

        if best_matches and bool(height_cfg.get("save_matches", True)):
            pd.concat(best_matches, ignore_index=True).drop(
                columns="geometry",
                errors="ignore",
            ).to_csv(
                tables_dir / "gba_lod1_height_matches_segmented.csv",
                index=False,
            )

    return enriched, summary


def _prepare_segment_buildings(
    buildings: gpd.GeoDataFrame,
    aoi_segment_metric: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    buildings_metric = buildings.to_crs(aoi_segment_metric.crs)
    buildings_clean = clean_building_geometries(buildings_metric)
    buildings_clipped = gpd.clip(buildings_clean, aoi_segment_metric)
    buildings_clipped = buildings_clipped[
        buildings_clipped.geometry.notna()
        & ~buildings_clipped.geometry.is_empty
    ].copy()

    if buildings_clipped.empty:
        return buildings_clipped

    buildings_clipped = add_footprint_area(buildings_clipped)
    return buildings_clipped


def _empty_neighbor_distance(grid: gpd.GeoDataFrame) -> pd.DataFrame:
    out = grid[["unit_id"]].copy()
    out["avg_neighbor_distance_m"] = pd.NA
    out["median_neighbor_distance_m"] = pd.NA
    out["neighbor_distance_valid_count"] = 0
    return out


def _nearest_neighbors_for_target_buildings(
    target_buildings: gpd.GeoDataFrame,
    context_buildings: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    target = target_buildings[["building_id", "geometry"]].copy()
    out = target.copy()
    out["neighbor_building_id"] = pd.NA
    out["neighbor_distance_m"] = pd.NA

    if target.empty or len(context_buildings) < 2:
        return gpd.GeoDataFrame(out, geometry="geometry", crs=target_buildings.crs)

    right = context_buildings[["building_id", "geometry"]].copy()
    right = right.rename(columns={"building_id": "neighbor_building_id"})

    nearest = gpd.sjoin_nearest(
        target,
        right,
        how="left",
        distance_col="neighbor_distance_m",
        exclusive=True,
    )

    nearest = nearest[
        nearest["building_id"] != nearest["neighbor_building_id"]
    ].copy()

    if nearest.empty:
        return gpd.GeoDataFrame(out, geometry="geometry", crs=target_buildings.crs)

    nearest = (
        nearest.sort_values(["building_id", "neighbor_distance_m"])
        .drop_duplicates("building_id")
        .loc[:, ["building_id", "neighbor_building_id", "neighbor_distance_m"]]
    )

    out = out.drop(columns=["neighbor_building_id", "neighbor_distance_m"])
    out = out.merge(nearest, on="building_id", how="left")
    return gpd.GeoDataFrame(out, geometry="geometry", crs=target_buildings.crs)


def _calculate_segment_neighbor_context(
    buildings: gpd.GeoDataFrame,
    buildings_segment: gpd.GeoDataFrame,
    aoi_segment_metric: gpd.GeoDataFrame,
    context_buffer_m: float,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    if buildings_segment.empty:
        return buildings_segment, {
            "context_buffer_m": float(context_buffer_m),
            "n_target_buildings": 0,
            "n_context_buildings": 0,
            "valid_neighbor_distance_count": 0,
            "valid_neighbor_distance_share": None,
            "target_buildings_without_neighbor_count": 0,
        }

    buildings_metric = clean_building_geometries(buildings.to_crs(aoi_segment_metric.crs))
    target_ids = buildings_segment["building_id"].drop_duplicates()
    target_buildings = buildings_metric[
        buildings_metric["building_id"].isin(target_ids)
    ].copy()

    segment_geometry = aoi_segment_metric.geometry.union_all()
    context_geometry = segment_geometry.buffer(context_buffer_m)
    context_buildings = buildings_metric[
        buildings_metric.geometry.intersects(context_geometry)
    ].copy()
    context_buildings = context_buildings[
        context_buildings.geometry.notna()
        & ~context_buildings.geometry.is_empty
    ].copy()

    neighbor_table = _nearest_neighbors_for_target_buildings(
        target_buildings=target_buildings,
        context_buildings=context_buildings,
    )

    valid_count = int(neighbor_table["neighbor_distance_m"].notna().sum())
    n_target = int(len(neighbor_table))
    n_context = int(context_buildings["building_id"].nunique())
    summary = {
        "context_buffer_m": float(context_buffer_m),
        "n_target_buildings": n_target,
        "n_context_buildings": n_context,
        "valid_neighbor_distance_count": valid_count,
        "valid_neighbor_distance_share": (
            float(valid_count / n_target) if n_target > 0 else None
        ),
        "target_buildings_without_neighbor_count": int(n_target - valid_count),
        "context_building_to_target_building_ratio": (
            float(n_context / n_target) if n_target > 0 else None
        ),
    }

    neighbor_attrs = neighbor_table[
        ["building_id", "neighbor_building_id", "neighbor_distance_m"]
    ].copy()
    buildings_with_neighbors = buildings_segment.drop(
        columns=["neighbor_building_id", "neighbor_distance_m"],
        errors="ignore",
    ).merge(neighbor_attrs, on="building_id", how="left")

    return (
        gpd.GeoDataFrame(
            buildings_with_neighbors,
            geometry="geometry",
            crs=buildings_segment.crs,
        ),
        summary,
    )


def _aggregate_neighbor_distance_from_buildings(
    buildings_segment: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
) -> pd.DataFrame:
    if buildings_segment.empty or "neighbor_distance_m" not in buildings_segment.columns:
        return _empty_neighbor_distance(grid)

    assigned = buildings_segment[
        ["building_id", "neighbor_distance_m", "geometry"]
    ].copy()
    assigned["geometry"] = assigned.geometry.representative_point()

    assigned = gpd.sjoin(
        assigned,
        grid[["unit_id", "geometry"]],
        how="left",
        predicate="within",
    )
    assigned_valid = assigned[assigned["neighbor_distance_m"].notna()].copy()

    if assigned_valid.empty:
        return _empty_neighbor_distance(grid)

    summary = (
        assigned_valid.groupby("unit_id")
        .agg(
            avg_neighbor_distance_m=("neighbor_distance_m", "mean"),
            median_neighbor_distance_m=("neighbor_distance_m", "median"),
            neighbor_distance_valid_count=("neighbor_distance_m", "count"),
        )
        .reset_index()
    )

    out = grid[["unit_id"]].copy().merge(summary, on="unit_id", how="left")
    out["neighbor_distance_valid_count"] = (
        out["neighbor_distance_valid_count"].fillna(0).astype(int)
    )
    return out


STREET_PROFILE_GRID_COLUMNS = [
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

STREET_PROFILE_BUILDING_COLUMNS = [
    "street_id",
    "street_profile_width_m",
    "street_profile_openness",
    "street_profile_momepy_height_m",
    "street_profile_hw_ratio_momepy",
    "street_profile_width_is_capped",
    "has_opposite_profile_evidence",
    "building_to_street_centerline_m",
    "street_profile_height_to_width_ratio_prelim",
    "street_profile_height_to_width_ratio_strict",
    "has_valid_street_profile_ratio_prelim",
    "has_valid_street_profile_ratio_strict",
]


def _empty_street_profile_grid(grid: gpd.GeoDataFrame) -> pd.DataFrame:
    out = grid[["unit_id"]].copy()

    for col in STREET_PROFILE_GRID_COLUMNS:
        out[col] = 0 if col.endswith("_count") else float("nan")

    return out


def _empty_street_context_summary(
    context_buffer_m: float,
    status: str = "not_calculated",
    error: str | None = None,
) -> dict[str, Any]:
    summary = {
        "street_context_status": status,
        "street_context_error": error,
        "context_buffer_m": float(context_buffer_m),
        "n_target_buildings": 0,
        "n_context_buildings": 0,
        "n_street_segments": 0,
        "valid_width_count": 0,
        "valid_width_share": None,
        "n_buildings": 0,
        "valid_ratio_prelim_count": 0,
        "valid_ratio_prelim_share": None,
        "valid_ratio_strict_count": 0,
        "valid_ratio_strict_share": None,
        "n_grid_cells": 0,
        "grid_cells_with_prelim_ratio_count": 0,
        "grid_cells_with_prelim_ratio_share": None,
        "grid_cells_with_strict_ratio_count": 0,
        "grid_cells_with_strict_ratio_share": None,
    }

    return summary


def _is_expected_no_graph_error(exc: Exception) -> bool:
    message = str(exc).lower()
    expected_fragments = [
        "found no graph nodes within the requested polygon",
        "no graph nodes",
        "found no edges",
        "graph contains no edges",
    ]
    return isinstance(exc, ValueError) and any(
        fragment in message for fragment in expected_fragments
    )


def _calculate_segment_street_context(
    buildings: gpd.GeoDataFrame,
    buildings_segment: gpd.GeoDataFrame,
    aoi_segment_metric: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    config: dict[str, Any],
    context_buffer_m: float,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, dict[str, Any]]:
    if buildings_segment.empty:
        return (
            buildings_segment,
            _empty_street_profile_grid(grid),
            _empty_street_context_summary(context_buffer_m),
        )

    street_cfg = config.get("street_context", {})
    source = street_cfg.get("source", "osmnx")

    if source != "osmnx":
        raise ValueError(f"Unsupported street_context source: {source}.")

    buildings_metric = clean_building_geometries(buildings.to_crs(aoi_segment_metric.crs))
    target_ids = buildings_segment["building_id"].drop_duplicates()
    target_buildings = buildings_metric[
        buildings_metric["building_id"].isin(target_ids)
    ].copy()

    segment_geometry = aoi_segment_metric.geometry.union_all()
    context_geometry = segment_geometry.buffer(context_buffer_m)
    context_area = gpd.GeoDataFrame(
        {"context_id": [1]},
        geometry=[context_geometry],
        crs=aoi_segment_metric.crs,
    )
    context_buildings = buildings_metric[
        buildings_metric.geometry.intersects(context_geometry)
    ].copy()

    if target_buildings.empty or context_buildings.empty:
        return (
            buildings_segment,
            _empty_street_profile_grid(grid),
            _empty_street_context_summary(context_buffer_m),
        )

    context_area_wgs84 = context_area.to_crs("EPSG:4326")

    try:
        streets = fetch_streets_from_osmnx(
            context_area_wgs84,
            network_type=street_cfg.get("network_type", "drive"),
            target_crs=aoi_segment_metric.crs,
        )
    except Exception as exc:
        if not _is_expected_no_graph_error(exc):
            raise

        summary = _empty_street_context_summary(
            context_buffer_m=context_buffer_m,
            status="no_osm_graph",
            error=str(exc),
        )
        summary.update(
            {
                "n_target_buildings": int(len(target_buildings)),
                "n_context_buildings": int(context_buildings["building_id"].nunique()),
            }
        )
        return buildings_segment, _empty_street_profile_grid(grid), summary

    streets = streets.to_crs(aoi_segment_metric.crs)

    if streets.empty:
        summary = _empty_street_context_summary(
            context_buffer_m=context_buffer_m,
            status="no_street_segments",
            error="Street fetch returned an empty street layer.",
        )
        summary.update(
            {
                "n_target_buildings": int(len(target_buildings)),
                "n_context_buildings": int(context_buildings["building_id"].nunique()),
            }
        )
        return buildings_segment, _empty_street_profile_grid(grid), summary

    streets_profile = calculate_street_profile_segments(
        streets=streets,
        buildings=context_buildings,
        height_col="height_m",
        distance=float(street_cfg.get("distance_m", 10)),
        tick_length=float(street_cfg.get("tick_length_m", 60)),
    )
    building_street, join_summary = assign_buildings_to_street_profiles(
        buildings=target_buildings,
        streets_profile=streets_profile,
        building_id_col="building_id",
        height_col="height_m",
    )
    building_street = calculate_building_street_profile_ratio(
        building_street,
        height_col="height_m",
    )
    grid_street_profile = aggregate_street_profile_ratio_to_units(
        building_street=building_street,
        units=grid,
        unit_id_col="unit_id",
        building_id_col="building_id",
    )
    street_summary = summarize_street_profile_quality(
        streets_profile=streets_profile,
        building_street=building_street,
        grid_street_profile=grid_street_profile,
        join_summary=join_summary,
    )
    street_summary.update(
        {
            "street_context_status": "ok",
            "street_context_error": None,
            "context_buffer_m": float(context_buffer_m),
            "n_target_buildings": int(len(target_buildings)),
            "n_context_buildings": int(context_buildings["building_id"].nunique()),
        }
    )

    street_attrs = building_street[
        ["building_id", *STREET_PROFILE_BUILDING_COLUMNS]
    ].copy()
    buildings_with_street = buildings_segment.drop(
        columns=STREET_PROFILE_BUILDING_COLUMNS,
        errors="ignore",
    ).merge(street_attrs, on="building_id", how="left")

    grid_street_table = grid_street_profile[
        ["unit_id", *STREET_PROFILE_GRID_COLUMNS]
    ].copy()

    return (
        gpd.GeoDataFrame(
            buildings_with_street,
            geometry="geometry",
            crs=buildings_segment.crs,
        ),
        grid_street_table,
        street_summary,
    )


def _empty_gsi(grid: gpd.GeoDataFrame) -> pd.DataFrame:
    out = grid[["unit_id"]].copy()
    out["building_footprint_area_m2"] = 0.0
    out["gsi"] = 0.0
    return out


def _empty_far(grid: gpd.GeoDataFrame) -> pd.DataFrame:
    out = grid[["unit_id"]].copy()
    out["floor_area_sum_m2"] = 0.0
    out["far_fsi"] = 0.0
    out["floor_data_valid_area_share"] = pd.NA
    return out


def _empty_bvd(grid: gpd.GeoDataFrame) -> pd.DataFrame:
    out = grid[["unit_id"]].copy()
    out["built_volume_m3"] = 0.0
    out["built_volume_density"] = 0.0
    out["height_valid_area_share"] = pd.NA
    return out


def _calculate_segment_core_indicators(
    buildings_segment: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    config: dict[str, Any],
) -> gpd.GeoDataFrame:
    result = grid.copy()
    params = config.get("indicator_parameters", config.get("parameters", {}))

    if _indicator_enabled(config, "gsi"):
        gsi_result = (
            _empty_gsi(grid)
            if buildings_segment.empty
            else calculate_gsi(buildings_segment, grid, params)
        )
        result = result.merge(gsi_result, on="unit_id", how="left")

    if _indicator_enabled(config, "far_fsi") and "num_floors" in buildings_segment.columns:
        far_result = (
            _empty_far(grid)
            if buildings_segment.empty
            else calculate_far_fsi(buildings_segment, grid, params)
        )
        result = result.merge(far_result, on="unit_id", how="left")

    if _indicator_enabled(config, "built_volume_density") and "height_m" in buildings_segment.columns:
        bvd_result = (
            _empty_bvd(grid)
            if buildings_segment.empty
            else calculate_built_volume_density(buildings_segment, grid, params)
        )
        result = result.merge(bvd_result, on="unit_id", how="left")

    if _indicator_enabled(config, "neighbor_distance"):
        neighbor_result = _aggregate_neighbor_distance_from_buildings(
            buildings_segment=buildings_segment,
            grid=grid,
        )
        result = result.merge(neighbor_result, on="unit_id", how="left")

    return result


def _summarize_segmented_neighbor_distance(
    segment_summaries: list[dict[str, Any]],
    merged_grid: gpd.GeoDataFrame,
    context_buffer_m: float,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {
            "enabled": False,
            "context_buffer_m": float(context_buffer_m),
        }

    neighbor_summaries = [
        segment.get("neighbor_distance", {})
        for segment in segment_summaries
    ]
    n_target = int(
        sum(item.get("n_target_buildings", 0) or 0 for item in neighbor_summaries)
    )
    valid_count = int(
        sum(
            item.get("valid_neighbor_distance_count", 0) or 0
            for item in neighbor_summaries
        )
    )
    grid_valid = (
        int((merged_grid["neighbor_distance_valid_count"] > 0).sum())
        if "neighbor_distance_valid_count" in merged_grid.columns
        else 0
    )

    return {
        "enabled": True,
        "context_buffer_m": float(context_buffer_m),
        "n_target_buildings": n_target,
        "valid_neighbor_distance_count": valid_count,
        "valid_neighbor_distance_share": (
            float(valid_count / n_target) if n_target > 0 else None
        ),
        "target_buildings_without_neighbor_count": int(n_target - valid_count),
        "n_grid_cells": int(len(merged_grid)),
        "grid_cells_with_neighbor_distance": grid_valid,
        "grid_cell_neighbor_distance_coverage_share": (
            float(grid_valid / len(merged_grid)) if len(merged_grid) > 0 else None
        ),
        "segments": neighbor_summaries,
    }


def _summarize_segmented_street_context(
    segment_summaries: list[dict[str, Any]],
    merged_grid: gpd.GeoDataFrame,
    context_buffer_m: float,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {
            "enabled": False,
            "context_buffer_m": float(context_buffer_m),
        }

    street_summaries = [
        segment.get("street_context", {})
        for segment in segment_summaries
    ]
    status_counts: dict[str, int] = {}

    for item in street_summaries:
        status = item.get("street_context_status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    def sum_key(key: str) -> int:
        return int(sum(item.get(key, 0) or 0 for item in street_summaries))

    n_buildings = sum_key("n_buildings")
    n_streets = sum_key("n_street_segments")
    valid_width_count = sum_key("valid_width_count")
    valid_prelim = sum_key("valid_ratio_prelim_count")
    valid_strict = sum_key("valid_ratio_strict_count")
    n_grid_cells = int(len(merged_grid))
    grid_prelim = (
        int((merged_grid["street_profile_ratio_prelim_valid_count"] > 0).sum())
        if "street_profile_ratio_prelim_valid_count" in merged_grid.columns
        else 0
    )
    grid_strict = (
        int((merged_grid["street_profile_ratio_strict_valid_count"] > 0).sum())
        if "street_profile_ratio_strict_valid_count" in merged_grid.columns
        else 0
    )

    return {
        "enabled": True,
        "context_buffer_m": float(context_buffer_m),
        "street_context_status_counts": status_counts,
        "no_graph_segment_count": int(status_counts.get("no_osm_graph", 0)),
        "no_street_segment_count": int(status_counts.get("no_street_segments", 0)),
        "n_street_segments": n_streets,
        "valid_width_count": valid_width_count,
        "valid_width_share": (
            float(valid_width_count / n_streets) if n_streets > 0 else None
        ),
        "n_buildings": n_buildings,
        "valid_ratio_prelim_count": valid_prelim,
        "valid_ratio_prelim_share": (
            float(valid_prelim / n_buildings) if n_buildings > 0 else None
        ),
        "valid_ratio_strict_count": valid_strict,
        "valid_ratio_strict_share": (
            float(valid_strict / n_buildings) if n_buildings > 0 else None
        ),
        "n_grid_cells": n_grid_cells,
        "grid_cells_with_prelim_ratio_count": grid_prelim,
        "grid_cells_with_prelim_ratio_share": (
            float(grid_prelim / n_grid_cells) if n_grid_cells > 0 else None
        ),
        "grid_cells_with_strict_ratio_count": grid_strict,
        "grid_cells_with_strict_ratio_share": (
            float(grid_strict / n_grid_cells) if n_grid_cells > 0 else None
        ),
        "segments": street_summaries,
    }


def process_segmented_core_indicators(
    buildings: gpd.GeoDataFrame,
    aoi: gpd.GeoDataFrame,
    config: dict[str, Any],
    output_dir: Path | None = None,
    save_outputs: bool = True,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """
    Process core grid indicators per UTM segment and merge outputs in WGS84.
    """
    validate_segmented_core_config(config)

    if buildings.crs is None:
        raise ValueError("Buildings GeoDataFrame has no CRS.")

    aggregation_cfg = config.get("aggregation", {})
    cell_size_m = int(aggregation_cfg["cell_size_m"])
    segments = segment_aoi_by_utm_zone(aoi)
    neighbor_distance_enabled = _indicator_enabled(config, "neighbor_distance")
    street_context_enabled = bool(config.get("street_context", {}).get("enabled", False))
    context_buffer_m = _segmented_context_buffer_m(config)
    buildings, height_enrichment_summary = enrich_segmented_building_heights(
        buildings=buildings,
        config=config,
        output_dir=output_dir,
        save_outputs=save_outputs,
        project_root=project_root,
    )

    merged_grids = []
    segment_summaries = []
    global_cell_index = 1

    for _, segment in segments.iterrows():
        segment_id = int(segment["segment_id"])
        epsg = int(segment["epsg"])
        segment_label = f"segment_{segment_id:03d}"
        segment_wgs84 = gpd.GeoDataFrame(
            {
                "segment_id": [segment_id],
                "utm_zone": [int(segment["utm_zone"])],
                "hemisphere": [segment["hemisphere"]],
                "epsg": [epsg],
            },
            geometry=[segment.geometry],
            crs="EPSG:4326",
        )
        aoi_segment_metric = segment_wgs84.to_crs(f"EPSG:{epsg}")
        buildings_segment = _prepare_segment_buildings(
            buildings=buildings,
            aoi_segment_metric=aoi_segment_metric,
        )
        neighbor_summary = None
        street_summary = None
        grid_street_profile = None

        if neighbor_distance_enabled:
            buildings_segment, neighbor_summary = _calculate_segment_neighbor_context(
                buildings=buildings,
                buildings_segment=buildings_segment,
                aoi_segment_metric=aoi_segment_metric,
                context_buffer_m=context_buffer_m,
            )

        grid = create_grid(aoi_segment_metric, cell_size_m=cell_size_m)
        grid["cell_id_local"] = grid["unit_id"]
        grid["unit_id"] = [
            f"seg{segment_id:03d}_{cell_id}"
            for cell_id in grid["cell_id_local"]
        ]

        if street_context_enabled:
            buildings_segment, grid_street_profile, street_summary = (
                _calculate_segment_street_context(
                    buildings=buildings,
                    buildings_segment=buildings_segment,
                    aoi_segment_metric=aoi_segment_metric,
                    grid=grid,
                    config=config,
                    context_buffer_m=context_buffer_m,
                )
            )

        indicator_grid = _calculate_segment_core_indicators(
            buildings_segment=buildings_segment,
            grid=grid,
            config=config,
        )

        if street_context_enabled and grid_street_profile is not None:
            indicator_grid = indicator_grid.merge(
                grid_street_profile,
                on="unit_id",
                how="left",
            )

        indicator_grid["segment_id"] = segment_id
        indicator_grid["utm_zone"] = int(segment["utm_zone"])
        indicator_grid["hemisphere"] = segment["hemisphere"]
        indicator_grid["segment_epsg"] = epsg
        indicator_grid["calculation_crs"] = f"EPSG:{epsg}"
        indicator_grid["cell_area_m2"] = indicator_grid["unit_area_m2"]
        indicator_grid["cell_id_global"] = [
            f"cell_{idx:06d}"
            for idx in range(global_cell_index, global_cell_index + len(indicator_grid))
        ]
        global_cell_index += len(indicator_grid)

        if save_outputs and output_dir is not None:
            segment_dir = output_dir / "segments" / segment_label
            processed_dir = segment_dir / "processed"
            indicators_dir = segment_dir / "indicators"
            processed_dir.mkdir(parents=True, exist_ok=True)
            indicators_dir.mkdir(parents=True, exist_ok=True)
            aoi_segment_metric.to_file(
                processed_dir / "aoi_segment_metric.gpkg",
                layer="aoi_segment_metric",
                driver="GPKG",
            )
            buildings_segment.to_file(
                processed_dir / "buildings_segment_metric.gpkg",
                layer="buildings_segment_metric",
                driver="GPKG",
            )
            indicator_grid.to_file(
                indicators_dir / "grid_indicators.gpkg",
                layer="grid_indicators",
                driver="GPKG",
            )

        merged_grids.append(indicator_grid.to_crs("EPSG:4326"))
        segment_summaries.append(
            {
                "segment_id": segment_id,
                "utm_zone": int(segment["utm_zone"]),
                "hemisphere": segment["hemisphere"],
                "epsg": epsg,
                "n_buildings": int(len(buildings_segment)),
                "n_grid_cells": int(len(indicator_grid)),
                "calculation_crs": f"EPSG:{epsg}",
                "neighbor_distance": neighbor_summary,
                "street_context": street_summary,
            }
        )

    if not merged_grids:
        raise ValueError("Segmented UTM processing produced no grid cells.")

    merged_grid = gpd.GeoDataFrame(
        pd.concat(merged_grids, ignore_index=True),
        geometry="geometry",
        crs="EPSG:4326",
    )

    ordered_cols = [
        "cell_id_global",
        "cell_id_local",
        "unit_id",
        "segment_id",
        "utm_zone",
        "segment_epsg",
        "calculation_crs",
        "cell_area_m2",
    ]
    remaining_cols = [
        col
        for col in merged_grid.columns
        if col not in ordered_cols and col != "geometry"
    ]
    merged_grid = merged_grid[ordered_cols + remaining_cols + ["geometry"]]
    neighbor_distance_summary = _summarize_segmented_neighbor_distance(
        segment_summaries=segment_summaries,
        merged_grid=merged_grid,
        context_buffer_m=context_buffer_m,
        enabled=neighbor_distance_enabled,
    )
    street_profile_summary = _summarize_segmented_street_context(
        segment_summaries=segment_summaries,
        merged_grid=merged_grid,
        context_buffer_m=context_buffer_m,
        enabled=street_context_enabled,
    )

    summary = {
        "processing_mode": "segmented_utm",
        "supported_indicators": sorted(
            CORE_INDICATORS
            | ({"neighbor_distance"} if neighbor_distance_enabled else set())
            | (
                {"street_profile_height_to_width_ratio"}
                if street_context_enabled
                else set()
            )
        ),
        "unsupported_branches": [
            "height_to_distance_ratio",
            "external cache reuse",
            "static maps",
        ],
        "height_enrichment_enabled": height_enrichment_summary is not None,
        "height_enrichment_summary": height_enrichment_summary,
        "segmented_neighbor_distance_enabled": neighbor_distance_enabled,
        "segmented_context_buffer_m": float(context_buffer_m),
        "neighbor_distance_summary": neighbor_distance_summary,
        "segmented_street_context_enabled": street_context_enabled,
        "segmented_street_context_buffer_m": float(context_buffer_m),
        "street_profile_summary": street_profile_summary,
        "n_segments": int(len(segment_summaries)),
        "segment_epsg_list": [item["epsg"] for item in segment_summaries],
        "segment_utm_zones": [item["utm_zone"] for item in segment_summaries],
        "n_grid_cells": int(len(merged_grid)),
        "segments": segment_summaries,
    }

    if save_outputs and output_dir is not None:
        indicators_dir = output_dir / "indicators"
        reports_dir = output_dir / "reports"
        indicators_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        merged_grid.to_file(
            indicators_dir / "grid_indicators_segmented_wgs84.gpkg",
            layer="grid_indicators_segmented_wgs84",
            driver="GPKG",
        )
        (reports_dir / "segmented_crs_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

    return {
        "indicator_grid": merged_grid,
        "summary": summary,
    }
