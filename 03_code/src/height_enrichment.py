from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import requests


DEFAULT_GBA_BASE_URL = "https://data.source.coop/tge-labs/globalbuildingatlas-lod1"


def _is_valid_height(series: pd.Series, min_height_m: float = 0.0) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.notna() & (values > min_height_m)


def _format_lon(value: int) -> str:
    prefix = "e" if value >= 0 else "w"
    return f"{prefix}{abs(int(value)):03d}"


def _format_lat(value: int) -> str:
    prefix = "n" if value >= 0 else "s"
    return f"{prefix}{abs(int(value)):02d}"


def gba_lod1_tile_names_for_bbox(
    bounds_wgs84: tuple[float, float, float, float],
) -> list[str]:
    """
    Return 5-degree GBA LoD1 Parquet tile names intersecting a WGS84 bbox.

    Example for Vienna:
    e015_n50_e020_n45.parquet
    """
    minx, miny, maxx, maxy = bounds_wgs84

    eps = 1e-12

    west_start = int(math.floor(minx / 5.0) * 5)
    west_end = int(math.floor((maxx - eps) / 5.0) * 5)

    south_start = int(math.floor(miny / 5.0) * 5)
    south_end = int(math.floor((maxy - eps) / 5.0) * 5)

    tile_names: list[str] = []

    for west in range(west_start, west_end + 1, 5):
        east = west + 5

        for south in range(south_start, south_end + 1, 5):
            north = south + 5

            tile_name = (
                f"{_format_lon(west)}_"
                f"{_format_lat(north)}_"
                f"{_format_lon(east)}_"
                f"{_format_lat(south)}.parquet"
            )

            tile_names.append(tile_name)

    return sorted(set(tile_names))


def _has_valid_parquet_magic(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 8:
        return False

    with open(path, "rb") as f:
        start = f.read(4)
        f.seek(-4, 2)
        end = f.read(4)

    return start == b"PAR1" and end == b"PAR1"


def download_gba_lod1_tile(
    tile_name: str,
    cache_dir: Path,
    base_url: str = DEFAULT_GBA_BASE_URL,
    max_download_size_mb: float = 2000,
) -> Path:
    """
    Download one GBA LoD1 Parquet tile from Source Cooperative data proxy.

    The function refuses obvious HTML pages and validates Parquet magic bytes.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    out_path = cache_dir / tile_name

    if _has_valid_parquet_magic(out_path):
        return out_path

    if out_path.exists():
        out_path.unlink()

    url = f"{base_url.rstrip('/')}/{tile_name}"

    head = requests.head(url, allow_redirects=True, timeout=120)
    content_length = head.headers.get("Content-Length")
    content_type = (head.headers.get("Content-Type") or "").lower()

    if "html" in content_type:
        raise RuntimeError(f"GBA tile URL returned HTML, not Parquet: {url}")

    if content_length is not None:
        size_mb = int(content_length) / 1024 / 1024
        if size_mb > max_download_size_mb:
            raise RuntimeError(
                f"GBA tile is too large for automatic download: "
                f"{size_mb:.1f} MB > {max_download_size_mb:.1f} MB. "
                f"Tile: {tile_name}"
            )

    with requests.get(url, stream=True, allow_redirects=True, timeout=120) as r:
        if r.status_code != 200:
            raise RuntimeError(
                f"Could not download GBA tile {tile_name}. "
                f"HTTP status: {r.status_code}. First response text: {r.text[:300]}"
            )

        response_content_type = (r.headers.get("Content-Type") or "").lower()
        if "html" in response_content_type:
            raise RuntimeError(f"GBA tile download returned HTML, not Parquet: {url}")

        downloaded = 0
        max_bytes = max_download_size_mb * 1024 * 1024

        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue

                downloaded += len(chunk)

                if downloaded > max_bytes:
                    f.close()
                    out_path.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"GBA tile download exceeded size limit "
                        f"({max_download_size_mb} MB): {tile_name}"
                    )

                f.write(chunk)

    if not _has_valid_parquet_magic(out_path):
        first_bytes = out_path.read_bytes()[:300]
        out_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded file is not a valid Parquet tile: {tile_name}. "
            f"First bytes: {first_bytes[:200]}"
        )

    return out_path


def _load_duckdb_spatial() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()

    try:
        con.execute("LOAD spatial;")
    except Exception:
        try:
            con.execute("INSTALL spatial;")
            con.execute("LOAD spatial;")
        except Exception as exc:
            raise RuntimeError(
                "DuckDB spatial extension is required for reading GBA geometry. "
                "Try installing/loading DuckDB spatial extension manually."
            ) from exc

    return con


def _normalize_wkb(value: Any) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    return value


def query_gba_lod1_subset_from_tile(
    parquet_path: Path,
    bounds_wgs84: tuple[float, float, float, float],
) -> gpd.GeoDataFrame:
    """
    Read only GBA LoD1 buildings whose bbox intersects the requested WGS84 bbox.
    """
    minx, miny, maxx, maxy = bounds_wgs84
    parquet_path_str = str(parquet_path).replace("\\", "/")

    con = _load_duckdb_spatial()

    try:
        df = con.execute(
            f"""
            SELECT
                source,
                id,
                height,
                var,
                region,
                struct_extract(bbox, 'xmin') AS xmin,
                struct_extract(bbox, 'ymin') AS ymin,
                struct_extract(bbox, 'xmax') AS xmax,
                struct_extract(bbox, 'ymax') AS ymax,
                ST_AsWKB(geometry) AS geometry_wkb
            FROM read_parquet('{parquet_path_str}')
            WHERE
                struct_extract(bbox, 'xmax') >= {minx}
                AND struct_extract(bbox, 'xmin') <= {maxx}
                AND struct_extract(bbox, 'ymax') >= {miny}
                AND struct_extract(bbox, 'ymin') <= {maxy}
            """
        ).df()
    finally:
        con.close()

    if df.empty:
        return gpd.GeoDataFrame(
            columns=[
                "source",
                "id",
                "height",
                "var",
                "region",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
            ],
            geometry=[],
            crs="EPSG:4326",
        )

    df["geometry_wkb"] = df["geometry_wkb"].apply(_normalize_wkb)

    geometry = gpd.GeoSeries.from_wkb(df["geometry_wkb"], crs="EPSG:4326")

    gba = gpd.GeoDataFrame(
        df.drop(columns=["geometry_wkb"]),
        geometry=geometry,
        crs="EPSG:4326",
    )

    gba["height"] = pd.to_numeric(gba["height"], errors="coerce")

    gba = gba[
        gba.geometry.notna()
        & (~gba.geometry.is_empty)
        & gba["height"].notna()
    ].copy()

    return gba


def load_gba_lod1_subset_for_aoi(
    aoi_gdf: gpd.GeoDataFrame,
    cache_dir: Path,
    base_url: str = DEFAULT_GBA_BASE_URL,
    bbox_buffer_deg: float = 0.002,
    max_download_size_mb: float = 2000,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """
    Automatically identify, download/cache, and spatially filter GBA LoD1 tiles for an AOI.
    """
    if aoi_gdf.empty:
        raise ValueError("AOI/building GeoDataFrame is empty; cannot query GBA.")

    if aoi_gdf.crs is None:
        raise ValueError("AOI/building GeoDataFrame has no CRS; cannot query GBA.")

    aoi_wgs84 = aoi_gdf.to_crs("EPSG:4326")

    minx, miny, maxx, maxy = aoi_wgs84.total_bounds

    bounds_buffered = (
        float(minx - bbox_buffer_deg),
        float(miny - bbox_buffer_deg),
        float(maxx + bbox_buffer_deg),
        float(maxy + bbox_buffer_deg),
    )

    tile_names = gba_lod1_tile_names_for_bbox(bounds_buffered)

    subsets: list[gpd.GeoDataFrame] = []
    downloaded_tiles: list[str] = []
    failed_tiles: list[dict[str, str]] = []

    for tile_name in tile_names:
        try:
            tile_path = download_gba_lod1_tile(
                tile_name=tile_name,
                cache_dir=Path(cache_dir),
                base_url=base_url,
                max_download_size_mb=max_download_size_mb,
            )
            downloaded_tiles.append(tile_name)

            subset = query_gba_lod1_subset_from_tile(
                parquet_path=tile_path,
                bounds_wgs84=bounds_buffered,
            )
            if not subset.empty:
                subsets.append(subset)

        except Exception as exc:
            failed_tiles.append({"tile": tile_name, "error": str(exc)})

    if subsets:
        gba = gpd.GeoDataFrame(
            pd.concat(subsets, ignore_index=True),
            geometry="geometry",
            crs="EPSG:4326",
        )

        if "id" in gba.columns:
            gba = gba.drop_duplicates(subset=["id"]).copy()
    else:
        gba = gpd.GeoDataFrame(
            columns=["source", "id", "height", "var", "region"],
            geometry=[],
            crs="EPSG:4326",
        )

    metadata = {
        "source": "global_building_atlas_lod1_parquet",
        "base_url": base_url,
        "bounds_wgs84_buffered": bounds_buffered,
        "bbox_buffer_deg": bbox_buffer_deg,
        "tile_names": tile_names,
        "downloaded_or_cached_tiles": downloaded_tiles,
        "failed_tiles": failed_tiles,
        "gba_features_in_aoi_bbox": int(len(gba)),
    }

    return gba, metadata


def enrich_missing_heights_with_gba_lod1(
    buildings: gpd.GeoDataFrame,
    cache_dir: Path,
    base_url: str = DEFAULT_GBA_BASE_URL,
    height_col: str = "height_m",
    min_overlap_share: float = 0.2,
    min_valid_height_m: float = 2.0,
    bbox_buffer_deg: float = 0.002,
    max_download_size_mb: float = 2000,
    replace_existing_height: bool = False,
) -> tuple[gpd.GeoDataFrame, dict[str, Any], gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Fill missing Overture heights using GBA LoD1, without overwriting valid Overture heights.

    Returns:
    - enriched buildings GeoDataFrame
    - summary dictionary
    - GBA subset GeoDataFrame
    - best matches GeoDataFrame
    """
    if replace_existing_height:
        raise ValueError(
            "replace_existing_height=True is not allowed for the thesis workflow. "
            "Use fill_missing_only."
        )

    if buildings.empty:
        raise ValueError("Buildings GeoDataFrame is empty.")

    if buildings.crs is None:
        raise ValueError("Buildings GeoDataFrame has no CRS.")

    if buildings.crs.is_geographic:
        raise ValueError(
            "Buildings must be in a projected metric CRS before GBA matching. "
            "Run CRS conversion/preprocessing first."
        )

    if height_col not in buildings.columns:
        raise ValueError(f"Height column not found: {height_col}")

    original_index = buildings.index

    out = buildings.reset_index(drop=True).copy()
    out["overture_row_id"] = np.arange(len(out), dtype=int)

    out["height_m_original"] = pd.to_numeric(out[height_col], errors="coerce")

    original_valid = _is_valid_height(out["height_m_original"], min_height_m=0.0)

    gba, gba_metadata = load_gba_lod1_subset_for_aoi(
        aoi_gdf=out,
        cache_dir=Path(cache_dir),
        base_url=base_url,
        bbox_buffer_deg=bbox_buffer_deg,
        max_download_size_mb=max_download_size_mb,
    )

    gba["height"] = pd.to_numeric(gba.get("height"), errors="coerce")

    gba_valid_all = gba[
        gba.geometry.notna()
        & (~gba.geometry.is_empty)
        & gba["height"].notna()
        & (gba["height"] > 0)
    ].copy()

    gba_valid_for_enrichment = gba_valid_all[
        gba_valid_all["height"] >= min_valid_height_m
    ].copy()

    missing_overture = out.loc[~original_valid].copy()

    best_matches = gpd.GeoDataFrame(
        columns=[
            "overture_row_id",
            "gba_id",
            "height_gba_candidate_m",
            "gba_source",
            "overlap_area_m2",
            "overlap_share_of_overture",
            "match_quality",
            "geometry",
        ],
        geometry="geometry",
        crs=buildings.crs,
    )

    strict_matches = best_matches.copy()

    if not missing_overture.empty and not gba_valid_for_enrichment.empty:
        missing_metric = gpd.GeoDataFrame(
            missing_overture,
            geometry="geometry",
            crs=buildings.crs,
        ).copy()

        gba_metric = gba_valid_for_enrichment.to_crs(buildings.crs).copy()

        missing_metric["overture_area_m2"] = missing_metric.geometry.area

        gba_metric = gba_metric.rename(
            columns={
                "id": "gba_id",
                "height": "height_gba_candidate_m",
                "source": "gba_source",
            }
        )

        missing_for_overlay = missing_metric[
            ["overture_row_id", "overture_area_m2", "geometry"]
        ].copy()

        gba_for_overlay = gba_metric[
            ["gba_id", "height_gba_candidate_m", "gba_source", "geometry"]
        ].copy()

        intersections = gpd.overlay(
            missing_for_overlay,
            gba_for_overlay,
            how="intersection",
            keep_geom_type=False,
        )

        if not intersections.empty:
            intersections["overlap_area_m2"] = intersections.geometry.area
            intersections["overlap_share_of_overture"] = (
                intersections["overlap_area_m2"] / intersections["overture_area_m2"]
            )

            intersections = intersections[
                intersections["overlap_area_m2"] > 1.0
            ].copy()

        if not intersections.empty:
            best_matches = (
                intersections
                .sort_values(
                    ["overture_row_id", "overlap_area_m2"],
                    ascending=[True, False],
                )
                .drop_duplicates("overture_row_id")
                .copy()
            )

            best_matches["match_quality"] = "weak_overlap"
            best_matches.loc[
                best_matches["overlap_share_of_overture"] >= min_overlap_share,
                "match_quality",
            ] = "strict_overlap"

            best_matches["height_gba_candidate_m"] = pd.to_numeric(
                best_matches["height_gba_candidate_m"],
                errors="coerce",
            )

            strict_matches = best_matches[
                (best_matches["match_quality"] == "strict_overlap")
                & best_matches["height_gba_candidate_m"].notna()
                & (best_matches["height_gba_candidate_m"] >= min_valid_height_m)
            ].copy()

    out["height_gba_m"] = np.nan
    out["gba_match_id"] = None
    out["gba_match_overlap_area_m2"] = np.nan
    out["gba_match_overlap_share"] = np.nan
    out["gba_match_quality"] = None

    if not strict_matches.empty:
        match_indexed = strict_matches.set_index("overture_row_id")

        out.loc[match_indexed.index, "height_gba_m"] = match_indexed[
            "height_gba_candidate_m"
        ]

        out.loc[match_indexed.index, "gba_match_id"] = match_indexed[
            "gba_id"
        ].astype(str)

        out.loc[match_indexed.index, "gba_match_overlap_area_m2"] = match_indexed[
            "overlap_area_m2"
        ]

        out.loc[match_indexed.index, "gba_match_overlap_share"] = match_indexed[
            "overlap_share_of_overture"
        ]

        out.loc[match_indexed.index, "gba_match_quality"] = match_indexed[
            "match_quality"
        ]

    gba_height_valid = _is_valid_height(
        out["height_gba_m"],
        min_height_m=min_valid_height_m,
    )

    out["height_m_enriched"] = out["height_m_original"]

    fill_mask = (
        (~original_valid)
        & gba_height_valid
        & (out["gba_match_quality"] == "strict_overlap")
    )

    out.loc[fill_mask, "height_m_enriched"] = out.loc[fill_mask, "height_gba_m"]

    out["height_source"] = "missing"
    out.loc[original_valid, "height_source"] = "overture"
    out.loc[fill_mask, "height_source"] = "gba_lod1"

    out["height_was_enriched"] = fill_mask

    # Use enriched final height in the existing workflow height column.
    # This does NOT overwrite valid Overture heights; it only fills missing rows.
    out[height_col] = out["height_m_enriched"]

    changed_existing_overture = (
        original_valid
        & (
            pd.to_numeric(out["height_m_original"], errors="coerce")
            != pd.to_numeric(out["height_m_enriched"], errors="coerce")
        )
    )

    if int(changed_existing_overture.sum()) != 0:
        raise AssertionError(
            "Some valid Overture heights were changed. "
            "This violates fill_missing_only policy."
        )

    final_valid = _is_valid_height(out[height_col], min_height_m=0.0)

    summary: dict[str, Any] = {
        "source": "global_building_atlas_lod1_parquet",
        "method": "gba_lod1_largest_overlap_matching",
        "policy": "fill_missing_only",
        "replace_existing_overture_height": False,
        "n_buildings": int(len(out)),
        "valid_height_count_before": int(original_valid.sum()),
        "missing_height_count_before": int((~original_valid).sum()),
        "valid_height_share_before": float(original_valid.mean()),
        "missing_height_share_before": float((~original_valid).mean()),
        "gba_lod1_candidates_in_bbox_total": int(len(gba)),
        "gba_lod1_candidates_with_positive_height": int(len(gba_valid_all)),
        "gba_lod1_candidates_after_min_height_filter": int(
            len(gba_valid_for_enrichment)
        ),
        "min_valid_height_m": float(min_valid_height_m),
        "gba_lod1_best_matches_any_overlap": int(len(best_matches)),
        "gba_lod1_strict_matches_used": int(len(strict_matches)),
        "min_overlap_share_for_enrichment": float(min_overlap_share),
        "height_enriched_count": int(fill_mask.sum()),
        "height_enriched_share_of_all_buildings": float(fill_mask.mean()),
        "height_enriched_share_of_missing_before": (
            float(fill_mask.sum() / (~original_valid).sum())
            if int((~original_valid).sum()) > 0
            else None
        ),
        "valid_height_count_after": int(final_valid.sum()),
        "missing_height_count_after": int((~final_valid).sum()),
        "valid_height_share_after": float(final_valid.mean()),
        "missing_height_share_after": float((~final_valid).mean()),
        "height_source_overture_count": int((out["height_source"] == "overture").sum()),
        "height_source_gba_lod1_count": int((out["height_source"] == "gba_lod1").sum()),
        "height_source_missing_count": int((out["height_source"] == "missing").sum()),
        "changed_existing_overture_height_count": int(changed_existing_overture.sum()),
        "gba_metadata": gba_metadata,
    }

    if not best_matches.empty:
        summary["gba_match_overlap_share_median_all_matches"] = float(
            best_matches["overlap_share_of_overture"].median()
        )
        summary["gba_match_overlap_share_min_all_matches"] = float(
            best_matches["overlap_share_of_overture"].min()
        )
        summary["gba_match_overlap_share_max_all_matches"] = float(
            best_matches["overlap_share_of_overture"].max()
        )

    # Store diagnostic information for all best matches, including weak matches.
# Only strict matches will be used to fill missing heights.
    if not best_matches.empty:
        best_match_indexed = best_matches.set_index("overture_row_id")
    
        out.loc[best_match_indexed.index, "height_gba_m"] = best_match_indexed[
            "height_gba_candidate_m"
        ]
    
        out.loc[best_match_indexed.index, "gba_match_id"] = best_match_indexed[
            "gba_id"
        ].astype(str)
    
        out.loc[best_match_indexed.index, "gba_match_overlap_area_m2"] = best_match_indexed[
            "overlap_area_m2"
        ]
    
        out.loc[best_match_indexed.index, "gba_match_overlap_share"] = best_match_indexed[
            "overlap_share_of_overture"
        ]
    
        out.loc[best_match_indexed.index, "gba_match_quality"] = best_match_indexed[
            "match_quality"
        ]

    if "footprint_area_m2" in out.columns:
        area = pd.to_numeric(out["footprint_area_m2"], errors="coerce").fillna(0)
        total_area = float(area.sum())

        summary["height_valid_area_share_before"] = (
            float(area[original_valid].sum()) / total_area if total_area > 0 else None
        )
        summary["height_valid_area_share_after"] = (
            float(area[final_valid].sum()) / total_area if total_area > 0 else None
        )
        summary["height_enriched_area_share"] = (
            float(area[fill_mask].sum()) / total_area if total_area > 0 else None
        )

    out = gpd.GeoDataFrame(out, geometry="geometry", crs=buildings.crs)
    out.index = original_index

    return out, summary, gba, best_matches


def write_height_enrichment_outputs(
    buildings_enriched: gpd.GeoDataFrame,
    summary: dict[str, Any],
    gba_subset: gpd.GeoDataFrame,
    best_matches: gpd.GeoDataFrame,
    output_dirs: dict[str, Path],
    save_enriched_buildings: bool = True,
    save_gba_subset: bool = True,
    save_matches: bool = True,
) -> None:
    processed_dir = Path(output_dirs["processed"])
    tables_dir = Path(output_dirs["tables"])
    reports_dir = Path(output_dirs["reports"])

    processed_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if save_enriched_buildings:
        buildings_enriched.to_file(
            processed_dir / "buildings_height_enriched.gpkg",
            layer="buildings_height_enriched",
            driver="GPKG",
        )

    if save_gba_subset and not gba_subset.empty:
        gba_subset.to_file(
            processed_dir / "gba_lod1_subset.gpkg",
            layer="gba_lod1_subset",
            driver="GPKG",
        )

    if save_matches and not best_matches.empty:
        best_matches.drop(columns="geometry").to_csv(
            tables_dir / "gba_lod1_height_matches.csv",
            index=False,
        )

    pd.DataFrame([summary]).to_csv(
        tables_dir / "height_enrichment_summary.csv",
        index=False,
    )

    (reports_dir / "height_enrichment_quality.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
