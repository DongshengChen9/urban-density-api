"""Stable, configuration-independent identities for metric AOIs and grids."""
from __future__ import annotations

import hashlib
import math
from typing import Any

import geopandas as gpd
from shapely import normalize, set_precision
from shapely.geometry import box


CORE_SPATIAL_IDENTITY_SCHEMA = "canonical_spatial_identity_v1"
CANONICAL_GRID_ID_CONVENTION = "row_col_0_based_south_to_north_v1"
METRIC_PRECISION_M = 0.001
WGS84_QUERY_PRECISION_DEG = 1e-8


def _hash_geometry(geometry, precision: float) -> str:
    normalized = normalize(set_precision(geometry, precision))
    return hashlib.sha256(normalized.wkb).hexdigest()


def _union(frame: gpd.GeoDataFrame):
    valid = frame.geometry[frame.geometry.notna() & ~frame.geometry.is_empty]
    if valid.empty:
        raise ValueError("Spatial identity requires non-empty geometry.")
    union = valid.union_all()
    return union.make_valid() if not union.is_valid else union


def canonical_metric_aoi_identity(aoi: gpd.GeoDataFrame, target_crs: Any) -> dict[str, Any]:
    metric = aoi.to_crs(target_crs)
    geometry = _union(metric)
    return {
        "spatial_identity_schema": CORE_SPATIAL_IDENTITY_SCHEMA,
        "canonical_aoi_metric_hash": _hash_geometry(geometry, METRIC_PRECISION_M),
        "canonical_aoi_metric_area_m2": float(geometry.area),
        "target_metric_crs": str(target_crs),
    }


def acquisition_query_wgs84_identity(aoi: gpd.GeoDataFrame) -> dict[str, Any]:
    geometry = _union(aoi.to_crs("EPSG:4326"))
    return {"acquisition_query_wgs84_hash": _hash_geometry(geometry, WGS84_QUERY_PRECISION_DEG)}


def canonical_cell_id(row_index: int, column_index: int) -> str:
    return f"r{int(row_index):05d}_c{int(column_index):05d}"


def attach_canonical_grid_ids(grid: gpd.GeoDataFrame, *, origin_x: float, origin_y: float, cell_size_m: float) -> gpd.GeoDataFrame:
    """Attach stable lattice IDs; row/column are determined before clipping."""
    result = grid.copy()
    if "row_index" not in result or "column_index" not in result:
        anchors = result.geometry.representative_point()
        result["column_index"] = ((anchors.x - origin_x) / cell_size_m).map(math.floor).astype(int)
        result["row_index"] = ((anchors.y - origin_y) / cell_size_m).map(math.floor).astype(int)
    result["row_index"] = result["row_index"].astype(int)
    result["column_index"] = result["column_index"].astype(int)
    if result.duplicated(["row_index", "column_index"]).any():
        raise ValueError("Canonical grid row/column pairs must be unique.")
    if "unit_id" in result:
        result["legacy_unit_id"] = result["unit_id"].astype(str)
    result["unit_id"] = [canonical_cell_id(row, col) for row, col in zip(result.row_index, result.column_index)]
    return result.sort_values(["row_index", "column_index"]).reset_index(drop=True)


def canonical_grid_identity(grid: gpd.GeoDataFrame, *, origin_x: float, origin_y: float, cell_size_m: float) -> dict[str, Any]:
    if grid.crs is None or grid.crs.is_geographic:
        raise ValueError("Canonical grid identity requires a metric CRS.")
    required = {"unit_id", "row_index", "column_index"}
    if not required.issubset(grid.columns):
        raise ValueError("Canonical grid identity requires unit_id, row_index and column_index.")
    ordered = grid.sort_values(["row_index", "column_index"])
    if ordered.unit_id.duplicated().any():
        raise ValueError("Canonical grid identifiers must be unique.")
    geometry_hash = _hash_geometry(_union(ordered), METRIC_PRECISION_M)
    pairs = "\n".join(f"{row.unit_id}:{_hash_geometry(row.geometry, METRIC_PRECISION_M)}" for row in ordered.itertuples())
    return {
        "canonical_grid_geometry_hash": geometry_hash,
        "canonical_grid_identity_hash": hashlib.sha256(pairs.encode("utf-8")).hexdigest(),
        "grid_identity": {"grid_id_convention": CANONICAL_GRID_ID_CONVENTION, "grid_origin_x_m": float(origin_x), "grid_origin_y_m": float(origin_y), "grid_cell_size_m": float(cell_size_m), "grid_rows": int(ordered.row_index.max()) + 1, "grid_columns": int(ordered.column_index.max()) + 1},
    }


def compatibility_status(current: dict[str, Any], source: dict[str, Any]) -> tuple[str, list[str]]:
    fields = ("spatial_identity_schema", "canonical_aoi_metric_hash", "target_metric_crs")
    reasons = [f"Spatial identity mismatch for `{field}`." for field in fields if current.get(field) != source.get(field)]
    return ("COMPATIBLE" if not reasons else "INCOMPATIBLE_GEOMETRY", reasons)
