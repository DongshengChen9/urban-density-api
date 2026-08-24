from __future__ import annotations

import math

import geopandas as gpd
from shapely.geometry import box

from spatial_identity import CANONICAL_GRID_ID_CONVENTION, attach_canonical_grid_ids


def _validate_projected_aoi(aoi: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Validate AOI before grid generation.

    The AOI must already be in a projected CRS because grid cell size is
    interpreted in metres.
    """
    if aoi is None:
        raise ValueError("AOI GeoDataFrame is None.")

    if aoi.empty:
        raise ValueError("AOI GeoDataFrame is empty.")

    if aoi.crs is None:
        raise ValueError("AOI GeoDataFrame has no CRS.")

    if aoi.crs.is_geographic:
        raise ValueError(
            "AOI is in a geographic CRS. Reproject to a metric CRS before creating a grid."
        )

    aoi_clean = aoi.copy()
    aoi_clean = aoi_clean[aoi_clean.geometry.notna()]
    aoi_clean = aoi_clean[~aoi_clean.geometry.is_empty]

    if aoi_clean.empty:
        raise ValueError("AOI contains no non-empty geometries.")

    invalid_count = (~aoi_clean.geometry.is_valid).sum()
    if invalid_count > 0:
        aoi_clean["geometry"] = aoi_clean.geometry.make_valid()

    return aoi_clean


def create_grid(
    aoi: gpd.GeoDataFrame,
    cell_size_m: int,
    clip_to_aoi: bool = True,
    max_cells: int = 200_000,
    grid_id_convention: str = CANONICAL_GRID_ID_CONVENTION,
) -> gpd.GeoDataFrame:
    """
    Create a regular square aggregation grid for an AOI.

    Parameters
    ----------
    aoi:
        AOI GeoDataFrame. Must be in a projected CRS with metre-based units.
    cell_size_m:
        Grid cell size in metres.
    clip_to_aoi:
        If True, grid cells are clipped to the AOI boundary.
    max_cells:
        Safety limit to prevent accidentally creating a very large grid.

    Returns
    -------
    geopandas.GeoDataFrame
        Grid with columns:
        - unit_id
        - cell_size_m
        - unit_area_m2
        - is_partial_cell
        - geometry

    Scientific note
    ---------------
    The grid is an aggregation unit, not a source dataset. If cells are clipped
    to the AOI, edge cells may have smaller areas. Indicator denominators must
    therefore use unit_area_m2, not cell_size_m ** 2.
    """
    if cell_size_m <= 0:
        raise ValueError("cell_size_m must be a positive number.")

    aoi_clean = _validate_projected_aoi(aoi)

    aoi_geometry = aoi_clean.geometry.union_all()
    minx, miny, maxx, maxy = aoi_geometry.bounds

    width = maxx - minx
    height = maxy - miny

    n_cols = math.ceil(width / cell_size_m)
    n_rows = math.ceil(height / cell_size_m)

    estimated_cells = n_cols * n_rows
    if estimated_cells > max_cells:
        raise ValueError(
            f"Grid would create {estimated_cells} cells, which exceeds max_cells={max_cells}. "
            "Increase cell_size_m or use a smaller AOI."
        )

    cells = []

    for col in range(n_cols):
        x0 = minx + col * cell_size_m
        x1 = x0 + cell_size_m

        for row in range(n_rows):
            y0 = miny + row * cell_size_m
            y1 = y0 + cell_size_m

            cells.append({"row_index": row, "column_index": col, "geometry": box(x0, y0, x1, y1)})

    grid = gpd.GeoDataFrame(
        cells,
        crs=aoi_clean.crs,
    )

    if clip_to_aoi:
        aoi_single = gpd.GeoDataFrame(
            {"aoi_id": [1]},
            geometry=[aoi_geometry],
            crs=aoi_clean.crs,
        )

        grid = gpd.overlay(
            grid,
            aoi_single,
            how="intersection",
            keep_geom_type=True,
        )

        if "aoi_id" in grid.columns:
            grid = grid.drop(columns=["aoi_id"])

    grid = grid[grid.geometry.notna()]
    grid = grid[~grid.geometry.is_empty]

    if grid.empty:
        raise ValueError("Grid is empty after clipping to AOI.")

    grid = grid[grid.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    grid = grid.reset_index(drop=True)

    if grid_id_convention == CANONICAL_GRID_ID_CONVENTION:
        grid = attach_canonical_grid_ids(
            grid,
            origin_x=minx,
            origin_y=miny,
            cell_size_m=cell_size_m,
        )
    elif grid_id_convention == "legacy_cell_sequence":
        grid["unit_id"] = [f"cell_{i:05d}" for i in range(1, len(grid) + 1)]
    else:
        raise ValueError(f"Unsupported grid ID convention: {grid_id_convention}")
    grid["cell_size_m"] = cell_size_m
    grid["unit_area_m2"] = grid.geometry.area

    full_cell_area = cell_size_m * cell_size_m
    grid["is_partial_cell"] = grid["unit_area_m2"] < full_cell_area * 0.999

    grid = grid[
        [
            "unit_id", "row_index", "column_index",
            "cell_size_m",
            "unit_area_m2",
            "is_partial_cell",
            "geometry",
        ]
    ]

    return grid
