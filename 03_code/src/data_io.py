"""
Data input/output utilities for the MSc urban density workflow.

This module contains data acquisition adapters. The first implemented adapter is
OvertureAdapter, which downloads Overture Maps Buildings for a user-provided AOI
and normalizes the output schema for later preprocessing, quality checks, and
indicator calculation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import geopandas as gpd
import pandas as pd

from overture_releases import (
    OvertureReleaseDiscoveryError,
    OvertureReleaseUnavailableError,
    resolve_overture_release,
)


logger = logging.getLogger(__name__)


class OvertureAcquisitionError(RuntimeError):
    """Acquisition failure with a stable category for reports and the dashboard."""

    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


OVERTURE_BUILDING_COLUMNS = [
    "building_id",
    "geometry",
    "height_m",
    "num_floors",
    "source",
    "source_release",
    "raw_source",
]


@dataclass
class OvertureAdapter:
    """
    Adapter for automated acquisition of Overture Maps Buildings.

    The adapter:
    - accepts an AOI GeoDataFrame in any valid CRS,
    - converts the AOI to EPSG:4326 for cloud-based querying,
    - queries Overture Maps Buildings from cloud GeoParquet using DuckDB,
    - returns a normalized GeoDataFrame with a stable schema.

    Important methodological note:
    The returned geometries remain in EPSG:4326. Area-based indicators such as
    GSI, FAR/FSI, and Built Volume Density must be calculated only after
    reprojection to a suitable projected CRS with metre-based units.
    """

    release: str = "auto"
    provider: Literal["aws"] = "aws"
    exclude_underground: bool = True
    clip_to_aoi: bool = False
    verify_release: bool = True

    def fetch_buildings(self, aoi_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Fetch Overture building footprints intersecting an AOI.

        Parameters
        ----------
        aoi_gdf:
            Area of interest as a GeoDataFrame. The AOI must have a defined CRS.

        Returns
        -------
        geopandas.GeoDataFrame
            Building features with normalized schema:
            building_id, geometry, height_m, num_floors, source,
            source_release, raw_source.

        Raises
        ------
        ValueError
            If the AOI is empty or has no CRS.
        ImportError
            If DuckDB is not installed.
        RuntimeError
            If the Overture query fails.
        """
        if self.verify_release:
            try:
                self.release = resolve_overture_release(
                    self.release, provider=self.provider
                ).resolved_release
            except OvertureReleaseUnavailableError as exc:
                raise OvertureAcquisitionError("release_unavailable", str(exc)) from exc
            except OvertureReleaseDiscoveryError as exc:
                raise OvertureAcquisitionError("release_discovery_failed", str(exc)) from exc
        aoi_wgs84 = self._prepare_aoi_wgs84(aoi_gdf)
        aoi_geometry = aoi_wgs84.geometry.union_all()

        minx, miny, maxx, maxy = aoi_geometry.bounds
        aoi_wkt = aoi_geometry.wkt.replace("'", "''")

        logger.info(
            "Fetching Overture buildings for bbox: "
            "(%.6f, %.6f, %.6f, %.6f), release=%s",
            minx,
            miny,
            maxx,
            maxy,
            self.release,
        )

        query = self._build_query(
            minx=minx,
            miny=miny,
            maxx=maxx,
            maxy=maxy,
            aoi_wkt=aoi_wkt,
        )

        raw_df = self._run_duckdb_query(query)

        if raw_df.empty:
            logger.warning("Overture query returned zero building features.")
            return self._empty_building_gdf()

        buildings = self._normalize_schema(raw_df)

        if self.clip_to_aoi:
            # Clipping is optional and disabled by default because source geometry
            # preservation is preferable during acquisition. Boundary treatment
            # should normally be handled explicitly in preprocessing/aggregation.
            buildings = gpd.clip(buildings, aoi_wgs84)

        return buildings[OVERTURE_BUILDING_COLUMNS]

    def _prepare_aoi_wgs84(self, aoi_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Validate AOI and convert it to EPSG:4326 for Overture querying.
        """
        if aoi_gdf is None:
            raise ValueError("AOI GeoDataFrame is None.")

        if aoi_gdf.empty:
            raise ValueError("AOI GeoDataFrame is empty.")

        if aoi_gdf.crs is None:
            raise ValueError(
                "AOI GeoDataFrame has no CRS. "
                "Define the CRS before using OvertureAdapter."
            )

        aoi = aoi_gdf.copy()
        aoi = aoi[aoi.geometry.notna()]
        aoi = aoi[~aoi.geometry.is_empty]

        if aoi.empty:
            raise ValueError("AOI contains no valid non-empty geometries.")

        # For the AOI only, basic geometry repair is acceptable because it is
        # needed for the query geometry. Building geometries are not silently
        # repaired here; they should be assessed in quality.py/preprocessing.py.
        invalid_count = (~aoi.geometry.is_valid).sum()
        if invalid_count > 0:
            logger.warning(
                "AOI contains %s invalid geometries. Applying make_valid().",
                invalid_count,
            )
            aoi["geometry"] = aoi.geometry.make_valid()

        try:
            aoi_wgs84 = aoi.to_crs("EPSG:4326")
        except Exception as exc:
            raise RuntimeError("Failed to convert AOI to EPSG:4326.") from exc

        minx, miny, maxx, maxy = aoi_wgs84.geometry.union_all().bounds

        if not (-180 <= minx <= 180 and -180 <= maxx <= 180):
            raise ValueError("AOI longitude bounds are outside EPSG:4326 range.")

        if not (-90 <= miny <= 90 and -90 <= maxy <= 90):
            raise ValueError("AOI latitude bounds are outside EPSG:4326 range.")

        return aoi_wgs84

    def _build_query(
        self,
        minx: float,
        miny: float,
        maxx: float,
        maxy: float,
        aoi_wkt: str,
    ) -> str:
        """
        Build a DuckDB SQL query for Overture Maps Buildings.

        The bbox filter reduces the amount of scanned data. The ST_Intersects
        filter then limits the result to buildings intersecting the AOI polygon.
        """
        if self.provider != "aws":
            raise ValueError("Only provider='aws' is implemented in this minimal adapter.")

        parquet_path = (
            f"s3://overturemaps-us-west-2/release/{self.release}/"
            "theme=buildings/type=building/*"
        )

        underground_filter = ""
        if self.exclude_underground:
            underground_filter = "AND (is_underground IS NULL OR is_underground = FALSE)"

        query = f"""
        SELECT
            id AS building_id,
            ST_AsWKB(geometry) AS geometry_wkb,
            height,
            num_floors,
            CAST(sources AS JSON) AS raw_source
        FROM read_parquet(
            '{parquet_path}',
            filename=true,
            hive_partitioning=1
        )
        WHERE
            bbox.xmin <= {maxx}
            AND bbox.xmax >= {minx}
            AND bbox.ymin <= {maxy}
            AND bbox.ymax >= {miny}
            {underground_filter}
            AND ST_Intersects(
                geometry,
                ST_GeomFromText('{aoi_wkt}')
            )
        """
        return query

    def _run_duckdb_query(self, query: str) -> pd.DataFrame:
        """
        Execute a DuckDB query and return a pandas DataFrame.
        """
        try:
            import duckdb
        except ImportError as exc:
            raise ImportError(
                "duckdb is required for OvertureAdapter. "
                "Install it with: pip install duckdb"
            ) from exc

        try:
            conn = duckdb.connect()

            self._load_duckdb_extension(conn, "httpfs")
            self._load_duckdb_extension(conn, "spatial")

            conn.execute("SET s3_region='us-west-2';")

            result = conn.execute(query).fetchdf()
            conn.close()

            return result

        except Exception as exc:
            text = str(exc).lower()
            category = "network_failure" if any(token in text for token in ("http", "ssl", "timeout", "connection")) else "query_failed"
            raise OvertureAcquisitionError(
                category,
                f"Overture Maps Buildings query failed; category={category}; release={self.release}.",
            ) from exc

    @staticmethod
    def _load_duckdb_extension(conn, extension_name: str) -> None:
        """
        Load a DuckDB extension. If it is not installed, install it first.
        """
        try:
            conn.execute(f"LOAD {extension_name};")
        except Exception:
            conn.execute(f"INSTALL {extension_name};")
            conn.execute(f"LOAD {extension_name};")

    def _normalize_schema(self, raw_df: pd.DataFrame) -> gpd.GeoDataFrame:
        """
        Convert raw Overture output to the workflow's normalized building schema.
        """
        required_columns = {
            "building_id",
            "geometry_wkb",
            "height",
            "num_floors",
            "raw_source",
        }

        missing_columns = required_columns - set(raw_df.columns)
        if missing_columns:
            raise OvertureAcquisitionError(
                "schema_mismatch",
                f"Overture output is missing expected columns: {missing_columns}",
            )

        geometry_wkb = raw_df["geometry_wkb"].apply(
            lambda value: bytes(value) if isinstance(value, bytearray) else value
        )

        geometry = gpd.GeoSeries.from_wkb(
            geometry_wkb,
            crs="EPSG:4326",
        )

        gdf = gpd.GeoDataFrame(
            raw_df.drop(columns=["geometry_wkb"]),
            geometry=geometry,
            crs="EPSG:4326",
        )

        # Preserve missing values as missing. Do not replace missing height or
        # floor-count values with zero or assumed defaults.
        gdf["height_m"] = pd.to_numeric(gdf["height"], errors="coerce")
        gdf.loc[gdf["height_m"] <= 0, "height_m"] = pd.NA

        floors = pd.to_numeric(gdf["num_floors"], errors="coerce")
        floors = floors.mask(floors <= 0)
        gdf["num_floors"] = floors.astype("Int64")

        gdf["source"] = "Overture Maps Buildings"
        gdf["source_release"] = self.release

        # Keep raw_source for traceability, but keep the normalized schema small.
        gdf = gdf.rename(columns={"raw_source": "raw_source"})

        normalized = gdf[
            [
                "building_id",
                "geometry",
                "height_m",
                "num_floors",
                "source",
                "source_release",
                "raw_source",
            ]
        ].copy()

        return normalized

    def _empty_building_gdf(self) -> gpd.GeoDataFrame:
        """
        Return an empty GeoDataFrame with the normalized building schema.
        """
        return gpd.GeoDataFrame(
            {
                "building_id": pd.Series(dtype="string"),
                "height_m": pd.Series(dtype="float64"),
                "num_floors": pd.Series(dtype="Int64"),
                "source": pd.Series(dtype="string"),
                "source_release": pd.Series(dtype="string"),
                "raw_source": pd.Series(dtype="object"),
            },
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
            crs="EPSG:4326",
        )



def load_buildings_from_file(path, layer=None) -> gpd.GeoDataFrame:
    ...

def save_layer(gdf, path, layer_name):
    ...
