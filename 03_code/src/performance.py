from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import geopandas as gpd
import pandas as pd
import shapely

try:
    import psutil
except ImportError:  # pragma: no cover - dependency is present in the project env
    psutil = None


INDICATOR_DEFINITION_VERSION = "2"
# Version 4 introduces artifact-specific contracts and separate physical layer
# hashes. Earlier manifests remain readable but are not eligible for new
# cross-run artifact reuse.
CACHE_MANIFEST_VERSION = 4
ARTIFACT_HASH_ALGORITHM = "semantic_geodataframe_v2"


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hash_geodataframe(
    frame: gpd.GeoDataFrame,
    id_column: str | None = None,
    attribute_columns: list[str] | None = None,
    *,
    semantic: bool,
) -> str:
    """Hash IDs, selected attributes, CRS, and geometry deterministically."""
    if frame.crs is None:
        raise ValueError("Cannot hash a GeoDataFrame without a CRS.")

    columns = list(attribute_columns or [])
    if id_column and id_column not in frame.columns:
        raise ValueError(f"ID column not found for cache hash: {id_column}")

    sort_columns = [id_column] if id_column else []
    value_columns = [*sort_columns, *columns]
    working = frame[[*value_columns, "geometry"]].copy()
    if id_column:
        working = working.sort_values(id_column, kind="mergesort")
    else:
        working = working.sort_index()

    digest = hashlib.sha256()
    authority = frame.crs.to_authority()
    crs_identity = ":".join(authority) if authority else frame.crs.to_wkt()
    digest.update(crs_identity.encode("utf-8"))
    digest.update("|".join(columns).encode("utf-8"))

    if semantic and value_columns:
        # GeoPackage represents nullable integer fields as floating point on
        # read-back. Normalize numeric values and categorical values before
        # hashing so storage dtype changes do not invalidate the same artifact.
        for column in value_columns:
            series = working[column]
            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                working[column] = pd.to_numeric(series, errors="coerce").astype("float64")
            else:
                working[column] = series.astype("string").fillna("<missing>")

    chunk_size = 10_000
    for start in range(0, len(working), chunk_size):
        chunk = working.iloc[start : start + chunk_size]
        if value_columns:
            value_hashes = pd.util.hash_pandas_object(
                chunk[value_columns],
                index=False,
                categorize=False,
            ).to_numpy(dtype="uint64", copy=False)
            digest.update(value_hashes.tobytes())
        geometry = shapely.normalize(chunk.geometry.array) if semantic else chunk.geometry.array
        geometry_wkb = shapely.to_wkb(geometry, byte_order=1)
        digest.update(
            b"".join(
                len(value or b"").to_bytes(8, "little") + (value or b"")
                for value in geometry_wkb
            )
        )

    return digest.hexdigest()


def stable_geodataframe_hash(
    frame: gpd.GeoDataFrame,
    id_column: str | None = None,
    attribute_columns: list[str] | None = None,
) -> str:
    """Hash semantic artifact content across supported storage round trips."""
    return _hash_geodataframe(
        frame, id_column=id_column, attribute_columns=attribute_columns, semantic=True
    )


def legacy_stable_geodataframe_hash(
    frame: gpd.GeoDataFrame,
    id_column: str | None = None,
    attribute_columns: list[str] | None = None,
) -> str:
    """Reproduce the v0.3.0 dtype-sensitive hash for verified migration only."""
    return _hash_geodataframe(
        frame, id_column=id_column, attribute_columns=attribute_columns, semantic=False
    )


def legacy_storage_compatible_hashes(
    frame: gpd.GeoDataFrame,
    id_column: str | None = None,
    attribute_columns: list[str] | None = None,
) -> set[str]:
    """Return strict legacy candidates for known GeoPackage nullable-integer coercion."""
    columns = list(attribute_columns or [])
    candidates = {legacy_stable_geodataframe_hash(frame, id_column, columns)}
    normalized = frame.copy()
    changed = False
    for column in columns:
        values = pd.to_numeric(normalized[column], errors="coerce")
        finite = values.dropna()
        if not finite.empty and ((finite % 1) == 0).all():
            normalized[column] = values.astype("Int64")
            changed = True
    if changed:
        candidates.add(legacy_stable_geodataframe_hash(normalized, id_column, columns))
    return candidates


def cache_signature(
    *,
    aoi_hash: str,
    source_release: str | None,
    processing_mode: str,
    target_crs: str,
    preprocessing: dict[str, Any],
    height_enrichment: dict[str, Any],
    street_context: dict[str, Any],
    input_layer_hashes: dict[str, str | None],
    indicator_definition_version: str = INDICATOR_DEFINITION_VERSION,
) -> dict[str, Any]:
    """Build the cache-relevant identity used by reusable AOI-level stages."""
    signature = {
        "aoi_hash": aoi_hash,
        "source_release": source_release,
        "processing_mode": processing_mode,
        "target_crs": target_crs,
        "preprocessing": preprocessing,
        "height_enrichment": height_enrichment,
        "street_context": street_context,
        "input_layer_hashes": input_layer_hashes,
        "indicator_definition_version": indicator_definition_version,
    }
    return {**signature, "signature_hash": _json_hash(signature)}


def compare_cache_signatures(
    current: dict[str, Any],
    cached: dict[str, Any] | None,
    required_fields: list[str] | None = None,
) -> dict[str, Any]:
    """Report cache compatibility without silently ignoring substantive fields."""
    if cached is None:
        return {
            "compatible": False,
            "status": "manifest_missing",
            "reasons": ["Compatible cache metadata were not found."],
        }

    fields = required_fields or [
        "aoi_hash",
        "source_release",
        "processing_mode",
        "target_crs",
        "preprocessing",
        "height_enrichment",
        "street_context",
        "input_layer_hashes",
        "indicator_definition_version",
    ]
    reasons = []
    for field_name in fields:
        if current.get(field_name) != cached.get(field_name):
            reasons.append(
                f"Cache field `{field_name}` differs: "
                f"current={current.get(field_name)!r}; cached={cached.get(field_name)!r}."
            )
    return {
        "compatible": not reasons,
        "status": "compatible" if not reasons else "mismatch_detected",
        "reasons": reasons,
    }


def geodataframe_memory_bytes(frame: gpd.GeoDataFrame | pd.DataFrame | None) -> int | None:
    if frame is None:
        return None
    return int(frame.memory_usage(index=True, deep=True).sum())


def file_bytes(paths: list[Path] | None) -> int | None:
    if not paths:
        return None
    return int(sum(path.stat().st_size for path in paths if path.exists() and path.is_file()))


def process_rss_bytes() -> int | None:
    if psutil is None:
        return None
    return int(psutil.Process(os.getpid()).memory_info().rss)


@dataclass
class PerformanceRecorder:
    """Collect detailed stage metrics while leaving scientific processing unchanged."""

    records: list[dict[str, Any]] = field(default_factory=list)

    @contextmanager
    def stage(
        self,
        stage: str,
        *,
        input_rows: int | None = None,
        cached: bool = False,
        bytes_read: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        started = time.perf_counter()
        rss_before = process_rss_bytes()
        details: dict[str, Any] = {}
        status = "completed"
        try:
            yield details
        except Exception:
            status = "failed"
            raise
        finally:
            rss_after = process_rss_bytes()
            peak_approx = max(v for v in [rss_before, rss_after] if v is not None) if any(
                v is not None for v in [rss_before, rss_after]
            ) else None
            self.records.append(
                {
                    "stage": stage,
                    "status": status,
                    "wall_clock_seconds": float(time.perf_counter() - started),
                    "input_rows": input_rows,
                    "output_rows": details.get("output_rows"),
                    "candidate_pair_count": details.get("candidate_pair_count"),
                    "peak_process_memory_bytes_approx": peak_approx,
                    "dataframe_memory_bytes": details.get("dataframe_memory_bytes"),
                    "bytes_read": bytes_read,
                    "bytes_written": details.get("bytes_written"),
                    "cache_status": "loaded_from_cache" if cached else "computed",
                    "notes": details.get("notes"),
                }
            )

    def add_record(self, stage: str, **values: Any) -> None:
        self.records.append({"stage": stage, **values})

    def write(self, reports_dir: Path, tables_dir: Path) -> tuple[Path, Path]:
        reports_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)
        json_path = reports_dir / "performance_metrics.json"
        csv_path = tables_dir / "performance_metrics.csv"
        json_path.write_text(json.dumps(self.records, indent=2), encoding="utf-8")
        pd.DataFrame(self.records).to_csv(csv_path, index=False)
        return json_path, csv_path


@dataclass
class StageStateTracker:
    """Persist resumable stage status independently from indicator availability."""

    path: Path
    run_name: str
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path, run_name: str) -> "StageStateTracker":
        tracker = cls(path=path, run_name=run_name)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("run_name") == run_name:
                tracker.stages = dict(data.get("stages", {}))
        return tracker

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_name": self.run_name,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "stages": self.stages,
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def mark(self, stage: str, status: str, **metadata: Any) -> None:
        if status not in {"pending", "processing", "completed", "failed", "unavailable"}:
            raise ValueError(f"Unsupported stage status: {status}")
        self.stages[stage] = {
            "status": status,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            **metadata,
        }
        self._write()

    def can_resume(self, stage: str, signature_hash: str) -> bool:
        record = self.stages.get(stage, {})
        return (
            record.get("status") == "completed"
            and record.get("signature_hash") == signature_hash
        )


def write_geodata_cache(frame: gpd.GeoDataFrame, path: Path) -> Path:
    """Write a compact internal GeoParquet cache with CRS and nulls preserved."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def read_geodata_cache(path: Path) -> gpd.GeoDataFrame:
    return gpd.read_parquet(path)


def resolve_geodata_cache_path(gpkg_path: Path) -> Path:
    """Prefer an equivalent GeoParquet cache, then fall back to GeoPackage."""
    parquet_path = gpkg_path.with_suffix(".parquet")
    return parquet_path if parquet_path.exists() else gpkg_path


def read_geodata_layer(path: Path, layer: str | None = None) -> gpd.GeoDataFrame:
    if path.suffix.lower() in {".parquet", ".geoparquet"}:
        frame = gpd.read_parquet(path)
    else:
        frame = gpd.read_file(path, layer=layer)
    if frame.crs is not None and frame.crs.to_authority():
        authority, code = frame.crs.to_authority()
        frame = frame.set_crs(f"{authority}:{code}", allow_override=True)
    return frame


def restore_singlepart_polygon_types(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Undo GeoPackage single-part MultiPolygon promotion on a temporary copy."""
    result = frame.copy()
    multi_mask = (
        result.geometry.geom_type.eq("MultiPolygon")
        & (shapely.get_num_geometries(result.geometry.array) == 1)
    )
    if multi_mask.any():
        result.loc[multi_mask, "geometry"] = shapely.get_geometry(
            result.loc[multi_mask, "geometry"].array,
            0,
        )
    return result


def dashboard_required_columns() -> set[str]:
    return {
        "unit_id",
        "unit_area_m2",
        "is_partial_cell",
        "gsi",
        "far_fsi",
        "built_volume_density",
        "avg_neighbor_distance_m",
        "avg_street_profile_height_to_width_ratio_strict",
        "floor_data_valid_area_share",
        "height_valid_area_share",
        "geometry",
    }


def compact_grid_for_dashboard(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep only dashboard-required columns without changing rows or geometry."""
    keep = [column for column in grid.columns if column in dashboard_required_columns()]
    return grid[keep].copy()
