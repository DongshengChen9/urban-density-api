"""Single, provenance-safe OSMnx street-network acquisition adapter."""
from __future__ import annotations

from datetime import datetime, timezone
import platform
from typing import Any


GRAPH_QUERY_FLAGS = {"simplify": True, "retain_all": False, "truncate_by_edge": True}


class OSMStreetAcquisitionError(RuntimeError):
    """A failed street query with safe diagnostic provenance."""
    def __init__(self, message: str, provenance: dict[str, Any]):
        super().__init__(message)
        self.provenance = provenance


def _settings_value(settings: Any, name: str) -> Any:
    return getattr(settings, name, None)


def acquire_osmnx_street_edges(aoi, *, network_type: str, acquisition_config: dict[str, Any] | None = None, target_crs: Any = None, query_context: dict[str, Any] | None = None):
    """Query OSMnx once with an optional explicit endpoint and record it.

    Explicit settings are restored immediately after the query.  There is no
    endpoint fallback: an error describes the selected effective endpoint.
    """
    import geopandas as gpd
    import osmnx as ox

    if aoi.empty or aoi.crs is None:
        raise ValueError("Street acquisition requires a non-empty AOI with a CRS.")
    cfg = acquisition_config or {}
    endpoint = cfg.get("overpass_endpoint")
    timeout = cfg.get("timeout_seconds")
    previous_endpoint = _settings_value(ox.settings, "overpass_url")
    previous_timeout = _settings_value(ox.settings, "timeout")
    if endpoint:
        ox.settings.overpass_url = str(endpoint)
    if timeout is not None:
        ox.settings.timeout = int(timeout)
    aoi_wgs84 = aoi.to_crs("EPSG:4326")
    polygon = aoi_wgs84.geometry.union_all()
    provenance = {
        "data_source": "OpenStreetMap", "acquisition_adapter": "OSMnx",
        "endpoint_selection_mode": "explicit" if endpoint else "default",
        "effective_overpass_endpoint": _settings_value(ox.settings, "overpass_url"),
        "timeout_seconds": _settings_value(ox.settings, "timeout"),
        "acquisition_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "osmnx_version": getattr(ox, "__version__", None),
        "geopandas_version": getattr(gpd, "__version__", None),
        "python_version": platform.python_version(), "network_type": network_type,
        "query_flags": dict(GRAPH_QUERY_FLAGS), "target_crs": str(target_crs or aoi.crs),
        "query_context": query_context or {}, "cache_reused": False,
    }
    try:
        graph = ox.graph_from_polygon(polygon, network_type=network_type, **GRAPH_QUERY_FLAGS)
        edges = ox.graph_to_gdfs(graph, nodes=False, edges=True).reset_index()
    except Exception as exc:
        provenance["failure_category"] = "external_street_acquisition_service_failure"
        provenance["technical_reason"] = f"{type(exc).__name__}: {exc}"
        raise OSMStreetAcquisitionError("OSM street acquisition failed without endpoint fallback.", provenance) from exc
    finally:
        ox.settings.overpass_url = previous_endpoint
        ox.settings.timeout = previous_timeout
    return edges, provenance
