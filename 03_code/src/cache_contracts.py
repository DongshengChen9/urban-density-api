"""Artifact-specific cache contracts for compatible workflow reuse.

The contracts intentionally separate scientific artifacts.  Enabling a later
contextual indicator therefore does not invalidate compatible preprocessing or
height-enrichment results.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


ARTIFACT_CACHE_SCHEMA_VERSION = 1
ARTIFACT_NAMES = (
    "building_core",
    "enriched_buildings",
    "neighbor_context",
    "canonical_grid",
    "street_network",
    "street_profiles",
    "street_assignments",
)


def _digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def build_artifact_contracts(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build deterministic compatibility contracts from a run manifest."""
    identity = {
        key: manifest.get(key)
        for key in (
            "spatial_identity_schema", "canonical_aoi_metric_hash",
            "acquisition_query_wgs84_hash", "target_metric_crs", "processing_mode",
        )
    }
    data_source = manifest.get("data_source", {})
    preprocessing = manifest.get("preprocessing", {})
    height = manifest.get("height_enrichment", {})
    street = manifest.get("street_context", {})
    aggregation = manifest.get("aggregation", {})
    building_core = {"identity": identity, "data_source": data_source, "preprocessing": preprocessing,
                     "indicator_definition_version": manifest.get("indicator_definition_version")}
    enriched = {"building_core": _digest(building_core), "height_enrichment": height}
    neighbor = {"enriched_buildings": _digest(enriched), "algorithm_version": "1"}
    grid = {"identity": identity, "aggregation": aggregation}
    street_network = {"identity": identity, "source": street.get("source", "osmnx"),
                      "network_type": street.get("network_type", "drive")}
    street_profiles = {"street_network": _digest(street_network), "enriched_buildings": _digest(enriched),
                       "distance_m": street.get("distance_m"), "tick_length_m": street.get("tick_length_m"),
                       "topology_rule_version": street.get("topology_rule_version", 1)}
    assignments = {"street_profiles": _digest(street_profiles), "enriched_buildings": _digest(enriched)}
    payloads = {
        "building_core": building_core,
        "enriched_buildings": enriched,
        "neighbor_context": neighbor,
        "canonical_grid": grid,
        "street_network": street_network,
        "street_profiles": street_profiles,
        "street_assignments": assignments,
    }
    return {name: {"signature": _digest(payload), "inputs": payload} for name, payload in payloads.items()}


def refresh_artifact_contracts(manifest: dict[str, Any]) -> dict[str, Any]:
    manifest["artifact_cache_schema_version"] = ARTIFACT_CACHE_SCHEMA_VERSION
    manifest["artifact_contracts"] = build_artifact_contracts(manifest)
    return manifest


def compare_artifact_contracts(current_manifest: dict[str, Any], source_manifest: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Return an explicit compatibility decision for every reusable artifact."""
    current = current_manifest.get("artifact_contracts") or build_artifact_contracts(current_manifest)
    if not source_manifest or source_manifest.get("artifact_cache_schema_version") != ARTIFACT_CACHE_SCHEMA_VERSION:
        return {name: {"compatible": False, "reasons": ["Artifact-aware cache metadata are required for cross-run reuse."]} for name in ARTIFACT_NAMES}
    source = source_manifest.get("artifact_contracts") or {}
    result = {}
    for name in ARTIFACT_NAMES:
        expected = (current.get(name) or {}).get("signature")
        observed = (source.get(name) or {}).get("signature")
        result[name] = {"compatible": bool(expected and expected == observed),
                        "reasons": [] if expected and expected == observed else [f"Incompatible `{name}` contract."]}
    return result


def contract_is_compatible(plan: dict[str, dict[str, Any]], artifact_name: str) -> bool:
    return bool(plan.get(artifact_name, {}).get("compatible", False))


def artifact_contract_signature(manifest: dict[str, Any], artifact_name: str) -> str | None:
    return ((manifest.get("artifact_contracts") or build_artifact_contracts(manifest)).get(artifact_name) or {}).get("signature")
