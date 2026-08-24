# v0.3.0 migration

Version 0.3.0 adds generic provenance and reuse safeguards without changing
indicator formulas or readiness thresholds.

## Cache artifacts

New runs record independent compatibility contracts for cleaned buildings,
height-enriched buildings, neighbour context, canonical grids, street
networks, street profiles, and building-to-street assignments. A run that
enables contextual indicators later can therefore reuse scientifically
compatible upstream artifacts. Older manifests without artifact contracts are
not accepted for new cross-run artifact reuse.

Cleaned and enriched building layers now have separate stable hashes. A change
to height-enrichment settings may retain a compatible cleaned layer but rejects
the enriched layer, so enrichment restarts from the genuine pre-enrichment
artifact.

## Spatial and street provenance

Canonical metric AOI and grid identities are stored with cache provenance.
Regular-grid row and column identities are assigned on the full lattice before
AOI clipping, which makes clipped edge cells stable across output ordering.

OSMnx street acquisition records the selected endpoint, timeout, query
settings, timestamp, and software version. Omit the optional acquisition block
to use installed OSMnx defaults. When an endpoint is explicitly configured,
the workflow records it and does not silently fall back to another service.
