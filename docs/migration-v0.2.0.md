# v0.2.0 migration

Version 0.2.0 updates reusable workflow behaviour.

## Indicator and cache changes

- GSI now uses the union of building-footprint intersections in each aggregation unit. It can differ from v0.1.0 where footprints overlap. `gsi_raw_sum`, raw and union footprint areas, and overlap diagnostics remain available for review.
- Grid and indicator caches created for definition version 1 are not compatible with definition version 2.
- Street-profile values with non-finite, negative, zero, at-or-below `1e-6 m`, empty, or zero-length denominators are missing. Profiles whose sampled origin intersects or touches a mapped building are also missing. High valid H/W values are retained and never capped.
- Missing values are not zeros. In particular, missing height, floor, or street-profile support must not be interpreted as a low indicator value.

## Building releases and repeated runs

New interactive analyses request `overture_release: auto`. The workflow resolves one currently available official dated Overture Buildings release at run start and writes that exact release to provenance outputs. For a reproducible study, supply a dated `overture_release` explicitly.

Grid-size reruns retain the original resolved release and can reuse compatible raw, prepared, enriched, neighbour, and street-context artifacts. They do not resolve a newer release merely because the grid changes.

## Cartography

Dashboard, static, and exported web maps share one presentation registry. Grey denotes missing or insufficient input, near-white denotes true zero, and higher valid values use progressively darker teal. These styles do not alter scientific values.
