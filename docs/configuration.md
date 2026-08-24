# Configuration

The workflow reads YAML. The public example is
`03_code/config/example_urban_area_100m.yaml`.

- `project` defines the run name, output directory, and overwrite behavior.
- `aoi` defines the analysis area in WGS84, normally as a bounding box.
- `crs_strategy` selects single-CRS or segmented UTM routing.
- `data_source` configures the implemented Overture Buildings source.
- `preprocessing` controls geometry cleaning, clipping, and metric reprojection.
- `aggregation` sets the regular-grid cell size in metres.
- `indicators` enables GSI, FAR/FSI, Built Volume Density, and neighbour distance.
- `height_enrichment` optionally fills missing heights from GBA LoD1 without
  replacing valid source heights.
- `street_context` enables the street-profile height-to-width indicator using
  OpenStreetMap streets acquired through OSMnx.
- `cache` permits reuse only when the saved manifest is compatible.

Missing floor or height attributes remain missing; they are not converted to
zero. FAR/FSI and Built Volume Density should be interpreted together with their
coverage and readiness information.

