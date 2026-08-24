# Attribution

This repository contains software only. It does not redistribute building,
street, basemap, height-enrichment, or generated workflow datasets.

The REST API directory is a later software-layer contribution that wraps the
existing Urban Density Workflow.

## Overture Maps Buildings

The implemented automated building source is the Overture Maps Buildings
theme. Overture documents the Buildings theme as Open Database License (ODbL)
data with source-specific attribution requirements.

When publishing maps, databases, or analyses derived from Overture data, review
the attribution for the exact release and sources used:

- https://docs.overturemaps.org/attribution/
- https://docs.overturemaps.org/guides/buildings/

A commonly applicable attribution is:

> OpenStreetMap contributors, Overture Maps Foundation

Confirm the required wording for the selected Overture release before
publication or redistribution.

## OpenStreetMap Street Data

Street-context acquisition uses OSMnx to obtain OpenStreetMap data. OpenStreetMap
data are licensed under ODbL. Credit OpenStreetMap and its contributors and make
the data license clear:

- https://www.openstreetmap.org/copyright

## Basemap Tiles

The local interface can use CARTO or OpenStreetMap basemap tiles. Data licenses
and tile-service usage policies are separate. Follow the selected provider's
current terms and retain the attribution inserted by the mapping library.

- CARTO basemap information: https://docs.carto.com/faqs/carto-basemaps
- OpenStreetMap tile policy: https://operations.osmfoundation.org/policies/tiles/

## Optional GlobalBuildingAtlas Height Enrichment

Optional height enrichment reads GlobalBuildingAtlas LoD1 data hosted by Source
Cooperative. Upstream material has mixed ODbL and CC BY-NC 4.0 implications.
Users must review the current license notice and cite both the original dataset
and hosted conversion where required.

- https://github.com/zhu-xlab/GlobalBuildingAtlas
- https://source.coop/tge-labs/globalbuildingatlas-lod1

No GlobalBuildingAtlas data or cache is included in this repository.
