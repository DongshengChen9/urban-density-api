# Third-Party Notices

## Scope

The workflow depends on third-party Python packages and accesses external data
services. Those packages and data remain governed by their own licenses. The
software license in `LICENSE` does not relicense them.

## Direct Python Dependencies

The direct runtime dependency families are listed in `pyproject.toml`:

- Branca, DuckDB, Folium, GeoPandas, Matplotlib, Momepy, NumPy, OSMnx,
  pandas, Pillow, Psutil, PyArrow, Pyogrio, PyProj, PyYAML, Rasterio, Requests,
  Shapely, Streamlit, and streamlit-folium.

The final release should retain the license and notice files supplied by the
installed packages. Exact tested versions are recorded in
`requirements-tested.txt` after clean-environment validation.

## External Data And Services

See `ATTRIBUTION.md` for Overture Maps Buildings, OpenStreetMap, basemap, and
optional GlobalBuildingAtlas obligations. No external datasets are bundled.

## Provenance Review

No explicit copied-source or adapted-source notice was found in the production
files selected for this candidate. The repository owner should retain normal
provenance and third-party license review for future contributions.
