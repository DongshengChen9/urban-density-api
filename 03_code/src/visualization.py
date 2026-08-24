from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from map_styles import color_for_value, legend_entries, resolved_style, style_for_column


def _prepare_output_path(output_path: Path) -> Path:
    """
    Ensure that the output directory exists.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _column_has_valid_values(gdf, column: str) -> bool:
    """
    Check whether a column exists and contains at least one non-missing value.
    """
    if column not in gdf.columns:
        return False

    values = pd.to_numeric(gdf[column], errors="coerce")
    return not values.dropna().empty


def _missing_summary(gdf, column: str) -> str:
    """
    Return a short missing-data summary for a map title.
    """
    if column not in gdf.columns:
        return "column not available"

    values = pd.to_numeric(gdf[column], errors="coerce")
    total = len(values)
    missing = int(values.isna().sum())

    if total == 0:
        return "no cells"

    missing_share = missing / total
    return f"missing cells: {missing}/{total} ({missing_share:.1%})"


def calculate_shared_value_range(indicator_grids: list[Any], column: str) -> tuple[float, float]:
    """
    Return a shared numeric display range for a map comparison series.

    This helper is only for visualization scaling. It does not alter indicator
    values or aggregation results.
    """
    values: list[pd.Series] = []
    for grid in indicator_grids:
        if column in grid.columns:
            numeric = pd.to_numeric(grid[column], errors="coerce").dropna()
            if not numeric.empty:
                values.append(numeric)

    if not values:
        raise ValueError(f"No valid values found for shared legend column `{column}`.")

    combined = pd.concat(values, ignore_index=True)
    return float(combined.min()), float(combined.max())


def plot_indicator_map(
    indicator_grid,
    column: str,
    output_path: Path,
    title: str | None = None,
    legend_label: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    map_note: str | None = None,
    aoi=None,
    cmap: str = "viridis",
    dpi: int = 300,
    figsize: tuple[float, float] = (8, 8),
    missing_color: str = "#d0d0d0",
    edgecolor: str = "#d9d9d9",
    linewidth: float = 0.15,
) -> Path | None:
    """
    Save a static choropleth-style map for one grid-level indicator.
    """
    output_path = _prepare_output_path(output_path)

    if indicator_grid.empty:
        logging.warning("Cannot map %s: indicator grid is empty.", column)
        return None

    if column not in indicator_grid.columns:
        logging.warning("Cannot map %s: column not found.", column)
        return None

    gdf = indicator_grid.copy()
    gdf[column] = pd.to_numeric(gdf[column], errors="coerce")

    fig, ax = plt.subplots(figsize=figsize)

    shared_style = style_for_column(column)
    if shared_style is not None:
        shared_style = resolved_style(shared_style, gdf[column])
        missing_color = shared_style.missing_color
    valid = gdf[gdf[column].notna()].copy()
    missing = gdf[gdf[column].isna()].copy()

    # 1. Draw missing cells first, so valid cells are drawn above them.
    if not missing.empty:
        missing.plot(
            ax=ax,
            color=missing_color,
            edgecolor="#999999",
            linewidth=0.25,
            hatch="///",
        )

    # 2. Draw valid indicator values.
    if valid.empty:
        gdf.boundary.plot(
            ax=ax,
            linewidth=linewidth,
            color=edgecolor,
        )
        ax.text(
            0.5,
            0.5,
            f"No valid values for {column}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )
    elif shared_style is not None:
        valid["_map_display_color"] = valid[column].map(
            lambda value: color_for_value(shared_style, value)
        )
        valid.plot(
            ax=ax,
            color=valid["_map_display_color"],
            edgecolor=edgecolor,
            linewidth=linewidth,
        )
        ax.legend(
            handles=[Patch(facecolor=color, edgecolor="#777777", label=label) for _key, label, color in legend_entries(shared_style)],
            title=shared_style.public_label,
            loc="lower left",
            fontsize=7,
            title_fontsize=8,
        )
    else:
        valid.plot(
            ax=ax,
            column=column,
            cmap=cmap,
            legend=True,
            edgecolor=edgecolor,
            linewidth=linewidth,
            vmin=vmin,
            vmax=vmax,
            legend_kwds={
                "label": legend_label or column,
                "shrink": 0.75,
            },
        )

    # 3. Draw AOI boundary.
    if aoi is not None and not aoi.empty:
        try:
            aoi.boundary.plot(
                ax=ax,
                color="black",
                linewidth=0.8,
            )
        except Exception as exc:
            logging.warning("Could not draw AOI boundary on %s map: %s", column, exc)

    missing_text = _missing_summary(gdf, column)

    if title is None:
        title = column

    ax.set_title(
        f"{title}\n{missing_text}",
        fontsize=11,
    )
    ax.set_axis_off()

    # 4. Add explicit missing-data and legend-scaling explanation.
    # This is more reliable than ax.legend(), because GeoPandas already creates
    # a colorbar/legend for the mapped indicator.
    note_lines: list[str] = []
    if not missing.empty:
        missing_count = int(gdf[column].isna().sum())
        total_count = int(len(gdf))
        missing_share = missing_count / total_count if total_count > 0 else 0

        note_lines.append(
            f"No data / not calculated: {missing_count}/{total_count} "
            f"cells ({missing_share:.1%}); shown in grey with hatching."
        )

    if map_note:
        note_lines.append(map_note)

    if note_lines:
        fig.text(
            0.5,
            0.03,
            "\n".join(note_lines),
            ha="center",
            va="bottom",
            fontsize=7.5,
            bbox={
                "facecolor": "white",
                "edgecolor": "#999999",
                "boxstyle": "round,pad=0.3",
                "alpha": 0.9,
            },
        )

    fig.tight_layout(rect=(0, 0.1 if note_lines else 0.02, 1, 1))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logging.info("Saved map: %s", output_path)
    return output_path


def plot_gsi_sanity_map(
    indicator_grid,
    output_path: Path,
    aoi=None,
    dpi: int = 300,
    figsize: tuple[float, float] = (8, 8),
) -> Path | None:
    """
    Save a diagnostic map highlighting cells where raw-sum GSI exceeds 1.

    This map is not a primary result map. It is a data-quality / sanity-check
    map used to inspect theoretically suspicious GSI values.
    """
    output_path = _prepare_output_path(output_path)

    if indicator_grid.empty:
        logging.warning("Cannot create GSI sanity map: indicator grid is empty.")
        return None

    diagnostic_column = "gsi_raw_sum" if "gsi_raw_sum" in indicator_grid.columns else "gsi"
    if diagnostic_column not in indicator_grid.columns:
        logging.warning("Cannot create GSI sanity map: no GSI diagnostic column found.")
        return None

    gdf = indicator_grid.copy()
    gdf["gsi_diagnostic"] = pd.to_numeric(gdf[diagnostic_column], errors="coerce")

    suspicious = gdf[gdf["gsi_diagnostic"] > 1].copy()
    normal = gdf[(gdf["gsi_diagnostic"].notna()) & (gdf["gsi_diagnostic"] <= 1)].copy()
    missing = gdf[gdf["gsi_diagnostic"].isna()].copy()

    fig, ax = plt.subplots(figsize=figsize)

    if not normal.empty:
        normal.plot(
            ax=ax,
            color="#f7f7f7",
            edgecolor="#d9d9d9",
            linewidth=0.15,
        )

    if not missing.empty:
        missing.plot(
            ax=ax,
            color="#eeeeee",
            edgecolor="#d9d9d9",
            linewidth=0.15,
        )

    if not suspicious.empty:
        suspicious.plot(
            ax=ax,
            color="#d7191c",
            edgecolor="black",
            linewidth=0.5,
        )

    if aoi is not None and not aoi.empty:
        try:
            aoi.boundary.plot(
                ax=ax,
                color="black",
                linewidth=0.8,
            )
        except Exception as exc:
            logging.warning("Could not draw AOI boundary on GSI sanity map: %s", exc)

    ax.set_title(
        f"GSI sanity check: cells with raw-sum GSI > 1\n"
        f"suspicious cells: {len(suspicious)}/{len(gdf)}",
        fontsize=11,
    )
    ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logging.info("Saved GSI sanity map: %s", output_path)
    return output_path


def save_default_workflow_maps(
    indicator_grid,
    output_dir: Path,
    config: dict[str, Any] | None = None,
    aoi=None,
) -> list[Path]:
    """
    Save the default set of static maps for the workflow.

    The function skips maps if the required indicator column is missing or
    contains no valid values.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = {}

    visualization_config = config.get("visualization", {})
    dpi = int(visualization_config.get("figure_dpi", 300))
    figure_format = visualization_config.get("figure_format", "png")
    shared_legend_ranges = visualization_config.get("shared_legend_ranges", {})
    standalone_map_note = visualization_config.get(
        "standalone_map_note",
        "Legend range is scaled for this map; colours should not be compared "
        "directly with independently scaled maps.",
    )
    conditional_map_note = visualization_config.get(
        "conditional_map_note",
        "Blank or hatched cells indicate missing or insufficient input data, "
        "not zero or low indicator values. "
        "Legend range is scaled for this map; colours should not be compared "
        "directly with independently scaled maps.",
    )

    map_specs = [
        {
            "column": "gsi",
            "filename": f"gsi_map.{figure_format}",
            "title": "GSI / Building Coverage Ratio",
            "legend_label": "GSI",
            "map_note": standalone_map_note,
        },
        {
            "column": "far_fsi",
            "filename": f"far_fsi_map.{figure_format}",
            "title": "FAR/FSI",
            "legend_label": "FAR / FSI",
            "map_note": conditional_map_note,
        },
        {
            "column": "built_volume_density",
            "filename": f"built_volume_density_map.{figure_format}",
            "title": "Built Volume Density",
            "legend_label": "m³ / m²",
        },
        {
            "column": "avg_neighbor_distance_m",
            "filename": f"avg_neighbor_distance_m_map.{figure_format}",
            "title": "Average nearest-neighbour distance",
            "legend_label": "metres",
        },
        {
            "column": "height_valid_area_share",
            "filename": f"height_valid_area_share_map.{figure_format}",
            "title": "Height-valid footprint area share",
            "legend_label": "share",
        },
        {
            "column": "floor_data_valid_area_share",
            "filename": f"floor_data_valid_area_share_map.{figure_format}",
            "title": "Floor-data-valid footprint area share",
            "legend_label": "share",
        },
    ]

    for spec in map_specs:
        if spec["column"] == "built_volume_density":
            spec["legend_label"] = "m3 / m2"
            spec["map_note"] = conditional_map_note
        elif spec["column"] == "avg_neighbor_distance_m":
            spec["title"] = "Average neighbour distance"
            spec["map_note"] = conditional_map_note
        elif spec["column"] in {
            "height_valid_area_share",
            "floor_data_valid_area_share",
        }:
            spec["map_note"] = standalone_map_note

    saved_paths: list[Path] = []

    for spec in map_specs:
        column = spec["column"]

        if not _column_has_valid_values(indicator_grid, column):
            logging.warning(
                "Skipping map for %s: column missing or no valid values.",
                column,
            )
            continue

        path = plot_indicator_map(
            indicator_grid=indicator_grid,
            column=column,
            output_path=output_dir / spec["filename"],
            title=spec["title"],
            legend_label=spec["legend_label"],
            vmin=shared_legend_ranges.get(column, [None, None])[0],
            vmax=shared_legend_ranges.get(column, [None, None])[1],
            map_note=spec.get("map_note"),
            aoi=aoi,
            dpi=dpi,
        )

        if path is not None:
            saved_paths.append(path)

    if "gsi" in indicator_grid.columns:
        gsi_values = pd.to_numeric(indicator_grid["gsi"], errors="coerce")
        if int((gsi_values > 1).sum()) > 0:
            path = plot_gsi_sanity_map(
                indicator_grid=indicator_grid,
                output_path=output_dir / f"gsi_sanity_map.{figure_format}",
                aoi=aoi,
                dpi=dpi,
            )
            if path is not None:
                saved_paths.append(path)

    return saved_paths
