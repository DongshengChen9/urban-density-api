"""Shared presentation-only styles for workflow, web, and dashboard maps."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Any, Iterable

import numpy as np
import pandas as pd


CARTOGRAPHIC_STYLE_VERSION = "unified_cartography_v1"
MISSING_COLOR = "#d9d9d9"
ZERO_COLOR = "#fbfbf7"
VALID_PALETTE = ("#e4f2f1", "#b7ded9", "#7cc8c0", "#3e9e9d", "#116b73")


@dataclass(frozen=True)
class MapStyle:
    key: str
    column: str
    public_label: str
    unit: str
    fixed_breaks: tuple[float, ...] | None = None
    valid_colors: tuple[str, ...] = VALID_PALETTE
    missing_color: str = MISSING_COLOR
    zero_color: str = ZERO_COLOR
    missing_label: str = "Missing / insufficient input"
    zero_label: str = "Zero"
    contextual: bool = False


MAP_STYLES = {
    "gsi": MapStyle("gsi", "gsi", "GSI / Building Coverage Ratio", "ratio", (0.1, 0.25, 0.5, 0.75, 1.0)),
    "far": MapStyle("far", "far_fsi", "FAR/FSI", "ratio"),
    "built_volume_density": MapStyle("built_volume_density", "built_volume_density", "Built Volume Density", "m3/m2"),
    "neighbour_distance": MapStyle("neighbour_distance", "avg_neighbor_distance_m", "Average nearest-building distance", "m", contextual=True, missing_label="Missing / insufficient contextual input"),
    "street_profile_ratio": MapStyle("street_profile_ratio", "avg_street_profile_height_to_width_ratio_strict", "Street-profile height-to-width ratio", "ratio", contextual=True, missing_label="Missing / insufficient contextual input"),
}

_ALIASES = {
    "far_fsi": "far",
    "average_neighbor_distance": "neighbour_distance",
    "average_neighbour_distance": "neighbour_distance",
    "avg_neighbor_distance_m": "neighbour_distance",
    "avg_neighbour_distance_m": "neighbour_distance",
    "avg_street_profile_height_to_width_ratio_strict": "street_profile_ratio",
}
_COLUMN_TO_KEY = {style.column: key for key, style in MAP_STYLES.items()}


def style_for_key(key: str) -> MapStyle:
    canonical = _ALIASES.get(key, key)
    if canonical not in MAP_STYLES:
        raise KeyError(f"No map style is registered for {key!r}.")
    return MAP_STYLES[canonical]


def style_for_column(column: str) -> MapStyle | None:
    key = _COLUMN_TO_KEY.get(column) or _ALIASES.get(column)
    return style_for_key(key) if key else None


def resolve_breaks(
    style: MapStyle | str,
    values: Iterable[Any],
    explicit_breaks: Iterable[float] | None = None,
) -> tuple[float, ...]:
    style = style_for_key(style) if isinstance(style, str) else style
    if explicit_breaks is not None:
        values_to_use = explicit_breaks
    elif style.fixed_breaks is not None:
        values_to_use = style.fixed_breaks
    else:
        numeric = pd.to_numeric(pd.Series(list(values)), errors="coerce")
        positive = numeric[np.isfinite(numeric) & (numeric > 0)]
        if positive.empty:
            return tuple()
        values_to_use = np.quantile(positive, [0.2, 0.4, 0.6, 0.8, 1.0])
    breaks = tuple(sorted({float(value) for value in values_to_use if isfinite(float(value)) and float(value) > 0}))
    if not breaks:
        raise ValueError("Map class breaks must contain finite positive values.")
    return breaks


def resolved_style(style: MapStyle | str, values: Iterable[Any], explicit_breaks: Iterable[float] | None = None) -> MapStyle:
    source = style_for_key(style) if isinstance(style, str) else style
    breaks = resolve_breaks(source, values, explicit_breaks)
    return replace(source, fixed_breaks=breaks, valid_colors=source.valid_colors[: len(breaks)])


def classify_value(style: MapStyle | str, value: Any) -> str:
    source = style_for_key(style) if isinstance(style, str) else style
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric) or not isfinite(float(numeric)) or numeric < 0:
        return "missing"
    if numeric == 0:
        return "zero"
    for index, upper in enumerate(source.fixed_breaks or ()):
        if numeric <= upper:
            return f"valid_{index}"
    return f"valid_{len(source.valid_colors) - 1}"


def color_for_value(style: MapStyle | str, value: Any) -> str:
    source = style_for_key(style) if isinstance(style, str) else style
    category = classify_value(source, value)
    if category == "missing":
        return source.missing_color
    if category == "zero":
        return source.zero_color
    return source.valid_colors[int(category.removeprefix("valid_"))]


def format_break(value: float) -> str:
    return f"{value:.4g}"


def legend_entries(style: MapStyle | str) -> tuple[tuple[str, str, str], ...]:
    source = style_for_key(style) if isinstance(style, str) else style
    lower = 0.0
    intervals = []
    for index, upper in enumerate(source.fixed_breaks or ()):
        suffix = f" {source.unit}" if source.unit else ""
        intervals.append((f"valid_{index}", f"> {format_break(lower)} to {format_break(upper)}{suffix}", source.valid_colors[index]))
        lower = upper
    return (("missing", source.missing_label, source.missing_color), ("zero", source.zero_label, source.zero_color), *intervals)
