from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


STATUS_OK = "OK"
STATUS_LIMITED = "LIMITED"
STATUS_WEAK = "WEAK"
STATUS_NOT_AVAILABLE = "NOT_AVAILABLE"
STATUS_DO_NOT_INTERPRET = "DO_NOT_INTERPRET"

OK_THRESHOLD = 0.8
LIMITED_THRESHOLD = 0.5
WEAK_THRESHOLD = 0.2


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric


def _as_int(value: Any) -> int | None:
    numeric = _as_float(value)
    if numeric is None:
        return None
    return int(numeric)


def _has_value(value: Any) -> bool:
    return _as_float(value) is not None


def _coverage_status(coverage: float | None) -> str:
    """
    Classify coverage using pragmatic workflow thresholds.

    These thresholds are reporting aids for this workflow, not universal
    scientific validity thresholds.
    """
    if coverage is None:
        return STATUS_LIMITED
    if coverage >= OK_THRESHOLD:
        return STATUS_OK
    if coverage >= LIMITED_THRESHOLD:
        return STATUS_LIMITED
    if coverage >= WEAK_THRESHOLD:
        return STATUS_WEAK
    return STATUS_DO_NOT_INTERPRET


def _blank_record(indicator: str) -> dict[str, Any]:
    return {
        "indicator": indicator,
        "calculated": False,
        "status": STATUS_NOT_AVAILABLE,
        "status_reason": "",
        "data_requirements": [],
        "main_quality_checks": [],
        "key_warnings": [],
        "coverage_explanation": None,
        "interpretation_advice": "",
        "recommended_use": "Do not use for interpretation in this run.",
        "do_not_interpret_reason": None,
    }


def _format_share(value: float | None) -> str:
    if value is None:
        return "not reported"
    return f"{value:.3f}"


def _format_percent(value: float | None) -> str:
    if value is None:
        return "not reported"
    return f"{value * 100:.1f}%"


def _append_low_coverage_warning(
    record: dict[str, Any],
    coverage_label: str,
    status: str,
) -> None:
    if status == STATUS_WEAK:
        record["key_warnings"].append(f"{coverage_label} coverage is low.")
    elif status == STATUS_DO_NOT_INTERPRET:
        record["key_warnings"].append(
            f"{coverage_label} coverage is below the interpretation threshold."
        )


STREET_PROFILE_COMPLETENESS_MESSAGE = (
    "Street-profile height-to-width ratio is available only where both building "
    "height and street-profile width are available. Missing areas indicate "
    "insufficient input data, not low values."
)


def _street_profile_limiting_factor(
    matched_to_street_share: float | None,
    valid_width_share: float | None,
    valid_height_share: float | None,
) -> str | None:
    if matched_to_street_share is not None and matched_to_street_share < OK_THRESHOLD:
        return "Main limitation: incomplete street/building matching."
    if valid_width_share is not None and valid_width_share < OK_THRESHOLD:
        return "Main limitation: incomplete street-profile width estimates."
    if (
        valid_height_share is not None
        and valid_height_share < OK_THRESHOLD
        and (matched_to_street_share is None or matched_to_street_share >= OK_THRESHOLD)
        and (valid_width_share is None or valid_width_share >= OK_THRESHOLD)
    ):
        return "Main limitation: incomplete building height data."
    return None


def _gsi_record(summary: dict[str, Any]) -> dict[str, Any]:
    record = _blank_record("GSI / Building Coverage Ratio")
    record.update(
        {
            "data_requirements": [
                "valid building footprints",
                "valid aggregation grid cells",
            ],
            "interpretation_advice": (
                "GSI shows the share of each grid cell covered by building "
                "footprints. Values should usually be read between 0 and 1."
            ),
            "main_quality_checks": [
                "valid grid-cell count",
                "cells with GSI greater than 1",
                "geometry and overlap diagnostics",
            ],
        }
    )

    n_grid_cells = _as_int(summary.get("n_grid_cells"))
    calculated = _has_value(summary.get("gsi_mean")) or _has_value(
        summary.get("gsi_median")
    )
    record["calculated"] = calculated

    if not calculated:
        record["status"] = STATUS_NOT_AVAILABLE
        record["status_reason"] = "GSI values were not found in the workflow summary."
        return record

    if n_grid_cells is not None and n_grid_cells <= 0:
        record["status"] = STATUS_DO_NOT_INTERPRET
        record["status_reason"] = "No valid grid cells were available."
        record["do_not_interpret_reason"] = "No valid grid cells were available."
        return record

    cells_over_1 = _as_int(
        summary.get("cells_with_gsi_over_1", summary.get("gsi_cells_over_1_count"))
    )
    if cells_over_1 is not None and cells_over_1 > 0:
        record["status"] = STATUS_LIMITED
        record["status_reason"] = (
            f"{cells_over_1} grid cells have raw-sum GSI greater than 1."
        )
        record["key_warnings"].append(
            "Some grid cells have raw-sum GSI greater than 1; check overlap or geometry diagnostics."
        )
        record["recommended_use"] = (
            "Use with caution and review the raw-sum GSI overlap diagnostics before interpretation."
        )
        return record

    record["status"] = STATUS_OK
    record["status_reason"] = "GSI was calculated and no raw-sum GSI overlap warning was reported."
    record["recommended_use"] = "Suitable as a primary physical density indicator."
    return record


def _far_record(summary: dict[str, Any]) -> dict[str, Any]:
    record = _blank_record("FAR/FSI")
    record.update(
        {
            "data_requirements": [
                "building footprints",
                "reliable floor counts or floor-area attributes",
            ],
            "main_quality_checks": [
                "floor-valid area share",
                "missing floor count share",
                "FAR/FSI value availability",
            ],
            "interpretation_advice": (
                "FAR/FSI relates floor area to land area. It should only be "
                "treated as a primary indicator when floor or floor-area data are reliable."
            ),
        }
    )

    calculated = _has_value(summary.get("far_fsi_mean")) or _has_value(
        summary.get("far_fsi_median")
    )
    floor_coverage = _as_float(summary.get("floor_valid_area_share"))
    if floor_coverage is None:
        missing_floor_share = _as_float(summary.get("missing_num_floors_share"))
        if missing_floor_share is not None:
            floor_coverage = max(0.0, min(1.0, 1.0 - missing_floor_share))

    record["calculated"] = calculated
    if not calculated:
        record["status"] = STATUS_NOT_AVAILABLE
        record["status_reason"] = (
            "FAR/FSI was not calculated or no valid FAR/FSI values were reported."
        )
        record["do_not_interpret_reason"] = "Reliable floor or floor-area data are unavailable."
        return record

    status = _coverage_status(floor_coverage)
    record["status"] = status
    if floor_coverage is None:
        record["status_reason"] = (
            "Floor coverage diagnostics are missing or incomplete, so FAR/FSI "
            "is conservatively marked as LIMITED."
        )
        record["key_warnings"].append(
            "Floor coverage diagnostics are missing or incomplete, so FAR/FSI is conservatively marked as LIMITED."
        )
    else:
        record["status_reason"] = (
            "Floor-data coverage used for interpretation: "
            f"{_format_share(floor_coverage)}."
        )

    if status == STATUS_OK:
        record["recommended_use"] = "Suitable as a primary density indicator."
    elif status == STATUS_LIMITED:
        if floor_coverage is None:
            record["recommended_use"] = (
                "Treat as provisional only until floor coverage diagnostics are available."
            )
        else:
            record["recommended_use"] = (
                "Use for broad comparison, but report floor-data limitations."
            )
            record["key_warnings"].append("Floor/floor-area coverage is incomplete.")
    elif status == STATUS_WEAK:
        record["recommended_use"] = (
            "Use only as a weak supporting indicator; avoid strong interpretation."
        )
        record["key_warnings"].append("Floor/floor-area coverage is low.")
    else:
        record["recommended_use"] = "Do not interpret FAR/FSI for this run."
        record["do_not_interpret_reason"] = "Floor/floor-area coverage is too low."
        record["key_warnings"].append("Floor/floor-area coverage is below the weak threshold.")

    return record


def _built_volume_record(summary: dict[str, Any]) -> dict[str, Any]:
    record = _blank_record("Built Volume Density")
    record.update(
        {
            "data_requirements": [
                "building footprints",
                "reliable building heights",
                "valid aggregation grid cells",
            ],
            "main_quality_checks": [
                "area-weighted height completeness",
                "height enrichment quality",
                "height source counts",
            ],
            "interpretation_advice": (
                "Built Volume Density relates estimated building volume to land "
                "area. Its interpretation depends directly on height completeness."
            ),
        }
    )

    calculated = _has_value(summary.get("built_volume_density_mean")) or _has_value(
        summary.get("built_volume_density_median")
    )
    height_coverage = _as_float(
        summary.get(
            "height_valid_area_share_after_enrichment",
            summary.get("height_valid_area_share"),
        )
    )
    if height_coverage is None:
        height_coverage = _as_float(summary.get("height_valid_share_after_enrichment"))
    if height_coverage is None:
        height_coverage = _as_float(summary.get("height_valid_area_share"))

    record["calculated"] = calculated
    if not calculated:
        record["status"] = STATUS_NOT_AVAILABLE
        record["status_reason"] = "Built Volume Density values were not reported."
        record["do_not_interpret_reason"] = "Height data are unavailable."
        return record

    status = _coverage_status(height_coverage)
    record["status"] = status
    record["status_reason"] = (
        "Height-data coverage used for interpretation: "
        f"{_format_share(height_coverage)}."
    )

    if status == STATUS_OK:
        record["recommended_use"] = "Suitable as a primary height-based density indicator."
    elif status == STATUS_LIMITED:
        record["recommended_use"] = (
            "Use with caution and cite height completeness or enrichment diagnostics."
        )
        record["key_warnings"].append("Height coverage is incomplete.")
    elif status == STATUS_WEAK:
        record["recommended_use"] = (
            "Use only as a weak supporting indicator; avoid fine-grained conclusions."
        )
        record["key_warnings"].append("Height coverage is low.")
    else:
        record["recommended_use"] = "Do not interpret Built Volume Density for this run."
        record["do_not_interpret_reason"] = "Height coverage is too low."
        record["key_warnings"].append("Height coverage is below the weak threshold.")

    return record


def _neighbor_distance_record(summary: dict[str, Any]) -> dict[str, Any]:
    record = _blank_record("Neighbour distance")
    record.update(
        {
            "data_requirements": [
                "valid building geometries",
                "stable building identifiers",
                "nearest-neighbour distance calculation",
            ],
            "main_quality_checks": [
                "valid building share",
                "grid-cell coverage",
                "zero-distance diagnostics",
            ],
            "interpretation_advice": (
                "Neighbour distance describes spacing between adjacent buildings. "
                "Very compact, attached, or overlapping fabric can produce zero distances."
            ),
        }
    )

    segmented_enabled = bool(summary.get("segmented_processing_enabled"))
    if segmented_enabled:
        calculated = bool(summary.get("segmented_neighbor_distance_enabled"))
        coverage = _as_float(
            summary.get(
                "segmented_neighbor_distance_grid_cell_coverage_share",
                summary.get("segmented_neighbor_distance_valid_grid_cell_share"),
            )
        )
        building_share = _as_float(
            summary.get("segmented_neighbor_distance_valid_building_share")
        )
    else:
        calculated = _has_value(summary.get("avg_neighbor_distance_mean_m")) or _has_value(
            summary.get("avg_neighbor_distance_median_m")
        )
        coverage = None
        building_share = None

    record["calculated"] = calculated
    if not calculated:
        record["status"] = STATUS_NOT_AVAILABLE
        record["status_reason"] = "Neighbour-distance values were not calculated."
        return record

    status = _coverage_status(coverage)
    record["status"] = status
    if coverage is None:
        record["status_reason"] = (
            "Neighbour distance was calculated, but grid-cell coverage was not reported."
        )
    else:
        record["status_reason"] = (
            "Neighbour-distance grid-cell coverage: "
            f"{_format_share(coverage)}; valid building share: {_format_share(building_share)}."
        )
        if building_share is not None and abs(coverage - building_share) >= 0.1:
            grid_status = _coverage_status(coverage).lower().replace("_", " ")
            if building_share >= OK_THRESHOLD:
                record["coverage_explanation"] = (
                    "Although {_building_share} of target buildings received a "
                    "neighbour-distance value, the indicator is present in only "
                    "{_grid_share} of grid cells. Grid-level interpretation is "
                    "therefore {grid_status}."
                ).format(
                    _building_share=_format_percent(building_share),
                    _grid_share=_format_percent(coverage),
                    grid_status=grid_status,
                )
            else:
                record["coverage_explanation"] = (
                    "Building-level coverage is {_building_share}, while "
                    "grid-cell coverage is {_grid_share}. Building-level coverage "
                    "counts target buildings with valid neighbour-distance values; "
                    "grid-cell coverage counts cells that contain at least one "
                    "valid aggregated value."
                ).format(
                    _building_share=_format_percent(building_share),
                    _grid_share=_format_percent(coverage),
                )

    zero_share = _as_float(summary.get("zero_neighbor_distance_share_building_level"))
    if zero_share is not None and zero_share > 0:
        record["key_warnings"].append(
            "Some building-level neighbour distances are zero; attached or overlapping fabric may affect interpretation."
        )

    if status == STATUS_OK:
        record["recommended_use"] = "Suitable as a contextual morphology indicator."
    elif status == STATUS_LIMITED:
        record["recommended_use"] = "Use as contextual evidence and report coverage."
    elif status == STATUS_WEAK:
        record["recommended_use"] = (
            "Use only as weak contextual evidence; coverage is low."
        )
        _append_low_coverage_warning(record, "Neighbour-distance grid-cell", status)
    else:
        record["recommended_use"] = "Do not interpret neighbour distance for this run."
        record["do_not_interpret_reason"] = "Neighbour-distance coverage is too low."
        _append_low_coverage_warning(record, "Neighbour-distance grid-cell", status)

    return record


def _street_profile_record(summary: dict[str, Any]) -> dict[str, Any]:
    record = _blank_record("Street-profile height-to-width ratio")
    record.update(
        {
            "data_requirements": [
                "valid building heights",
                "street-profile width estimates",
                "building-to-street-profile assignment",
            ],
            "main_quality_checks": [
                "reliable street-profile ratio coverage",
                "valid street-profile width share",
                "valid building height share",
                "matched building share",
            ],
            "interpretation_advice": (
                "This is the official contextual height-width indicator: "
                "building height divided by street-profile width. "
                f"{STREET_PROFILE_COMPLETENESS_MESSAGE}"
            ),
        }
    )

    if bool(summary.get("segmented_processing_enabled")):
        enabled = bool(summary.get("segmented_street_context_enabled"))
        coverage = _as_float(summary.get("segmented_street_profile_grid_cell_coverage_share"))
        building_share = _as_float(summary.get("segmented_street_profile_valid_building_share"))
    else:
        availability = summary.get("official_contextual_height_width_availability")
        enabled = bool(summary.get("street_context_enabled")) or availability == "available"
        coverage = _as_float(summary.get("street_profile_valid_grid_cell_share"))
        building_share = _as_float(summary.get("street_profile_valid_building_share"))

    calculated = enabled and coverage is not None
    record["calculated"] = calculated

    if not enabled:
        record["status"] = STATUS_NOT_AVAILABLE
        record["status_reason"] = "Street-context processing was disabled for this run."
        record["do_not_interpret_reason"] = "The official contextual height-width indicator was not calculated."
        return record

    if coverage is None:
        record["status"] = STATUS_NOT_AVAILABLE
        record["status_reason"] = "Street-profile ratio coverage was not reported."
        record["do_not_interpret_reason"] = "Street-profile ratio coverage is unavailable."
        return record

    matched_to_street_share = _as_float(
        summary.get("street_profile_matched_to_street_share")
    )
    valid_width_share = _as_float(summary.get("street_profile_valid_width_share"))
    valid_height_share = _as_float(summary.get("street_profile_valid_height_share"))
    limiting_factor = _street_profile_limiting_factor(
        matched_to_street_share=matched_to_street_share,
        valid_width_share=valid_width_share,
        valid_height_share=valid_height_share,
    )
    status = _coverage_status(coverage)
    record["status"] = status
    record["status_reason"] = (
        "Street-profile grid-cell coverage: "
        f"{_format_share(coverage)}; valid building share: {_format_share(building_share)}; "
        f"valid height share: {_format_share(valid_height_share)}; "
        f"valid width share: {_format_share(valid_width_share)}; "
        f"matched-to-street share: {_format_share(matched_to_street_share)}."
    )
    record["coverage_explanation"] = STREET_PROFILE_COMPLETENESS_MESSAGE
    record["key_warnings"].append(STREET_PROFILE_COMPLETENESS_MESSAGE)
    if limiting_factor:
        record["key_warnings"].append(limiting_factor)

    if status == STATUS_OK:
        record["recommended_use"] = "Suitable as the official contextual height-width indicator."
    elif status == STATUS_LIMITED:
        record["recommended_use"] = (
            "Use as the contextual height-width indicator, but report data completeness."
        )
    elif status == STATUS_WEAK:
        record["recommended_use"] = (
            "Use only as weak contextual evidence; street-profile coverage is low."
        )
        _append_low_coverage_warning(record, "Street-profile grid-cell", status)
    else:
        record["recommended_use"] = (
            "Do not interpret street-profile height-to-width ratio for this run."
        )
        record["do_not_interpret_reason"] = (
            "Street-profile ratio coverage is below the weak threshold."
        )
        if coverage == 0:
            record["key_warnings"].append(
                "Street-profile coverage is zero, so this indicator should not be interpreted."
            )
        else:
            record["key_warnings"].append(
                "Street-profile coverage is below the interpretation threshold."
            )

    return record


def build_indicator_readiness_records(
    workflow_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Build user-facing indicator readiness records from existing diagnostics.

    The function intentionally reads already-produced workflow summary fields.
    It does not recalculate indicators or alter workflow outputs.
    """
    summary = workflow_summary or {}
    return [
        _gsi_record(summary),
        _far_record(summary),
        _built_volume_record(summary),
        _neighbor_distance_record(summary),
        _street_profile_record(summary),
    ]


def _run_context_items(workflow_summary: dict[str, Any] | None) -> list[tuple[str, Any]]:
    if not workflow_summary:
        return []

    source_type = workflow_summary.get("data_source_type")
    source_release = workflow_summary.get("data_source_release")
    if source_type and source_release:
        source = f"{source_type} / {source_release}"
    else:
        source = source_type or source_release

    return [
        ("Run", workflow_summary.get("run_name")),
        ("AOI", workflow_summary.get("aoi_name")),
        (
            "Processing mode",
            workflow_summary.get("crs_resolved_processing_mode")
            or workflow_summary.get("crs_requested_processing_mode"),
        ),
        ("Grid size", workflow_summary.get("cell_size_m")),
        ("Building data source", source),
        ("Height enrichment enabled", workflow_summary.get("height_enrichment_enabled")),
        (
            "Report generated",
            workflow_summary.get("report_generated_at")
            or workflow_summary.get("generated_at"),
        ),
    ]


def _format_context_value(label: str, value: Any) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if label == "Grid size":
        return f"{value} m"
    return str(value)


def _render_run_context(workflow_summary: dict[str, Any] | None) -> list[str]:
    items = [(label, value) for label, value in _run_context_items(workflow_summary) if value is not None]
    if not items:
        return []

    lines = ["", "## Run context", ""]
    for label, value in items:
        lines.append(f"- **{label}:** {_format_context_value(label, value)}")
    return lines


def _group_records_for_summary(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    groups = {
        "Safe to use": [],
        "Use with limitations": [],
        "Weak evidence only": [],
        "Do not interpret": [],
    }
    for record in records:
        indicator = str(record.get("indicator"))
        status = record.get("status")
        if status == STATUS_OK:
            groups["Safe to use"].append(indicator)
        elif status == STATUS_LIMITED:
            groups["Use with limitations"].append(indicator)
        elif status == STATUS_WEAK:
            groups["Weak evidence only"].append(indicator)
        else:
            groups["Do not interpret"].append(indicator)
    return groups


def _render_quick_summary(records: list[dict[str, Any]]) -> list[str]:
    groups = _group_records_for_summary(records)
    lines = ["", "## Quick interpretation summary", ""]
    for label, indicators in groups.items():
        rendered = ", ".join(indicators) if indicators else "None"
        lines.append(f"- **{label}:** {rendered}")
    return lines


def _render_overall_interpretation(records: list[dict[str, Any]]) -> list[str]:
    status_counts = {
        status: sum(1 for record in records if record.get("status") == status)
        for status in [
            STATUS_OK,
            STATUS_LIMITED,
            STATUS_WEAK,
            STATUS_NOT_AVAILABLE,
            STATUS_DO_NOT_INTERPRET,
        ]
    }

    if status_counts[STATUS_DO_NOT_INTERPRET]:
        headline = (
            "This run is partly suitable for substantive interpretation, but "
            "some indicators should be treated mainly as technical diagnostics "
            "or not interpreted at all. Use the remaining indicators according "
            "to their individual status."
        )
    elif status_counts[STATUS_WEAK]:
        headline = (
            "This run can support cautious substantive interpretation, but at "
            "least one indicator is weak. Keep interpretation broad and report "
            "coverage limitations."
        )
    elif status_counts[STATUS_LIMITED]:
        headline = (
            "This run is suitable for substantive interpretation, with at least "
            "one indicator that should be reported with limitations."
        )
    elif status_counts[STATUS_OK]:
        headline = "The available indicators are broadly interpretable for this run."
    else:
        headline = "No indicators are currently ready for interpretation."

    return [
        "",
        "## Overall interpretation",
        "",
        headline,
        "",
        "- **OK:** {ok}".format(ok=status_counts[STATUS_OK]),
        "- **LIMITED:** {limited}".format(limited=status_counts[STATUS_LIMITED]),
        "- **WEAK:** {weak}".format(weak=status_counts[STATUS_WEAK]),
        "- **NOT_AVAILABLE:** {missing}".format(
            missing=status_counts[STATUS_NOT_AVAILABLE]
        ),
        "- **DO_NOT_INTERPRET:** {do_not}".format(
            do_not=status_counts[STATUS_DO_NOT_INTERPRET]
        ),
    ]


def _ordered_warnings(record: dict[str, Any]) -> list[str]:
    warnings = list(record.get("key_warnings") or [])
    if record.get("status") != STATUS_DO_NOT_INTERPRET:
        return warnings

    severity_order = ("zero", "below", "too low", "do not interpret")
    severe = []
    for term in severity_order:
        severe.extend(
            warning
            for warning in warnings
            if term in warning.lower()
        )

    ordered = []
    for warning in severe + warnings:
        if warning not in ordered:
            ordered.append(warning)
    if not ordered and record.get("do_not_interpret_reason"):
        ordered.append(str(record.get("do_not_interpret_reason")))
    return ordered


def render_indicator_readiness_markdown(
    records: list[dict[str, Any]],
    workflow_summary: dict[str, Any] | None = None,
) -> str:
    lines = [
        "# Indicator readiness and interpretation",
        "",
        "These statuses use pragmatic workflow thresholds for reporting: OK at "
        "0.8 or higher coverage, LIMITED at 0.5 or higher, WEAK at 0.2 or "
        "higher, and DO_NOT_INTERPRET below 0.2. They are not universal "
        "scientific thresholds.",
    ]

    lines.extend(_render_run_context(workflow_summary))
    lines.extend(_render_quick_summary(records))
    lines.extend(_render_overall_interpretation(records))
    lines.extend([
        "",
        "## Indicator status table",
        "",
        "| Indicator | Status | Recommended use |",
        "| --- | --- | --- |",
    ])

    for record in records:
        lines.append(
            "| {indicator} | {status} | {recommended_use} |".format(
                indicator=record.get("indicator"),
                status=record.get("status"),
                recommended_use=record.get("recommended_use"),
            )
        )

    for record in records:
        lines.extend(
            [
                "",
                f"## {record.get('indicator')}",
                "",
                f"**Status:** {record.get('status')}",
                "",
                f"**What it means:** {record.get('interpretation_advice')}",
                "",
                f"**Why:** {record.get('status_reason')}",
                "",
                f"**Use:** {record.get('recommended_use')}",
            ]
        )

        coverage_explanation = record.get("coverage_explanation")
        if coverage_explanation:
            lines.extend(["", f"**Coverage note:** {coverage_explanation}"])

        data_requirements = record.get("data_requirements") or []
        if data_requirements:
            lines.extend(["", "**Data requirements:**"])
            lines.extend(f"- {item}" for item in data_requirements)

        checks_to_review = record.get("main_quality_checks") or []
        if checks_to_review:
            lines.extend(["", "**Checks to review:**"])
            lines.extend(f"- {item}" for item in checks_to_review)

        warnings = _ordered_warnings(record)
        if warnings:
            lines.extend(["", "**Warnings:**"])
            lines.extend(f"- {warning}" for warning in warnings)
        do_not_interpret_reason = record.get("do_not_interpret_reason")
        if do_not_interpret_reason:
            lines.extend(["", f"**Do not interpret because:** {do_not_interpret_reason}"])

    return "\n".join(lines) + "\n"


def _records_for_csv(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    csv_records = []
    for record in records:
        csv_record = dict(record)
        for key in ["data_requirements", "main_quality_checks", "key_warnings"]:
            csv_record[key] = "; ".join(record.get(key) or [])
        csv_records.append(csv_record)
    return csv_records


def write_indicator_readiness_outputs(
    records: list[dict[str, Any]],
    reports_dir: Path,
    tables_dir: Path,
    save_reports: bool = True,
    save_tables: bool = True,
    workflow_summary: dict[str, Any] | None = None,
) -> None:
    report_context = dict(workflow_summary or {})
    if "report_generated_at" not in report_context:
        report_context["report_generated_at"] = datetime.now().isoformat(
            timespec="seconds"
        )

    if save_reports:
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "indicator_readiness.json").write_text(
            json.dumps(records, indent=2),
            encoding="utf-8",
        )
        (reports_dir / "indicator_readiness.md").write_text(
            render_indicator_readiness_markdown(
                records,
                workflow_summary=report_context,
            ),
            encoding="utf-8",
        )

    if save_tables:
        tables_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(_records_for_csv(records)).to_csv(
            tables_dir / "indicator_readiness.csv",
            index=False,
        )
