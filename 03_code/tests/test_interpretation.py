from __future__ import annotations

import json
from pathlib import Path
import sys


TEST_DIR = Path(__file__).resolve().parent
CODE_DIR = TEST_DIR.parent
SRC_DIR = CODE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from interpretation import (  # noqa: E402
    STATUS_DO_NOT_INTERPRET,
    STATUS_LIMITED,
    STATUS_NOT_AVAILABLE,
    STATUS_OK,
    STATUS_WEAK,
    build_indicator_readiness_records,
    render_indicator_readiness_markdown,
    write_indicator_readiness_outputs,
)


def record_by_indicator(records, indicator):
    return next(record for record in records if record["indicator"] == indicator)


def base_summary():
    return {
        "run_name": "test_run",
        "aoi_name": "test_aoi",
        "crs_resolved_processing_mode": "single_crs",
        "cell_size_m": 100,
        "data_source_type": "fixture",
        "data_source_release": "test_release",
        "height_enrichment_enabled": True,
        "report_generated_at": "2026-07-08T12:00:00",
        "n_grid_cells": 10,
        "gsi_mean": 0.35,
        "gsi_median": 0.3,
        "cells_with_gsi_over_1": 0,
        "far_fsi_mean": 1.4,
        "far_fsi_median": 1.2,
        "floor_valid_area_share": 0.9,
        "built_volume_density_mean": 4.2,
        "built_volume_density_median": 3.9,
        "height_valid_area_share_after_enrichment": 0.9,
        "avg_neighbor_distance_mean_m": 7.0,
        "street_context_enabled": True,
        "official_contextual_height_width_availability": "available",
        "street_profile_valid_grid_cell_share": 0.9,
        "street_profile_valid_building_share": 0.85,
        "street_profile_valid_width_share": 1.0,
        "street_profile_matched_to_street_share": 1.0,
        "street_profile_valid_height_share": 0.9,
        "street_profile_valid_ratio_strict_share": 0.85,
        "street_profile_opposite_profile_evidence_share": 0.93,
    }


def test_high_quality_gsi_is_ok():
    records = build_indicator_readiness_records(base_summary())

    gsi = record_by_indicator(records, "GSI / Building Coverage Ratio")

    assert gsi["calculated"] is True
    assert gsi["status"] == STATUS_OK


def test_gsi_over_one_warning_makes_gsi_limited():
    summary = base_summary()
    summary["cells_with_gsi_over_1"] = 2

    records = build_indicator_readiness_records(summary)
    gsi = record_by_indicator(records, "GSI / Building Coverage Ratio")

    assert gsi["status"] == STATUS_LIMITED
    assert "GSI greater than 1" in gsi["status_reason"]


def test_low_height_completeness_makes_built_volume_density_weak():
    summary = base_summary()
    summary["height_valid_area_share_after_enrichment"] = 0.3

    records = build_indicator_readiness_records(summary)
    bvd = record_by_indicator(records, "Built Volume Density")

    assert bvd["status"] == STATUS_WEAK
    assert "Height-data coverage" in bvd["status_reason"]


def test_very_low_height_completeness_makes_built_volume_density_do_not_interpret():
    summary = base_summary()
    summary["height_valid_area_share_after_enrichment"] = 0.1

    records = build_indicator_readiness_records(summary)
    bvd = record_by_indicator(records, "Built Volume Density")

    assert bvd["status"] == STATUS_DO_NOT_INTERPRET
    assert bvd["do_not_interpret_reason"]


def test_missing_floor_data_makes_far_not_available():
    summary = base_summary()
    summary["far_fsi_mean"] = None
    summary["far_fsi_median"] = None
    summary["floor_valid_area_share"] = 0.0
    summary["missing_num_floors_share"] = 1.0

    records = build_indicator_readiness_records(summary)
    far = record_by_indicator(records, "FAR/FSI")

    assert far["status"] == STATUS_NOT_AVAILABLE
    assert far["do_not_interpret_reason"]


def test_zero_strict_street_profile_coverage_is_do_not_interpret():
    summary = base_summary()
    summary["street_profile_valid_grid_cell_share"] = 0.0
    summary["street_profile_valid_building_share"] = 0.0

    records = build_indicator_readiness_records(summary)
    street = record_by_indicator(records, "Street-profile height-to-width ratio")

    assert street["status"] == STATUS_DO_NOT_INTERPRET
    assert "Street-profile grid-cell coverage" in street["status_reason"]
    assert any("coverage is zero" in warning for warning in street["key_warnings"])
    assert any("insufficient input data, not low values" in warning for warning in street["key_warnings"])


def test_street_profile_low_height_coverage_identifies_height_limitation():
    summary = base_summary()
    summary.update(
        {
            "street_profile_valid_width_share": 1.0,
            "street_profile_matched_to_street_share": 1.0,
            "street_profile_opposite_profile_evidence_share": 0.93,
            "street_profile_valid_height_share": 0.525,
            "street_profile_valid_ratio_strict_share": 0.525,
            "street_profile_valid_building_share": 0.525,
            "street_profile_valid_grid_cell_share": 0.399,
        }
    )

    records = build_indicator_readiness_records(summary)
    street = record_by_indicator(records, "Street-profile height-to-width ratio")

    assert street["status"] == STATUS_WEAK
    assert any(
        warning == "Main limitation: incomplete building height data."
        for warning in street["key_warnings"]
    )
    assert street["coverage_explanation"] == (
        "Street-profile height-to-width ratio is available only where both "
        "building height and street-profile width are available. Missing areas "
        "indicate insufficient input data, not low values."
    )


def test_segmented_neighbor_distance_coverage_is_used():
    summary = base_summary()
    summary.update(
        {
            "segmented_processing_enabled": True,
            "segmented_neighbor_distance_enabled": True,
            "segmented_neighbor_distance_grid_cell_coverage_share": 0.6,
            "segmented_neighbor_distance_valid_building_share": 0.75,
            "segmented_street_context_enabled": False,
            "street_context_enabled": False,
        }
    )

    records = build_indicator_readiness_records(summary)
    neighbor = record_by_indicator(records, "Neighbour distance")

    assert neighbor["calculated"] is True
    assert neighbor["status"] == STATUS_LIMITED
    assert "0.600" in neighbor["status_reason"]


def test_low_neighbor_distance_coverage_warning_is_explicit():
    summary = base_summary()
    summary.update(
        {
            "segmented_processing_enabled": True,
            "segmented_neighbor_distance_enabled": True,
            "segmented_neighbor_distance_grid_cell_coverage_share": 0.3,
            "segmented_neighbor_distance_valid_building_share": 0.4,
            "segmented_street_context_enabled": False,
            "street_context_enabled": False,
        }
    )

    records = build_indicator_readiness_records(summary)
    neighbor = record_by_indicator(records, "Neighbour distance")

    assert neighbor["status"] == STATUS_WEAK
    assert any("coverage is low" in warning for warning in neighbor["key_warnings"])


def test_missing_far_coverage_explanation_is_conservative():
    summary = base_summary()
    summary["floor_valid_area_share"] = None
    summary["missing_num_floors_share"] = None

    records = build_indicator_readiness_records(summary)
    far = record_by_indicator(records, "FAR/FSI")

    assert far["status"] == STATUS_LIMITED
    assert far["status_reason"] == (
        "Floor coverage diagnostics are missing or incomplete, so FAR/FSI "
        "is conservatively marked as LIMITED."
    )
    assert far["key_warnings"] == [
        "Floor coverage diagnostics are missing or incomplete, so FAR/FSI is conservatively marked as LIMITED."
    ]


def test_quick_summary_groups_indicators_by_status():
    summary = base_summary()
    summary["cells_with_gsi_over_1"] = 1
    summary["height_valid_area_share_after_enrichment"] = 0.3
    summary["street_profile_valid_grid_cell_share"] = 0.0
    records = build_indicator_readiness_records(summary)

    markdown = render_indicator_readiness_markdown(
        records,
        workflow_summary=summary,
    )

    assert "## Quick interpretation summary" in markdown
    assert "**Use with limitations:** GSI / Building Coverage Ratio" in markdown
    assert "**Weak evidence only:** Built Volume Density" in markdown
    assert "**Do not interpret:** Street-profile height-to-width ratio" in markdown


def test_overall_interpretation_section_is_rendered():
    summary = base_summary()
    summary["street_profile_valid_grid_cell_share"] = 0.0
    records = build_indicator_readiness_records(summary)

    markdown = render_indicator_readiness_markdown(
        records,
        workflow_summary=summary,
    )

    assert "## Overall interpretation" in markdown
    assert "partly suitable for substantive interpretation" in markdown
    assert "technical diagnostics" in markdown
    assert "- **DO_NOT_INTERPRET:** 1" in markdown


def test_run_context_renders_when_metadata_are_available():
    summary = base_summary()
    records = build_indicator_readiness_records(summary)

    markdown = render_indicator_readiness_markdown(
        records,
        workflow_summary=summary,
    )

    assert "## Run context" in markdown
    assert "**Run:** test_run" in markdown
    assert "**AOI:** test_aoi" in markdown
    assert "**Processing mode:** single_crs" in markdown
    assert "**Grid size:** 100 m" in markdown
    assert "**Building data source:** fixture / test_release" in markdown
    assert "**Height enrichment enabled:** Yes" in markdown
    assert "**Report generated:** 2026-07-08T12:00:00" in markdown


def test_run_context_renders_false_boolean_as_no():
    summary = base_summary()
    summary["height_enrichment_enabled"] = False
    records = build_indicator_readiness_records(summary)

    markdown = render_indicator_readiness_markdown(
        records,
        workflow_summary=summary,
    )

    assert "**Height enrichment enabled:** No" in markdown


def test_warnings_render_as_bullet_list():
    summary = base_summary()
    summary["street_profile_valid_grid_cell_share"] = 0.0
    records = build_indicator_readiness_records(summary)

    markdown = render_indicator_readiness_markdown(
        records,
        workflow_summary=summary,
    )

    assert (
        "- Street-profile coverage is zero, so this indicator should not be interpreted."
        in markdown
    )
    assert "insufficient input data, not low values" in markdown
    assert "Use strict street-profile ratios" not in markdown
    assert "preliminary ratios" not in markdown
    assert "| Indicator | Status | Recommended use |" in markdown
    assert "Main warnings" not in markdown


def test_building_and_grid_coverage_difference_is_explained_for_neighbor_distance():
    summary = base_summary()
    summary.update(
        {
            "segmented_processing_enabled": True,
            "segmented_neighbor_distance_enabled": True,
            "segmented_neighbor_distance_grid_cell_coverage_share": 0.385,
            "segmented_neighbor_distance_valid_building_share": 1.0,
            "segmented_street_context_enabled": False,
            "street_context_enabled": False,
        }
    )

    records = build_indicator_readiness_records(summary)
    neighbor = record_by_indicator(records, "Neighbour distance")
    markdown = render_indicator_readiness_markdown(
        records,
        workflow_summary=summary,
    )

    assert neighbor["coverage_explanation"]
    assert (
        "Although 100.0% of target buildings received a neighbour-distance value, "
        "the indicator is present in only 38.5% of grid cells. Grid-level "
        "interpretation is therefore weak."
    ) in markdown


def test_do_not_interpret_warning_lists_most_severe_warning_first():
    summary = base_summary()
    summary["street_profile_valid_grid_cell_share"] = 0.0
    records = build_indicator_readiness_records(summary)

    markdown = render_indicator_readiness_markdown(
        records,
        workflow_summary=summary,
    )
    street_section = markdown.split("## Street-profile height-to-width ratio", 1)[1]
    warnings_block = street_section.split("**Warnings:**", 1)[1]
    first_warning = next(
        line for line in warnings_block.splitlines() if line.startswith("- ")
    )

    assert first_warning == (
        "- Street-profile coverage is zero, so this indicator should not be interpreted."
    )


def test_writes_indicator_readiness_outputs(tmp_path):
    records = build_indicator_readiness_records(base_summary())
    reports_dir = tmp_path / "reports"
    tables_dir = tmp_path / "tables"

    write_indicator_readiness_outputs(
        records,
        reports_dir,
        tables_dir,
        workflow_summary=base_summary(),
    )

    json_records = json.loads(
        (reports_dir / "indicator_readiness.json").read_text(encoding="utf-8")
    )
    markdown = (reports_dir / "indicator_readiness.md").read_text(encoding="utf-8")
    csv_text = (tables_dir / "indicator_readiness.csv").read_text(encoding="utf-8")

    assert json_records[0]["indicator"] == "GSI / Building Coverage Ratio"
    assert "Indicator readiness and interpretation" in markdown
    assert "## Run context" in markdown
    assert "Street-profile height-to-width ratio" in csv_text


def test_markdown_mentions_thresholds():
    records = build_indicator_readiness_records(base_summary())

    markdown = render_indicator_readiness_markdown(records)

    assert "pragmatic workflow thresholds" in markdown
    assert "DO_NOT_INTERPRET below 0.2" in markdown
