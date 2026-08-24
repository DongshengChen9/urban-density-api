from pathlib import Path
import sys

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box


# Make src modules importable when tests are run from project root
SRC_DIR = Path(__file__).resolve().parents[1] / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import height_enrichment as he


def make_buildings(heights, geometries=None):
    """
    Create a small synthetic building GeoDataFrame in a projected metric CRS.
    """
    if geometries is None:
        geometries = [
            box(i * 20, 0, i * 20 + 10, 10)
            for i in range(len(heights))
        ]

    return gpd.GeoDataFrame(
        {
            "building_id": [f"b{i}" for i in range(len(heights))],
            "height_m": heights,
            "footprint_area_m2": [geom.area for geom in geometries],
        },
        geometry=geometries,
        crs="EPSG:32633",
    )


def make_gba(ids, heights, geometries):
    """
    Create a small synthetic GBA-like GeoDataFrame.
    """
    return gpd.GeoDataFrame(
        {
            "source": ["mock_gba"] * len(ids),
            "id": ids,
            "height": heights,
            "var": [0.0] * len(ids),
            "region": ["test"] * len(ids),
        },
        geometry=geometries,
        crs="EPSG:32633",
    )


def patch_gba_loader(monkeypatch, gba):
    """
    Replace real GBA loading/downloading with synthetic test data.

    This keeps the tests fast, deterministic, and independent of network access.
    """
    def fake_loader(*args, **kwargs):
        return gba, {
            "source": "mock_gba",
            "tile_names": ["mock_tile.parquet"],
            "failed_tiles": [],
            "gba_features_in_aoi_bbox": int(len(gba)),
        }

    monkeypatch.setattr(
        he,
        "load_gba_lod1_subset_for_aoi",
        fake_loader,
    )


def test_gba_tile_name_for_vienna_bbox():
    """
    Vienna should fall into the GBA 5-degree tile e015_n50_e020_n45.parquet.
    """
    tile_names = he.gba_lod1_tile_names_for_bbox(
        (16.36, 48.20, 16.39, 48.22)
    )

    assert tile_names == ["e015_n50_e020_n45.parquet"]


def test_fill_missing_only_preserves_valid_overture_height(
    monkeypatch,
    tmp_path,
):
    """
    Valid Overture heights must not be overwritten,
    even if GBA has a different value.

    Missing Overture heights may be filled from GBA
    if the match is valid.
    """
    buildings = make_buildings(
        heights=[10.0, None],
        geometries=[
            box(0, 0, 10, 10),
            box(20, 0, 30, 10),
        ],
    )

    gba = make_gba(
        ids=["gba_valid_overlap", "gba_missing_overlap"],
        heights=[99.0, 8.0],
        geometries=[
            box(0, 0, 10, 10),
            box(20, 0, 30, 10),
        ],
    )

    patch_gba_loader(monkeypatch, gba)

    enriched, summary, _, _ = he.enrich_missing_heights_with_gba_lod1(
        buildings=buildings,
        cache_dir=tmp_path,
        min_overlap_share=0.2,
        min_valid_height_m=2.0,
    )

    by_id = enriched.set_index("building_id")

    # Existing valid Overture height must be preserved.
    assert by_id.loc["b0", "height_m"] == 10.0
    assert by_id.loc["b0", "height_m_original"] == 10.0
    assert by_id.loc["b0", "height_gba_m"] != 99.0
    assert by_id.loc["b0", "height_source"] == "overture"
    assert bool(by_id.loc["b0", "height_was_enriched"]) is False

    # Missing Overture height should be filled from GBA.
    assert by_id.loc["b1", "height_m"] == 8.0
    assert pd.isna(by_id.loc["b1", "height_m_original"])
    assert by_id.loc["b1", "height_gba_m"] == 8.0
    assert by_id.loc["b1", "height_source"] == "gba_lod1"
    assert bool(by_id.loc["b1", "height_was_enriched"]) is True

    # Summary should correctly report before/after completeness.
    assert summary["valid_height_count_before"] == 1
    assert summary["missing_height_count_before"] == 1
    assert summary["height_enriched_count"] == 1
    assert summary["valid_height_count_after"] == 2
    assert summary["missing_height_count_after"] == 0
    assert summary["changed_existing_overture_height_count"] == 0


def test_weak_overlap_is_not_used_for_enrichment(monkeypatch, tmp_path):
    """
    A GBA building with too small spatial overlap must not be used for enrichment.
    However, the best weak match should still be stored as diagnostic information.
    """
    buildings = make_buildings(
        heights=[None],
        geometries=[box(0, 0, 10, 10)],
    )

    # Building area is 100 m².
    # Overlap area is 4 m², so overlap share is 0.04.
    # With min_overlap_share=0.2 this match must not be used for enrichment.
    gba = make_gba(
        ids=["gba_weak"],
        heights=[12.0],
        geometries=[box(0, 0, 2, 2)],
    )

    patch_gba_loader(monkeypatch, gba)

    enriched, summary, _, _ = he.enrich_missing_heights_with_gba_lod1(
        buildings=buildings,
        cache_dir=tmp_path,
        min_overlap_share=0.2,
        min_valid_height_m=2.0,
    )

    row = enriched.iloc[0]

    # Final height is still missing because the match is weak.
    assert pd.isna(row["height_m"])
    assert row["height_source"] == "missing"
    assert bool(row["height_was_enriched"]) is False

    # But the weak GBA match is stored as diagnostic information.
    assert row["height_gba_m"] == 12.0
    assert row["gba_match_quality"] == "weak_overlap"
    assert row["gba_match_overlap_share"] < 0.2

    assert summary["height_enriched_count"] == 0
    assert summary["missing_height_count_after"] == 1
    assert summary["gba_lod1_best_matches_any_overlap"] == 1
    assert summary["gba_lod1_strict_matches_used"] == 0


def test_low_gba_height_is_ignored(
    monkeypatch,
    tmp_path,
):
    """
    GBA heights below min_valid_height_m must not be used for enrichment.
    """
    buildings = make_buildings(
        heights=[None],
        geometries=[
            box(0, 0, 10, 10),
        ],
    )

    # Exact spatial overlap, but height is below min_valid_height_m=2.0.
    gba = make_gba(
        ids=["gba_low"],
        heights=[1.5],
        geometries=[
            box(0, 0, 10, 10),
        ],
    )

    patch_gba_loader(monkeypatch, gba)

    enriched, summary, _, _ = he.enrich_missing_heights_with_gba_lod1(
        buildings=buildings,
        cache_dir=tmp_path,
        min_overlap_share=0.2,
        min_valid_height_m=2.0,
    )

    row = enriched.iloc[0]

    assert pd.isna(row["height_m"])
    assert row["height_source"] == "missing"
    assert bool(row["height_was_enriched"]) is False

    assert summary["height_enriched_count"] == 0
    assert summary["missing_height_count_after"] == 1
    assert summary["gba_lod1_candidates_after_min_height_filter"] == 0


def test_geographic_crs_is_rejected(tmp_path):
    """
    GBA matching must be done in a projected metric CRS,
    not in geographic degrees.
    """
    buildings = make_buildings(
        heights=[None],
        geometries=[
            box(0, 0, 10, 10),
        ],
    ).to_crs("EPSG:4326")

    with pytest.raises(ValueError, match="projected metric CRS"):
        he.enrich_missing_heights_with_gba_lod1(
            buildings=buildings,
            cache_dir=tmp_path,
        )


def test_replace_existing_height_is_rejected(tmp_path):
    """
    The thesis workflow must not allow replacement of valid Overture heights.
    """
    buildings = make_buildings(
        heights=[10.0],
        geometries=[
            box(0, 0, 10, 10),
        ],
    )

    with pytest.raises(ValueError, match="replace_existing_height=True"):
        he.enrich_missing_heights_with_gba_lod1(
            buildings=buildings,
            cache_dir=tmp_path,
            replace_existing_height=True,
        )