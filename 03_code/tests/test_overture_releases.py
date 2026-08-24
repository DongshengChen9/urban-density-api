"""Deterministic tests for Overture release resolution."""

from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from overture_releases import OvertureReleaseUnavailableError, resolve_overture_release  # noqa: E402


def _listing(url: str) -> bytes:
    if "theme%3Dbuildings" in url:
        return b"<ListBucketResult><KeyCount>1</KeyCount><Contents><Key>x</Key></Contents></ListBucketResult>"
    return b"<ListBucketResult><CommonPrefixes><Prefix>release/2026-01-01.0/</Prefix></CommonPrefixes><CommonPrefixes><Prefix>release/2026-02-01.0/</Prefix></CommonPrefixes></ListBucketResult>"


def test_auto_resolves_one_exact_dated_release():
    resolved = resolve_overture_release("auto", fetcher=_listing)
    assert resolved.mode == "auto_resolved"
    assert resolved.resolved_release == "2026-02-01.0"


def test_pinned_release_stays_exact_and_latest_is_rejected():
    assert resolve_overture_release("2026-01-01.0", fetcher=_listing).resolved_release == "2026-01-01.0"
    with pytest.raises(ValueError, match="auto"):
        resolve_overture_release("latest", fetcher=_listing)


def test_unavailable_pinned_release_fails_explicitly():
    with pytest.raises(OvertureReleaseUnavailableError):
        resolve_overture_release("2026-01-01.0", fetcher=lambda _url: b"<ListBucketResult><KeyCount>0</KeyCount></ListBucketResult>")
