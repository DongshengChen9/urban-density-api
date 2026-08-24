"""Official Overture Buildings release discovery and resolution."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen


AWS_BUCKET_URL = "https://overturemaps-us-west-2.s3.us-west-2.amazonaws.com/"
BUILDINGS_PREFIX = "theme=buildings/type=building/"
_RELEASE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}\.\d+$")


class OvertureReleaseError(RuntimeError):
    category = "release_resolution_failed"


class OvertureReleaseDiscoveryError(OvertureReleaseError):
    category = "release_discovery_failed"


class OvertureReleaseUnavailableError(OvertureReleaseError):
    category = "release_unavailable"


@dataclass(frozen=True)
class ResolvedOvertureRelease:
    requested_release: str
    resolved_release: str
    mode: str
    provider: str
    discovery_url: str


def is_dated_release(value: object) -> bool:
    return bool(_RELEASE_PATTERN.fullmatch(str(value or "").strip()))


def _listing_url(prefix: str, delimiter: str | None = "/") -> str:
    parameters = {"list-type": "2", "prefix": prefix}
    if delimiter:
        parameters["delimiter"] = delimiter
    return f"{AWS_BUCKET_URL}?{urlencode(parameters)}"


def _fetch(url: str, timeout_seconds: float = 30.0) -> bytes:
    try:
        with urlopen(Request(url, headers={"User-Agent": "urban-density-workflow/0.2"}), timeout=timeout_seconds) as response:  # noqa: S310
            return response.read()
    except Exception as exc:  # pragma: no cover - network dependent
        raise OvertureReleaseDiscoveryError("Could not reach the official Overture AWS release listing.") from exc


def _root(payload: bytes) -> ET.Element:
    try:
        return ET.fromstring(payload)
    except ET.ParseError as exc:
        raise OvertureReleaseDiscoveryError("The official Overture AWS listing returned invalid XML.") from exc


def _sort_key(release: str) -> tuple[int, int, int, int]:
    date, revision = release.split(".", maxsplit=1)
    year, month, day = (int(part) for part in date.split("-"))
    return year, month, day, int(revision)


def list_official_aws_releases(fetcher: Callable[[str], bytes] | None = None) -> list[str]:
    root = _root((fetcher or _fetch)(_listing_url("release/")))
    releases = []
    for item in root.findall("{*}CommonPrefixes/{*}Prefix"):
        candidate = (item.text or "").strip().removeprefix("release/").rstrip("/")
        if is_dated_release(candidate):
            releases.append(candidate)
    if not releases:
        raise OvertureReleaseDiscoveryError("The official Overture AWS listing did not contain dated releases.")
    return sorted(set(releases), key=_sort_key)


def release_has_buildings(release: str, fetcher: Callable[[str], bytes] | None = None) -> bool:
    if not is_dated_release(release):
        return False
    root = _root((fetcher or _fetch)(_listing_url(f"release/{release}/{BUILDINGS_PREFIX}", delimiter=None)))
    try:
        key_count = int(root.findtext("{*}KeyCount", default="0"))
    except ValueError:
        key_count = 0
    return key_count > 0 or root.find("{*}Contents") is not None


def resolve_overture_release(requested_release: object, provider: str = "aws", fetcher: Callable[[str], bytes] | None = None) -> ResolvedOvertureRelease:
    requested = str(requested_release or "").strip()
    if provider != "aws":
        raise ValueError("Only provider='aws' is implemented for Overture Buildings.")
    if requested.lower() == "latest":
        raise ValueError("Use overture_release: auto instead of latest.")
    if requested.lower() == "auto":
        for candidate in reversed(list_official_aws_releases(fetcher)):
            if release_has_buildings(candidate, fetcher):
                return ResolvedOvertureRelease("auto", candidate, "auto_resolved", provider, _listing_url("release/"))
        raise OvertureReleaseUnavailableError("No currently listed Overture release exposed Buildings.")
    if not is_dated_release(requested):
        raise ValueError("overture_release must be auto or a dated identifier such as 2026-06-17.0.")
    if not release_has_buildings(requested, fetcher):
        raise OvertureReleaseUnavailableError(f"Pinned Overture release {requested} is not currently available.")
    return ResolvedOvertureRelease(requested, requested, "pinned", provider, _listing_url(f"release/{requested}/{BUILDINGS_PREFIX}", delimiter=None))
