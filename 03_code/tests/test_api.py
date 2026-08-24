from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from uuid import UUID

import geopandas as gpd
import pytest
from fastapi.testclient import TestClient
from shapely.geometry import box


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from api.main import create_app
from api.models import AnalysisRequest
from api.services import WorkflowService


VALID_REQUEST = {
    "bbox": {
        "min_lon": 4.477,
        "min_lat": 51.918,
        "max_lon": 4.483,
        "max_lat": 51.922,
    },
    "grid_size": 100,
    "mode": "quick_2d",
}


def wait_for_status(client: TestClient, analysis_id: str, expected: str) -> dict:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/analyses/{analysis_id}/status")
        assert response.status_code == 200
        if response.json()["status"] == expected:
            return response.json()
        time.sleep(0.01)
    pytest.fail(f"Analysis did not reach {expected!r}")


def write_synthetic_results(output_directory: Path) -> None:
    reports = output_directory / "reports"
    indicators = output_directory / "indicators"
    reports.mkdir(parents=True, exist_ok=True)
    indicators.mkdir(parents=True, exist_ok=True)
    (reports / "workflow_summary.json").write_text(
        json.dumps({"n_grid_cells": 1, "gsi_mean": 0.25}), encoding="utf-8"
    )
    (reports / "stage_timings.json").write_text(
        json.dumps({"total_runtime_seconds": 0.01}), encoding="utf-8"
    )
    (reports / "indicator_readiness.json").write_text(
        json.dumps([{"indicator": "gsi", "status": "OK"}]), encoding="utf-8"
    )
    grid = gpd.GeoDataFrame(
        {
            "unit_id": ["r0_c0"],
            "gsi": [0.25],
            "far_fsi": [1.2],
            "built_volume_density": [3.4],
            "avg_neighbor_distance_m": [8.5],
            "avg_street_profile_height_to_width_ratio_strict": [0.6],
        },
        geometry=[box(4.477, 51.918, 4.478, 51.919)],
        crs="EPSG:4326",
    )
    grid.to_file(indicators / "grid_indicators.gpkg", layer="grid_indicators")


def test_workflow_service_reuses_bbox_config_generator(tmp_path, monkeypatch):
    import run_workflow as workflow_module

    received_config_path = None

    def fake_run_workflow(config_path):
        nonlocal received_config_path
        received_config_path = config_path

    monkeypatch.setattr(workflow_module, "run_workflow", fake_run_workflow)
    request = AnalysisRequest.model_validate(VALID_REQUEST)
    WorkflowService().run_analysis("00000000-0000-0000-0000-000000000001", request, tmp_path)

    assert received_config_path == tmp_path / "workflow_config.yaml"
    config_text = received_config_path.read_text(encoding="utf-8")
    assert "mode: bbox" in config_text
    assert "cell_size_m: 100" in config_text
    assert "gsi: true" in config_text
    assert "height_enrichment:\n  enabled: false" in config_text
    assert "street_context:\n  enabled: false" in config_text


def test_health_does_not_run_workflow(tmp_path):
    called = False

    def runner(*_args):
        nonlocal called
        called = True

    with TestClient(create_app(output_root=tmp_path, workflow_runner=runner)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "urban-density-api"}
    assert called is False


def test_valid_analysis_is_accepted_with_uuid(tmp_path):
    release = threading.Event()

    def runner(_analysis_id, _request, _output_directory):
        release.wait(timeout=5)

    with TestClient(create_app(output_root=tmp_path, workflow_runner=runner)) as client:
        response = client.post("/api/v1/analyses", json=VALID_REQUEST)
        release.set()

    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    UUID(response.json()["analysis_id"])


@pytest.mark.parametrize(
    "request_update",
    [
        {"bbox": {**VALID_REQUEST["bbox"], "min_lon": 4.5}},
        {"bbox": {**VALID_REQUEST["bbox"], "min_lat": 52.0}},
    ],
)
def test_invalid_bbox_is_rejected(tmp_path, request_update):
    request = {**VALID_REQUEST, **request_update}
    with TestClient(create_app(output_root=tmp_path)) as client:
        response = client.post("/api/v1/analyses", json=request)
    assert response.status_code == 422


def test_invalid_mode_is_rejected(tmp_path):
    with TestClient(create_app(output_root=tmp_path)) as client:
        response = client.post(
            "/api/v1/analyses", json={**VALID_REQUEST, "mode": "invented"}
        )
    assert response.status_code == 422


def test_non_positive_grid_size_is_rejected(tmp_path):
    with TestClient(create_app(output_root=tmp_path)) as client:
        response = client.post(
            "/api/v1/analyses", json={**VALID_REQUEST, "grid_size": 0}
        )
    assert response.status_code == 422


def test_unknown_analysis_id_returns_404(tmp_path):
    with TestClient(create_app(output_root=tmp_path)) as client:
        response = client.get(
            "/api/v1/analyses/00000000-0000-0000-0000-000000000000/status"
        )
    assert response.status_code == 404


def test_map_unknown_analysis_returns_404(tmp_path):
    with TestClient(create_app(output_root=tmp_path)) as client:
        response = client.get(
            "/api/v1/analyses/00000000-0000-0000-0000-000000000000/map"
        )
    assert response.status_code == 404


def test_status_transitions_and_running_results_conflict(tmp_path):
    started = threading.Event()
    release = threading.Event()

    def runner(_analysis_id, _request, output_directory):
        started.set()
        release.wait(timeout=5)
        write_synthetic_results(output_directory)

    with TestClient(create_app(output_root=tmp_path, workflow_runner=runner)) as client:
        created = client.post("/api/v1/analyses", json=VALID_REQUEST).json()
        assert started.wait(timeout=2)
        running = wait_for_status(client, created["analysis_id"], "running")
        conflict = client.get(f"/api/v1/analyses/{created['analysis_id']}/results")
        release.set()
        completed = wait_for_status(client, created["analysis_id"], "completed")

    assert running["elapsed_seconds"] is not None
    assert conflict.status_code == 409
    assert completed["error"] is None


def test_map_incomplete_analysis_returns_409(tmp_path):
    started = threading.Event()
    release = threading.Event()

    def runner(_analysis_id, _request, _output_directory):
        started.set()
        release.wait(timeout=5)

    with TestClient(create_app(output_root=tmp_path, workflow_runner=runner)) as client:
        created = client.post("/api/v1/analyses", json=VALID_REQUEST).json()
        assert started.wait(timeout=2)
        wait_for_status(client, created["analysis_id"], "running")
        response = client.get(f"/api/v1/analyses/{created['analysis_id']}/map")
        release.set()

    assert response.status_code == 409


def test_failed_workflow_has_safe_public_error(tmp_path):
    def runner(*_args):
        raise RuntimeError("secret internal path C:/sensitive")

    with TestClient(create_app(output_root=tmp_path, workflow_runner=runner)) as client:
        created = client.post("/api/v1/analyses", json=VALID_REQUEST).json()
        failed = wait_for_status(client, created["analysis_id"], "failed")
        map_response = client.get(f"/api/v1/analyses/{created['analysis_id']}/map")

    assert "secret" not in failed["error"]
    assert "service logs" in failed["error"]
    assert map_response.status_code == 409


def test_completed_results_reuse_workflow_metadata(tmp_path):
    def runner(_analysis_id, _request, output_directory):
        write_synthetic_results(output_directory)

    with TestClient(create_app(output_root=tmp_path, workflow_runner=runner)) as client:
        created = client.post("/api/v1/analyses", json=VALID_REQUEST).json()
        wait_for_status(client, created["analysis_id"], "completed")
        response = client.get(f"/api/v1/analyses/{created['analysis_id']}/results")

    assert response.status_code == 200
    assert response.json()["workflow_summary"]["gsi_mean"] == 0.25
    assert response.json()["stage_timings"]["total_runtime_seconds"] == 0.01


def test_geojson_endpoint_returns_feature_collection(tmp_path):
    def runner(_analysis_id, _request, output_directory):
        write_synthetic_results(output_directory)

    with TestClient(create_app(output_root=tmp_path, workflow_runner=runner)) as client:
        created = client.post("/api/v1/analyses", json=VALID_REQUEST).json()
        wait_for_status(client, created["analysis_id"], "completed")
        response = client.get(
            f"/api/v1/analyses/{created['analysis_id']}/grid.geojson"
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/geo+json")
    assert response.json()["type"] == "FeatureCollection"
    assert len(response.json()["features"]) == 1


def test_completed_map_returns_folium_html(tmp_path):
    def runner(_analysis_id, _request, output_directory):
        write_synthetic_results(output_directory)

    with TestClient(create_app(output_root=tmp_path, workflow_runner=runner)) as client:
        created = client.post("/api/v1/analyses", json=VALID_REQUEST).json()
        wait_for_status(client, created["analysis_id"], "completed")
        response = client.get(f"/api/v1/analyses/{created['analysis_id']}/map")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "folium-map" in response.text
    assert "leaflet" in response.text.lower()
    assert "GSI / Building Coverage Ratio" in response.text
    assert "FAR / FSI" in response.text
    assert "Built Volume Density" in response.text
    assert "Average nearest-building distance" in response.text
    assert "Street-profile height-to-width ratio" in response.text
    assert "exclusiveGroups" in response.text
