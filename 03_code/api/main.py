from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator
from uuid import UUID

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.responses import HTMLResponse

from .jobs import JobManager, JobStore, WorkflowRunner
from .models import (
    AnalysisCreatedResponse,
    AnalysisRequest,
    AnalysisResultsResponse,
    AnalysisStatusResponse,
    ErrorResponse,
    HealthResponse,
)
from .services import (
    WorkflowService,
    grid_feature_collection,
    read_analysis_results,
    render_grid_map,
)

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _configured_max_workers() -> int:
    raw_value = os.getenv("UDW_MAX_WORKERS", "1")
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError("UDW_MAX_WORKERS must be an integer") from exc
    if value < 1:
        raise ValueError("UDW_MAX_WORKERS must be at least 1")
    return value


def _elapsed_seconds(metadata: dict[str, object]) -> float | None:
    started = metadata.get("started_at")
    if not isinstance(started, str):
        return None
    completed = metadata.get("completed_at")
    end = datetime.fromisoformat(completed) if isinstance(completed, str) else datetime.now().astimezone()
    return max(0.0, (end - datetime.fromisoformat(started)).total_seconds())


def create_app(
    *,
    output_root: Path | None = None,
    max_workers: int | None = None,
    workflow_runner: WorkflowRunner | None = None,
) -> FastAPI:
    resolved_output_root = output_root or Path(
        os.getenv("UDW_OUTPUT_DIR", str(PROJECT_ROOT / "04_outputs" / "api"))
    )
    service = WorkflowService()
    store = JobStore(resolved_output_root)
    manager = JobManager(
        store,
        workflow_runner or service.run_analysis,
        max_workers=max_workers if max_workers is not None else _configured_max_workers(),
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        yield
        manager.shutdown(wait=True)

    application = FastAPI(
        title="Urban Density Workflow API",
        version="1.0.0",
        description=(
            "Local research/demo REST wrapper around the existing Urban Density "
            "Workflow. Jobs use a small in-process executor and JSON metadata; "
            "this is not a horizontally scaled multi-user execution platform."
        ),
        lifespan=lifespan,
    )
    application.state.job_store = store
    application.state.job_manager = manager

    def require_job(analysis_id: UUID) -> tuple[str, dict[str, object]]:
        identifier = str(analysis_id)
        metadata = store.get(identifier)
        if metadata is None:
            raise HTTPException(status_code=404, detail="Analysis not found")
        return identifier, metadata

    def require_completed(analysis_id: UUID) -> tuple[str, dict[str, object]]:
        identifier, metadata = require_job(analysis_id)
        if metadata.get("status") != "completed":
            raise HTTPException(
                status_code=409,
                detail=f"Analysis results are unavailable while status is {metadata.get('status')}",
            )
        return identifier, metadata

    @application.get(
        "/health", response_model=HealthResponse, summary="Check service health"
    )
    def health() -> HealthResponse:
        """Return service health without contacting any geospatial data source."""
        return HealthResponse(status="ok", service="urban-density-api")

    @application.post(
        "/api/v1/analyses",
        response_model=AnalysisCreatedResponse,
        status_code=status.HTTP_202_ACCEPTED,
        summary="Queue a bounding-box analysis",
        responses={422: {"model": ErrorResponse, "description": "Invalid request"}},
    )
    def create_analysis(request: AnalysisRequest) -> AnalysisCreatedResponse:
        metadata = manager.submit(request)
        return AnalysisCreatedResponse(
            analysis_id=str(metadata["analysis_id"]), status="queued"
        )

    @application.get(
        "/api/v1/analyses/{analysis_id}/status",
        response_model=AnalysisStatusResponse,
        summary="Get analysis status",
        responses={404: {"model": ErrorResponse, "description": "Unknown analysis"}},
    )
    def analysis_status(analysis_id: UUID) -> AnalysisStatusResponse:
        identifier, metadata = require_job(analysis_id)
        return AnalysisStatusResponse(
            analysis_id=identifier,
            status=metadata["status"],
            elapsed_seconds=_elapsed_seconds(metadata),
            error=metadata.get("error"),
        )

    @application.get(
        "/api/v1/analyses/{analysis_id}/results",
        response_model=AnalysisResultsResponse,
        summary="Get completed workflow metadata",
        responses={
            404: {"model": ErrorResponse, "description": "Unknown analysis"},
            409: {"model": ErrorResponse, "description": "Analysis not completed"},
        },
    )
    def analysis_results(analysis_id: UUID) -> AnalysisResultsResponse:
        identifier, _ = require_completed(analysis_id)
        try:
            result = read_analysis_results(store.output_directory(identifier), identifier)
        except (OSError, ValueError) as exc:
            LOGGER.exception("Completed analysis has unreadable results: %s", identifier)
            raise HTTPException(status_code=500, detail="Completed result metadata is unavailable") from exc
        return AnalysisResultsResponse(**result)

    @application.get(
        "/api/v1/analyses/{analysis_id}/grid.geojson",
        summary="Get the completed indicator grid as GeoJSON",
        response_class=Response,
        responses={
            200: {"content": {"application/geo+json": {}}, "description": "GeoJSON FeatureCollection"},
            404: {"model": ErrorResponse, "description": "Unknown analysis"},
            409: {"model": ErrorResponse, "description": "Analysis not completed"},
        },
    )
    def analysis_grid(analysis_id: UUID) -> Response:
        identifier, _ = require_completed(analysis_id)
        try:
            feature_collection = grid_feature_collection(store.output_directory(identifier))
        except (OSError, ValueError) as exc:
            LOGGER.exception("Completed analysis has unreadable grid: %s", identifier)
            raise HTTPException(status_code=500, detail="Completed grid result is unavailable") from exc
        import json

        return Response(
            content=json.dumps(feature_collection, ensure_ascii=False),
            media_type="application/geo+json",
        )

    @application.get(
        "/api/v1/analyses/{analysis_id}/map",
        summary="View the completed indicator grid on an interactive map",
        response_class=HTMLResponse,
        responses={
            200: {"content": {"text/html": {}}, "description": "Standalone Folium/Leaflet map"},
            404: {"model": ErrorResponse, "description": "Unknown analysis"},
            409: {
                "model": ErrorResponse,
                "description": "Analysis is queued, running, or failed",
            },
        },
    )
    def analysis_map(analysis_id: UUID) -> HTMLResponse:
        identifier, _ = require_completed(analysis_id)
        try:
            html = render_grid_map(store.output_directory(identifier))
        except (OSError, ValueError) as exc:
            LOGGER.exception("Completed analysis has an unreadable map grid: %s", identifier)
            raise HTTPException(status_code=500, detail="Completed map result is unavailable") from exc
        return HTMLResponse(content=html)

    return application


app = create_app()
