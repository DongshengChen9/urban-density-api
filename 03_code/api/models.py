from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AnalysisMode(str, Enum):
    """Existing workflow modes exposed by the service."""

    QUICK_2D = "quick_2d"
    STANDARD = "standard"
    FULL_CONTEXT = "full_context"


class BoundingBox(BaseModel):
    """A WGS84 longitude/latitude bounding box."""

    min_lon: float = Field(..., ge=-180, le=180, description="Western longitude in WGS84 degrees.")
    min_lat: float = Field(..., ge=-90, le=90, description="Southern latitude in WGS84 degrees.")
    max_lon: float = Field(..., ge=-180, le=180, description="Eastern longitude in WGS84 degrees.")
    max_lat: float = Field(..., ge=-90, le=90, description="Northern latitude in WGS84 degrees.")

    @model_validator(mode="after")
    def validate_order(self) -> "BoundingBox":
        if self.min_lon >= self.max_lon:
            raise ValueError("min_lon must be smaller than max_lon")
        if self.min_lat >= self.max_lat:
            raise ValueError("min_lat must be smaller than max_lat")
        return self


class AnalysisRequest(BaseModel):
    """Parameters for one bounding-box workflow run."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "bbox": {
                    "min_lon": 4.477,
                    "min_lat": 51.918,
                    "max_lon": 4.483,
                    "max_lat": 51.922,
                },
                "grid_size": 100,
                "mode": "quick_2d",
            }
        }
    )

    bbox: BoundingBox = Field(description="Analysis bounds in WGS84 coordinates.")
    grid_size: float = Field(..., gt=0, description="Regular grid cell size in metres.")
    mode: AnalysisMode = Field(description="One of the workflow's existing analysis modes.")


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: Literal["urban-density-api"]


class AnalysisCreatedResponse(BaseModel):
    analysis_id: str = Field(description="UUID identifying the local analysis job.")
    status: Literal["queued"]


class AnalysisStatusResponse(BaseModel):
    analysis_id: str
    status: Literal["queued", "running", "completed", "failed"]
    elapsed_seconds: float | None = None
    error: str | None = None


class AnalysisResultsResponse(BaseModel):
    analysis_id: str
    workflow_summary: dict[str, Any]
    stage_timings: dict[str, Any]
    indicator_readiness: list[dict[str, Any]] | dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    detail: str

