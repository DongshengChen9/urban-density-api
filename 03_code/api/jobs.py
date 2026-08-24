from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from .models import AnalysisRequest

LOGGER = logging.getLogger(__name__)
PUBLIC_FAILURE_MESSAGE = "The analysis failed. See the service logs for diagnostic details."


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    """Persistent, JSON-per-analysis metadata with thread-safe updates."""

    def __init__(self, output_root: Path) -> None:
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def output_directory(self, analysis_id: str) -> Path:
        return self.output_root / analysis_id

    def metadata_path(self, analysis_id: str) -> Path:
        return self.output_directory(analysis_id) / "analysis.json"

    def create(self, request: AnalysisRequest) -> dict[str, Any]:
        analysis_id = str(uuid4())
        output_directory = self.output_directory(analysis_id)
        output_directory.mkdir(parents=True, exist_ok=False)
        metadata: dict[str, Any] = {
            "analysis_id": analysis_id,
            "status": "queued",
            "created_at": utc_now(),
            "started_at": None,
            "completed_at": None,
            "mode": request.mode.value,
            "grid_size": request.grid_size,
            "bbox": request.bbox.model_dump(),
            "output_directory": str(output_directory),
            "error": None,
        }
        self._write(metadata)
        return metadata

    def get(self, analysis_id: str) -> dict[str, Any] | None:
        path = self.metadata_path(analysis_id)
        with self._lock:
            if not path.is_file():
                return None
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                LOGGER.exception("Could not read job metadata for %s", analysis_id)
                return None
        return data if isinstance(data, dict) else None

    def update(self, analysis_id: str, **changes: Any) -> dict[str, Any]:
        with self._lock:
            metadata = self.get(analysis_id)
            if metadata is None:
                raise KeyError(analysis_id)
            metadata.update(changes)
            self._write(metadata)
            return metadata

    def _write(self, metadata: dict[str, Any]) -> None:
        path = self.metadata_path(str(metadata["analysis_id"]))
        temporary_path = path.with_suffix(".json.tmp")
        with self._lock:
            temporary_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            temporary_path.replace(path)


WorkflowRunner = Callable[[str, AnalysisRequest, Path], None]


class JobManager:
    """Small in-process executor intended for local/demo deployments."""

    def __init__(self, store: JobStore, runner: WorkflowRunner, max_workers: int = 1) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        self.store = store
        self.runner = runner
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="udw-analysis"
        )

    def submit(self, request: AnalysisRequest) -> dict[str, Any]:
        metadata = self.store.create(request)
        analysis_id = str(metadata["analysis_id"])
        LOGGER.info("Analysis created: %s", analysis_id)
        self.executor.submit(self._execute, analysis_id, request)
        return metadata

    def _execute(self, analysis_id: str, request: AnalysisRequest) -> None:
        started_at = utc_now()
        self.store.update(analysis_id, status="running", started_at=started_at)
        LOGGER.info("Workflow starting: %s", analysis_id)
        try:
            self.runner(analysis_id, request, self.store.output_directory(analysis_id))
        except Exception:
            LOGGER.exception("Workflow failed: %s", analysis_id)
            self.store.update(
                analysis_id,
                status="failed",
                completed_at=utc_now(),
                error=PUBLIC_FAILURE_MESSAGE,
            )
            return
        self.store.update(
            analysis_id, status="completed", completed_at=utc_now(), error=None
        )
        LOGGER.info("Workflow completed: %s", analysis_id)

    def shutdown(self, wait: bool = True) -> None:
        self.executor.shutdown(wait=wait, cancel_futures=False)

