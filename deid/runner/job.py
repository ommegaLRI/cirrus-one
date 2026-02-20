"""
deid.runner.job
---------------

Job state model for the pipeline runner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class JobState:
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


class StageState:
    NOT_STARTED = "NOT_STARTED"
    RUNNING = "RUNNING"
    DONE = "DONE"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    job_id: str
    session_id: str
    run_id: str
    state: str = JobState.PENDING

    created_at_utc: str = field(default_factory=utc_now_iso)
    started_at_utc: Optional[str] = None
    finished_at_utc: Optional[str] = None

    stage_status: Dict[str, str] = field(default_factory=dict)
    stage_timings_s: Dict[str, float] = field(default_factory=dict)
    errors: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # stage_id -> error dict

    def mark_started(self) -> None:
        self.state = JobState.RUNNING
        self.started_at_utc = utc_now_iso()

    def mark_finished(self, succeeded: bool) -> None:
        self.finished_at_utc = utc_now_iso()
        self.state = JobState.SUCCEEDED if succeeded else JobState.FAILED