"""
deid.inference.findings
-----------------------

Evidence-first findings contract.

Rule:
- No finding without evidence pointers unless explicitly marked unsupported.

This file defines:
- EvidenceRefs
- Finding
- FindingsReport
- helper make_finding(...)
- validation helpers for defensibility

All structures are JSON-serializable via .to_dict().
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class EvidenceRefs:
    event_ids: List[str] = field(default_factory=list)
    frame_indices: List[int] = field(default_factory=list)
    artifact_refs: List[str] = field(default_factory=list)  # paths within run bundle

    def is_empty(self) -> bool:
        return (len(self.event_ids) == 0) and (len(self.frame_indices) == 0) and (len(self.artifact_refs) == 0)


@dataclass(frozen=True)
class Finding:
    finding_id: str
    title: str
    summary: str
    time_range_utc: Tuple[str, str]  # ISO strings
    evidence: EvidenceRefs
    confidence: float
    qc_penalties: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # Defensibility flags
    supported: bool = True
    unsupported_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # asdict nests dataclasses; good for JSON
        return d


@dataclass(frozen=True)
class FindingsReport:
    schema_version: str
    findings: List[Finding]
    session_qc: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "findings": [f.to_dict() for f in self.findings],
            "session_qc": dict(self.session_qc),
        }


def _iso(dt: Any) -> str:
    if isinstance(dt, str):
        return dt
    if isinstance(dt, datetime):
        return dt.isoformat()
    raise TypeError("time values must be datetime or ISO string")


def validate_finding(f: Finding) -> None:
    """
    Enforce evidence-first rule.

    If supported=True, evidence must not be empty.
    If supported=False, unsupported_reason must be set.
    """
    if f.supported:
        if f.evidence.is_empty():
            raise ValueError("Finding is marked supported=True but has no evidence pointers.")
    else:
        if not f.unsupported_reason:
            raise ValueError("Finding is marked supported=False but unsupported_reason is missing.")


def make_finding(
    *,
    finding_id: str,
    title: str,
    summary: str,
    time_range_utc: Tuple[Any, Any],
    event_ids: Optional[List[str]] = None,
    frame_indices: Optional[List[int]] = None,
    artifact_refs: Optional[List[str]] = None,
    confidence: float = 0.5,
    qc_penalties: Optional[Dict[str, float]] = None,
    tags: Optional[List[str]] = None,
    supported: bool = True,
    unsupported_reason: Optional[str] = None,
) -> Finding:
    """
    Helper builder that enforces the evidence-first contract.
    """
    ev = EvidenceRefs(
        event_ids=list(event_ids or []),
        frame_indices=list(frame_indices or []),
        artifact_refs=list(artifact_refs or []),
    )
    f = Finding(
        finding_id=str(finding_id),
        title=str(title),
        summary=str(summary),
        time_range_utc=(_iso(time_range_utc[0]), _iso(time_range_utc[1])),
        evidence=ev,
        confidence=float(confidence),
        qc_penalties=dict(qc_penalties or {}),
        tags=list(tags or []),
        supported=bool(supported),
        unsupported_reason=unsupported_reason,
    )
    validate_finding(f)
    return f


def make_findings_report(
    *,
    findings: List[Finding],
    session_qc: Optional[Dict[str, Any]] = None,
    schema_version: str = "findings_v1",
) -> FindingsReport:
    # validate all findings before emitting report
    for f in findings:
        validate_finding(f)

    return FindingsReport(
        schema_version=schema_version,
        findings=list(findings),
        session_qc=dict(session_qc or {}),
    )