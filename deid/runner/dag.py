"""
deid.runner.dag
---------------

Stage DAG definition and topological ordering.

Rules:
- DAG is declarative (no execution here).
- Stages list expected outputs (relative to run_dir, e.g. "inputs/manifest.json").
- Optional inputs are expressed so stages can run in degraded mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass(frozen=True)
class StageDef:
    id: str
    depends_on: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)          # relative paths within run bundle
    optional_inputs: List[str] = field(default_factory=list)  # relative paths within run bundle
    gated_by: List[str] = field(default_factory=list)         # relative paths within run bundle


PIPELINE_VERSION: str = "v1"


def get_stage_defs() -> Dict[str, StageDef]:
    """
    Normative v1 DAG definition (runner-ready).
    """
    stages = [
        StageDef(
            id="ingest",
            outputs=[
                "inputs/manifest.json",
                "inputs/thermal_ref.json",
                "inputs/particle.parquet",
                "inputs/processed.parquet",
            ],
        ),
        StageDef(
            id="alignment",
            depends_on=["ingest"],
            outputs=["intermediate/alignment.json", "intermediate/integrity.json"],
        ),
        StageDef(
            id="plate_state",
            depends_on=["alignment"],
            outputs=["intermediate/plate_state.npz", "intermediate/plate_state.json", "intermediate/instrument_health.json"],
        ),
        StageDef(
            id="event_extract",
            depends_on=["plate_state"],
            outputs=["intermediate/event_catalog.parquet", "intermediate/event_qc.json", "intermediate/event_masks/index.json"],
        ),
        StageDef(
            id="fusion",
            depends_on=["event_extract"],
            optional_inputs=["inputs/particle.parquet"],
            outputs=["intermediate/matched_events.parquet", "intermediate/particle_validation.json"],
        ),
        StageDef(
            id="swe_closure",
            depends_on=["event_extract", "alignment"],
            optional_inputs=["inputs/processed.parquet"],
            outputs=["outputs/swe_products.parquet", "outputs/closure_report.json"],
        ),
        StageDef(
            id="inference",
            depends_on=["swe_closure", "plate_state"],
            gated_by=["intermediate/instrument_health.json", "outputs/closure_report.json"],
            outputs=["outputs/findings.json", "outputs/regimes.json", "outputs/latent.json", "outputs/inference_skipped.json"],
        ),
        StageDef(
            id="report",
            depends_on=["ingest", "alignment", "plate_state", "event_extract", "fusion", "swe_closure", "inference"],
            outputs=["outputs/qc_summary.json"],
        ),
    ]

    return {s.id: s for s in stages}


def topo_order(stage_defs: Dict[str, StageDef]) -> List[str]:
    """
    Compute a deterministic topological order.
    Raises ValueError on cycles/missing deps.
    """
    # Kahn's algorithm with deterministic ordering
    deps: Dict[str, Set[str]] = {sid: set(sd.depends_on) for sid, sd in stage_defs.items()}
    rev: Dict[str, Set[str]] = {sid: set() for sid in stage_defs}
    for sid, sd in stage_defs.items():
        for d in sd.depends_on:
            if d not in stage_defs:
                raise ValueError(f"Stage '{sid}' depends on missing stage '{d}'")
            rev[d].add(sid)

    ready = sorted([sid for sid, ds in deps.items() if not ds])
    out: List[str] = []

    while ready:
        sid = ready.pop(0)
        out.append(sid)
        for child in sorted(rev[sid]):
            deps[child].discard(sid)
            if not deps[child]:
                if child not in out and child not in ready:
                    ready.append(child)
                    ready.sort()

    if len(out) != len(stage_defs):
        missing = sorted(set(stage_defs.keys()) - set(out))
        raise ValueError(f"DAG has a cycle or unresolved dependencies. Unresolved: {missing}")

    return out