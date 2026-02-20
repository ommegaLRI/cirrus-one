"""
deid.inference.gating
---------------------

Centralized inference gating.

Inputs (Tier 2 artifacts payloads):
- instrument_health payload (from intermediate/instrument_health.json)
- closure_report payload (from outputs/closure_report.json)
- alignment payload (from intermediate/alignment.json) for gaps + confidence

Outputs:
- GatingDecision dict (serializable)
- helper to write inference_skipped.json (stage will do writing)

v1:
- hard thresholds on:
    health_score
    closure_score (if present)
    alignment_confidence
    gap_fraction (if computable)
- soft gating option:
    allow_inference=True but apply qc_penalty multiplier

Important: gating is a *policy* layer; it does not run any model.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class GatingDecision:
    allow_inference: bool
    mode: str  # "hard" | "soft"
    reasons: List[str]
    metrics: Dict[str, Any]
    qc_penalties: Dict[str, float]  # multipliers or additive penalties for downstream confidence


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _gap_fraction_from_alignment(alignment_payload: Dict[str, Any]) -> Optional[float]:
    """
    v1 gap fraction heuristic:
    - If dt_seconds is known and we have gap intervals with delta_seconds, estimate:
        gap_frames / T  (T unknown here) -> return None
      Since we don't know T in this layer, we instead return:
        total_gap_seconds / total_span_seconds (span from t0 to last mapped ts)
    - If insufficient, return None.
    """
    gaps = alignment_payload.get("gaps", []) or []
    if not gaps:
        return 0.0

    # total gap seconds
    total_gap = 0.0
    for g in gaps:
        if g.get("kind") == "missing":
            ds = _safe_float(g.get("delta_seconds"))
            if ds is not None and ds > 0:
                total_gap += ds

    # estimate span from frame_timebase if possible (constant cadence + count unknown),
    # else from gap endpoints (very rough).
    ft = alignment_payload.get("frame_timebase", {}) or {}
    t0 = ft.get("t0_utc")
    dt = _safe_float(ft.get("dt_seconds"))

    # We do not have T. If alignment maps processed/particle times, estimate span using those map keys.
    # We'll use gap endpoints: min start, max end.
    starts = []
    ends = []
    for g in gaps:
        s = g.get("t_start_utc")
        e = g.get("t_end_utc")
        if isinstance(s, str) and isinstance(e, str):
            starts.append(s)
            ends.append(e)

    if not starts or not ends:
        return None

    # Parse ISO strings with numpy datetime64 (UTC assumed)
    try:
        t_min = np.min(np.array(starts, dtype="datetime64[ns]"))
        t_max = np.max(np.array(ends, dtype="datetime64[ns]"))
        span_s = float((t_max - t_min) / np.timedelta64(1, "s"))
        if span_s <= 0:
            return None
        return float(total_gap / span_s)
    except Exception:
        return None


def evaluate_gating(
    *,
    instrument_health: Dict[str, Any],
    closure_report: Dict[str, Any],
    alignment_payload: Dict[str, Any],
    config: Dict[str, Any],
) -> GatingDecision:
    """
    Evaluate whether inference is allowed.

    Config keys (v1; defaults are conservative):
      mode: "hard"|"soft"
      health_min: float (default 0.7)
      closure_min: float (default 0.6)  (if closure_score is None, don't block by closure)
      alignment_min: float (default 0.6)
      gap_fraction_max: float (default 0.05) (if gap_fraction computable)

    Soft gating:
      - allow_inference=True but adds qc_penalties["gating_penalty"] in (0,1]
    """
    mode = str(config.get("mode", "hard")).lower()
    if mode not in ("hard", "soft"):
        mode = "hard"

    health_min = float(config.get("health_min", 0.70))
    closure_min = float(config.get("closure_min", 0.60))
    alignment_min = float(config.get("alignment_min", 0.60))
    gap_fraction_max = float(config.get("gap_fraction_max", 0.05))

    reasons: List[str] = []
    metrics: Dict[str, Any] = {}

    health_score = _safe_float(instrument_health.get("overall_score"))
    if health_score is None:
        reasons.append("missing_health_score")
        health_score = 0.0
    metrics["health_score"] = health_score

    closure_score = _safe_float(closure_report.get("closure_score"))
    metrics["closure_score"] = closure_score

    alignment_conf = _safe_float(alignment_payload.get("confidence"))
    if alignment_conf is None:
        reasons.append("missing_alignment_confidence")
        alignment_conf = 0.0
    metrics["alignment_confidence"] = alignment_conf

    gap_fraction = _gap_fraction_from_alignment(alignment_payload)
    metrics["gap_fraction_est"] = gap_fraction

    # Hard checks
    if health_score < health_min:
        reasons.append(f"health_below_threshold:{health_score:.3f}<{health_min:.3f}")

    if closure_score is not None and closure_score < closure_min:
        reasons.append(f"closure_below_threshold:{closure_score:.3f}<{closure_min:.3f}")

    if alignment_conf < alignment_min:
        reasons.append(f"alignment_below_threshold:{alignment_conf:.3f}<{alignment_min:.3f}")

    if gap_fraction is not None and gap_fraction > gap_fraction_max:
        reasons.append(f"gap_fraction_above_threshold:{gap_fraction:.3f}>{gap_fraction_max:.3f}")

    hard_fail = len(reasons) > 0

    if mode == "hard":
        allow = not hard_fail
        penalties: Dict[str, float] = {}
        return GatingDecision(
            allow_inference=allow,
            mode="hard",
            reasons=reasons if not allow else [],
            metrics=metrics,
            qc_penalties=penalties,
        )

    # Soft mode: always allow, but apply penalty scaling to downstream confidence
    # Penalty is multiplicative factor in (0,1]; more reasons -> stronger penalty.
    # Also penalize based on how far below thresholds we are.
    penalty = 1.0
    if health_score < health_min:
        penalty *= _clip01(max(0.1, health_score / (health_min + 1e-9)))
    if closure_score is not None and closure_score < closure_min:
        penalty *= _clip01(max(0.1, closure_score / (closure_min + 1e-9)))
    if alignment_conf < alignment_min:
        penalty *= _clip01(max(0.1, alignment_conf / (alignment_min + 1e-9)))
    if gap_fraction is not None and gap_fraction > gap_fraction_max:
        # if over max, reduce confidence
        penalty *= _clip01(max(0.1, gap_fraction_max / (gap_fraction + 1e-9)))

    return GatingDecision(
        allow_inference=True,
        mode="soft",
        reasons=reasons,  # keep reasons for transparency
        metrics=metrics,
        qc_penalties={"gating_penalty": float(penalty)},
    )


def gating_decision_to_dict(d: GatingDecision) -> Dict[str, Any]:
    return asdict(d)


def make_inference_skipped_payload(decision: GatingDecision) -> Dict[str, Any]:
    """
    Standard payload for outputs/inference_skipped.json (or intermediate/ depending on your spec).
    """
    return {
        "schema_version": "inference_skipped_v1",
        "allow_inference": decision.allow_inference,
        "mode": decision.mode,
        "reasons": list(decision.reasons),
        "metrics": dict(decision.metrics),
        "qc_penalties": dict(decision.qc_penalties),
    }