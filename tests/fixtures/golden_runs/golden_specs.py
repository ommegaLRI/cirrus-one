from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from tests.fixtures.synthetic_generator import SyntheticConfig, SyntheticEventSpec, default_event_set


@dataclass(frozen=True)
class GoldenSpec:
    """
    Defines one golden case:
      - synthetic config + events
      - expected metric bounds (ranges)
    """
    case_id: str
    cfg: SyntheticConfig
    events: List[SyntheticEventSpec]
    expected: Dict[str, Any]


def golden_cases() -> List[GoldenSpec]:
    # Keep T modest for tests; you can bump for “slow goldens” later
    base_cfg = SyntheticConfig(
        T=600,
        H=75,
        W=90,
        seed=42,
        baseline_step_frame=None,
        noise_jump_frame=None,
    )

    # 1) Clean-ish session: no step, no noise jump
    cfg_clean = base_cfg

    # 2) Overlap-heavy: uses default set (already contains overlap group)
    cfg_overlap = SyntheticConfig(**{**base_cfg.__dict__, "seed": 123})

    # 3) Drift + step + noise jump: stresses plate_state + closure + gating
    cfg_drift_jump = SyntheticConfig(
        **{
            **base_cfg.__dict__,
            "seed": 7,
            "baseline_step_frame": 300,
            "baseline_step_delta": -250.0,
            "noise_jump_frame": 450,
            "noise_jump_multiplier": 2.5,
            "baseline_drift_per_frame": 0.12,
        }
    )

    # Event sets
    events_clean = default_event_set(cfg_clean)
    events_overlap = default_event_set(cfg_overlap)

    # Drift/jump: add a couple extra events to stress rate + gating
    events_drift_jump = default_event_set(cfg_drift_jump) + [
        SyntheticEventSpec(
            t_start=520, t_peak=525, t_end=560,
            y0=cfg_drift_jump.H * 0.7, x0=cfg_drift_jump.W * 0.2,
            amp=1600.0, sigma_y=4.0, sigma_x=5.0,
            mass_mg_truth=4.0
        ),
        SyntheticEventSpec(
            t_start=540, t_peak=545, t_end=590,
            y0=cfg_drift_jump.H * 0.72, x0=cfg_drift_jump.W * 0.25,
            amp=1100.0, sigma_y=3.5, sigma_x=4.5,
            overlap_group="B",
            mass_mg_truth=2.5
        ),
    ]

    # Expected metric bounds:
    # Keep ranges broad enough to tolerate minor algorithmic changes,
    # but tight enough to catch regressions. You’ll tune these once you run them.
    return [
        GoldenSpec(
            case_id="session_clean",
            cfg=cfg_clean,
            events=events_clean,
            expected={
                "event_count_range": [1, 200],
                "closure_score_range": [0.0, 1.0],
                "health_score_range": [0.0, 1.0],
                "baseline_drift_abs_range": [0.0, 500.0],
                "n_regimes_range": [1, 10],
            },
        ),
        GoldenSpec(
            case_id="session_overlap",
            cfg=cfg_overlap,
            events=events_overlap,
            expected={
                "event_count_range": [1, 250],
                "closure_score_range": [0.0, 1.0],
                "health_score_range": [0.0, 1.0],
                "baseline_drift_abs_range": [0.0, 500.0],
                "n_regimes_range": [1, 10],
            },
        ),
        GoldenSpec(
            case_id="session_drift_jump",
            cfg=cfg_drift_jump,
            events=events_drift_jump,
            expected={
                "event_count_range": [1, 300],
                "closure_score_range": [0.0, 1.0],
                "health_score_range": [0.0, 1.0],
                "baseline_drift_abs_range": [10.0, 1500.0],
                "n_regimes_range": [1, 10],
            },
        ),
    ]