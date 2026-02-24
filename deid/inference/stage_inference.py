"""
deid.inference.stage_inference
------------------------------

Tier 3 runner stage:

Stage G — Optional Inference (regimes/latent/findings), QC-gated

Consumes Tier 2 artifacts only:
  intermediate/alignment.json
  intermediate/instrument_health.json
  intermediate/event_catalog.parquet
  intermediate/plate_state.json (optional context)
  outputs/closure_report.json
  outputs/swe_products.parquet (optional for phenotype)

Produces (under outputs/):
  regimes.json
  regime_events.parquet
  phenotype_vector.json
  findings.json
  inference_skipped.json (if hard gate fails)
  latent.json (Phase 3)

v1 scope:
- gating decision
- regimes discovery (kmeans)
- phenotype vector
- latent trajectory inference (Kalman + RTS)
- evidence-first findings
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

from deid.config.models import DEIDConfig
from deid.core.ids import stable_id
from deid.core.logging import log_info
from deid.storage.io import read_json, read_parquet, write_json, write_parquet, wrap_artifact
from deid.storage.paths import intermediate_dir, outputs_dir, inputs_dir

from deid.inference.gating import evaluate_gating, gating_decision_to_dict, make_inference_skipped_payload
from deid.inference.regimes import run_regime_discovery
from deid.inference.phenotype_vector import build_phenotype_vector
from deid.inference.findings import make_finding, make_findings_report, Finding
from deid.inference.latent import run_latent_inference 


def _load_wrapped_payload(path: Path) -> Dict[str, Any]:
    w = read_json(path)
    return w.get("payload", {})


def _maybe_read_parquet(path: Path) -> Optional[pd.DataFrame]:
    return read_parquet(path) if path.exists() else None


def _detect_regime_shift_findings(
    regimes_payload: Dict[str, Any],
    *,
    shift_threshold: float = 0.35,
) -> List[Finding]:
    """
    Evidence-first regime transition findings from regimes_payload timeline fractions.
    """
    timeline = regimes_payload.get("timeline", {}) or {}
    bins = timeline.get("bins", []) or []
    fractions = timeline.get("fractions", {}) or {}

    if len(bins) < 2 or not fractions:
        return []

    k = sorted(fractions.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    F = {r: fractions.get(r, []) for r in k}
    n = min(len(bins), *(len(F[r]) for r in k))
    if n < 2:
        return []

    out: List[Finding] = []
    for i in range(1, n):
        max_d = 0.0
        max_reg = None
        for r in k:
            d = abs(float(F[r][i]) - float(F[r][i - 1]))
            if d > max_d:
                max_d = d
                max_reg = r

        if max_d >= shift_threshold:
            t0 = bins[i - 1]
            t1 = bins[i]
            fid = stable_id("finding", {"kind": "regime_shift", "t0": t0, "t1": t1, "regime": max_reg, "d": round(max_d, 3)})
            out.append(
                make_finding(
                    finding_id=fid,
                    title="Regime transition detected",
                    summary=f"Regime fraction shift of {max_d:.2f} in regime {max_reg} between adjacent time bins.",
                    time_range_utc=(t0, t1),
                    event_ids=[],
                    frame_indices=[],
                    artifact_refs=["outputs/regimes.json"],
                    confidence=float(min(0.95, 0.5 + max_d)),
                    qc_penalties={},
                    tags=["regimes", "transition"],
                    supported=True,
                )
            )
    return out


def inference_stage(
    run_dir: Path,
    inputs: Dict[str, Any],
    config: DEIDConfig,
    context: Dict[str, Any],
) -> None:
    """
    Runner stage entrypoint.
    """
    log_info("inference_stage_start", run_dir=str(run_dir))

    config_hash: str = inputs["config_hash"]
    input_hashes: Dict[str, str] = inputs.get("input_hashes", {})

    inter = intermediate_dir(run_dir)
    out = outputs_dir(run_dir)
    inp = inputs_dir(run_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Load Tier 2 artifacts ----
    alignment_payload = _load_wrapped_payload(inter / "alignment.json")
    instrument_health = _load_wrapped_payload(inter / "instrument_health.json")

    closure_path_out = out / "closure_report.json"
    closure_path_inter = inter / "closure_report.json"
    if closure_path_out.exists():
        closure_report = _load_wrapped_payload(closure_path_out)
    elif closure_path_inter.exists():
        closure_report = _load_wrapped_payload(closure_path_inter)
    else:
        closure_report = {}

    event_df = read_parquet(inter / "event_catalog.parquet")
    swe_products_df = _maybe_read_parquet(out / "swe_products.parquet")
    if swe_products_df is None:
        swe_products_df = _maybe_read_parquet(inter / "swe_products.parquet")

    # NEW: optional processed series for latent observation
    processed_df = _maybe_read_parquet(inp / "processed.parquet")

    plate_state_meta = _load_wrapped_payload(inter / "plate_state.json") if (inter / "plate_state.json").exists() else {}

    # ---- Configs ----
    # Inference config lives under config.inference
    try:
        inf_cfg = dict(getattr(config, "inference", {}))
    except Exception:
        inf_cfg = {}

    gating_cfg = dict(inf_cfg.get("gating", {}))
    regimes_cfg = dict(inf_cfg.get("regimes", {}))

    try:
        regimes_cfg = dict(config.regimes)
    except Exception:
        regimes_cfg = {}

    try:
        phen_cfg = dict(getattr(config, "phenotype", {}))
    except Exception:
        phen_cfg = {}

    try:
        latent_cfg = dict(getattr(config, "latent", {}))
    except Exception:
        latent_cfg = {}

    # ---- Gating decision ----
    decision = evaluate_gating(
        instrument_health=instrument_health,
        closure_report=closure_report,
        alignment_payload=alignment_payload,
        config=gating_cfg,
    )

    decision_payload = gating_decision_to_dict(decision)

    # If hard gated, write inference_skipped and return
    if (decision.mode == "hard") and (not decision.allow_inference):
        skipped = make_inference_skipped_payload(decision)
        skipped_wrapped = wrap_artifact(
            payload=skipped,
            schema_version="inference_skipped_v1",
            config_hash=config_hash,
            input_hashes=input_hashes,
            artifact_type="inference_skipped",
        )
        write_json(out / "inference_skipped.json", skipped_wrapped)

        fid = stable_id("finding", {"kind": "inference_skipped", "reasons": "|".join(decision.reasons)})
        f = make_finding(
            finding_id=fid,
            title="Inference skipped due to gating",
            summary="Inference outputs were not generated because QC gating failed.",
            time_range_utc=(
                alignment_payload.get("gaps", [{}])[0].get("t_start_utc", "1970-01-01T00:00:00+00:00"),
                alignment_payload.get("gaps", [{}])[-1].get("t_end_utc", "1970-01-01T00:00:00+00:00"),
            ),
            event_ids=[],
            frame_indices=[],
            artifact_refs=["outputs/inference_skipped.json"],
            confidence=0.0,
            qc_penalties=dict(decision.qc_penalties),
            tags=["gating"],
            supported=False,
            unsupported_reason="Gating failed; findings are not supported by downstream inference outputs.",
        )
        rep = make_findings_report(findings=[f], session_qc={"gating_decision": decision_payload})
        rep_wrapped = wrap_artifact(
            payload=rep.to_dict(),
            schema_version="findings_v1",
            config_hash=config_hash,
            input_hashes=input_hashes,
            artifact_type="findings",
        )
        write_json(out / "findings.json", rep_wrapped)

        log_info("inference_stage_skipped", reasons=decision.reasons)
        return

    # ---- Regimes ----
    regime_events_df, regimes_payload = run_regime_discovery(
        event_df=event_df,
        alignment_payload=alignment_payload,
        config=regimes_cfg,
    )

    write_parquet(out / "regime_events.parquet", regime_events_df)

    regimes_wrapped = wrap_artifact(
        payload=regimes_payload,
        schema_version="regimes_v1",
        config_hash=config_hash,
        input_hashes=input_hashes,
        artifact_type="regimes",
    )
    write_json(out / "regimes.json", regimes_wrapped)

    # ---- Phenotype vector ----
    phenotype_payload = build_phenotype_vector(
        instrument_health=instrument_health,
        closure_report=closure_report,
        event_df=event_df,
        swe_products_df=swe_products_df,
        regimes_payload=regimes_payload,
        extra={"plate_state_method": plate_state_meta.get("method")},
    )

    phenotype_wrapped = wrap_artifact(
        payload=phenotype_payload,
        schema_version="phenotype_vector_v1",
        config_hash=config_hash,
        input_hashes=input_hashes,
        artifact_type="phenotype_vector",
    )
    write_json(out / "phenotype_vector.json", phenotype_wrapped)

    # ---- NEW: Latent inference (Phase 3) ----
    try:
        with np.load(inter / "plate_state.npz") as z:
            latent_payload = run_latent_inference(
                alignment_payload=alignment_payload,
                plate_state_npz=z,
                event_df=event_df,
                swe_products_df=swe_products_df,
                processed_df=processed_df,
                qc_penalties=decision.qc_penalties,
                config=latent_cfg,
            )

        latent_wrapped = wrap_artifact(
            payload=latent_payload,
            schema_version="latent_v1",
            config_hash=config_hash,
            input_hashes=input_hashes,
            artifact_type="latent",
        )
        write_json(out / "latent.json", latent_wrapped)
    except Exception as e:
        log_info("latent_inference_failed_nonfatal", error=str(e))

    # ---- Findings (v1) ----
    findings: List[Finding] = []

    findings.extend(_detect_regime_shift_findings(regimes_payload, shift_threshold=float(regimes_cfg.get("shift_threshold", 0.35))))

    if decision.mode == "soft" and decision.reasons:
        fid = stable_id("finding", {"kind": "soft_gating", "reasons": "|".join(decision.reasons)})
        findings.append(
            make_finding(
                finding_id=fid,
                title="Inference ran under soft gating",
                summary="QC gating raised concerns; inference outputs should be interpreted with reduced confidence.",
                time_range_utc=(
                    regimes_payload.get("timeline", {}).get("bins", ["1970-01-01T00:00:00+00:00"])[0],
                    regimes_payload.get("timeline", {}).get("bins", ["1970-01-01T00:00:00+00:00"])[-1],
                ),
                event_ids=[],
                frame_indices=[],
                artifact_refs=["outputs/regimes.json", "outputs/phenotype_vector.json"],
                confidence=float(decision.qc_penalties.get("gating_penalty", 0.5)),
                qc_penalties=dict(decision.qc_penalties),
                tags=["gating", "soft"],
                supported=False,
                unsupported_reason="This is a policy statement (soft gating), not a data-driven finding.",
            )
        )

    rep = make_findings_report(
        findings=findings,
        session_qc={"gating_decision": decision_payload},
        schema_version="findings_v1",
    )

    rep_wrapped = wrap_artifact(
        payload=rep.to_dict(),
        schema_version="findings_v1",
        config_hash=config_hash,
        input_hashes=input_hashes,
        artifact_type="findings",
    )
    write_json(out / "findings.json", rep_wrapped)

    log_info(
        "inference_stage_complete",
        allow_inference=decision.allow_inference,
        mode=decision.mode,
        n_regime_events=int(len(regime_events_df)),
        n_findings=int(len(findings)),
    )