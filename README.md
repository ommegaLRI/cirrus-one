# Cirrus One

**Cirrus One** is a research-grade pipeline for analyzing thermal data alongside derived particle and SWE products. It is designed for scientifically defensible, reproducible analysis of spatiotemporal thermal measurements and their macro-scale hydrometeorological implications.

This project is not a visualization tool or a simple data converter, it is a deterministic scientific analysis engine built for research workflows.

---

## What This Project Does

The service ingests three kinds of session data:

1. **Thermal cube (HDF5)** — raw uint16 spatiotemporal field (authoritative source of truth)
2. **Particle table** — event-level measurements such as centroids, mass, area, evaporation time, and temperature
3. **Processed SWE series** — cumulative SWE and SWE rate estimates

From these inputs, the pipeline produces:

- Instrument QC and integrity diagnostics
- Baseline drift, noise, and nonuniformity estimation
- Authoritative event detection directly from thermal data
- Cross-validation against provided particle tables
- SWE reconstruction and closure analysis
- Optional regime and latent-state inference (QC-gated)
- A fully versioned, provenance-rich run bundle

---

## Design Philosophy

### Scientific Defensibility

Every output is traceable to:

- input data
- configuration parameters
- algorithms
- QC gates

Nothing is tuned silently. The processed SWE table is never used to influence event detection thresholds.

### HDF5-First Truth Pipeline

The thermal cube is treated as the authoritative measurement source.

Particle and processed tables are validated against extracted events — not used to drive segmentation. This avoids circular reasoning and preserves research integrity.

### Deterministic and Reproducible

Same inputs plus same config produce identical outputs.

Each run stores:

- config hash
- input checksums
- environment versions
- pipeline version
- artifact schemas

### Modular Architecture

Major components are pluggable:

- Event extraction algorithms
- Mass calibration models
- SWE rate estimators
- Inference models

The API layer is optional. All core computation runs independently of HTTP services.

---

## High-Level Pipeline

Ingest → Alignment → Plate State → Event Extraction  
↓  
Fusion / Validation  
↓  
SWE Reconstruction and Closure  
↓  
Inference (QC-gated)  
↓  
Reporting and Run Bundle

### Key Concepts

**Plate State**  
Estimates baseline drift, spatial nonuniformity, and noise from quiescent frames.

**Authoritative Events**  
Events extracted directly from the thermal cube using baseline-corrected signals.

**Closure**  
Comparison between reconstructed SWE and provided processed SWE to evaluate consistency.

**Findings**  
Evidence-first outputs referencing events, frames, and artifacts — not black-box claims.

---

## Repository Structure

Core computation is independent of the API. The API is an orchestrator, not the scientific core.

- `deid/api/` — thin FastAPI layer 
- `deid/cli/` — command-line interface
- `deid/config/` — Pydantic config models + canonical hashing
- `deid/core/` — shared utilities (time, coords, units, IDs, errors, logging, versions)
- `deid/ingest/` — input readers + manifest generation
- `deid/alignment/` — cadence inference, alignment, integrity checks
- `deid/plate_state/` — quiescent selection, baseline/noise/nonuniformity estimation, health metrics
- `deid/events/` — event extraction, masks store, features, QC reports
- `deid/fusion/` — particle ↔ extracted event matching + validation diagnostics
- `deid/swe/` — mass calibration, SWE reconstruction, rate estimators, closure analysis
- `deid/inference/` — QC gating, regime/latent inference, evidence-first findings
- `deid/reporting/` — run bundle writer, figures, exporters
- `deid/runner/` — stage DAG, job state machine, cached execution
- `deid/storage/` — canonical run paths, IO helpers, hashing

---

## Inputs

### Thermal Cube (Required)

- HDF5 dataset with raw uint16 frames
- Expected shape: `(T, H, W)`
- Accessed through a `ThermalCubeReader` (HDF5 now, Zarr later)

### Particle Table (Optional but Recommended)

Event-level derived measurements. Typical fields include:

- Date, Time
- Time to Evaporate (seconds)
- Centroid location (x, y)
- Mass (mg)
- Max Area (mm²)
- Temperature (°C)

### Processed SWE Series (Optional)

Macro products, typically:

- SWE (mm), cumulative
- SWE Rate (mm/hr)

Note: processed SWE is never used to tune detection thresholds (prevents circularity).

---

## Outputs: The Run Bundle

Each analysis produces a deterministic run directory:

`run_<session_id>_<run_id>/`

with:

- `inputs/` — normalized inputs and manifest
- `intermediate/` — stage outputs (alignment, plate state, event catalog, masks, matches)
- `outputs/` — SWE products, closure, findings, QC summary, figures
- `provenance/` — config, hashes, environment, code version, timings

Key artifacts include:

- `event_catalog.parquet` — authoritative extracted events
- `plate_state.npz` — baseline, noise, and nonuniformity maps
- `swe_products.parquet` — reconstructed SWE and multiple rate estimators
- `closure_report.json` — macro consistency diagnostics
- `findings.json` — QC-gated hypotheses with evidence pointers

---

## Quick Start (CLI)

Analyze a session:

`deid analyze --particle particle.xlsx --processed processed.xlsx --hdf5 thermal.h5 --config config.yaml`

Generate a report bundle:

`deid report --run <run_id>`

Export tables:

`deid export --run <run_id> --format parquet`

---

## Why This Tool Exists

Traditional workflows often rely on derived particle tables or processed SWE products without verifying physical consistency against the underlying thermal field.

This tool provides:

- A ground-truth-first workflow (HDF5 is authoritative)
- Explicit QC, integrity checks, and gating
- Evidence-backed findings instead of opaque metrics
- Deterministic, provenance-rich run bundles suitable for publication and long-term research

---

## Intended Audience

This project is aimed at:

- Atmospheric and hydrometeorological researchers
- Instrumentation scientists
- Physics or data science students working with spatiotemporal thermal datasets

Undergraduate readers should be able to follow the core ideas, while advanced users can extend the architecture through modular interfaces.

---

## Contributing

Contributions should preserve:

- deterministic behavior
- provenance tracking
- clear module boundaries

New algorithms should be added by implementing the defined interfaces (extractor, calibrator, rate estimator, inferencer) rather than bypassing pipeline stages.

---

## License and Citation

This project is currently under active research and development.

No open-source license has been granted at this time.
All rights are reserved by the author(s) unless explicitly stated otherwise.

A formal license may be added in the future once the long-term distribution model (open source, source-available, or proprietary) is finalized.

If you are interested in collaboration, research use, or early access, please contact the maintainers.
