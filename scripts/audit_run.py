import sys
from pathlib import Path
import json
import pandas as pd


def read_json_payload(path: Path):
    with open(path, "r") as f:
        obj = json.load(f)
    return obj.get("payload", obj)


def load_run(run_dir: Path):
    inputs = run_dir / "inputs"
    intermediate = run_dir / "intermediate"
    outputs = run_dir / "outputs"

    particle = None
    processed = None
    events = None
    swe = None

    if (inputs / "particle.parquet").exists():
        particle = pd.read_parquet(inputs / "particle.parquet")

    if (inputs / "processed.parquet").exists():
        processed = pd.read_parquet(inputs / "processed.parquet")

    if (intermediate / "event_catalog.parquet").exists():
        events = pd.read_parquet(intermediate / "event_catalog.parquet")

    if (outputs / "swe_products.parquet").exists():
        swe = pd.read_parquet(outputs / "swe_products.parquet")

    qc_summary = None
    if (outputs / "qc_summary.json").exists():
        qc_summary = read_json_payload(outputs / "qc_summary.json")

    closure_report = None
    if (outputs / "closure_report.json").exists():
        closure_report = read_json_payload(outputs / "closure_report.json")

    return particle, processed, events, swe, qc_summary, closure_report


def summarize_counts(particle, events):
    print("\n--- Event Counts ---")

    if particle is not None:
        print("Particle table rows:", len(particle))

    if events is not None:
        print("Extracted events:", len(events))

    if particle is not None and events is not None:
        ratio = len(events) / max(len(particle), 1)
        print("Extraction ratio (events/particle):", round(ratio, 3))


def summarize_event_features(events):
    if events is None:
        return

    print("\n--- Event Feature Summary ---")

    cols = [
        "delta_peak",
        "energy_proxy_E",
        "duration_s",
        "area_peak_px",
        "snr",
    ]

    for c in cols:
        if c in events.columns:
            print(f"\n{c}")
            print(events[c].describe())


def compare_swe(processed, swe):
    if processed is None or swe is None:
        return

    print("\n--- SWE Comparison ---")

    merged = swe.merge(
        processed,
        on="t_utc",
        how="inner",
        suffixes=("_recon", "_processed"),
    )

    if len(merged) == 0:
        print("No overlapping timestamps")
        return

    # Determine correct column names
    recon_col = "swe_reconstructed_mm"

    if "swe_mm" in merged.columns:
        proc_col = "swe_mm"
    elif "swe_processed_mm" in merged.columns:
        proc_col = "swe_processed_mm"
    else:
        print("Processed SWE column not found")
        print("Columns:", list(merged.columns))
        return

    merged["diff"] = merged[recon_col] - merged[proc_col]

    print("Points compared:", len(merged))
    print("Mean SWE difference (mm):", merged["diff"].mean())
    print("Std SWE difference (mm):", merged["diff"].std())
    print("Max SWE difference (mm):", merged["diff"].abs().max())


def summarize_closure(closure_report):
    if not closure_report:
        return

    print("\n--- Closure Report ---")

    print("Closure score:", closure_report.get("closure_score"))

    stats = closure_report.get("residual_stats", {})
    for k, v in stats.items():
        print(f"{k}: {v}")


def summarize_qc(qc_summary):
    if not qc_summary:
        return

    print("\n--- QC Summary ---")

    for k, v in qc_summary.items():
        print(f"{k}: {v}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python audit_run.py <run_dir>")
        sys.exit(1)

    run_dir = Path(sys.argv[1]).resolve()

    if not run_dir.exists():
        print("Run directory not found:", run_dir)
        sys.exit(1)

    particle, processed, events, swe, qc_summary, closure_report = load_run(run_dir)

    summarize_counts(particle, events)
    summarize_event_features(events)
    compare_swe(processed, swe)
    summarize_closure(closure_report)
    summarize_qc(qc_summary)


if __name__ == "__main__":
    main()