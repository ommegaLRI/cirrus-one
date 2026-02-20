"""
deid.cli.commands_report
------------------------

CLI entry:
    deid report --run <path>

Generates:
- figures
- qc_summary.json
- summary.json
"""

from __future__ import annotations

from pathlib import Path
import typer

from deid.reporting.bundle import build_run_bundle
from deid.reporting.figures import generate_figures

app = typer.Typer()


@app.command("report")
def report(
    run: Path = typer.Option(..., "--run", help="Run directory"),
):
    """
    Build report bundle + figures.
    """
    run = run.resolve()
    provenance = run / "provenance"

    # Load hashes from provenance
    import json

    with open(provenance / "config_hash.txt", "r") as f:
        config_hash = f.read().strip()

    with open(provenance / "input_hashes.json", "r") as f:
        input_hashes = json.load(f)

    generate_figures(run)
    build_run_bundle(
        run_dir=run,
        config_hash=config_hash,
        input_hashes=input_hashes,
    )

    typer.echo(f"Report generated for run: {run}")