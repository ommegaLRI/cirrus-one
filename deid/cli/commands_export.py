"""
deid.cli.commands_export
------------------------

CLI entry:
    deid export --run <path> --format parquet|zarr
"""

from __future__ import annotations

from pathlib import Path
import typer

from deid.reporting.exporters import export_parquet_bundle, export_zarr

app = typer.Typer()


@app.command("export")
def export(
    run: Path = typer.Option(..., "--run", help="Run directory"),
    format: str = typer.Option("parquet", "--format", help="Export format: parquet|zarr"),
    out: Path = typer.Option(..., "--out", help="Output directory"),
):
    run = run.resolve()
    out = out.resolve()

    if format == "parquet":
        export_parquet_bundle(run, out)
    elif format == "zarr":
        export_zarr(run, out)
    else:
        raise typer.BadParameter("format must be parquet or zarr")

    typer.echo(f"Export complete → {out}")