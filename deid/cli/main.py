"""
deid.cli.main
-------------

CLI entrypoint.

Usage:
    deid analyze --particle ... --processed ... --hdf5 ...
"""

from __future__ import annotations

import typer

from deid.cli.commands_analyze import analyze_command

app = typer.Typer(help="DEID Service CLI")

app.command("analyze")(analyze_command)


def main() -> None:
    app()


if __name__ == "__main__":
    main()