from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    # Force regeneration mode for the pytest run
    env = os.environ.copy()
    env["DEID_REGEN_GOLDEN"] = "1"

    cmd = [sys.executable, "-m", "pytest", "-q", "tests/golden_runs/test_golden_pipeline.py"]
    print("Running:", " ".join(cmd))
    p = subprocess.run(cmd, env=env)
    return int(p.returncode)


if __name__ == "__main__":
    raise SystemExit(main())