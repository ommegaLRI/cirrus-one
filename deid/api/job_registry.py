"""
deid.api.job_registry.py
------------

"""

from __future__ import annotations

from typing import Dict
from deid.runner.job import Job

JOBS: Dict[str, Job] = {}