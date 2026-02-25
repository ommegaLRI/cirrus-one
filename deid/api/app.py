"""
deid.api.app
------------

FastAPI entrypoint.

Responsibilities:
- orchestration only
- no scientific computation here
"""

from __future__ import annotations

from fastapi import FastAPI

from deid.api.routes_sessions import router as sessions_router
from deid.api.routes_runs import router as runs_router

app = FastAPI(title="Cirrus One API", version="v1")

app.include_router(sessions_router)
app.include_router(runs_router)


@app.get("/")
def root():
    return {"service": "deid-service", "status": "ok"}