"""
deid.api.app
------------

FastAPI entrypoint.

Responsibilities:
- orchestration only
- no scientific computation here
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from deid.api.routes_sessions import router as sessions_router
from deid.api.routes_runs import router as runs_router
from deid.api.routes_jobs import router as jobs_router

app = FastAPI(title="Cirrus One API", version="v1")

# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "https://cirrus-dash.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions_router)
app.include_router(runs_router)
app.include_router(jobs_router)


@app.get("/")
def root():
    return {"service": "cirrus-one", "status": "ok"}