"""AG-UI bridge: exposes the social_poster agent to the CopilotKit frontend.

Run from the repo root:  uv run uvicorn backend.main:app --port 8000
"""

import os
import pathlib
import re

from ag_ui_adk import ADKAgent, add_adk_fastapi_endpoint
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from google.adk.memory import VertexAiMemoryBankService
from google.adk.sessions import VertexAiSessionService

load_dotenv(pathlib.Path(__file__).parent / "social_poster" / ".env")

from backend.social_poster import db  # noqa: E402  (needs .env first)
from backend.social_poster.agent import root_agent  # noqa: E402  (needs .env first)

app = FastAPI(title="social-agent AG-UI backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Agent Runtime injects APP_URL with its own resource name. Keep local runs
# in-memory; the session region comes from Runtime, not Gemini's global endpoint.
session_service = None
memory_service = None

app_url = os.environ.get("APP_URL", "").strip()
if app_url:
    runtime_resource = re.search(
        r"(?:^|/)projects/(?P<project>\d+)/locations/(?P<location>[a-z0-9-]+)/"
        r"reasoningEngines/(?P<engine>\d+)(?:/|$)",
        app_url,
    )
    if runtime_resource is None:
        raise ValueError("APP_URL must contain a valid Agent Runtime resource name")
    session_service = VertexAiSessionService(
        project=runtime_resource["project"],
        location=runtime_resource["location"],
        agent_engine_id=runtime_resource["engine"],
    )
    memory_service = VertexAiMemoryBankService(
        project=runtime_resource["project"],
        location=runtime_resource["location"],
        agent_engine_id=runtime_resource["engine"],
    )

adk_agent = ADKAgent(
    adk_agent=root_agent,
    app_name="social_poster",
    user_id="devcamp-user",  # single-user POC; extract from auth in real apps
    session_service=session_service,
    memory_service=memory_service,
    # Recover threads from stored state; arbitrary thread IDs may not be valid
    # Vertex custom session IDs, so let the service generate them.
    use_thread_id_as_session_id=False,
    # Evict cloud sessions from the cache without deleting their durable data.
    delete_session_on_cleanup=session_service is None,
)

add_adk_fastapi_endpoint(app, adk_agent, path="/api/adk")

# Serve generate_image's output files so the frontend's image gallery
# (PostGallery.tsx) can render them — tools.py writes local file paths, which
# the browser has no way to read directly otherwise.
GALLERY_DIR = pathlib.Path(__file__).resolve().parents[1] / "gallery"
GALLERY_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=GALLERY_DIR), name="outputs")


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/api/posts")
def posts() -> dict:
    """Published posts (with whatever image they used), read from the local
    SQLite store — durable across reloads/restarts, unlike ADK session state
    or the browser's own chat history."""
    return {"posts": db.list_posts()}
