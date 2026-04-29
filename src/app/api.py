from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agents.advisor import AdvisorAgent
from app.agents.analyst import AnalystAgent
from app.agents.client import ClientAgent
from app.chat_handler import new_session_id, run_client_turn
from app.graph.workflow import build_workflow
from app.knowledge.chroma_store import ChromaKnowledgeStore, seed_default_knowledge
from app.logging_config import configure_logging
from app.models.schemas import ClientProfile
from app.storage.db import init_db

_STATIC = Path(__file__).resolve().parent / "static"
_DEFAULT_PROFILE_PLACEHOLDER = "@@@DEFAULT_PROFILE_JSON@@@"


class ChatRequest(BaseModel):
    message: str = ""
    session_id: str | None = None
    client_profile: ClientProfile | None = None


class SessionResponse(BaseModel):
    session_id: str
    profile: ClientProfile


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    init_db()
    base_client = ClientAgent()
    profile = base_client.profile
    advisor = AdvisorAgent()
    client_for_workflow = ClientAgent(profile=profile)
    store = ChromaKnowledgeStore()
    seed_default_knowledge(store)
    analyst = AnalystAgent(store)
    graph = build_workflow(advisor, analyst, client_for_workflow)
    app.state.client_profile = profile
    app.state.base_client = base_client
    app.state.store = store
    app.state.workflow = graph
    yield


app = FastAPI(title="Investment advisor chat", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.get("/")
def chat_page() -> HTMLResponse:
    index = _STATIC / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=500, detail="Chat UI not found.")
    template = index.read_text(encoding="utf-8")
    if _DEFAULT_PROFILE_PLACEHOLDER not in template:
        raise HTTPException(status_code=500, detail="Chat UI template missing profile placeholder.")
    embedded = json.dumps(app.state.client_profile.model_dump(), ensure_ascii=False)
    html = template.replace(_DEFAULT_PROFILE_PLACEHOLDER, embedded)
    return HTMLResponse(content=html, media_type="text/html")


@app.post("/api/chat")
async def post_chat(body: ChatRequest) -> JSONResponse:
    store = app.state.store
    workflow = app.state.workflow
    default_profile: ClientProfile = app.state.client_profile
    prompt_client: ClientAgent = app.state.base_client

    session_id = body.session_id or new_session_id()
    profile = body.client_profile or default_profile

    result = await run_client_turn(
        session_id=session_id,
        user_input=body.message,
        client_profile=profile,
        store=store,
        workflow=workflow,
        prompt_client=prompt_client,
    )
    return JSONResponse(result)


@app.get("/api/profile", response_model=ClientProfile)
def get_profile() -> ClientProfile:
    return app.state.client_profile


@app.post("/api/session", response_model=SessionResponse)
def create_browser_session() -> SessionResponse:
    profile: ClientProfile = app.state.client_profile
    return SessionResponse(session_id=new_session_id(), profile=profile)


def run() -> None:
    import uvicorn

    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=False)
