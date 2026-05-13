"""FastAPI entrypoint for the affinity ranking UI.

Agent backend selection
-----------------------
Stock ``uvicorn`` does not accept custom flags such as ``--agent-model``. Use
one of:

1. **Environment variable** (works with plain ``uvicorn``)::

     export AGENT_MODEL=IabAgentInferenceModel
     uvicorn app.main:app --reload --app-dir .

   Valid values: ``ImageRankingAgentModel`` (default), ``IabAgentInferenceModel``.

2. **Module runner** (parses ``--agent-model`` then invokes uvicorn)::

     python -m app --agent-model ImageRankingAgentModel --reload --app-dir .
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routers import web
from app.services import model_service

_APP_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def _lifespan(app: FastAPI):
    model_service.configure_agent_model()
    yield


app = FastAPI(title="Affinity ranking UI", lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=_APP_DIR / "static"), name="static")
app.include_router(web.router)
