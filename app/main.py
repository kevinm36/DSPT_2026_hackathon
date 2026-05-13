from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routers import web
from app.services.model_service import set_model

_APP_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Affinity ranking UI")
app.mount("/static", StaticFiles(directory=_APP_DIR / "static"), name="static")
app.include_router(web.router)


@app.on_event("startup")
def _load_model():
    """Swap in AgentModel if ARN is configured, otherwise keep StubModel."""
    try:
        from app.services.agent_model import AgentModel
        set_model(AgentModel())
        print("✓ Using AgentModel (image ranking agent)")
    except (RuntimeError, Exception) as e:
        print(f"⚠ AgentModel not available ({e}), using StubModel")
