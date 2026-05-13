from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routers import web

_APP_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Affinity ranking UI")
app.mount("/static", StaticFiles(directory=_APP_DIR / "static"), name="static")
app.include_router(web.router)


@app.on_event("startup")
def _load_model():
    """Swap in AgentModel if ARN is configured, otherwise keep stub."""
    import app.services.model_service as ms
    try:
        from image_ranking_agent_pipeline.image_ranking_agent_model import ImageRankingAgentModel
        ms.default_agent_model = ImageRankingAgentModel()
        print("\u2713 Using AgentModel (image ranking agent)")
    except (RuntimeError, Exception) as e:
        print(f"\u26a0 AgentModel not available ({e}), using stub")
