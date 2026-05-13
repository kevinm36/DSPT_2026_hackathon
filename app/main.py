from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routers import web

_APP_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Affinity ranking UI")
app.mount("/static", StaticFiles(directory=_APP_DIR / "static"), name="static")
app.include_router(web.router)
