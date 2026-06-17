"""Point d'entrée de l'API FastAPI.

Lancement en dev :
    uv run uvicorn backend.main:app --reload
"""

from backend.core import config

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.core import model_loader
from backend.routers import colorizer

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"

app = FastAPI()

# Sert les fichiers statiques (css, js, images) si le dossier existe
_static = FRONTEND_DIR / "static"
if _static.exists():
    app.mount("/static", StaticFiles(directory=_static), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(colorizer.router)


@app.get("/")
def root():
    """Sert la page d'accueil."""
    return FileResponse(FRONTEND_DIR / "templates" / "index.html")


@app.get("/health")
def health():
    """État de l'API et modèles disponibles."""
    return {"status": "ok", "modeles_disponibles": model_loader.available_models()}
