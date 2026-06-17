"""Point d'entrée de l'API FastAPI.

Lancement en dev :
    uv run uvicorn backend.main:app --reload
"""

from backend.core import config  # force KERAS_BACKEND avant tout

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.core import model_loader
from backend.routers import colorizer

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"

app = FastAPI(
    title="IA Créative — API",
    description="Classer, coloriser et générer des images (Keras 3 + JAX)",
    version="0.1.0",
)

# Sert les fichiers statiques (css, js, images).
# Chemin absolu dérivé de l'emplacement du fichier : fonctionne quel que soit
# le répertoire depuis lequel uvicorn est lancé.
app.mount("/static", StaticFiles(directory=FRONTEND_DIR / "static"), name="static")

# CORS ouvert en dev (à restreindre en prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(colorizer.router)


@app.get("/")
def root():
    """Page de sélection des outils."""
    return FileResponse(FRONTEND_DIR / "templates" / "index.html")


@app.get("/colorizer")
def colorizer_page():
    """Page de l'outil de colorisation."""
    return FileResponse(FRONTEND_DIR / "templates" / "colorizer.html")


@app.get("/health")
def health():
    """État de l'API et modèles disponibles."""
    return {"status": "ok", "modeles_disponibles": model_loader.available_models()}