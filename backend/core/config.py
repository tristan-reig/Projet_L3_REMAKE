"""Configuration centrale de l'application."""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "models"

# Chemins des modèles entraînés
MODEL_PATHS = {
    # "classifier": MODELS_DIR / "model_classifier.keras",
    "colorizer": MODELS_DIR / "model_colorizer.keras",
    # "generator": MODELS_DIR / "model_vae.keras",  # à venir
}

# Tailles d'entrée attendues par les modèles (à ajuster selon l'entraînement)
INPUT_SIZES = {
    # "classifier": (128, 128),
    "colorizer": (256, 256),
}
