"""Endpoints du classifieur d'images."""

import io

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from backend.core import model_loader
from ml.classifier.model import CLASS_NAMES
from ml.classifier.predict import classify_image

router = APIRouter(prefix="/classifier", tags=["classifier"])


@router.get("/classes")
async def classes():
    """Liste les classes que le modèle peut reconnaître."""
    return {"classes": CLASS_NAMES}


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Prédit les classes les plus probables d'une image uploadée (top-3)."""
    try:
        img = Image.open(io.BytesIO(await file.read()))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Image invalide : {exc}") from exc

    model = model_loader.get_model("classifier")
    predictions = classify_image(model, img, top_k=3)
    return {"predictions": predictions}