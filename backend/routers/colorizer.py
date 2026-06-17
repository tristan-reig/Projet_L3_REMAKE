"""Endpoints du coloriseur d'images noir & blanc."""

import io

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image

from backend.core import model_loader
from ml.colorizer.colorize import colorize_image

router = APIRouter(prefix="/colorizer", tags=["colorizer"])

@router.post("/colorize")
async def colorize(file: UploadFile = File(...)):
    """Colorise une image noir & blanc et renvoie le PNG résultat."""
    try:
        img = Image.open(io.BytesIO(await file.read()))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Image invalide : {exc}") from exc

    model = model_loader.get_model("colorizer")
    result = colorize_image(model, img)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
