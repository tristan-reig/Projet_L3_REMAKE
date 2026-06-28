"""Endpoints du débruiteur d'images."""

import io

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image

from backend.core import model_loader
from ml.denoiser.predict import denoise_image

router = APIRouter(prefix="/denoiser", tags=["denoiser"])

@router.post("/denoise")
async def denoise(file: UploadFile = File(...)):
    """Débruite une image et renvoie le PNG restauré."""
    try:
        img = Image.open(io.BytesIO(await file.read()))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Image invalide : {exc}") from exc

    model = model_loader.get_model("denoiser")
    result = denoise_image(model, img)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
