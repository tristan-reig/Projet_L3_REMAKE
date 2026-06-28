"""Inférence du débruiteur : image bruitée -> image restaurée."""

import numpy as np
from PIL import Image

from ml.denoiser.model import IMG_SIZE

def denoise_image(model, pil_image: Image.Image) -> Image.Image:
    """Débruite une image et la renvoie à sa résolution d'origine."""
    img_orig = pil_image.convert("RGB")
    orig_size = img_orig.size

    img_small = img_orig.resize((IMG_SIZE, IMG_SIZE))
    x = np.asarray(img_small, dtype="float32") / 255.0
    x = np.expand_dims(x, axis=0)

    out = model.predict(x, verbose=0)[0]
    out = np.clip(out * 255.0, 0, 255).astype("uint8")

    result = Image.fromarray(out).resize(orig_size, Image.Resampling.LANCZOS)
    return result
