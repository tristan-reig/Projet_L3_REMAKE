"""Inférence du coloriseur : luminance (N&B) -> image RGB colorisée."""

import numpy as np
from PIL import Image
from skimage.color import lab2rgb, rgb2lab


def colorize_image(model, pil_image: Image.Image, img_size: int = 128) -> Image.Image:
    """Colorise une image et la renvoie à sa résolution d'origine.

    Le modèle prédit les canaux ab à partir du canal L, on recombine avec le L
    d'origine puis on reconvertit en RGB. Le résultat est rééchantillonné à la
    taille initiale pour préserver les détails.
    """
    img_orig = pil_image.convert("RGB")
    orig_size = img_orig.size  # (largeur, hauteur)

    img_small = img_orig.resize((img_size, img_size))
    img_array = np.asarray(img_small, dtype="float32") / 255.0

    lab = rgb2lab(img_array).astype("float32")
    L = (lab[..., :1] / 50.0 - 1.0)[np.newaxis]  # normalisation + batch

    ab_pred = model.predict(L, verbose=0)[0] * 128.0
    lab_out = np.concatenate([lab[..., :1], ab_pred], axis=-1)

    rgb = (lab2rgb(lab_out) * 255).astype("uint8")
    result = Image.fromarray(rgb).resize(orig_size, Image.Resampling.LANCZOS)
    return result
