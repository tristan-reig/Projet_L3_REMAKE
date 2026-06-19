import numpy as np
from PIL import Image

from ml.classifier.model import CLASS_NAMES, IMG_SIZE

def classify_image(model, pil_image: Image.Image, top_k: int = 3) -> list[dict]:
    """Classe une image et renvoie les top_k prédictions triées par score."""
    img = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    x = np.expand_dims(np.asarray(img, dtype="float32"), axis=0)

    preds = model.predict(x, verbose=0)[0]
    top_idx = np.argsort(preds)[::-1][:top_k]

    return [
        {"classe": CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i),
         "score": float(preds[i])}
        for i in top_idx
    ]
