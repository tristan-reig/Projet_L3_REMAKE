"""Chargement paresseux (lazy) et mise en cache des modèles Keras.

Les modèles ne sont chargés en mémoire qu'au premier appel, puis réutilisés.
Cela évite de tout charger au démarrage et accélère le lancement de l'API.
"""

from backend.core import config

import keras

_CACHE: dict[str, keras.Model] = {}

def _custom_objects(name: str) -> dict:
    """Objets custom nécessaires au chargement de certains modèles."""
    if name == "colorizer":
        from ml.colorizer.model import colorization_loss

        return {"colorization_loss": colorization_loss}
    return {}

def get_model(name: str) -> keras.Model:
    """Retourne le modèle demandé, en le chargeant une seule fois."""
    if name not in config.MODEL_PATHS:
        raise KeyError(f"Modèle inconnu : {name!r}. Disponibles : {list(config.MODEL_PATHS)}")

    if name not in _CACHE:
        path = config.MODEL_PATHS[name]
        if not path.exists():
            raise FileNotFoundError(
                f"Fichier modèle introuvable : {path}. "
            )
        _CACHE[name] = keras.models.load_model(path, custom_objects=_custom_objects(name))

    return _CACHE[name]

def available_models() -> list[str]:
    """Liste les modèles dont le fichier .keras est réellement présent."""
    return [name for name, path in config.MODEL_PATHS.items() if path.exists()]
