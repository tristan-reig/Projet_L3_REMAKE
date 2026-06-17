"""Chargement paresseux (lazy) et mise en cache des modèles Keras.

Les modèles ne sont chargés en mémoire qu'au premier appel, puis réutilisés.
"""

from backend.core import config

import keras
from typing import cast

_CACHE: dict[str, keras.Model] = {}


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
        _CACHE[name] = cast(keras.Model, keras.models.load_model(path))

    return _CACHE[name]


def available_models() -> list[str]:
    """Liste les modèles dont le fichier .keras est réellement présent."""
    return [name for name, path in config.MODEL_PATHS.items() if path.exists()]
