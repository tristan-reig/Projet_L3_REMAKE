"""CNN maison pour la classification multi-classes (Imagenette, 10 classes)."""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras
from keras import layers, models

IMG_SIZE = 160
NUM_CLASSES = 10

CLASS_NAMES = [
    "tanche",
    "springer anglais",
    "lecteur cassette",
    "tronçonneuse",
    "église",
    "cor d'harmonie",
    "camion poubelle",
    "pompe à essence",
    "balle de golf",
    "parachute",
]


def _conv_block(x, filters: int, name: str):
    """Bloc convolutif : 2x (Conv 3x3 + BN + ReLU) puis MaxPooling."""
    for i in (1, 2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_c{i}")(x)
        x = layers.BatchNormalization(name=f"{name}_bn{i}")(x)
        x = layers.Activation("relu", name=f"{name}_r{i}")(x)
    x = layers.MaxPooling2D(2, name=f"{name}_pool")(x)
    return x


def build_cnn(img_size: int = IMG_SIZE, num_classes: int = NUM_CLASSES) -> models.Model:
    """CNN maison à 4 blocs convolutifs + tête de classification.

    Filtres : 32 → 64 → 128 → 256.
    """
    inp = layers.Input(shape=(img_size, img_size, 3), name="image")

    # Normalisation des pixels [0,255] -> [0,1]
    x = layers.Rescaling(1.0 / 255, name="rescale")(inp)

    x = _conv_block(x, 32, name="block1")
    x = _conv_block(x, 64, name="block2")
    x = _conv_block(x, 128, name="block3")
    x = _conv_block(x, 256, name="block4")

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    x = layers.Dense(256, activation="relu", name="fc")(x)
    out = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return models.Model(inp, out, name="CNN_Classifier")


def compile_model(model: models.Model, initial_lr: float = 1e-3) -> models.Model:
    """Compile avec Adam + perte d'entropie croisée catégorielle."""
    model.compile(
        optimizer=keras.optimizers.Adam(initial_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
