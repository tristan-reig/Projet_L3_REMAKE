"""Autoencodeur U-Net pour le débruitage d'images."""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras
from keras import layers, models

IMG_SIZE = 128
FILTERS = (64, 128, 256, 512)

@keras.saving.register_keras_serializable(package="denoiser", name="psnr_metric")
def psnr_metric(y_true, y_pred):
    """PSNR (Peak Signal-to-Noise Ratio) : mesure la qualité de reconstruction."""
    mse = keras.ops.mean(keras.ops.square(y_true - y_pred))
    return 10.0 * keras.ops.log(1.0 / (mse + 1e-8)) / keras.ops.log(10.0)


def _conv_block(x, filters: int, name: str):
    """Deux convolutions 3x3 (BN + ReLU)."""
    for i in (1, 2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_c{i}")(x)
        x = layers.BatchNormalization(name=f"{name}_bn{i}")(x)
        x = layers.Activation("relu", name=f"{name}_r{i}")(x)
    return x

def build_denoiser(img_size: int = IMG_SIZE, filters: tuple = FILTERS) -> models.Model:
    """U-Net : encodeur contractant, décodeur expansif, skip connections."""
    inp = layers.Input(shape=(img_size, img_size, 3), name="noisy_input")
    skips = []
    x = inp

    # Encodeur
    for i, f in enumerate(filters):
        x = _conv_block(x, f, name=f"enc{i}")
        skips.append(x)
        x = layers.MaxPooling2D(2, name=f"pool{i}")(x)

    # Goulot
    x = _conv_block(x, filters[-1] * 2, name="bottleneck")

    # Décodeur avec skip connections
    for i, f in enumerate(reversed(filters)):
        x = layers.Conv2DTranspose(f, 2, strides=2, padding="same", name=f"up{i}")(x)
        x = layers.Concatenate(name=f"skip{i}")([x, skips[-(i + 1)]])
        x = _conv_block(x, f, name=f"dec{i}")

    out = layers.Conv2D(3, 1, activation="sigmoid", name="clean_output")(x)
    return models.Model(inp, out, name="UNet_Denoiser")

def compile_model(model: models.Model, initial_lr: float = 1e-3) -> models.Model:
    """Compile avec Adam, perte MAE (préserve mieux les contours que la MSE) + PSNR."""
    model.compile(
        optimizer=keras.optimizers.Adam(initial_lr),
        loss="mae",
        metrics=[psnr_metric],
    )
    return model
