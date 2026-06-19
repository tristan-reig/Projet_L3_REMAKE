"""Architecture U-Net pour la colorisation (espace Lab, canal L -> canaux ab)."""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras
from keras import layers, models

IMG_SIZE = 128
FILTERS = (64, 128, 256, 512)

@keras.saving.register_keras_serializable(package="colorizer", name="colorization_loss")
def colorization_loss(y_true, y_pred):
    """Coût de colorisation sur les canaux ab.

    MAE + pénalité de désaturation (évite le gris) - bonus de variance (encourage
    des couleurs franches).
    """
    mae = keras.ops.mean(keras.ops.abs(y_true - y_pred))
    desat_penalty = keras.ops.mean(keras.ops.exp(-keras.ops.square(y_pred) * 20))
    variance = keras.ops.mean(keras.ops.var(y_pred, axis=[1, 2]))
    return mae + 0.05 * desat_penalty - 0.05 * variance

def _conv_block(x, filters: int, dropout_rate: float = 0.1, name: str = ""):
    """Deux convolutions 3x3 avec BatchNorm, ReLU et dropout optionnel."""
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_c1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_r1")(x)
    if dropout_rate:
        x = layers.Dropout(dropout_rate, name=f"{name}_drop")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_c2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_r2")(x)
    return x

def _attention_block(x, name: str = ""):
    """Attention par recalibration de canaux (squeeze-and-excitation)."""
    channels = int(x.shape[-1])
    gap = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    gap = layers.Dense(channels // 4, activation="relu", name=f"{name}_fc1")(gap)
    gap = layers.Dense(channels, activation="sigmoid", name=f"{name}_fc2")(gap)
    gap = layers.Reshape((1, 1, channels), name=f"{name}_rs")(gap)
    return layers.Multiply(name=f"{name}_mul")([x, gap])

def build_unet(img_size: int = IMG_SIZE, filters: tuple = FILTERS) -> models.Model:
    """Construit le modèle U-Net."""
    inp = layers.Input(shape=(img_size, img_size, 1), name="L_input")
    skips = []
    x = inp

    for i, f in enumerate(filters):
        x = _conv_block(x, f, dropout_rate=0.2, name=f"enc{i}")
        skips.append(x)
        x = layers.MaxPooling2D(2, name=f"pool{i}")(x)

    x = _conv_block(x, filters[-1] * 2, dropout_rate=0.3, name="bottleneck")
    x = _attention_block(x, name="att")

    for i, f in enumerate(reversed(filters)):
        x = layers.Conv2DTranspose(f, 2, strides=2, padding="same", name=f"up{i}")(x)
        x = layers.Concatenate(name=f"skip{i}")([x, skips[-(i + 1)]])
        x = _conv_block(x, f, dropout_rate=0.2, name=f"dec{i}")

    out = layers.Conv2D(2, 1, activation="tanh", name="ab_output")(x)
    return models.Model(inp, out, name="UNet_Colorizer")

def compile_model(
    model: models.Model,
    train_size: int,
    batch_size: int,
    epochs: int,
    initial_lr: float = 3e-4,
) -> models.Model:
    """Compile le modèle avec Adam + décroissance cosinus du learning rate."""
    total_steps = (train_size // batch_size) * epochs
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=total_steps,
        alpha=1e-6,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(lr_schedule),
        loss=colorization_loss,
        metrics=["mae"],
    )
    return model
