"""
Coloriseur U-Net - espace Lab - STL-10
========================================================
Usage :
  python colorizer.py --create  my_model
  python colorizer.py --train   my_model 20
  python colorizer.py --plot    my_model
  python colorizer.py --predict my_model path/to/image.jpg
"""

import os
import argparse
import pickle
from typing import List, Dict, Tuple, Any, cast

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
from keras import layers, models
from keras.callbacks import ModelCheckpoint, EarlyStopping
from skimage.color import rgb2lab, lab2rgb

# ─────────────────────────────────────────────────────────────
# 0. CONFIGURATION GLOBALE
# ─────────────────────────────────────────────────────────────

IMG_SIZE   = 128
FILTERS    = (64, 128, 256, 512)
BATCH_SIZE = 128
TRAIN_SIZE = 90000
EPOCHS     = 20

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
HISTORY_PATH = os.path.join(SCRIPT_DIR, "history.pkl")

# ─────────────────────────────────────────────────────────────
# 1. DATASET
# ─────────────────────────────────────────────────────────────

def build_dataset(
    batch_size: int = BATCH_SIZE,
    img_size:   int = IMG_SIZE,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Construit les datasets d'entraînement et de validation depuis STL-10."""
    print("Chargement de STL-10...")
    ds = cast(
        tf.data.Dataset,
        tfds.load("stl10", split="unlabelled", as_supervised=False)
    )

    def process_image(features: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        """Prétraite une image RGB et la convertit en espace colorimétrique Lab normalisé."""
        img = tf.image.resize(features["image"], [img_size, img_size])
        img = img / 255.0

        mask_rgb = img > 0.04045
        img_lin = tf.where(mask_rgb, tf.pow((img + 0.055) / 1.055, 2.4), img / 12.92)

        mat = tf.constant(
            [[0.412453, 0.212671, 0.019334],
            [0.357580, 0.715160, 0.119193],
            [0.180423, 0.072169, 0.950227]],
            dtype=tf.float32
        )
        xyz = tf.linalg.matmul(img_lin, mat)

        xyz_norm = xyz / tf.constant([0.950456, 1.0, 1.088754], dtype=tf.float32)
        mask_xyz = xyz_norm > 0.008856
        f_xyz = tf.where(mask_xyz, tf.pow(xyz_norm, 1.0/3.0), (7.787 * xyz_norm) + (16.0 / 116.0))

        x, y, z = tf.unstack(f_xyz, axis=-1)
        L = (116.0 * y) - 16.0
        a = 500.0 * (x - y)
        b = 200.0 * (y - z)

        L_norm = tf.expand_dims(L / 50.0 - 1.0, axis=-1)
        ab_norm = tf.stack([a, b], axis=-1) / 128.0

        return L_norm, ab_norm

    def augment_fn(l: tf.Tensor, ab: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Augmentation symétrique sur L et ab : flip + bruit gaussien."""
        seed = tf.random.uniform(shape=(), minval=0, maxval=2**31-1, dtype=tf.int32)
        l  = tf.image.stateless_random_flip_left_right(l,  seed=(seed, 0))
        ab = tf.image.stateless_random_flip_left_right(ab, seed=(seed, 0))

        ab = ab + tf.random.normal(tf.shape(ab), stddev=0.02)
        ab = tf.clip_by_value(ab, -1.0, 1.0)

        l = tf.image.random_brightness(l, 0.15)
        l = tf.clip_by_value(l + tf.random.normal(tf.shape(l), stddev=0.02), -1.0, 1.0)
        return l, ab
    
    
    print("Préparation du mapping et de l'augmentation...")
    ds = ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(10000, reshuffle_each_iteration=False)
    train_ds = ds.take(90000)
    val_ds = ds.skip(90000)

    train_ds = train_ds.shuffle(5000, reshuffle_each_iteration=True)
    train_ds = train_ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    print("Dataset prêt")
    return train_ds, val_ds

# ─────────────────────────────────────────────────────────────
# 2. MODÈLE
# ─────────────────────────────────────────────────────────────

def colorization_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Fonction de coût pour la colorisation d’images en espace Lab (canaux ab)."""
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    desat_penalty = tf.reduce_mean(tf.exp(-tf.square(y_pred) * 20))
    variance = tf.reduce_mean(tf.math.reduce_variance(y_pred, axis=[1, 2]))
    return mae + 0.05 * desat_penalty - 0.05 * variance

def _conv_block(
        x: Any,
        filters: int,
        dropout_rate: float = 0.1,
        name: str = ""
    ) -> tf.Tensor:
    """Bloc convolutionnel standard composé de deux convolutions 2D avec normalisation,
    activation ReLU et dropout optionnel."""
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_c1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_r1")(x)
    if dropout_rate:
        x = layers.Dropout(dropout_rate, name=f"{name}_drop")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_c2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_r2")(x)
    return x

def _attention_block(x: tf.Tensor, name: str = "") -> tf.Tensor:
    """Bloc d'attention par compression de canal"""
    gap = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)

    channels = cast(int, x.shape[-1])

    gap = layers.Dense(channels // 4, activation="relu", name=f"{name}_fc1")(gap)
    gap = layers.Dense(channels, activation="sigmoid", name=f"{name}_fc2")(gap)
    gap = layers.Reshape((1, 1, channels), name=f"{name}_rs")(gap)
    return layers.Multiply(name=f"{name}_mul")([x, gap])

def build_unet(
        img_size: int = IMG_SIZE,
        filters: tuple = FILTERS,
        epochs: int = EPOCHS
    ) -> models.Model:
    """Construit et compile l'architecture U-Net pour la colorisation"""
    inp = layers.Input(shape=(img_size, img_size, 1), name="L_input")
    skips: List[tf.Tensor] = []
    x = inp
    
    for i, f in enumerate(filters):
        x = _conv_block(x, f, dropout_rate=0.2, name=f"enc{i}")
        skips.append(x)
        x = layers.MaxPooling2D(2, name=f"pool{i}")(x)
        
    x = _conv_block(x, filters[-1]*2, dropout_rate=0.3, name="bottleneck")
    x = _attention_block(x, name="att")
    
    for i, f in enumerate(reversed(filters)):
        x = layers.Conv2DTranspose(f, 2, strides=2, padding="same", name=f"up{i}")(x)
        x = layers.Concatenate(name=f"skip{i}")([x, skips[-(i+1)]])
        x = _conv_block(x, f, dropout_rate=0.2, name=f"dec{i}")
        
    out = layers.Conv2D(2, 1, activation="tanh", name="ab_output")(x)
    model = models.Model(inp, out, name="UNet_Colorizer")
    
    total_steps = (TRAIN_SIZE // BATCH_SIZE) * epochs
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=3e-4,
        decay_steps=total_steps,
        alpha=1e-6
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        loss=colorization_loss,
        metrics=["mae"],
    )
    return model


# ─────────────────────────────────────────────────────────────
# 3. CLASSE Colorizer
# ─────────────────────────────────────────────────────────────

class Colorizer:
    """Classe pour encapsuler le cycle de vie du modèle de colorisation"""
    def __init__(self, img_size: int = IMG_SIZE):
        self.img_size = img_size
        self.model: models.Model | None = None

    def create(self, epochs: int = EPOCHS) -> None:
        """Instancie et compile un nouveau U-Net."""
        self.model = build_unet(self.img_size, FILTERS, epochs)
        self.model.summary()
        print(f"\nModèle U-Net créé ({self.img_size}x{self.img_size}, filtres={FILTERS})")

    def save(self, path: str) -> None:
        """Sauvegarde le modèle au format Keras (.keras)."""
        if self.model is None:
            raise RuntimeError("Aucun modèle à sauvegarder.")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)

        self.model.save(path)
        print(f"Modèle sauvegardé → {path}")

    def load(self, path: str) -> None:
        """Charge un modèle Keras à partir de son chemin"""
        loaded = models.load_model(
            path,
            custom_objects={"colorization_loss": colorization_loss},
        )
        self.model = cast(models.Model, loaded)
        self.img_size = self.model.input_shape[1]
        print(f"Modèle {path} chargé")

    def train(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        epochs: int,
        save_path: str,
        history_path: str
    ) -> None:
        """Entraîne le modèle et accumule l'historique dans un fichier Pickle."""
        if self.model is None:
            raise RuntimeError("Créez ou chargez un modèle d'abord.")

        callbacks = [
            ModelCheckpoint(save_path, monitor="val_loss", save_best_only=True, save_weights_only=False, verbose=1),
            EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        ]

        print(f"\n─── Entraînement : {epochs} époques max ───")
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
        )
        print("─── Terminé ───\n")

        new_history = history.history
        if os.path.exists(history_path):
            with open(history_path, "rb") as f:
                old_history = pickle.load(f)
            for key in new_history:
                if key in old_history:
                    old_history[key] += new_history[key]
            new_history = old_history

        with open(history_path, "wb") as f:
            pickle.dump(new_history, f)
        print(f"Historique sauvegardé dans {history_path} ({len(new_history['loss'])} époques au total)")

    def predict(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Colorise une image à partir de son canal de luminance."""
        if self.model is None:
            raise RuntimeError("Chargez un modèle d'abord.")

        img_orig  = Image.open(image_path).convert("RGB")
        img_small = img_orig.resize((self.img_size, self.img_size))
        img_array = np.array(img_small) / 255.0
        
        lab = rgb2lab(img_array).astype("float32")
        L   = (lab[..., :1] / 50.0 - 1.0)[np.newaxis]
        
        ab_pred = self.model.predict(L, verbose="0")[0] * 128
        lab_out = np.concatenate([lab[..., :1], ab_pred], axis=-1)
        
        rgb_pred = np.array(
            Image.fromarray((lab2rgb(lab_out) * 255).astype("uint8")).resize(
                img_orig.size, Image.Resampling.LANCZOS
            )
        )
        
        lab_orig  = rgb2lab(np.array(img_orig) / 255.0)
        gray_full = (lab_orig[..., 0] / 100.0 * 255).clip(0, 255).astype("uint8")

        return gray_full, rgb_pred

    def plot_history(
        self,
        history_dict: Dict[str, List[float]],
        out_path: str = "courbe_apprentissage.png"
    ) -> None:
        """Trace et sauvegarde les courbes d'apprentissage."""
        mae_key     = next((k for k in history_dict if "mae" in k and "val" not in k), None)
        val_mae_key = next((k for k in history_dict if "mae" in k and "val"     in k), None)

        pairs = [("loss", "val_loss", "Loss (colorization)")]
        if mae_key and val_mae_key:
            pairs.append((mae_key, val_mae_key, "MAE"))

        fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 4))
        if len(pairs) == 1:
            axes = [axes]

        for ax, (train_key, val_key, title) in zip(axes, pairs):
            ax.plot(history_dict[train_key], label="Train", linewidth=2)
            ax.plot(history_dict[val_key],   label="Val",   linewidth=2, linestyle="--")
            ax.set_title(title)
            ax.set_xlabel("Époque")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.6)

        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        print(f"Courbes sauvegardées => {out_path}")
        plt.show()


# ─────────────────────────────────────────────────────────────
# 4. CLI
# ─────────────────────────────────────────────────────────────

def _ensure_keras(name: str) -> str:
    """Convertit un nom de modèle en chemin valide et s'assure qu'il est lisible par Keras"""
    if not name.endswith(".keras"):
        name = name + ".keras"
    if os.sep not in name and "/" not in name:
        name = os.path.join(MODELS_DIR, name)
    return name

def main():
    """Point d'entrée principal de l'interface en ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Coloriseur U-Net (Lab) - STL-10",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--create", metavar="MODEL")
    parser.add_argument("--train", nargs="+",  metavar="ARG", help="--train MODEL [EPOCHS=20]")
    parser.add_argument("--plot", metavar="MODEL")
    parser.add_argument("--predict", nargs=2, metavar=("MODEL", "IMAGE"))
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        return

    colorizer = Colorizer(img_size=IMG_SIZE)

    if args.create:
        path = _ensure_keras(args.create)
        epochs_for_schedule = EPOCHS
        if args.train and len(args.train) > 1:
            epochs_for_schedule = int(args.train[1])
            
        colorizer.create(epochs=epochs_for_schedule)
        colorizer.save(path)

    if args.train:
        path    = _ensure_keras(args.train[0])
        epochs  = int(args.train[1]) if len(args.train) > 1 else EPOCHS
        history_path = HISTORY_PATH

        if not os.path.exists(path):
            print(f"Erreur : '{path}' introuvable. Lancez --create d'abord.")
            print(f"        (Recherche effectuée dans : {os.path.dirname(path)})")
            return

        colorizer.load(path)
        train_ds, val_ds = build_dataset(
            batch_size=BATCH_SIZE,
            img_size=colorizer.img_size,
        )
        colorizer.train(train_ds, val_ds, epochs=epochs,
                        save_path=path, history_path=history_path)

    if args.plot:
        history_path = HISTORY_PATH
        if not os.path.exists(history_path):
            print(f"Erreur : historique '{history_path}' introuvable.")
            return
        with open(history_path, "rb") as f:
            history_data: dict = pickle.load(f)
        colorizer.plot_history(history_data)

    if args.predict:
        path, image_path = _ensure_keras(args.predict[0]), args.predict[1]
        for p, label in [(path, "modèle"), (image_path, "image")]:
            if not os.path.exists(p):
                print(f"Erreur : {label} '{p}' introuvable.")
                print(f"        (Recherche effectuée dans : {os.path.dirname(p)})")
                return

        colorizer.load(path)
        gray, colorized = colorizer.predict(image_path)
        
        img_orig = np.array(Image.open(image_path).convert("RGB"))

        is_color = not (np.allclose(img_orig[..., 0], img_orig[..., 1], atol=5) and 
                        np.allclose(img_orig[..., 1], img_orig[..., 2], atol=5))

        if is_color:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            axes[0].imshow(img_orig)
            axes[0].set_title("Original")
            axes[0].axis("off")
            
            axes[1].imshow(gray, cmap="gray")
            axes[1].set_title("Entrée (niveaux de gris)")
            axes[1].axis("off")
            
            axes[2].imshow(colorized)
            axes[2].set_title("Prédiction colorisée")
            axes[2].axis("off")
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            axes[0].imshow(img_orig)
            axes[0].set_title("Original (Déjà en niveaux de gris)")
            axes[0].axis("off")
            
            axes[1].imshow(colorized)
            axes[1].set_title("Prédiction colorisée")
            axes[1].axis("off")
            
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()