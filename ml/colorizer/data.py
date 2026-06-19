"""Pipeline de données Imagenette pour l'entraînement du classifieur.

Charge Imagenette directement depuis sa source (tarball d'images organisées en
dossiers par classe), SANS tensorflow_datasets — ce qui évite le conflit Protobuf
récurrent de tfds sur Colab. On s'appuie sur keras.utils.image_dataset_from_directory.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras
import tensorflow as tf

IMG_SIZE = 160
BATCH_SIZE = 64

DATASET_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"

SYNSET_TO_NAME = {
    "n01440764": "tanche",
    "n02102040": "springer anglais",
    "n02979186": "lecteur cassette",
    "n03000684": "tronçonneuse",
    "n03028079": "église",
    "n03394916": "cor d'harmonie",
    "n03417042": "camion poubelle",
    "n03425413": "pompe à essence",
    "n03445777": "balle de golf",
    "n03888257": "parachute",
}


def download_imagenette() -> str:
    """Télécharge et décompresse Imagenette, renvoie le chemin du dossier racine."""
    path = keras.utils.get_file(origin=DATASET_URL, extract=True)
    base = os.path.join(os.path.dirname(path), "imagenette2-320")
    if not os.path.isdir(base):
        for root, dirs, _ in os.walk(os.path.dirname(path)):
            if "train" in dirs and "val" in dirs:
                base = root
                break
    return base


def _augment(image, label):
    """Augmentation légère (flip, luminosité, contraste)"""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.15)
    image = tf.image.random_contrast(image, 0.85, 1.15)
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, label


def build_dataset(batch_size: int = BATCH_SIZE):
    """Retourne (train_ds, val_ds, class_names) prêts pour model.fit()."""
    base = download_imagenette()
    train_dir = os.path.join(base, "train")
    val_dir = os.path.join(base, "val")

    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        label_mode="int",
        shuffle=True,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
    )

    # Noms lisibles dans l'ordre des dossiers (= ordre des labels).
    synsets = train_ds.class_names
    class_names = [SYNSET_TO_NAME.get(s, s) for s in synsets]

    train_ds = train_ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names
