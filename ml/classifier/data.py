"""Pipeline de données Imagenette pour l'entraînement du classifieur."""

import tensorflow as tf
import tensorflow_datasets as tfds

IMG_SIZE = 160
BATCH_SIZE = 64

DATASET = "imagenette/320px-v2"

def _resize(image, label):
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return image, label

def _augment(image, label):
    """Augmentation légère : flip, variations de luminosité et contraste."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.15)
    image = tf.image.random_contrast(image, 0.85, 1.15)
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, label


def build_dataset(batch_size: int = BATCH_SIZE):
    """Retourne (train_ds, val_ds, num_classes) prêts pour model.fit()."""
    (train_ds, val_ds), info = tfds.load(
        DATASET,
        split=["train", "validation"],
        as_supervised=True,
        with_info=True,
    )
    num_classes = info.features["label"].num_classes

    train_ds = (
        train_ds.map(_resize, num_parallel_calls=tf.data.AUTOTUNE)
        .map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(2000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        val_ds.map(_resize, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, val_ds, num_classes
