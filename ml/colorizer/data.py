"""Pipeline de données STL-10 pour l'entraînement du coloriseur."""

import tensorflow as tf
import tensorflow_datasets as tfds

IMG_SIZE = 128
BATCH_SIZE = 128
TRAIN_SIZE = 90000

def _process_image(features):
    """RGB -> Lab normalisé (L dans [-1,1], ab dans [-1,1] après /128)."""
    img = tf.image.resize(features["image"], [IMG_SIZE, IMG_SIZE]) / 255.0

    mask_rgb = img > 0.04045
    img_lin = tf.where(mask_rgb, tf.pow((img + 0.055) / 1.055, 2.4), img / 12.92)

    mat = tf.constant(
        [[0.412453, 0.212671, 0.019334],
         [0.357580, 0.715160, 0.119193],
         [0.180423, 0.072169, 0.950227]],
        dtype=tf.float32,
    )
    xyz = tf.linalg.matmul(img_lin, mat)
    xyz_norm = xyz / tf.constant([0.950456, 1.0, 1.088754], dtype=tf.float32)

    mask_xyz = xyz_norm > 0.008856
    f_xyz = tf.where(mask_xyz, tf.pow(xyz_norm, 1.0 / 3.0), (7.787 * xyz_norm) + (16.0 / 116.0))

    x, y, z = tf.unstack(f_xyz, axis=-1)
    L = (116.0 * y) - 16.0
    a = 500.0 * (x - y)
    b = 200.0 * (y - z)

    L_norm = tf.expand_dims(L / 50.0 - 1.0, axis=-1)
    ab_norm = tf.stack([a, b], axis=-1) / 128.0
    return L_norm, ab_norm


def _augment(l, ab):
    """Flip horizontal synchronisé + léger bruit gaussien."""
    seed = tf.random.uniform(shape=(), minval=0, maxval=2**31 - 1, dtype=tf.int32)
    l = tf.image.stateless_random_flip_left_right(l, seed=(seed, 0))
    ab = tf.image.stateless_random_flip_left_right(ab, seed=(seed, 0))

    ab = tf.clip_by_value(ab + tf.random.normal(tf.shape(ab), stddev=0.02), -1.0, 1.0)
    l = tf.image.random_brightness(l, 0.15)
    l = tf.clip_by_value(l + tf.random.normal(tf.shape(l), stddev=0.02), -1.0, 1.0)
    return l, ab


def build_dataset(batch_size: int = BATCH_SIZE, train_size: int = TRAIN_SIZE):
    """Retourne (train_ds, val_ds) prêts pour model.fit()."""
    ds = tfds.load("stl10", split="unlabelled", as_supervised=False)
    ds = ds.map(_process_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(10000, reshuffle_each_iteration=False)

    train_ds = ds.take(train_size).shuffle(5000, reshuffle_each_iteration=True)
    train_ds = train_ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = ds.skip(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds
