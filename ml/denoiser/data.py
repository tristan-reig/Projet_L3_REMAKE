"""Pipeline de données pour le débruiteur : DIV2K + dégradations composites."""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras
import tensorflow as tf

IMG_SIZE = 128
BATCH_SIZE = 32

DATA_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"

def download_div2k() -> str:
    """Télécharge et décompresse DIV2K, renvoie le dossier des images HR."""
    path = keras.utils.get_file(origin=DATA_URL, extract=True)
    base = os.path.join(os.path.dirname(path), "DIV2K_train_HR")
    if not os.path.isdir(base):
        for root, dirs, files in os.walk(os.path.dirname(path)):
            if any(f.endswith(".png") for f in files):
                base = root
                break
    return base

def _random_patch(image):
    """Extrait un patch carré aléatoire et le normalise dans [0, 1]."""
    image = tf.image.random_crop(image, [IMG_SIZE, IMG_SIZE, 3])
    return tf.cast(image, tf.float32) / 255.0

def _add_gaussian(x):
    sigma = tf.random.uniform([], 0.02, 0.2)
    return x + tf.random.normal(tf.shape(x), stddev=sigma)

def _add_salt_pepper(x):
    amount = tf.random.uniform([], 0.01, 0.08)
    r = tf.random.uniform(tf.shape(x))
    x = tf.where(r < amount / 2, 0.0, x)
    x = tf.where(r > 1.0 - amount / 2, 1.0, x)
    return x

def _add_poisson(x):
    lam = tf.random.uniform([], 10.0, 50.0)
    noisy = tf.random.poisson([], x * lam) / lam
    return noisy

def _add_speckle(x):
    sigma = tf.random.uniform([], 0.05, 0.25)
    return x + x * tf.random.normal(tf.shape(x), stddev=sigma)

def _add_blur(x):
    k = tf.constant([1.0, 2.0, 1.0]) / 4.0
    kx = tf.reshape(k, [3, 1, 1, 1]) * tf.reshape(k, [1, 3, 1, 1])
    kx = tf.tile(kx, [1, 1, 3, 1])
    x4 = x[tf.newaxis]
    blurred = tf.nn.depthwise_conv2d(x4, kx, [1, 1, 1, 1], "SAME")[0]
    return blurred

def _degrade(clean):
    """Applique 1 à 3 dégradations tirées au hasard sur le patch propre."""
    noisy = clean
    # Chaque type de bruit est activé aléatoirement (au moins un en moyenne).
    if tf.random.uniform([]) < 0.6:
        noisy = _add_gaussian(noisy)
    if tf.random.uniform([]) < 0.3:
        noisy = _add_salt_pepper(noisy)
    if tf.random.uniform([]) < 0.3:
        noisy = _add_poisson(noisy)
    if tf.random.uniform([]) < 0.3:
        noisy = _add_speckle(noisy)
    if tf.random.uniform([]) < 0.2:
        noisy = _add_blur(noisy)
    noisy = tf.clip_by_value(noisy, 0.0, 1.0)
    return noisy, clean

def _load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    return img


def build_dataset(batch_size: int = BATCH_SIZE, patches_per_image: int = 8):
    """Retourne (train_ds, val_ds) de paires (image_bruitée, image_propre)."""
    base = download_div2k()
    files = sorted(
        os.path.join(base, f) for f in os.listdir(base) if f.endswith(".png")
    )
    split = int(len(files) * 0.9)
    train_files, val_files = files[:split], files[split:]

    def make_ds(file_list, training):
        ds = tf.data.Dataset.from_tensor_slices(file_list)
        ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
        if training:
            ds = ds.repeat(patches_per_image)
        ds = ds.map(_random_patch, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(_degrade, num_parallel_calls=tf.data.AUTOTUNE)
        if training:
            ds = ds.shuffle(1000)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return make_ds(train_files, True), make_ds(val_files, False)
