"""
Classifieur Chiens/Chats — CNN personnalisé
========================================================
Usage :
  python classifier.py --create  my_model.keras
  python classifier.py --train   my_model.keras 10
  python classifier.py --plot    my_model.keras
  python classifier.py --predict my_model.keras path/to/image.jpg
"""


import os
import warnings
import argparse
import pickle
import random

import numpy as np
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import tensorflow as tf
import keras
from keras import layers, regularizers
from keras.utils import load_img, img_to_array
from typing import List, Dict, Tuple, cast

# ─────────────────────────────────────────────────────────────
# 0. CONFIGURATION GLOBALE
# ─────────────────────────────────────────────────────────────

IMAGE_SIZE   = (128, 128)
BATCH_SIZE   = 32
DATASET_DIR  = "PetImages"
MODEL_PATH   = "models/model.keras"
HISTORY_PATH = "history.pkl"

GREEN = "\033[32m"
RED   = "\033[31m"
RESET = "\033[0m"

# ─────────────────────────────────────────────────────────────
# 1. DATASET
# ─────────────────────────────────────────────────────────────

def delete_corrupted_images() -> None:
    """
    Parcourt les sous-dossiers 'Cat' et 'Dog' de DATASET_DIR et supprime
    toute image dont l'en-tête ne contient pas la signature JFIF (JPEG valide).
    """
    num_skipped = 0
    if not os.path.exists(DATASET_DIR):
        print(f"Erreur: Le dossier '{DATASET_DIR}' est introuvable.")
        return

    for folder_name in ("Cat", "Dog"):
        folder_path = os.path.join(DATASET_DIR, folder_name)
        if not os.path.exists(folder_path): 
            continue
        
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                with open(fpath, "rb") as f:
                    is_jfif = b"JFIF" in f.peek(10)
            except:
                is_jfif = False

            if not is_jfif:
                num_skipped += 1
                os.remove(fpath)

    if num_skipped > 0:
        print(f"Images corrompues supprimées : {num_skipped}")
    else:
        print("Aucune image corrompue trouvée.")

def build_datasets() -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Charge les images depuis DATASET_DIR et les répartit en trois jeux de données :

    - 80 % pour l'entraînement (train)
    - 10 % pour la validation (val)
    - 10 % pour le test final (test)
    """
    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"Le dossier {DATASET_DIR} est introuvable !")

    print(f"Chargement et split des données depuis : {DATASET_DIR} ...")
    
    train_ds = cast(tf.data.Dataset, keras.utils.image_dataset_from_directory(
        DATASET_DIR, validation_split=0.2, subset="training", seed=1337,
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE
    ))

    val_and_test_ds = cast(tf.data.Dataset, keras.utils.image_dataset_from_directory(
        DATASET_DIR, validation_split=0.2, subset="validation", seed=1337,
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE
    ))

    val_batches = tf.data.experimental.cardinality(val_and_test_ds)
    test_ds = val_and_test_ds.take(val_batches // 2)
    val_ds  = val_and_test_ds.skip(val_batches // 2)

    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_ds  = test_ds.cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds

# ─────────────────────────────────────────────────────────────
# 2. MODÈLE
# ─────────────────────────────────────────────────────────────

def build_model(input_shape: Tuple[int, int, int]) -> keras.Model:
    """
    Construit et retourne le CNN de classification binaire (Chat / Chien).

    Architecture :
        - Augmentation des données (flip, rotation, zoom, contraste)
        - Normalisation des pixels (rescaling [0, 255] → [0, 1])
        - 3 blocs Conv2D + MaxPooling (32, 64, 128 filtres)
        - GlobalAveragePooling2D
        - Dense 128 avec régularisation L2 + Dropout 0.5
        - Sortie sigmoid (1 neurone)
    """
    inputs = keras.Input(shape=input_shape)

    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")

    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(x)

    for filters in [32, 64, 128]:
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="PetClassifier")

# ─────────────────────────────────────────────────────────────
# 3. CLASSE PetClassifier
# ─────────────────────────────────────────────────────────────

class PetClassifier:
    """Classe pour encapsuler le cycle de vie du modèle de classification"""
    def __init__(self, model_path: str):
        self.model: keras.Model | None = None
        self.model_path = model_path

    def _get_model(self) -> keras.Model:
        if self.model is None:
            raise RuntimeError("Le modèle n'est pas chargé. Appelez load() ou créez-en un d'abord.")
        return self.model

    def load(self) -> None:
        """Charge le modèle depuis self.model_path et l'assigne à self.model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Le modèle {self.model_path} est introuvable. Lancez 'create' d'abord."
            )
        self.model = cast(keras.Model, keras.models.load_model(self.model_path))
        print(f"Modèle chargé depuis {self.model_path}")

    def save_history(self, new_history: Dict):
        """Sauvegarde l'historique d'entraînement en l'accumulant aux sessions précédentes."""
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, "rb") as f:
                old_history = pickle.load(f)
            for key in new_history:
                if key in old_history:
                    old_history[key] += new_history[key]
            new_history = old_history

        with open(HISTORY_PATH, "wb") as f:
            pickle.dump(new_history, f)
        print(f"Historique sauvegardé → {HISTORY_PATH}")

    def train_routine(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        epochs: int,
        learning_rate: float,
        patience: int,
        use_plateau: bool = False
    ) -> None:
        """Compile et entraîne le modèle, puis sauvegarde l'historique."""
        model = self._get_model()
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        callbacks: List[keras.callbacks.Callback] = [
            keras.callbacks.ModelCheckpoint(self.model_path, save_best_only=True, monitor="val_loss", verbose=1),
            keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True, verbose=1)
        ]

        if use_plateau:
            callbacks.append(
                keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)
            )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks
        )
        
        self.save_history(history.history)

    def evaluate(self, test_ds: tf.data.Dataset) -> None:
        """Évalue le modèle sur le jeu de test et affiche la loss et l'accuracy."""
        print("\n--- Evaluation avec le jeu de test inconnu ---")
        test_loss, test_acc = self._get_model().evaluate(test_ds)
        print(f"Résultat -> Loss: {test_loss:.4f} | Accuracy: {test_acc:.2%}")

    def plot_history(self) -> None:
        """
        Génère et sauvegarde les courbes d'évolution de la loss et de l'accuracy
        à partir de l'historique cumulé dans HISTORY_PATH.

        Le graphe est sauvegardé sous 'resultat_entrainement.png' puis affiché.
        """
        if not os.path.exists(HISTORY_PATH):
            print(f"Erreur: Historique '{HISTORY_PATH}' introuvable.")
            return

        with open(HISTORY_PATH, "rb") as f:
            hist: Dict = pickle.load(f)

        acc_key = "accuracy" if "accuracy" in hist else "acc"
        val_acc_key = "val_accuracy" if "val_accuracy" in hist else "val_acc"

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        if "loss" in hist:
            axes[0].plot(hist["loss"], label="Train", marker='o', markersize=4) 
        if "val_loss" in hist:
            axes[0].plot(hist["val_loss"], label="Val", marker='o', markersize=4, linestyle="--")
        axes[0].set_title("Évolution de l'Erreur (Loss)")
        axes[0].set_xlabel("Époques")
        axes[0].legend()
        axes[0].grid(True, linestyle="--", alpha=0.6)

        if acc_key in hist:
            axes[1].plot(hist[acc_key], label="Train", marker='o', markersize=4)
            if val_acc_key in hist:
                axes[1].plot(hist[val_acc_key], label="Val", marker='o', markersize=4, linestyle="--")
            axes[1].set_title("Évolution de la Précision (Accuracy)")
            axes[1].set_xlabel("Époques")
            axes[1].legend()
            axes[1].grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        filename = "resultat_entrainement.png"
        plt.savefig(filename, dpi=150)
        print(f"\nGraphe généré et sauvegardé sous : {filename}")
        plt.show()

    def predict_single(self, image_path: str | None = None) -> Tuple[str, str, str, float, bool]:
        """
        Effectue une prédiction sur une image unique.

        Si aucun chemin n'est fourni, une image est tirée aléatoirement depuis DATASET_DIR.
        La classe réelle est déduite du nom du fichier ('Cat' ou 'Dog').
        """
        if image_path is None:
            true_class = random.choice(["Cat", "Dog"])
            class_dir = os.path.join(DATASET_DIR, true_class)
            images = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            image_path = os.path.join(class_dir, random.choice(images))
        else:
            fname = os.path.basename(image_path)
            true_class = "Cat" if "Cat" in fname else "Dog"

        img = load_img(image_path, target_size=IMAGE_SIZE)
        img_array = img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)

        score: float = self._get_model().predict(img_batch, verbose="0")[0][0]
        pred_class = "Dog" if score > 0.5 else "Cat"
        correct = pred_class == true_class

        return image_path, true_class, pred_class, score, correct

# ─────────────────────────────────────────────────────────────
# 4. CLI
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Classifieur d'images Chiens/Chats")
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("--create", type=str, metavar="MODEL", 
                       help="Nettoie, crée et entraîne un nouveau modèle (10 époques)")
    group.add_argument("--train", nargs=2, metavar=("MODEL", "EPOCHS"), 
                       help="Reprend l'entraînement du modèle existant")
    group.add_argument("--plot", type=str, metavar="MODEL", 
                       help="Affiche les courbes d'apprentissage")
    group.add_argument("--predict", nargs="+", metavar=("MODEL", "IMAGE_PATH"), 
                       help="Teste le modèle sur une image spécifique ou au hasard")

    args = parser.parse_args()

    if args.create:
        model_path = args.create
        classifier = PetClassifier(model_path)
        
        delete_corrupted_images()
        
        if os.path.exists(model_path):
            print(f"Suppression de l'ancien modèle '{model_path}'.")
            os.remove(model_path)
        if os.path.exists(HISTORY_PATH):
            os.remove(HISTORY_PATH)

        train_ds, val_ds, test_ds = build_datasets()
        
        classifier.model = build_model(input_shape=IMAGE_SIZE + (3,))
        classifier.model.summary()
        
        print("\n--- Démarrage de l'entraînement initial ---")
        classifier.train_routine(train_ds, val_ds, epochs=10, learning_rate=1e-3, patience=4)
        classifier.evaluate(test_ds)

    elif args.train:
        model_path = args.train[0]
        epochs = int(args.train[1])
        classifier = PetClassifier(model_path)
        
        classifier.load()
        train_ds, val_ds, test_ds = build_datasets()
        
        print(f"\n--- Reprise de l'entraînement pour {epochs} époques ---")
        classifier.train_routine(train_ds, val_ds, epochs=epochs, learning_rate=1e-4, patience=5, use_plateau=True)
        classifier.evaluate(test_ds)

    elif args.plot:
        model_path = args.plot
        classifier = PetClassifier(model_path)
        classifier.plot_history()

    elif args.predict:
        model_path = args.predict[0]
        image_path = args.predict[1] if len(args.predict) > 1 else None
        
        classifier = PetClassifier(model_path)
        classifier.load()
        
        path, true_c, pred_c, score, correct = classifier.predict_single(image_path)
        symbol = f"{GREEN}Success{RESET}" if correct else f"{RED}Failure{RESET}"
        print(f"{true_c} → {pred_c} (Score: {score:.2f}) {symbol} | {path}")

if __name__ == "__main__":
    main()