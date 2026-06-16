from traitement_jd import *
import os

chemin_base = "./dataset_final"

print(f"Tentative de création manuelle de : {chemin_base}")
try:
    os.makedirs(chemin_base, exist_ok=True)
    print("Le dossier racine est prêt.")
except Exception as e:
    print(f"Erreur lors de la création du dossier: {e}.")

if os.path.exists(chemin_base):
    train_view, val_view = telecharger_jeu_equilibre(["Airplane", "Car"], 10000)
    redimensionner_jeu_organise(["Airplane", "Car"], train_view, os.path.join(chemin_base, "train"), (128, 128), 8000)
    redimensionner_jeu_organise(["Airplane", "Car"], val_view, os.path.join(chemin_base, "val"), (128, 128), 2000)
else:
    print("Impossible de trouver le dossier cible.")
