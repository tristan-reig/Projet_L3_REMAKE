from traitement_jd import *
from creer_modele import *
import tensorflow as tf

print("Nombre de GPU disponibles :", len(tf.config.list_physical_devices('GPU')))

print("Chargement des datasets Keras...")
donnees_entrainement = tf.keras.utils.image_dataset_from_directory(
    'dataset_final/train',
    labels='inferred',
    label_mode='int',
    image_size=(128, 128),
    batch_size=32,
    shuffle=True
)

donnees_test = tf.keras.utils.image_dataset_from_directory(
    'dataset_final/val',
    labels='inferred',
    label_mode='int',
    image_size=(128, 128),
    batch_size=32,
    shuffle=False
)


donnees_entrainement = donnees_entrainement.prefetch(buffer_size=tf.data.AUTOTUNE)
donnees_test = donnees_test.prefetch(buffer_size=tf.data.AUTOTUNE)

definition_modele = definirModele()
instanciation_modele = instancierModele(definition_modele[0],definition_modele[1])
creation_optimiseur = creerOptimiseur(0.0001)
compilation_modele = compilerModele(instanciation_modele,creation_optimiseur,'sparse_categorical_crossentropy','accuracy')

print("Début de l'entraînement.")
entrainement = entrainerModele(compilation_modele,donnees_entrainement,20,donnees_test)

nom_modele = input('Saisissez le nom du fichier .keras pour le sauvegarder : ')
nom_dossier = './modeles'

os.makedirs(nom_dossier, exist_ok=True)
chemin_sauvegarde = os.path.join(nom_dossier, nom_modele)

compilation_modele.save(nom_modele)
print(f"Le modèle a été sauvegardé sous le nom {nom_modele} !")

afficherHistorique(entrainement)
