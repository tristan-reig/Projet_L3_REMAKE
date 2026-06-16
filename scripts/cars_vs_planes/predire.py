import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

def predire_vehicule(chemin_image, chemin_modele):
  
    modele = tf.keras.models.load_model(chemin_modele)
    
   
    img = tf.keras.utils.load_img(chemin_image, target_size=(128, 128))
 
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) 

    predictions = modele.predict(img_array)
 
    score = tf.nn.softmax(predictions[0])
 
    classes = ["Avion", "Voiture"]
    classe_predite = classes[np.argmax(score)]
    confiance = np.max(score).item()

    return classe_predite, confiance

    """print(f"Résultat : {classe_predite}")
    print(f"Confiance : {confiance:.2f}%")
    
    plt.imshow(img)
    plt.title(f"{classe_predite} ({confiance:.1f}%)")
    plt.axis('off')
    plt.show()"""


"""
nom_modele = input("Saisissez le chemin vers le modèle :")
nom_image = input("Saisissez le chemin vers l'image :")

if os.path.exists(nom_image):
    predire_vehicule(nom_image, nom_modele)
else:
    print(f"Erreur : Le fichier '{nom_image}' est introuvable. Vérifie le nom ou le chemin !")"""

