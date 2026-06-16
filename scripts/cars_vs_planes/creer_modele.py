from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt
import keras

def definirModele():
    entree = layers.Input(shape=(128, 128,3))

    inversion = layers.RandomFlip("horizontal")(entree)
    rotation = layers.RandomRotation(0.1)(inversion)
    zoom = layers.RandomZoom(0.1)(rotation)

    redimensionner = layers.Rescaling(1./255)(zoom)

    convolution1 = layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu')(redimensionner)
    maxPool1 = layers.MaxPooling2D(pool_size=(2,2))(convolution1)

    convolution2 = layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu')(maxPool1)
    maxPool2 = layers.MaxPooling2D(pool_size=(2,2))(convolution2)

    convolution3 = layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu')(maxPool2)
    maxPool3 = layers.MaxPooling2D(pool_size=(2,2))(convolution3)

    aplatir = layers.Flatten()(maxPool3)

    sortie1 = layers.Dense(32, activation='relu')(aplatir)

    reguler = layers.Dropout(0.6)(sortie1)

    sortie2 = layers.Dense(2, activation='softmax')(reguler)

    return entree, sortie2



def instancierModele(couche_entree,couche_sortie):
    return keras.Model(inputs=couche_entree, outputs=couche_sortie)



def creerOptimiseur(pasApprentissage):
    return optimizers.Adam(learning_rate=pasApprentissage)



def compilerModele(modele, optimiseur, fonctionLoss, metrique1, metrique2=None, metrique3=None):
    liste_metriques = []
    liste_metriques.append(metrique1)

    if metrique2 is not None:
        liste_metriques.append(metrique2)
    if metrique3 is not None:
        liste_metriques.append(metrique3)

    modele.compile(optimizer=optimiseur, loss=fonctionLoss, metrics=liste_metriques)

    return modele

def entrainerModele(modele, donneesEntrainement, nbPassages, donneesTest):
    historique = modele.fit(
        donneesEntrainement, 
        epochs=nbPassages, 
        validation_data=donneesTest, 
        verbose=1
    )
    return historique


def afficherHistorique(historique):
    plt.subplot(1,2,1) 
    plt.plot(historique.history["accuracy"], label='Accuracy entrainement')
    plt.plot(historique.history["val_accuracy"], label='Accuracy validation')
    plt.xlabel('Époques')
    plt.legend()

    plt.subplot(1,2,2) 
    plt.plot(historique.history["loss"], label='Loss entrainement')
    plt.plot(historique.history["val_loss"], label='Loss validation')
    plt.xlabel('Époques')
    plt.legend()
    
    plt.tight_layout() 
    plt.show()