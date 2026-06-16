# Projet L3 : Classifieur & Coloriseur d'Images

Ce projet regroupe trois outils basés sur l'apprentissage profond (Deep Learning) : deux classifieurs d'images (chiens/chats et avions/voitures, via des réseaux convolutifs) et un coloriseur d'images en niveaux de gris (via une architecture U-Net). Une interface web Flask est également incluse pour tester les modèles de manière interactive.

## Prérequis et Installation

Voici les étapes pour préparer votre environnement et installer les dépendances nécessaires au fonctionnement du projet.

**1. Installation de l'environnement Python**

```sh
curl https://pyenv.run | bash
pyenv local 3.13.5
```

**2. Installation de Git LFS**

Le modèle de colorisation dépasse la taille standard autorisée par Git. Le projet utilise donc [Git LFS](https://git-lfs.com/) pour le stockage des fichiers volumineux. Il est nécessaire d'installer Git LFS avant de cloner le dépôt afin que les modèles soient correctement récupérés.

```sh
# Sur Debian/Ubuntu
sudo apt install git-lfs

# Sur macOS (via Homebrew)
brew install git-lfs

# Initialiser Git LFS
git lfs install
```

Si vous avez cloné le dépôt avant d'installer Git LFS, récupérez les fichiers volumineux avec :

```sh
git lfs pull
```

**3. Installation des dépendances Python et Node.js**

```sh
python3 -m pip install -r requirements.txt
npm install
```

**4. Téléchargement des jeux de données**

*Classifieur Chiens/Chats :*

```sh
curl -O https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip
```

> Note : Veillez à extraire l'archive et à placer le dossier `PetImages` à la racine du projet avant de lancer l'entraînement du classifieur.

*Classifieur Avions/Voitures :*

Le dataset est téléchargé dynamiquement depuis Open Images V7 via la bibliothèque FiftyOne. Il suffit de lancer le script dédié :

```sh
python scripts/cars_vs_planes/telecharger_jd.py
```

Le dossier `dataset_final/` contenant les images d'entraînement et de validation sera créé automatiquement à la racine du projet.

**5. Compilation du CSS (Tailwind)**

```sh
npx @tailwindcss/cli -i ./static/src/input.css -o ./static/dist/output.css --watch
```

-----

## Lancement de l'Application Web (Flask)

Pour utiliser l'interface graphique et tester les modèles de manière visuelle depuis votre navigateur, lancez le script principal :

```sh
python app.py
```

Accédez ensuite à l'adresse **[http://127.0.0.1:5000](http://127.0.0.1:5000)** dans votre navigateur web. Depuis cette interface, vous pourrez sélectionner le modèle souhaité, uploader vos images et visualiser directement les prédictions (pour la classification) ou les rendus (pour la colorisation).

-----

## Utilisation en Ligne de Commande (CLI)

Les modèles peuvent être créés, entraînés et évalués directement depuis votre terminal.

### 1\. Classifieur Chiens / Chats (`classifier_cats_vs_dog.py`)

Ce script gère le nettoyage automatique des données corrompues, l'entraînement du réseau de neurones et l'évaluation finale.

| Commande | Action | Exemple d'utilisation |
| :--- | :--- | :--- |
| `--create` | Nettoie les images corrompues, crée l'architecture et entraîne un nouveau modèle sur 10 époques. | `python classifier_cats_vs_dog.py --create classifier.keras` |
| `--train` | Reprend l'entraînement d'un modèle existant pour un nombre d'époques donné. | `python classifier_cats_vs_dog.py --train classifier.keras 15` |
| `--plot` | Génère, sauvegarde et affiche les courbes d'apprentissage (Loss/Accuracy). | `python classifier_cats_vs_dog.py --plot classifier.keras` |
| `--predict` | Teste le modèle sur une image. Si aucune image n'est spécifiée, une image du dataset est choisie au hasard. | `python classifier_cats_vs_dog.py --predict classifier.keras path/to/cat.jpg` |

### 2\. Classifieur Avions / Voitures

Ce classifieur est réparti en plusieurs scripts dans le dossier `scripts/cars_vs_planes/`, chacun avec un rôle dédié.

| Script | Action | Exemple d'utilisation |
| :--- | :--- | :--- |
| `telecharger_jd.py` | Télécharge et organise les images depuis Open Images V7 (8000 images d'entraînement et 2000 de validation par classe). | `python telecharger_jd.py` |
| `entrainer.py` | Crée l'architecture, entraîne le modèle sur 20 époques et sauvegarde le résultat. Affiche les courbes d'apprentissage à la fin. | `python entrainer.py` |
| `predire.py` | Charge un modèle existant et effectue une prédiction (Avion/Voiture) sur une image fournie au prompt. | `python predire.py` |

### 3\. Coloriseur U-Net (`colorizer.py`)

Ce script permet de coloriser des images en utilisant l'espace colorimétrique Lab. Il utilise par défaut le dataset STL-10 (téléchargé à la volée) mais supporte également les dossiers d'images personnalisés.

| Commande | Action | Exemple d'utilisation |
| :--- | :--- | :--- |
| `--create` | Instancie et sauvegarde un nouveau modèle U-Net non entraîné. | `python colorizer.py --create colorizer.keras` |
| `--train` | Entraîne le modèle. Utilisez l'argument `--data` pour cibler un dossier local au lieu de STL-10. | `python colorizer.py --train colorizer.keras 20` |
| `--plot` | Affiche l'historique de l'erreur absolue moyenne (MAE) et de la fonction de perte. | `python colorizer.py --plot colorizer.keras` |
| `--predict` | Applique la colorisation et affiche une comparaison visuelle (Original vs Niveaux de gris vs Prédiction). | `python colorizer.py --predict colorizer.keras path/to/image.jpg` |
