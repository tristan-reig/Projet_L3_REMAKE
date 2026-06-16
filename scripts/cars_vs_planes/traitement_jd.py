import os   
import fiftyone as fo  
import fiftyone.zoo as foz                       
from PIL import Image
from tqdm import tqdm



def telecharger_jeu_equilibre(classes, nb_images_par_classe):
    echantillon_train = []
    echantillon_val = []

    for classe in classes:
        print(f"Téléchargement de {nb_images_par_classe} images pour la classe : {classe}...")

        nom_collection_echantillons = f"open-images-{classe.lower()}-{nb_images_par_classe}"
        
        if fo.dataset_exists(nom_collection_echantillons):
            fo.delete_dataset(nom_collection_echantillons)
            
        jd = foz.load_zoo_dataset(
            "open-images-v7",
            split="train",
            label_types=["detections"],
            classes=[classe],
            max_samples=nb_images_par_classe,
            shuffle=True,
            dataset_name=nom_collection_echantillons
        )
        
        all_ids = jd.values("id")
        split_idx = int(len(all_ids) * 0.8)
        
        echantillon_train.extend(jd[:split_idx].select_fields("ground_truth"))
        echantillon_val.extend(jd[split_idx:].select_fields("ground_truth"))

    jd_train = fo.Dataset("train_equilibre")
    jd_train.add_samples(echantillon_train)
    jd_train.shuffle()

    jd_val = fo.Dataset("val_equilibre")
    jd_val.add_samples(echantillon_val)
    jd_val.shuffle()
    
    return jd_train, jd_val



def redimensionner_jeu_organise(classes_cibles, collection_echantillons, repertoire, taille, max_par_classe):
    print(f"\nTraitement vers : {repertoire}")
    
    compteurs = {c: 0 for c in classes_cibles}

    for echantillon in tqdm(collection_echantillons):
        try:
            if not echantillon.has_field("ground_truth") or echantillon.ground_truth is None:
                continue

            detection_valide = None

            for det in echantillon.ground_truth.detections:
                if det.label in classes_cibles and compteurs[det.label] < max_par_classe:
                    detection_valide = det
                    break 
            
            if not detection_valide:
                continue

            etiquette = detection_valide.label
            output_subfolder = os.path.join(repertoire, etiquette)
            os.makedirs(output_subfolder, exist_ok=True)

            filename = os.path.basename(echantillon.filepath)
            output_path = os.path.join(output_subfolder, filename)
            
            if not os.path.exists(output_path):
                with Image.open(echantillon.filepath) as img:
                    img = img.convert("RGB")
                    largeur, hauteur = img.size
                    
                    bbox = detection_valide.bounding_box
                    
                    gauche = max(0, bbox[0]*largeur) 
                    haut = max(0, bbox[1]*hauteur)
                    droite = min(largeur, (bbox[0]+bbox[2])*largeur)
                    bas = min(hauteur, (bbox[1]+bbox[3])*hauteur)
                    
                    img_rognee = img.crop((gauche, haut, droite, bas))
                    img_redimensionnee = img_rognee.resize(taille, Image.Resampling.LANCZOS)
                    img_redimensionnee.save(output_path, "JPEG", quality=90)
                    
                    compteurs[etiquette] += 1

            if all(c >= max_par_classe for c in compteurs.values()):
                break

        except Exception:
            continue

    print(f"Terminé : {compteurs} images créées dans {repertoire}")
