import os
import io
import keras
from flask import Flask, render_template, jsonify, request, send_file
from scripts.cats_vs_dogs.classifier_cats_vs_dog import PetClassifier
from scripts.cars_vs_planes.predire import predire_vehicule
from scripts.colorizer.colorizer import Colorizer
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/classifier", methods=["GET", "POST"])
def classifier():
    return render_template("classifier.html")

@app.route("/class_predict", methods=["POST"])
def class_predict():
    if 'file' not in request.files:
        return jsonify({"error": "Pas d'images"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Pas d'images selectionnee"}), 400
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    model = request.form.get('model')
    modelpath = "models/modele_cars_VS_planes.keras"
    if(model == "pets"):
        m = PetClassifier("models/classifier_model.keras")
        m.load()
        try:
            result = m.predict_single(filepath)
            os.remove(filepath)
            return jsonify({"result": result[2], "score": (2*abs((result[3].item())-0.5))})
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": str(e)}), 500
    elif(model == "cars"):
        try:
            result, score = predire_vehicule(filepath,modelpath)
            os.remove(filepath)
            return jsonify({"result": result, "score":score})
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": str(e)}), 500


@app.route("/coloriseur", methods=["GET", "POST"])
def coloriseur():
    return render_template("coloriseur.html")

@app.route("/coloriser_predict", methods=["POST"])
def coloriser_predict():
    if 'file' not in request.files:
        return jsonify({"error": "Pas d'images"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Pas d'images selectionnee"}), 400
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    modelpath = "models/model.keras"
    m = Colorizer()
    m.load(modelpath)
    try:
        result = m.predict(filepath)[1]
        img = Image.fromarray(result.astype('uint8'))
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        buffer.seek(0)
        os.remove(filepath)
        return send_file(buffer, mimetype='image/jpeg')
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
