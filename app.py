from flask import Flask, render_template, request
from keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

# Load model
model = load_model("lung_cancer_model.h5")

class_names = ["Malignant", "Benign", "Normal"]

def preprocess_image(img_path):
    img = Image.open(img_path)

    # Convert grayscale → RGB
    img = img.convert("RGB")

    # Resize to model input
    img = img.resize((128, 128))

    img_array = np.array(img).reshape(1, 128, 128, 3)
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    file_path = None

    if request.method == "POST":
        file = request.files['image']
        if file:

            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            img_array = preprocess_image(file_path)

            pred = model.predict(img_array)
            pred_class = np.argmax(pred)
            confidence = float(pred[0][pred_class] * 100)

            prediction = class_names[pred_class]

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           file_path=file_path)

if __name__ == "__main__":
    app.run(debug=True)
