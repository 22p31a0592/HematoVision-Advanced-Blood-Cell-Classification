from flask import Flask, render_template, request
from your_model import classify_image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def Hom():
    return render_template("Home.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return redirect(request.url)

    file = request.files["image"]
    if file.filename == "":
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join("static", filename)
        file.save(filepath)

        prediction = classify_image(filepath)
        image_url = f"/static/{filename}"
        return render_template("result.html", prediction=prediction, image_url=image_url)

if __name__ == "__main__":
    app.run(debug=True)
