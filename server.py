from flask import Flask, request, jsonify
from pathlib import Path
import uuid
import os

from run_pipeline import run_pipeline_api

app = Flask(__name__)

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True, parents=True)


@app.route("/", methods=["GET"])
def home():
    return {"status": "Palm Line API running"}, 200


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Endpoint principale:
    - riceve un file immagine via POST multipart/form-data
    - lo salva in uploads/
    - esegue la pipeline
    - ritorna il JSON completo
    """
    if "image" not in request.files:
        return jsonify({"error": "Missing file 'image'"}), 400

    file = request.files["image"]

    # genera nome univoco
    ext = Path(file.filename).suffix or ".jpg"
    filename = f"{uuid.uuid4().hex}{ext}"

    save_path = UPLOAD_FOLDER / filename
    file.save(save_path)

    # esegue la pipeline
    try:
        result = run_pipeline_api(save_path)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Debug su localhost
    app.run(host="0.0.0.0", port=5000, debug=True)
