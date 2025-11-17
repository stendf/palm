import os
import uuid
from pathlib import Path
from flask import Flask, request, jsonify

from run_pipeline import run_pipeline_api

# ============================================================
# 1) CONFIG
# ============================================================

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "best_model.pth"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1CcG7idf0yIbj697E7_JoBeP_BNJ6_KIi"

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True, parents=True)

# ============================================================
# 2) DOWNLOAD AUTOMATICO MODELLO (SOLO SE MANCANTE)
# ============================================================
def ensure_model_present():
    if MODEL_PATH.exists():
        print("✅ Model already exists.")
        return

    import requests

    print("📥 Downloading model from Google Drive...")
    r = requests.get(MODEL_URL, allow_redirects=True)

    if r.status_code != 200:
        raise RuntimeError("❌ Failed to download the model from Google Drive.")

    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

    print("✅ Model downloaded and saved.")

# Scarica all'avvio
ensure_model_present()

# ============================================================
# 3) FLASK APP
# ============================================================

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return {"status": "Palm Line API running"}, 200


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Endpoint principale:
    - riceve immagine via POST multipart/form-data
    - salva in uploads/
    - esegue la pipeline
    - ritorna JSON
    """
    if "image" not in request.files:
        return jsonify({"error": "Missing file 'image'"}), 400

    file = request.files["image"]

    # genera nome univoco
    ext = Path(file.filename).suffix or ".jpg"
    filename = f"{uuid.uuid4().hex}{ext}"
    save_path = UPLOAD_FOLDER / filename

    # salva file
    file.save(save_path)

    # esegue la pipeline
    try:
        result = run_pipeline_api(save_path)
        return jsonify(result)

    except Exception as e:
        print("❌ ERROR:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
