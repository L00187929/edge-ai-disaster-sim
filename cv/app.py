from flask import Flask, request, jsonify
import numpy as np
import cv2
from infer import analyse

app = Flask(__name__)

@app.post("/predict")
def predict_route():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "image file missing"}), 400
    file = request.files["image"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"ok": False, "error": "invalid image"}), 400
    try:
        out = analyse(img)
        return jsonify({"ok": True, "prediction": out})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.get("/health")
def health():
    return {"ok": True}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
