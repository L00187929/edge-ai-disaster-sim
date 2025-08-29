from flask import Flask, request, jsonify
from model import load, infer

app = Flask(__name__)
model = load()

@app.post("/predict")
def predict_route():
    data = request.get_json(silent=True) or {}
    text = data.get("text","")
    try:
        out = infer(model, text)
        return jsonify({"ok": True, "input": text, "prediction": out})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.get("/health")
def health():
    return {"ok": True}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
