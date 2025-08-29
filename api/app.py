import os, time, json
from flask import Flask, request, jsonify, render_template
import requests
from sqlalchemy import create_engine, text

NLP_URL = os.getenv("NLP_URL", "http://localhost:5001/predict")
CV_URL  = os.getenv("CV_URL",  "http://localhost:5002/predict")
DB_PATH = os.getenv("DB_PATH", "events.db")

app = Flask(__name__, static_folder="static", template_folder="templates")

engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
with engine.begin() as conn:
    conn.exec_driver_sql("""
    CREATE TABLE IF NOT EXISTS events(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts INTEGER NOT NULL,
        nlp_label TEXT, nlp_conf REAL,
        cv_label TEXT, cv_conf REAL,
        impact TEXT, priority TEXT, score REAL
    )
    """)

def fuse(nlp, cv):
    weights = {"request_help":1.0,"infrastructure_damage":0.9,"donation_offer":0.6,"other":0.2}
    nlp_label = nlp.get("label","other")
    nlp_conf  = float(nlp.get("confidence",0.0))
    cv_label  = cv.get("label","unknown")
    cv_conf   = float(cv.get("confidence",0.0))

    impact = "low"
    if cv_label in ["fire","flood","damage"] and cv_conf >= 0.5:
        impact = "high" if cv_conf >= 0.7 else "medium"

    score = weights.get(nlp_label,0.2)*nlp_conf
    if impact == "high": score += 0.4
    elif impact == "medium": score += 0.2

    if score >= 1.0: priority = "immediate"
    elif score >= 0.6: priority = "elevated"
    else: priority = "normal"

    return {"score": round(score,3), "priority": priority, "impact": impact}

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/analyse")
def analyse_route():
    text_in = request.form.get("text","")
    image   = request.files.get("image", None)

    # NLP
    nlp_pred = {"label":"other","confidence":0.0}
    if text_in:
        r = requests.post(NLP_URL, json={"text": text_in}, timeout=10)
        nlp_pred = r.json().get("prediction", nlp_pred)

    # CV
    cv_pred = {"label":"unknown","confidence":0.0}
    if image:
        files = {"image": (image.filename, image.stream, image.mimetype)}
        r = requests.post(CV_URL, files=files, timeout=15)
        cv_pred = r.json().get("prediction", cv_pred)

    fused = fuse(nlp_pred, cv_pred)

    # Persist
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO events (ts, nlp_label, nlp_conf, cv_label, cv_conf, impact, priority, score)
            VALUES (:ts,:nl,:nc,:cl,:cc,:im,:pr,:sc)
        """), dict(ts=int(time.time()), nl=nlp_pred["label"], nc=nlp_pred["confidence"],
                   cl=cv_pred["label"], cc=cv_pred["confidence"],
                   im=fused["impact"], pr=fused["priority"], sc=fused["score"]))

    return jsonify({"ok": True, "result":{"nlp":nlp_pred,"cv":cv_pred,"decision":fused}})

@app.get("/events")
def events():
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT ts,nlp_label,nlp_conf,cv_label,cv_conf,impact,priority,score FROM events ORDER BY id DESC LIMIT 200")).all()
    data = [dict(ts=r.ts, nlp_label=r.nlp_label, nlp_conf=r.nlp_conf, cv_label=r.cv_label,
                 cv_conf=r.cv_conf, impact=r.impact, priority=r.priority, score=r.score) for r in rows]
    return jsonify({"ok": True, "events": data})

@app.get("/health")
def health():
    return {"ok": True}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
