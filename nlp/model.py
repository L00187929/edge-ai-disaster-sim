from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

def dataset():
    X = [
        "please help trapped in house water rising need rescue",
        "urgent rescue needed at north bridge flooding rising fast",
        "we need food blankets and water at shelter",
        "bridge collapsed road blocked major structural damage",
        "power lines down substation fire structural damage",
        "donating blankets food and water available",
        "we can offer shelter volunteers available",
        "sunny day no issues here",
        "watching football tonight nothing urgent",
    ]
    y = [
        "request_help",
        "request_help",
        "request_help",
        "infrastructure_damage",
        "infrastructure_damage",
        "donation_offer",
        "donation_offer",
        "other",
        "other",
    ]
    return X, y

def train(path="nlp_model.joblib"):
    X, y = dataset()
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1)),
        ("clf", LogisticRegression(max_iter=300))
    ])
    pipe.fit(X, y)
    joblib.dump(pipe, path)
    return path

def load(path="nlp_model.joblib"):
    try:
        return joblib.load(path)
    except:
        train(path)
        return joblib.load(path)

def infer(model, text: str):
    if not text or not text.strip():
        return {"label": "other", "confidence": 0.0}
    proba = model.predict_proba([text])[0]
    idx = proba.argmax()
    return {"label": model.classes_[idx], "confidence": float(proba[idx])}
