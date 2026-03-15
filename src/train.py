from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "dataset.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
MODEL_INFO_PATH = MODEL_DIR / "model_info.json"


def build_models() -> dict:
    return {
        "LogisticRegression": Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=1000))
        ]),
        "MultinomialNB": Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", MultinomialNB())
        ]),
        "LinearSVC": Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", LinearSVC())
        ]),
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("dataset.csv must contain 'text' and 'label' columns")

    X = df["text"].astype(str)
    y = df["label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    models = build_models()

    best_model_name = None
    best_model = None
    best_score = -1.0
    all_scores = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = f1_score(y_test, y_pred, average="weighted")

        all_scores[model_name] = round(float(score), 4)

        if score > best_score:
            best_score = score
            best_model_name = model_name
            best_model = model

    if best_model is None or best_model_name is None:
        raise RuntimeError("Failed to select the best model")

    joblib.dump(best_model, MODEL_PATH)

    model_info = {
        "best_model": best_model_name,
        "best_f1_weighted": round(float(best_score), 4),
        "all_model_scores": all_scores
    }

    with open(MODEL_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)

    print(f"Best model: {best_model_name}")
    print(f"Best weighted F1: {best_score:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Model info saved to: {MODEL_INFO_PATH}")


if __name__ == "__main__":
    main()
