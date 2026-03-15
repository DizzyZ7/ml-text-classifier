from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "model.joblib"
MODEL_INFO_PATH = BASE_DIR / "models" / "model_info.json"
REPORTS_DIR = BASE_DIR / "reports"
METRICS_PATH = REPORTS_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
EXAMPLES_PATH = REPORTS_DIR / "examples.csv"


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df["text"].astype(str)
    y = df["label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    model = joblib.load(MODEL_PATH)

    with open(MODEL_INFO_PATH, "r", encoding="utf-8") as f:
        model_info = json.load(f)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    metrics = {
        "best_model": model_info["best_model"],
        "best_f1_weighted": model_info["best_f1_weighted"],
        "all_model_scores": model_info["all_model_scores"],
        "accuracy": round(float(accuracy), 4),
        "precision_weighted": round(float(precision), 4),
        "recall_weighted": round(float(recall), 4),
        "f1_weighted": round(float(f1), 4),
        "labels": labels,
        "classification_report": report
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix ({model_info['best_model']})")

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()

    examples_df = pd.DataFrame({
        "text": X_test.values,
        "true_label": y_test.values,
        "predicted_label": y_pred
    })
    examples_df.to_csv(EXAMPLES_PATH, index=False)

    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")
    print(f"Examples saved to: {EXAMPLES_PATH}")


if __name__ == "__main__":
    main()
