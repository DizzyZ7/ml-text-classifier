from pathlib import Path
import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.joblib"
PREDICTIONS_PATH = BASE_DIR / "data" / "predictions.csv"


def main() -> None:
    model = joblib.load(MODEL_PATH)

    sample_texts = [
        "Мне очень понравился этот сервис",
        "Это худшее приложение из всех",
        "Сегодня у нас обычная рабочая встреча",
        "Качество отличное и всё работает быстро",
        "Система ведёт себя нестабильно и раздражает"
    ]

    predictions = model.predict(sample_texts)

    df = pd.DataFrame({
        "text": sample_texts,
        "predicted_label": predictions
    })

    df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Predictions saved to: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
