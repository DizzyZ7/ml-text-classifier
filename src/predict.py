from pathlib import Path
import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.joblib"
INPUT_PATH = BASE_DIR / "data" / "input_texts.csv"
PREDICTIONS_PATH = BASE_DIR / "data" / "predictions.csv"


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    input_df = pd.read_csv(INPUT_PATH)

    if "text" not in input_df.columns:
        raise ValueError("input_texts.csv must contain a 'text' column")

    input_df["text"] = input_df["text"].astype(str)

    model = joblib.load(MODEL_PATH)

    predictions = model.predict(input_df["text"])

    result_df = input_df.copy()
    result_df["predicted_label"] = predictions

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(PREDICTIONS_PATH, index=False)

    print(f"Predictions saved to: {PREDICTIONS_PATH}")
    print(f"Processed {len(result_df)} texts")


if __name__ == "__main__":
    main()
