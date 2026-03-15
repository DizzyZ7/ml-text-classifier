from pathlib import Path
import json
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
README_PATH = BASE_DIR / "README.md"
METRICS_PATH = BASE_DIR / "reports" / "metrics.json"
EXAMPLES_PATH = BASE_DIR / "reports" / "examples.csv"


def replace_block(content: str, start_marker: str, end_marker: str, new_block: str) -> str:
    start_index = content.index(start_marker)
    end_index = content.index(end_marker) + len(end_marker)
    return content[:start_index] + new_block + content[end_index:]


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = []

    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines.append(header_row)
    lines.append(separator_row)

    for _, row in df.iterrows():
        values = [str(value).replace("\n", " ") for value in row.tolist()]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def main() -> None:
    readme = README_PATH.read_text(encoding="utf-8")

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    examples_df = pd.read_csv(EXAMPLES_PATH).head(5)

    model_scores_lines = "\n".join(
        [f"- {name}: **{score}**" for name, score in metrics["all_model_scores"].items()]
    )

    metrics_block = f"""<!-- METRICS_START -->
## Metrics

- Best Model: **{metrics['best_model']}**
- Accuracy: **{metrics['accuracy']}**
- Precision (weighted): **{metrics['precision_weighted']}**
- Recall (weighted): **{metrics['recall_weighted']}**
- F1-score (weighted): **{metrics['f1_weighted']}**
- Labels: **{', '.join(metrics['labels'])}**

### Model Comparison

{model_scores_lines}
<!-- METRICS_END -->"""

    examples_md = dataframe_to_markdown(examples_df)

    examples_block = f"""<!-- EXAMPLES_START -->
## Example Predictions

{examples_md}
<!-- EXAMPLES_END -->"""

    image_block = """<!-- IMAGE_START -->
## Confusion Matrix

![Confusion Matrix](reports/confusion_matrix.png)
<!-- IMAGE_END -->"""

    readme = replace_block(readme, "<!-- METRICS_START -->", "<!-- METRICS_END -->", metrics_block)
    readme = replace_block(readme, "<!-- EXAMPLES_START -->", "<!-- EXAMPLES_END -->", examples_block)
    readme = replace_block(readme, "<!-- IMAGE_START -->", "<!-- IMAGE_END -->", image_block)

    README_PATH.write_text(readme, encoding="utf-8")
    print("README updated successfully")


if __name__ == "__main__":
    main()
