# ML Text Classifier

Python + Machine Learning проект, который:

- обучает модель классификации текстов
- оценивает качество модели
- сохраняет метрики
- строит confusion matrix
- делает предсказания
- автоматически обновляет README через GitHub Actions

## Структура проекта

```text
ml-text-classifier/
│
├── data/
│   ├── dataset.csv
│   └── predictions.csv
│
├── models/
│   └── model.joblib
│
├── reports/
│   ├── metrics.json
│   ├── confusion_matrix.png
│   └── examples.csv
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   └── update_readme.py
│
├── .github/
│   └── workflows/
│       └── train.yml
│
├── README.md
├── requirements.txt
└── .gitignore
