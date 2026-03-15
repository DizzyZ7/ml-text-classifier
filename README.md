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
```
Описание

Модель обучается на CSV-датасете с двумя колонками:
	•	text — текст
	•	label — класс

По умолчанию используется pipeline:
	•	TfidfVectorizer
	•	LogisticRegression

Текущие результаты

Ниже блок будет автоматически обновляться после запуска GitHub Actions.
<!-- METRICS_START -->
Метрики появятся здесь после первого выполнения рабочего процесса.
<!-- METRICS_END -->
<!-- EXAMPLES_START -->
Примеры прогнозов появятся здесь после первого выполнения рабочего процесса.
<!-- EXAMPLES_END -->
<!-- IMAGE_START -->
Матрица путаницы

Изображение появится здесь после первого выполнения рабочего процесса.
<!-- IMAGE_END -->
