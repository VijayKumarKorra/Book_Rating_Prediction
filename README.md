# Book Rating Prediction

Predict a book’s **average rating** from metadata (title, authors, language, pages, counts, publication date, publisher).  
Includes an end-to-end ML pipeline (EDA → cleaning → feature engineering → model selection) and a production-style **Flask web app** + **JSON API**.  
Large artifacts (models/data/video) are versioned with **Git LFS**.

**Repo:** https://github.com/VijayKumarKorra/Book_Rating_Prediction  
**Author:** Vijay Kumar Korra (korravijay1004@gmail.com)  
**License:** MIT

---

## Project Structure

├── app.py # Flask app (single & batch predictions, /api endpoints)
├── best_pipeline.pkl # Trained sklearn Pipeline (via LFS)
├── books.csv # Dataset (via LFS if large)
├── artifacts/ # (Generated) predictions, retrain drops, etc.
├── notebooks/ # (Optional) Jupyter analysis
├── requirements.txt
├── README.md
└── LICENSE


> If you don’t see `best_pipeline.pkl`, the app can still run (with a dummy constant), or retrain if you use the training script in the notebook.

---

## Dataset (columns)

- `bookID`, `title`, `authors`, `average_rating`, `isbn`, `isbn13`, `language_code`,  
  `num_pages`, `ratings_count`, `text_reviews_count`, `publication_date`, `publisher`

Target: **`average_rating`** (float 0–5).

---

## Pipeline Overview

- **Cleaning:** type coercions, trimming, `year` extraction, ranges, log transforms.
- **Features:** numeric (`num_pages`, `ratings_count`, `text_reviews_count`, `log_*`, `authors_count`, `year`), categorical (`language_code` + optional `publisher_freq`).
- **Preprocessing:** `ColumnTransformer` with:
  - numeric → `SimpleImputer(median)` + `StandardScaler`
  - categorical → `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown="ignore")`
- **Models tried:** Dummy, Linear, RidgeCV, RandomForest, GradientBoosting (5-fold CV).
- **Artifact:** single sklearn `Pipeline` persisted as `best_pipeline.pkl`.

---

## Quickstart (Web App)

```bash
# (Windows PowerShell shown)
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
python app.py
# Open http://127.0.0.1:5000


(optional)
JSON API

POST /api/predict

{
  "rows": [
    {
      "title": "The Hobbit",
      "authors": "J.R.R. Tolkien",
      "language_code": "eng",
      "num_pages": 300,
      "ratings_count": 1000,
      "text_reviews_count": 120,
      "publication_date": "2001-01-01",
      "publisher": "HarperCollins"
    }
  ]
}


Response

{ "predictions": [4.23] }


POST /api/predict-batch

{ "rows": [ { ... }, { ... } ] }


Response

{ "predictions": [4.23, 3.98, ...] }
