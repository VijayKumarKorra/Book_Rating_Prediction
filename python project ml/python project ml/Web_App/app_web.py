import argparse
import io
import json
import traceback
from pathlib import Path

from flask import Flask, request, jsonify, render_template_string, send_file, redirect, url_for
import pandas as pd
import joblib

app = Flask(__name__)

# =========================
# Config
# =========================
PIPELINE_PATH = Path("best_pipeline.pkl")  # your saved model
BOOKS_CSV = Path(r"C:\Users\MSI PC\Desktop\python project ml\python project ml\Project 1\books.csv")  # <-- CHANGE

# =========================
# Load model (optional)
# =========================
pipeline = None
if PIPELINE_PATH.exists():
    try:
        pipeline = joblib.load(PIPELINE_PATH)
        print(f"[OK] Loaded pipeline from {PIPELINE_PATH}")
    except Exception as e:
        print("[WARN] Could not load pipeline:", e)
else:
    print(f"[WARN] {PIPELINE_PATH} not found. App will run using a fallback constant.")

# =========================
# Load books once (for search)
# =========================
def _read_books(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] Books CSV not found at {path}")
        return pd.DataFrame()

    # robust CSV read
    try:
        df = pd.read_csv(path, encoding="utf-8", engine="python", on_bad_lines="skip")
    except Exception as e:
        print("[WARN] Default read failed:", e)
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")

    # normalize standard column names if present
    rename_map = {
        "bookID": "book_id",
        "title": "title",
        "authors": "authors",
        "language_code": "language_code",
        "publisher": "publisher",
        "  num_pages": "num_pages",
        "num_pages": "num_pages",
        "ratings_count": "ratings_count",
        "text_reviews_count": "text_reviews_count",
        "publication_date": "publication_date",
        "average_rating": "average_rating",
        "isbn": "isbn",
        "isbn13": "isbn13",
    }
    df = df.rename(columns=rename_map)

    # keep only the columns we use in UI/prediction
    keep = [
        "title", "authors", "language_code", "publisher",
        "num_pages", "ratings_count", "text_reviews_count",
        "publication_date", "average_rating"
    ]
    existing = [c for c in keep if c in df.columns]
    df = df[existing].copy()

    # Coerce numerics
    for col in ["num_pages", "ratings_count", "text_reviews_count", "average_rating"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Strings
    for col in ["title", "authors", "language_code", "publisher", "publication_date"]:
        if col in df.columns:
            df[col] = df[col].astype("string")

    # helpful: lowercase copies for searching
    for col in ["title", "authors", "publisher"]:
        if col in df.columns:
            df[f"_{col}_lc"] = df[col].str.lower()

    print(f"[OK] Loaded books: {df.shape}")
    return df

BOOKS = _read_books(BOOKS_CSV)

# =========================
# Common helpers
# =========================
def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure numeric + string columns are the right dtype for prediction."""
    for col in ["num_pages", "ratings_count", "text_reviews_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["title", "authors", "language_code", "publisher", "publication_date"]:
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df

def _predict_df(df: pd.DataFrame):
    df = _prepare_dataframe(df.copy())
    if pipeline is not None:
        preds = pipeline.predict(df)
    else:
        # fallback constant
        preds = pd.Series([3.9] * len(df)).values
    return preds

def _search_books(q: str, top_n: int = 24) -> pd.DataFrame:
    """Simple contains-based search over title/authors/publisher (case-insensitive)."""
    if BOOKS.empty:
        return BOOKS

    q = (q or "").strip().lower()
    if not q:
        # default to something "popular"
        cols = [c for c in ["ratings_count", "average_rating"] if c in BOOKS.columns]
        if "ratings_count" in cols:
            return BOOKS.sort_values("ratings_count", ascending=False).head(top_n)
        return BOOKS.head(top_n)

    mask = False
    for col in ["_title_lc", "_authors_lc", "_publisher_lc"]:
        if col in BOOKS.columns:
            mask = mask | BOOKS[col].fillna("").str.contains(q, na=False, regex=False)
    res = BOOKS[mask].copy()

    # Rank by something “nice”: many ratings, then higher average_rating
    if "ratings_count" in res.columns:
        res = res.sort_values(["ratings_count", "average_rating"], ascending=[False, False], na_position="last")
    return res.head(top_n)

# =========================
# HTML (polished, app-like)
# =========================
BASE_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Book App · Search & Predict</title>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <style>
      :root {
        --bg1: #101728;
        --bg2: #0e1320;
        --card: #151c2f;
        --muted:#9aa7bd;
        --fg:#eef3ff;
        --accent:#6aa0ff;
        --accent2:#51d1c5;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        color: var(--fg);
        font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Arial;
        background: radial-gradient(1200px 600px at 10% -10%, #1b2338 5%, transparent 40%),
                    radial-gradient(1000px 700px at 100% -20%, #0a6c7f40 5%, transparent 50%),
                    linear-gradient(180deg, var(--bg1), var(--bg2));
        min-height: 100vh;
      }
      header {
        padding: 28px 18px;
        max-width: 1100px; margin: 0 auto;
        display:flex; align-items:center; justify-content:space-between;
      }
      .brand { font-size: 22px; letter-spacing: .4px; font-weight: 700; }
      .tag { font-size: 13px; color: var(--muted); margin-left: 10px; }
      .ok { color: #62d26f; } .warn { color: #f7b955; }

      .hero {
        max-width: 1100px; margin: 0 auto; padding: 10px 18px 24px;
      }
      .searchbar {
        display:flex; gap: 10px; align-items: center;
        background: #0b1222; border:1px solid #27304b; padding: 12px 12px;
        border-radius: 14px;
      }
      .searchbar input[type="text"] {
        flex: 1; background: transparent; color: var(--fg); border: none; outline: none; font-size: 16px;
      }
      .btn {
        background: linear-gradient(135deg, var(--accent), var(--accent2));
        border:none; color:white; padding: 10px 16px; border-radius:10px; font-weight:600;
        cursor:pointer; box-shadow: 0 8px 24px rgba(0,0,0,.2);
      }
      .btn.secondary { background: #202943; color: var(--fg); border:1px solid #2f3a5c; }
      .grid {
        max-width: 1100px; margin: 20px auto 60px; padding: 0 18px;
        display:grid; gap:16px; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
      }
      .card {
        background: var(--card); border:1px solid #263357; border-radius:12px; padding:14px;
      }
      .title { font-weight:700; line-height:1.3; margin-bottom:6px; }
      .meta { color: var(--muted); font-size: 13px; }
      .row { display:flex; gap:6px; margin-top:10px; }
      .pill { background:#202943; border:1px solid #2f3a5c; padding:2px 8px; border-radius:999px; font-size:12px; color:#c9d4ea; }
      .pred { margin-top:10px; font-weight:600; }
      .footer { padding:30px; text-align:center; color: var(--muted); border-top:1px solid #212a45; }
      a { color: #9dc1ff; text-decoration:none; }
      .hint { font-size: 13px; color: var(--muted); margin-top:6px;}
      .bar { display:flex; gap:10px; align-items:center; }
    </style>
  </head>
  <body>
    <header>
      <div class="bar">
        <div class="brand">📚 Book App</div>
        {% if model_loaded %}
          <div class="tag ok">model loaded</div>
        {% else %}
          <div class="tag warn">no model (using dummy 3.9)</div>
        {% endif %}
      </div>
      <nav class="bar">
        <a href="{{ url_for('explore') }}">Search</a>
        <a href="{{ url_for('home_docs') }}">Docs</a>
      </nav>
    </header>

    {% block content %}{% endblock %}

    <div class="footer">
      Built with Flask · Search your dataset and predict ratings inline.
    </div>

    <script>
      async function predictFromRow(btn, payload) {
        try {
          btn.disabled = true;
          btn.textContent = "Predicting...";
          const res = await fetch("{{ url_for('predict') }}", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({rows: [payload]})
          });
          const data = await res.json();
          let val = "error";
          if (data && data.predictions && data.predictions.length) {
            val = data.predictions[0].predicted_rating?.toFixed ? data.predictions[0].predicted_rating.toFixed(2) : data.predictions[0].predicted_rating;
          } else if (data.error) {
            val = "error";
          }
          const predSpan = btn.parentElement.querySelector(".pred-val");
          if (predSpan) predSpan.textContent = val;
        } catch (e) {
          console.error(e);
          alert("Prediction failed");
        } finally {
          btn.disabled = false;
          btn.textContent = "Predict";
        }
      }
    </script>
  </body>
</html>
"""

DOCS_HTML = """
{% extends "base.html" %}
{% block content %}
  <div class="hero">
    <h2>API Docs</h2>
    <p class="hint">POST JSON to <code>/predict</code> with body:</p>
    <pre>{
  "rows":[
    {"title":"T1","authors":"A1","language_code":"eng",
     "num_pages":123,"ratings_count":200,"text_reviews_count":20,
     "publication_date":"2001-01-01","publisher":"X"}
  ]
}</pre>
    <p class="hint">Health check: <a href="{{ url_for('health') }}">{{ url_for('health') }}</a></p>
  </div>
{% endblock %}
"""

EXPLORE_HTML = """
{% extends "base.html" %}
{% block content %}
  <div class="hero">
    <form class="searchbar" method="get" action="{{ url_for('explore') }}">
      <input name="q" type="text" placeholder="Search by title, author or publisher…" value="{{ q|default('') }}">
      <button class="btn" type="submit">Search</button>
      <a class="btn secondary" href="{{ url_for('explore') }}">Clear</a>
    </form>
    <div class="hint">Showing {{ results|length }} result(s){% if q %} for "<b>{{ q }}</b>"{% endif %}.</div>
  </div>

  <div class="grid">
    {% for r in results %}
      <div class="card">
        <div class="title">{{ r.title or 'Untitled' }}</div>
        <div class="meta">{{ r.authors or '—' }}</div>
        <div class="row">
          {% if r.language_code %}<div class="pill">{{ r.language_code }}</div>{% endif %}
          {% if r.publisher %}<div class="pill">{{ r.publisher }}</div>{% endif %}
          {% if r.publication_date %}<div class="pill">{{ r.publication_date }}</div>{% endif %}
        </div>
        <div class="row">
          {% if r.num_pages is not none %}<div class="pill">{{ r.num_pages|int }} pages</div>{% endif %}
          {% if r.ratings_count is not none %}<div class="pill">{{ r.ratings_count|int }} ratings</div>{% endif %}
          {% if r.average_rating is not none %}<div class="pill">avg {{ "%.2f"|format(r.average_rating) }}</div>{% endif %}
        </div>

        <div class="row" style="justify-content:space-between; align-items:center; margin-top:12px;">
          <div class="pred">Predicted: <span class="pred-val">—</span></div>
          <button class="btn" type="button"
            onclick='predictFromRow(this, {{ {
              "title": r.title, "authors": r.authors, "language_code": r.language_code,
              "publisher": r.publisher, "num_pages": r.num_pages, "ratings_count": r.ratings_count,
              "text_reviews_count": r.text_reviews_count, "publication_date": r.publication_date
            } | tojson }})'>
            Predict
          </button>
        </div>
      </div>
    {% endfor %}
  </div>
{% endblock %}
"""

HOME_HTML = """
{% extends "base.html" %}
{% block content %}
  <div class="hero">
    <h2>Welcome</h2>
    <p class="hint">Use <b>Search</b> to find books by title/author/publisher and predict ratings inline.
       The banner shows whether your model is loaded.</p>

    <h3 style="margin-top:24px">Batch prediction (CSV)</h3>
    <form action="{{ url_for('predict_csv') }}" method="post" enctype="multipart/form-data" class="searchbar" style="gap:14px;">
      <input type="file" name="file" accept=".csv" required>
      <button class="btn" type="submit">Upload & Predict</button>
    </form>
  </div>
{% endblock %}
"""

# Jinja template registration
app.jinja_env.globals["url_for"] = url_for
app.jinja_env.from_string(BASE_HTML)  # ensure compiled once
app.jinja_loader.mapping = {"base.html": BASE_HTML}  # in-memory base


# =========================
# Routes
# =========================
@app.get("/")
def root():
    return redirect(url_for("home_docs"))

@app.get("/home")
def home_docs():
    return render_template_string(HOME_HTML, model_loaded=pipeline is not None)

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": pipeline is not None, "books_loaded": not BOOKS.empty})

@app.get("/explore")
def explore():
    q = request.args.get("q", "", type=str)
    df = _search_books(q, top_n=24)
    # Convert to list of dicts for Jinja
    results = df.to_dict(orient="records")
    return render_template_string(EXPLORE_HTML, model_loaded=pipeline is not None, results=results, q=q)

# ---- existing APIs preserved ----
@app.post("/predict")
def predict():
    try:
        rows = None
        if request.is_json:
            payload = request.get_json(silent=True) or {}
            rows = payload.get("rows")
        if rows is None and request.form:
            single = {k: v for k, v in request.form.items()}
            rows = [single]
        if not rows:
            return jsonify({"error": "No input rows provided."}), 400

        df = pd.DataFrame(rows)
        preds = _predict_df(df)
        out = df.copy()
        out["predicted_rating"] = preds
        return jsonify({"predictions": out.to_dict(orient="records")})
    except Exception as e:
        print("Prediction error:", e)
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.post("/predict-csv")
def predict_csv():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part named 'file'"}), 400
        f = request.files["file"]
        if f.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        df = pd.read_csv(f)
        preds = _predict_df(df)
        out = df.copy()
        out["predicted_rating"] = preds

        csv_bytes = out.to_csv(index=False).encode("utf-8")
        return send_file(io.BytesIO(csv_bytes),
                         mimetype="text/csv",
                         as_attachment=True,
                         download_name="predictions.csv")
    except Exception as e:
        print("CSV prediction error:", e)
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


# =========================
# Entrypoint
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    # Make the base template available to Jinja
    app.jinja_loader.mapping = {"base.html": BASE_HTML}
    print(f"Serving on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=True, threaded=True)
