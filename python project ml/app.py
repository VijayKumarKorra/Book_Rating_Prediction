import argparse
import io
import traceback
from pathlib import Path

from flask import Flask, request, jsonify, render_template_string, send_file
import pandas as pd
import joblib

app = Flask(__name__)

# Path to trained pipeline
PIPELINE_PATH = Path("best_pipeline.pkl")

# Load model pipeline once
pipeline = None
if PIPELINE_PATH.exists():
    try:
        pipeline = joblib.load(PIPELINE_PATH)
        print(f"[OK] Loaded pipeline from {PIPELINE_PATH}")
    except Exception as e:
        print("[WARN] Could not load pipeline:", e)
else:
    print(f"[WARN] {PIPELINE_PATH} not found. Using dummy constant 3.9")

# Homepage HTML
HOME_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Book Rating Predictor</title>
    <style>
      body { font-family: system-ui, Arial, sans-serif; max-width: 800px; margin: 2rem auto; }
      h1 { color: #2c3e50; }
      form { margin-bottom: 1.5rem; padding: 1rem; border: 1px solid #ccc; border-radius: 8px; }
      input, button { margin: .25rem 0; padding: .5rem; }
    </style>
  </head>
  <body>
    <h1>📚 Book Rating Prediction</h1>
    <p>Use this form to predict the average rating of a book by entering its metadata.</p>

    <h2>Single Prediction</h2>
    <form action="/predict" method="post">
      <label>Title: <input type="text" name="title"></label><br>
      <label>Authors: <input type="text" name="authors"></label><br>
      <label>Publisher: <input type="text" name="publisher"></label><br>
      <label>Language Code: <input type="text" name="language_code" value="eng"></label><br>
      <label>Publication Date: <input type="text" name="publication_date" placeholder="YYYY-MM-DD"></label><br>
      <label>Number of Pages: <input type="number" name="num_pages"></label><br>
      <label>Ratings Count: <input type="number" name="ratings_count"></label><br>
      <label>Text Reviews Count: <input type="number" name="text_reviews_count"></label><br>
      <button type="submit">Predict</button>
    </form>

    <h2>Batch Prediction (Upload CSV)</h2>
    <form action="/predict-csv" method="post" enctype="multipart/form-data">
      <input type="file" name="file">
      <button type="submit">Upload & Predict</button>
    </form>
  </body>
</html>
"""

@app.get("/")
def home():
    return render_template_string(HOME_HTML, model_loaded=pipeline is not None)

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": pipeline is not None})

# --- Preprocessing helpers ---
def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
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
        preds = pd.Series([3.9] * len(df)).values  # fallback
    return preds

# --- Single prediction ---
@app.post("/predict")
def predict():
    try:
        rows = None
        if request.is_json:
            payload = request.get_json(silent=True) or {}
            rows = payload.get("rows")
        if rows is None and request.form:
            rows = [{k: v for k, v in request.form.items()}]
        if not rows:
            return jsonify({"error": "Send JSON {\"rows\":[{...}]} or submit the form."}), 400

        df = pd.DataFrame(rows)
        preds = _predict_df(df)
        out = df.copy()
        out["predicted_rating"] = preds
        return jsonify({"predictions": out.to_dict(orient="records")})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# --- Batch prediction (CSV upload) ---
@app.post("/predict-csv")
def predict_csv():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file named 'file'"}), 400
        f = request.files["file"]
        if f.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        # Robust CSV parsing
        try:
            df = pd.read_csv(f)
        except Exception:
            f.seek(0)  # reset file pointer
            df = pd.read_csv(f, engine="python", on_bad_lines="skip")

        preds = _predict_df(df)
        out = df.copy()
        out["predicted_rating"] = preds

        csv_bytes = out.to_csv(index=False).encode("utf-8")
        return send_file(io.BytesIO(csv_bytes),
                         mimetype="text/csv",
                         as_attachment=True,
                         download_name="predictions.csv")
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    print(f"Serving on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=True, threaded=True)
