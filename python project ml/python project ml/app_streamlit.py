
import streamlit as st
import pandas as pd
import numpy as np

from inference_utils import load_artifacts, build_features_from_payload, predict_rating

st.set_page_config(page_title="Book Rating Predictor", page_icon="📚", layout="centered")

st.title("Book Rating Predictor")
st.caption("Predict a book's average rating from its metadata.")

@st.cache_resource
def _load():
    return load_artifacts()

pipeline, meta, pub_map = _load()
numeric_cols = meta.get("numeric_cols", [])
categorical_cols = meta.get("categorical_cols", [])

st.subheader("Single Prediction")
with st.form("single_pred"):
    # Inputs expected by the pipeline
    col1, col2 = st.columns(2)
    with col1:
        num_pages = st.number_input("Number of pages (num_pages)", min_value=0, max_value=10000, value=300, step=1)
        ratings_count = st.number_input("Ratings count", min_value=0, value=1000, step=1)
        text_reviews_count = st.number_input("Text reviews count", min_value=0, value=100, step=1)
        authors_count = st.number_input("Authors count", min_value=1, value=1, step=1)
    with col2:
        year = st.number_input("Publication year (year)", min_value=1800, max_value=2100, value=2010, step=1)
        language_code = st.text_input("Language code (e.g., eng, spa, fre)", value="eng")
        publisher = st.text_input("Publisher", value="Unknown")

    submitted = st.form_submit_button("Predict rating")
    if submitted:
        payload = {
            "num_pages": num_pages,
            "ratings_count": ratings_count,
            "text_reviews_count": text_reviews_count,
            "authors_count": authors_count,
            "year": year,
            "language_code": language_code,
            "publisher": publisher
        }
        feat = build_features_from_payload(payload, meta, pub_map)
        pred = predict_rating(pipeline, feat)
        st.success(f"Predicted average_rating: {pred:.2f}")

st.divider()
st.subheader("Batch Prediction (CSV)")

st.markdown("Upload a CSV with columns you have (e.g., num_pages, ratings_count, text_reviews_count, authors_count, year, language_code, publisher). Missing numeric columns are imputed by the pipeline; unknown publishers get 0 frequency.")

file = st.file_uploader("Upload CSV", type=["csv"])
if file is not None:
    df_in = pd.read_csv(file)
    # Build rows one by one to respect publisher_freq and column order
    feats = []
    for _, row in df_in.iterrows():
        payload = row.to_dict()
        feat = build_features_from_payload(payload, meta, pub_map)
        feats.append(feat)
    X = pd.concat(feats, axis=0).reset_index(drop=True)
    preds = pipeline.predict(X)
    preds = np.clip(preds, 0.0, 5.0)
    out = df_in.copy()
    out["predicted_average_rating"] = np.round(preds, 3)
    st.dataframe(out.head(50))
    st.download_button("Download predictions", out.to_csv(index=False).encode("utf-8"), file_name="predictions.csv", mime="text/csv")

st.caption("Artifacts expected in ./artifacts/: final_pipeline.joblib, preprocessor_meta.json, and optionally publisher_freq_map.json")
