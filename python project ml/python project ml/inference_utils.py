
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

ARTIFACT_DIR = Path("artifacts")
FINAL_PIPELINE_PATH = ARTIFACT_DIR / "final_pipeline.joblib"
PREPROCESSOR_META_PATH = ARTIFACT_DIR / "preprocessor_meta.json"
PUBLISHER_FREQ_MAP_PATH = ARTIFACT_DIR / "publisher_freq_map.json"

def load_artifacts() -> Tuple[Any, Dict[str, Any], Optional[Dict[str, float]]]:
    if not FINAL_PIPELINE_PATH.exists():
        raise FileNotFoundError(f"Missing final pipeline: {FINAL_PIPELINE_PATH}")
    if not PREPROCESSOR_META_PATH.exists():
        raise FileNotFoundError(f"Missing preprocessor meta: {PREPROCESSOR_META_PATH}")

    pipeline = joblib.load(FINAL_PIPELINE_PATH)
    with open(PREPROCESSOR_META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    pub_map = None
    if PUBLISHER_FREQ_MAP_PATH.exists():
        with open(PUBLISHER_FREQ_MAP_PATH, "r", encoding="utf-8") as f:
            pub_map = json.load(f)

    return pipeline, meta, pub_map

def _to_float(x):
    if x is None or x == "" or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan

def _publisher_freq(publisher: Optional[str], pub_map: Optional[Dict[str, float]]) -> float:
    if pub_map is None:
        return 0.0
    if publisher is None:
        return 0.0
    return float(pub_map.get(str(publisher), 0.0))

def build_features_from_payload(payload: Dict[str, Any],
                                meta: Dict[str, Any],
                                pub_map: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    numeric_cols = meta.get("numeric_cols", [])
    categorical_cols = meta.get("categorical_cols", [])

    # We build exactly the columns the preprocessor expects
    cols_needed = list(numeric_cols) + list(categorical_cols)
    row: Dict[str, Any] = {}

    # Numeric fields (with special handling for publisher_freq if used)
    for c in numeric_cols:
        if c == "publisher_freq":
            pub_val = payload.get("publisher", None)
            row["publisher_freq"] = _publisher_freq(pub_val, pub_map)
        else:
            row[c] = _to_float(payload.get(c, None))

    # Categorical fields
    for c in categorical_cols:
        if c == "language_code":
            row[c] = (payload.get("language_code") or "eng")
        else:
            row[c] = payload.get(c, None)

    df = pd.DataFrame([row], columns=cols_needed)
    return df

def predict_rating(pipeline, features_df: pd.DataFrame) -> float:
    pred = float(pipeline.predict(features_df)[0])
    # clip into [0, 5] just to be safe
    pred = max(0.0, min(5.0, pred))
    return pred
