# train_books_model.py
# -----------------------------------------------------------
# Train a book-rating model on books.csv and save best_pipeline.pkl
# Works on older scikit-learn; computes RMSE without squared=False.
# -----------------------------------------------------------

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from joblib import dump


# -------------------------
# Helper: compute RMSE safely
# -------------------------
def rmse_score(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)  # always returns MSE
    return float(np.sqrt(mse))


def main(args):
    csv_path = Path(args.csv)
    out_path = Path(args.out)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"[info] Reading: {csv_path}")
    # Robust csv read (handles messy rows)
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")

    print(f"[info] Shape: {df.shape}")

    # ----------- choose/confirm target -----------
    # You can pass --target average_rating (recommended)
    target_col = args.target
    if target_col is None:
        # sensible default for Kaggle/Goodreads-style files
        if "average_rating" in df.columns:
            target_col = "average_rating"
        else:
            # pick first numeric-like column as last resort
            numeric_cols = [
                c for c in df.columns
                if pd.to_numeric(df[c], errors="coerce").notna().sum() > 0
            ]
            if not numeric_cols:
                raise ValueError(
                    "No numeric-like column found for target. "
                    "Please pass --target <column> (e.g. average_rating)."
                )
            target_col = numeric_cols[-1]
            print(f"[warn] Using fallback target column: {target_col}")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV.")

    # ------------- select features -------------
    # Keep a reasonable set; adjust as you like.
    wanted = [
        "title",
        "authors",
        "language_code",
        "publisher",
        "num_pages",
        "ratings_count",
        "text_reviews_count",
        "publication_date",
        target_col,
    ]
    present = [c for c in wanted if c in df.columns]
    df = df[present].copy()

    # --- parse / clean columns ---
    # convert numerics
    for col in ["num_pages", "ratings_count", "text_reviews_count", target_col]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # extract year from publication_date if present
    if "publication_date" in df.columns:
        # try dd/mm/yyyy, yyyy-mm-dd, etc.
        try:
            dt = pd.to_datetime(df["publication_date"], errors="coerce", dayfirst=True)
        except Exception:
            dt = pd.to_datetime(df["publication_date"], errors="coerce")
        df["pub_year"] = dt.dt.year
        df.drop(columns=["publication_date"], inplace=True)

    # define final feature list
    X_cols = [c for c in df.columns if c != target_col]

    # drop rows without target
    df = df[df[target_col].notna()].copy()
    y = df[target_col].astype(float)
    X = df[X_cols].copy()

    print(f"[info] After cleaning: X={X.shape}, y={y.shape}")

    # split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # column types
    num_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    # preprocessors
    numeric_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, num_cols),
            ("cat", categorical_tf, cat_cols),
        ],
        remainder="drop",
    )

    # candidates
    candidates = [
        ("Dummy(mean)", DummyRegressor(strategy="mean")),
        ("LinearRegression", LinearRegression()),
        ("RidgeCV", RidgeCV(alphas=np.logspace(-3, 3, 21), cv=5,
                            scoring="neg_mean_squared_error")),
        ("LassoCV", LassoCV(alphas=np.logspace(-3, 3, 21), cv=5, random_state=42)),
        ("RandomForest", RandomForestRegressor(n_estimators=200, random_state=42)),
        ("GradientBoosting", GradientBoostingRegressor(random_state=42)),
    ]

    results = []
    best = {"rmse": float("inf"), "name": None, "pipe": None}

    for name, est in candidates:
        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", est),
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_valid)
        r = rmse_score(y_valid, pred)
        r2 = r2_score(y_valid, pred)
        print(f"{name:>18s} | RMSE={r:.4f} | R²={r2:.4f}")
        results.append((name, r, r2, pipe))
        if r < best["rmse"]:
            best = {"rmse": r, "name": name, "pipe": pipe}

    print(f"\n[info] Best on validation: {best['name']} | RMSE={best['rmse']:.4f}")

    # refit on all
    final_pipe = best["pipe"]
    final_pipe.fit(X, y)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump(final_pipe, out_path)
    print(f"[ok] Saved pipeline -> {out_path.resolve()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to books.csv")
    p.add_argument("--out", default="best_pipeline.pkl", help="Output .pkl path")
    p.add_argument("--target", default=None,
                   help="Target column (default: average_rating if present)")
    args = p.parse_args()
    main(args)
