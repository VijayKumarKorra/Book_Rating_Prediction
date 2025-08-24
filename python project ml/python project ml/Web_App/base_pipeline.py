# base_pipeline.py  (fixed delimiter + simple pipeline)
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# ========= EDIT THESE =========
CSV_PATH = Path(r"C:\Users\MSI PC\Desktop\python project ml\python project ml\Project 1\books.csv")
TARGET_COL = "average_rating"      # e.g. "average_rating" or "ratings_count" or "num_pages"
TEST_SIZE = 0.2
RANDOM_STATE = 42
# ==============================

def read_books_csv(path: Path) -> pd.DataFrame:
    """Force comma as delimiter; skip bad lines."""
    assert path.exists(), f"File not found: {path}"
    # Use engine='python' because titles/authors contain commas/quotes
    df = pd.read_csv(path,
                     engine="python",
                     sep=",",
                     quotechar='"',
                     escapechar="\\",
                     on_bad_lines="skip")
    return df

def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # Identify numeric vs categorical columns
    numeric_cols, categorical_cols = [], []
    for c in X.columns:
        s = pd.to_numeric(X[c], errors="coerce")
        if s.notna().sum() > 0:
            numeric_cols.append(c)
            X[c] = s
        else:
            categorical_cols.append(c)

    print(f"[info] numeric_cols ({len(numeric_cols)}): {numeric_cols}")
    print(f"[info] categorical_cols ({len(categorical_cols)}): {categorical_cols}")

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer(
        [("num", num_pipe, numeric_cols),
         ("cat", cat_pipe, categorical_cols)],
        remainder="drop"
    )

def evaluate(name, estimator, Xtr, ytr, Xval, yval, preprocessor):
    pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
    pipe.fit(Xtr, ytr)
    yp = pipe.predict(Xval)
    rmse = mean_squared_error(yval, yp, squared=False)
    r2 = r2_score(yval, yp)
    print(f"{name:>20s} | RMSE={rmse:,.4f} | R²={r2:,.4f}")
    return {"name": name, "rmse": rmse, "r2": r2, "pipe": pipe}

def main():
    print("Reading:", CSV_PATH.resolve())
    df = read_books_csv(CSV_PATH)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    if TARGET_COL not in df.columns:
        raise ValueError(f"TARGET_COL='{TARGET_COL}' not found. Columns: {list(df.columns)}")

    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    X = df.drop(columns=[TARGET_COL])
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask]

    if len(y) < 50:
        raise ValueError("Too few numeric target rows after cleaning.")

    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    preprocessor = make_preprocessor(Xtr.copy())

    results = []
    results.append(evaluate("Dummy(mean)", DummyRegressor(strategy="mean"), Xtr, ytr, Xval, yval, preprocessor))
    results.append(evaluate("LinearRegression", LinearRegression(), Xtr, ytr, Xval, yval, preprocessor))
    results.append(evaluate("RidgeCV", RidgeCV(alphas=np.logspace(-3,3,21), cv=5,
                                              scoring="neg_mean_squared_error"),
                           Xtr, ytr, Xval, yval, preprocessor))
    results.append(evaluate("LassoCV", LassoCV(alphas=np.logspace(-3,3,21), cv=5, random_state=RANDOM_STATE),
                           Xtr, ytr, Xval, yval, preprocessor))
    results.append(evaluate("RandomForest", RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
                           Xtr, ytr, Xval, yval, preprocessor))
    results.append(evaluate("GradientBoosting", GradientBoostingRegressor(random_state=RANDOM_STATE),
                           Xtr, ytr, Xval, yval, preprocessor))

    best = min(results, key=lambda d: d["rmse"])
    print("\nBest:", best["name"], "| RMSE:", f"{best['rmse']:.4f}")

    # Refit on all and save
    X_all = pd.concat([Xtr, Xval], axis=0)
    y_all = pd.concat([ytr, yval], axis=0)
    best_pipe = best["pipe"].fit(X_all, y_all)

    out = Path("best_pipeline.pkl").resolve()
    dump(best_pipe, out)
    print("Saved:", out)

if __name__ == "__main__":
    main()
