import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

DATA_CANDIDATES = ["data/train.csv", "data/house_prices.csv"]
TARGET = "SalePrice"
FEATURES = [
    "OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "1stFlrSF",
    "FullBath", "YearBuilt", "YearRemodAdd", "GarageArea", "TotRmsAbvGrd"
]


def find_data():
    for path in DATA_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Training data not found. Put Kaggle Ames House Prices train.csv at "
        "data/train.csv (or rename your existing CSV to data/house_prices.csv)."
    )


def main():
    os.makedirs("models", exist_ok=True)
    path = find_data()
    df = pd.read_csv(path)

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ])

    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.035,
        max_depth=4,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=2,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_test)),
        "features": FEATURES,
    }

    joblib.dump(pipeline, "models/house_price_model.pkl")
    with open("models/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Model saved to models/house_price_model.pkl")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
