# House Price Prediction — Deployable ML App

An end-to-end machine-learning project that predicts residential sale prices from property characteristics.

## Project overview

- **Problem:** regression-based house price prediction
- **Target:** `SalePrice`
- **Model:** XGBoost Regressor
- **Preprocessing:** median imputation for numeric inputs and one-hot encoding for categorical inputs when present
- **Core features:** OverallQual, GrLivArea, GarageCars, TotalBsmtSF, 1stFlrSF, FullBath, YearBuilt, YearRemodAdd, GarageArea, TotRmsAbvGrd
- **UI:** Streamlit
- **Model persistence:** Joblib

## Dataset

This project is designed for the Kaggle **House Prices: Advanced Regression Techniques** Ames dataset. The standard training file contains 1,460 rows and 81 columns, with `SalePrice` as the target.

Place your dataset at:

```text
data/train.csv
```

You can also use `data/house_prices.csv`.

## Run locally

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository named `house-price-prediction`.
2. Upload this project folder, including `data/train.csv` and the generated `models/house_price_model.pkl`.
3. Open Streamlit Community Cloud and create a new app from the GitHub repository.
4. Set the main file to `app.py`.
5. Deploy.

The live URL can then be shared with an employer.

## Important submission note

Only claim this project as **personally developed and deployed** after you have run the training code yourself, verified the app, and deployed the resulting repository. In the application email, describe your actual contribution and technologies used.

## Suggested interview explanation

> I developed an end-to-end house price prediction system using Python, Pandas, Scikit-learn and XGBoost. I handled preprocessing and missing values, trained and evaluated an XGBoost regression model, serialized the trained pipeline with Joblib, and built a Streamlit interface so users can enter property characteristics and receive a predicted sale price.
