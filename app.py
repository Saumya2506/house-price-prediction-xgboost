import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/house_price_model.pkl")
METRICS_PATH = Path("models/metrics.json")

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

st.title("🏠 House Price Predictor")
st.caption("XGBoost regression model trained on the Ames housing dataset")

if not MODEL_PATH.exists():
    st.error("The trained model is not available yet.")
    st.markdown("Run `python train_model.py` after placing your `train.csv` in `data/`.")
    st.stop()

model = joblib.load(MODEL_PATH)

with st.sidebar:
    st.header("Model")
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        st.metric("Validation R²", f"{metrics['r2']:.3f}")
        st.metric("Validation RMSE", f"${metrics['rmse']:,.0f}")
        st.caption(f"Validation rows: {metrics['validation_rows']}")

st.subheader("Enter property details")

c1, c2 = st.columns(2)
with c1:
    overall_qual = st.slider("Overall Quality (1–10)", 1, 10, 6)
    gr_liv_area = st.number_input("Living Area (sq ft)", 300, 5000, 1500, step=50)
    total_bsmt_sf = st.number_input("Basement Area (sq ft)", 0, 3000, 1000, step=50)
    first_flr_sf = st.number_input("1st Floor Area (sq ft)", 300, 3000, 1000, step=50)
    garage_cars = st.number_input("Garage Capacity (cars)", 0, 5, 2, step=1)
with c2:
    full_bath = st.number_input("Full Bathrooms", 0, 5, 2, step=1)
    year_built = st.number_input("Year Built", 1850, 2026, 2000, step=1)
    year_remod = st.number_input("Year Remodeled", 1850, 2026, 2005, step=1)
    garage_area = st.number_input("Garage Area (sq ft)", 0, 1500, 500, step=25)
    total_rooms = st.number_input("Total Rooms Above Ground", 2, 15, 7, step=1)

if st.button("Predict House Price", type="primary", use_container_width=True):
    row = pd.DataFrame([{
        "OverallQual": overall_qual,
        "GrLivArea": gr_liv_area,
        "GarageCars": garage_cars,
        "TotalBsmtSF": total_bsmt_sf,
        "1stFlrSF": first_flr_sf,
        "FullBath": full_bath,
        "YearBuilt": year_built,
        "YearRemodAdd": year_remod,
        "GarageArea": garage_area,
        "TotRmsAbvGrd": total_rooms,
    }])
    prediction = float(model.predict(row)[0])
    st.success(f"Estimated Sale Price: ${prediction:,.0f}")
    st.info("This is a machine-learning estimate, not an appraisal or guaranteed market value.")

st.divider()
st.markdown("**Technologies:** Python · Pandas · Scikit-learn · XGBoost · Streamlit · Joblib")
