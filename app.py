import streamlit as st
import pandas as pd
import joblib

# ─── 1) LOAD CLEANED DATA & TRAINED MODEL ─────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("crop_yield_cleaned.csv")

@st.cache_resource
def load_model():
    return joblib.load("xgb_model.joblib")

# Load assets
 df = load_data()
 model = load_model()
 # Extract the model's expected feature names
 feature_names = model.get_booster().feature_names

# ─── 2) APP CONFIG & SIDEBAR INPUTS ───────────────────────────────────────
st.set_page_config(page_title="Crop Yield Explorer", layout="wide")
st.title("🌾 FAO Crop Yield Explorer")

# Sidebar selectors
# Country & Crop extracted from one-hot columns in df
countries = sorted(col.replace("Area_","") for col in df.columns if col.startswith("Area_"))
items     = sorted(col.replace("Item_","") for col in df.columns if col.startswith("Item_"))

country = st.sidebar.selectbox("Country", countries)
crop    = st.sidebar.selectbox("Crop", items)

years = sorted(df["Year"].unique())
year_min, year_max = st.sidebar.select_slider(
    "Historical Year Range", options=years, value=(years[0], years[-1])
)
predict_year = st.sidebar.number_input(
    "Predict for Year",
    min_value=year_max + 1,
    max_value=years[-1] + 5,
    value=year_max + 1
)

# ─── 3) FILTER & PLOT HISTORICAL TREND ───────────────────────────────────
mask = (
    df[f"Area_{country}"] &
    df[f"Item_{crop}"] &
    df["Year"].between(year_min, year_max)
)
history = df[mask]

st.subheader(f"📈 Historical Yield: {crop} in {country} ({year_min}–{year_max})")
st.line_chart(history.set_index("Year")["hg/ha_yield"], use_container_width=True)

# ─── 4) BUILD FEATURE VECTOR & ALIGN COLUMNS ─────────────────────────────
# Compute historical averages for numeric features
avg_vals = {
    "average_rain_fall_mm_per_year": history["average_rain_fall_mm_per_year"].mean(),
    "pesticides_tonnes":            history["pesticides_tonnes"].mean(),
    "avg_temp":                     history["avg_temp"].mean(),
}
# Start with Year + numeric averages
X_new = {"Year": predict_year, **avg_vals}
# Add one-hot features for all Item_ and Area_
for col in df.columns:
    if col.startswith("Item_"):
        X_new[col] = 1 if col == f"Item_{crop}" else 0
    elif col.startswith("Area_"):
        X_new[col] = 1 if col == f"Area_{country}" else 0

# Convert to DataFrame and reindex to match training features
X_new_df = pd.DataFrame([X_new])
X_new_df = X_new_df.reindex(columns=feature_names, fill_value=0)

# Predict with XGBoost model
y_pred = model.predict(X_new_df)[0]

st.subheader(f"📊 Predicted Yield for {predict_year}")
st.metric(label="Tonnes per hectare", value=f"{y_pred:.2f}")

# ─── 5) OPTIONAL: SHOW RAW DATA ──────────────────────────────────────────
if st.checkbox("Show raw historical data"):
    st.dataframe(history.reset_index(drop=True), use_container_width=True)
