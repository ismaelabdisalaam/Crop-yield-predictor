import streamlit as st
import pandas as pd
import joblib

# ─── 1) LOAD CLEANED DATA & TRAINED MODEL ───────────────────────────────
@st.cache
def load_data():
    return pd.read_csv("crop_yield_cleaned.csv")

@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load("xgb_model.joblib")

df    = load_data()
model = load_model()

# ─── 2) SIDEBAR INPUTS ───────────────────────────────────────────────────
st.sidebar.header("Scenario Inputs")

# derive the list of countries & crops from your one‑hot columns
area_cols  = [c for c in df.columns if c.startswith("Area_")]
country_list = sorted([c.replace("Area_","") for c in area_cols])
country = st.sidebar.selectbox("Country", country_list)

item_cols  = [c for c in df.columns if c.startswith("Item_")]
crop_list = sorted([c.replace("Item_","") for c in item_cols])
crop = st.sidebar.selectbox("Crop", crop_list)

years = sorted(df["Year"].unique())
year_min, year_max = st.sidebar.select_slider(
    "Historical Year Range",
    options=years,
    value=(years[0], years[-1])
)

predict_year = st.sidebar.number_input(
    "Predict for Year",
    min_value=year_max + 1,
    max_value=years[-1] + 5,
    value=year_max + 1
)

# ─── 3) FILTER HISTORICAL DATA ───────────────────────────────────────────
mask = (
        (df[f"Area_{country}"] == True) &
        (df[f"Item_{crop}"] == True) &
        (df["Year"].between(year_min, year_max))
)
history = df[mask]

# ─── 4) MAIN PANEL: TITLE & HISTORICAL TREND ─────────────────────────────
st.title("🌾 FAO Crop Yield Explorer")
st.subheader(f"📈 Historical Yield: {crop} in {country} ({year_min}–{year_max})")
st.line_chart(history.set_index("Year")["hg/ha_yield"])

# ─── 5) NEXT‑YEAR PREDICTION ─────────────────────────────────────────────
# use historical averages for your numeric features
numeric_feats = ["average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
avg_vals = {feat: history[feat].mean() for feat in numeric_feats}

# build the one‑row dict of inputs
X_new = {"Year": predict_year}
X_new.update(avg_vals)

# set the one‑hot columns
for col in item_cols:
    X_new[col] = 1 if col == f"Item_{crop}" else 0
for col in area_cols:
    X_new[col] = 1 if col == f"Area_{country}" else 0

X_new_df = pd.DataFrame([X_new])
y_pred = model.predict(X_new_df)[0]

st.subheader(f"📊 Predicted Yield for {predict_year}")
st.metric(label="Tonnes per hectare", value=f"{y_pred:.2f}")

