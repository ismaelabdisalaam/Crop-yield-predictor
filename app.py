import streamlit as st
import pandas as pd
import joblib

# ─── 1) LOAD CLEANED DATA & TRAINED MODEL ───────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("crop_yield_cleaned.csv")

@st.cache_resource
def load_model():
    return joblib.load("xgb_model.joblib")

df    = load_data()
model = load_model()

# Identify feature columns
numeric_feats = ["average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"]
item_cols     = [c for c in df.columns if c.startswith("Item_")]
area_cols     = [c for c in df.columns if c.startswith("Area_")]

# ─── 2) APP CONFIG & SIDEBAR ─────────────────────────────────────────────
st.set_page_config(page_title="Crop Yield Explorer", layout="wide")
st.sidebar.header("Scenario Inputs")

# Country & Crop selectors (pulling names from one-hot columns)
country = st.sidebar.selectbox(
    "Country", 
    sorted(col.replace("Area_","") for col in area_cols)
)
crop = st.sidebar.selectbox(
    "Crop",
    sorted(col.replace("Item_","") for col in item_cols)
)

# Historical year range slider
years = sorted(df["Year"].unique())
year_min, year_max = st.sidebar.select_slider(
    "Historical Year Range",
    options=years,
    value=(years[0], years[-1])
)

# Next-year prediction input
predict_year = st.sidebar.number_input(
    "Predict for Year",
    min_value=year_max + 1,
    max_value=years[-1] + 5,
    value=year_max + 1
)

# ─── 3) FILTER & PLOT HISTORICAL TREND ───────────────────────────────────
mask = (
    (df[f"Area_{country}"]) &
    (df[f"Item_{crop}"]) &
    (df["Year"].between(year_min, year_max))
)
history = df[mask]

st.title("🌾 FAO Crop Yield Explorer")
st.subheader(f"📈 Historical Yield: {crop} in {country} ({year_min}–{year_max})")
st.line_chart(
    history.set_index("Year")["hg/ha_yield"],
    use_container_width=True
)

# ─── 4) BUILD FEATURE VECTOR & PREDICT ───────────────────────────────────
# Compute averages of numeric features over the selected history
avg_vals = {feat: history[feat].mean() for feat in numeric_feats}

# Start with Year + numeric averages
X_new = {"Year": predict_year, **avg_vals}

# One-hot encode the chosen crop & country
for col in item_cols:
    X_new[col] = 1 if col == f"Item_{crop}" else 0
for col in area_cols:
    X_new[col] = 1 if col == f"Area_{country}" else 0

# Create DataFrame and predict
X_new_df = pd.DataFrame([X_new])
y_pred = model.predict(X_new_df)[0]

st.subheader(f"📊 Predicted Yield for {predict_year}")
st.metric(label="Tonnes per hectare", value=f"{y_pred:.2f}")
