import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# =========================================================
# PAGE SETTINGS
# =========================================================
st.set_page_config(
    page_title="Forest Fire Prediction",
    page_icon="🔥",
    layout="wide"
)

# =========================================================
# CUSTOM DESIGN
# =========================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #07111f, #101827);
    color: white;
}
.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}
.hero {
    background: linear-gradient(135deg, #172554, #7f1d1d);
    padding: 35px;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 30px;
}
.hero h1 { font-size: 42px; margin-bottom: 10px; }
.hero p { font-size: 18px; color: #dbeafe; }
.card {
    background: #111827;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #374151;
    margin-bottom: 20px;
}
.result-high {
    background: #7f1d1d;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    border: 2px solid #ef4444;
}
.result-medium {
    background: #78350f;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    border: 2px solid #f59e0b;
}
.result-low {
    background: #064e3b;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    border: 2px solid #10b981;
}
div.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 12px;
    font-size: 20px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <h1>🔥 Forest Fire Prediction System</h1>
    <p>AI Based Forest Fire Risk Prediction using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# DATASET
# IMPORTANT: use the folder containing app.py, not the
# terminal's current working directory.
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.csv"

@st.cache_data
def load_data(path):
    if not path.exists():
        return None, f"Dataset not found: {path}"

    # Try the normal CSV format first.
   
    try:
        from io import StringIO

        raw_text = path.read_text(encoding="utf-8-sig", errors="replace")

        # Remove Git conflict marker lines
        lines = []
        for line in raw_text.splitlines():
            s = line.strip()
            if (
                s.startswith("<<<<<<<")
                or s.startswith("=======")
                or s.startswith(">>>>>>>")
            ):
                continue
            lines.append(line)

        # Find the actual CSV header
        header_index = next(
            (
                i for i, line in enumerate(lines)
                if "Temperature" in line
                and "Classes" in line
                and "day" in line
            ),
            None,
        )

        if header_index is None:
            return None, "Could not find the CSV column header."

        df = pd.read_csv(
            StringIO("\n".join(lines[header_index:])),
            skipinitialspace=True,
        )

        df.columns = (
            df.columns.astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
        )

        # Clean column names and remove trailing spaces in text values
        if "Classes" in df.columns:
            df["Classes"] = df["Classes"].astype(str).str.strip()

        return df, None

    except Exception as e:
        return None, str(e)


df, load_error = load_data(DATASET_PATH)

if df is None:
    st.error("❌ dataset.csv could not be loaded.")
    st.write(load_error)
    st.info("Keep app.py and dataset.csv inside the same Project folder.")
    st.stop()

# =========================================================
# REQUIRED FEATURES
# =========================================================
features = [
    "Temperature", "RH", "Ws", "Rain",
    "FFMC", "DMC", "DC", "ISI", "BUI", "FWI"
]

missing_features = [f for f in features if f not in df.columns]

if missing_features:
    st.error("❌ The dataset loaded, but required columns are missing.")
    st.write("Missing columns:", missing_features)
    st.write("Available columns:", list(df.columns))
    st.info(
        "The working dataset should contain: "
        + ", ".join(features)
        + ", Classes, and Region."
    )
    st.stop()

# =========================================================
# TARGET COLUMN
# =========================================================
possible_targets = ["Classes", "class", "Class", "Target", "target"]
target_column = next((c for c in possible_targets if c in df.columns), None)

if target_column is None:
    st.error("❌ Target column 'Classes' was not found.")
    st.write("Available columns:", list(df.columns))
    st.stop()

# =========================================================
# DATA CLEANING
# =========================================================
data = df.copy()

for column in features:
    data[column] = pd.to_numeric(data[column], errors="coerce")

def encode_target(value):
    value = str(value).strip().lower()

    if value in {"0", "false", "not fire", "not_fire", "notfire"}:
        return 0
    if value in {"1", "true", "fire"}:
        return 1

    # Handles values such as "1.0" and "0.0".
    try:
        number = float(value)
        if number in (0, 1):
            return int(number)
    except ValueError:
        pass

    return np.nan

data["target"] = data[target_column].apply(encode_target)
data = data.dropna(subset=features + ["target"])
data["target"] = data["target"].astype(int)

if data.empty:
    st.error("❌ No usable rows remain after cleaning the dataset.")
    st.stop()

if data["target"].nunique() < 2:
    st.error("❌ The target column must contain both FIRE and NOT FIRE classes.")
    st.stop()

# =========================================================
# TRAIN MODELS
# =========================================================
@st.cache_resource
def train_models(data):
    X = data[features]
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        ),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier())
        ])
    }

    results = []
    best_model = None
    best_name = ""
    best_accuracy = -1

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        results.append({
            "Model": name,
            "Accuracy": accuracy
        })

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_name = name

    return (
        best_model,
        best_name,
        best_accuracy,
        pd.DataFrame(results)
    )

best_model, best_name, best_accuracy, results_df = train_models(data)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("📊 Model Information")
    st.success(f"Best Model: {best_name}")
    st.metric("Model Accuracy", f"{best_accuracy * 100:.2f}%")
    st.divider()
    st.write("### Dataset")
    st.write(f"Total Records: **{len(data)}**")
    st.write(f"Features: **{len(features)}**")

# =========================================================
# INPUT SECTION
# =========================================================
st.markdown("""
<div class="card">
    <h2>🌲 Enter Environmental Conditions</h2>
    <p>
        Enter weather and fire-weather values to predict
        the possibility of a forest fire.
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input(
        "🌡️ Temperature (°C)",
        min_value=0.0, max_value=60.0,
        value=30.0, step=0.1
    )
    rh = st.number_input(
        "💧 Relative Humidity (RH)",
        min_value=0.0, max_value=100.0,
        value=60.0, step=1.0
    )
    ws = st.number_input(
        "💨 Wind Speed (Ws)",
        min_value=0.0, max_value=100.0,
        value=15.0, step=1.0
    )
    rain = st.number_input(
        "🌧️ Rain",
        min_value=0.0, max_value=100.0,
        value=2.0, step=0.1
    )
    ffmc = st.number_input(
        "🔥 FFMC",
        min_value=0.0, max_value=100.0,
        value=75.0, step=0.1
    )

with col2:
    dmc = st.number_input(
        "🌲 DMC",
        min_value=0.0, max_value=100.0,
        value=15.0, step=0.1
    )
    dc = st.number_input(
        "🌳 DC",
        min_value=0.0, max_value=1000.0,
        value=40.0, step=0.1
    )
    isi = st.number_input(
        "🔥 ISI",
        min_value=0.0, max_value=100.0,
        value=5.0, step=0.1
    )
    bui = st.number_input(
        "🌿 BUI",
        min_value=0.0, max_value=100.0,
        value=15.0, step=0.1
    )
    fwi = st.number_input(
        "🔥 FWI",
        min_value=0.0, max_value=100.0,
        value=8.0, step=0.1
    )

# =========================================================
# PREDICTION
# =========================================================
st.write("")
predict_button = st.button("🔥 PREDICT FIRE RISK")

if predict_button:
    new_data = pd.DataFrame([{
        "Temperature": temperature,
        "RH": rh,
        "Ws": ws,
        "Rain": rain,
        "FFMC": ffmc,
        "DMC": dmc,
        "DC": dc,
        "ISI": isi,
        "BUI": bui,
        "FWI": fwi
    }], columns=features)

    prediction = int(best_model.predict(new_data)[0])

    if hasattr(best_model, "predict_proba"):
        probability = float(best_model.predict_proba(new_data)[0][1])
    else:
        probability = float(prediction)

    probability_percent = probability * 100

    if probability >= 0.70:
        st.markdown(f"""
        <div class="result-high">
            <h1>🔥 HIGH FIRE RISK</h1>
            <h3>Forest fire is highly possible</h3>
            <h2>{probability_percent:.2f}% Probability</h2>
        </div>
        """, unsafe_allow_html=True)
    elif probability >= 0.40:
        st.markdown(f"""
        <div class="result-medium">
            <h1>⚠️ MEDIUM FIRE RISK</h1>
            <h3>Fire possibility requires attention</h3>
            <h2>{probability_percent:.2f}% Probability</h2>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-low">
            <h1>🟢 LOW FIRE RISK</h1>
            <h3>Forest fire possibility is low</h3>
            <h2>{probability_percent:.2f}% Probability</h2>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Prediction",
            "🔥 FIRE" if prediction == 1 else "🟢 NOT FIRE"
        )
    with c2:
        st.metric("Fire Probability", f"{probability_percent:.2f}%")
    with c3:
        st.metric("AI Model", best_name)

# =========================================================
# MODEL COMPARISON
# =========================================================
with st.expander("📊 View Model Performance"):
    display_results = results_df.copy()
    display_results["Accuracy"] = (
        display_results["Accuracy"] * 100
    ).round(2)
    display_results = display_results.sort_values(
        "Accuracy", ascending=False
    )
    st.dataframe(display_results, use_container_width=True)

# =========================================================
# DATASET PREVIEW
# =========================================================
with st.expander("📁 View Dataset"):
    st.dataframe(data.head(20), use_container_width=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<hr>
<center>
<p>🔥 <b>Forest Fire Prediction System</b></p>
<p>Machine Learning Based Prediction Application</p>
</center>
""", unsafe_allow_html=True)