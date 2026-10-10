
# ============================================================
# FOREST FIRE PREDICTION USING MACHINE LEARNING
# Dataset: Algerian Forest Fires Dataset
# ============================================================

# -----------------------------
# 1. IMPORT LIBRARIES
# -----------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 2. LOAD THE DATASET
# ============================================================

file_path = "dataset.csv"

# The uploaded dataset contains two sections:
# 1. Bejaia Region Dataset
# 2. Sidi-Bel Abbes Region Dataset

# Read the file manually because the two sections have
# separate headers.

with open(file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()


# Find the two header rows
header_indices = []

for i, line in enumerate(lines):
    if line.strip().startswith("day,month,year"):
        header_indices.append(i)


# Make sure both headers were found
if len(header_indices) < 2:
    raise ValueError("Could not find both dataset sections.")


# First region
bejaia_start = header_indices[0]
bejaia_end = header_indices[1]

bejaia_data = lines[bejaia_start:bejaia_end]


# Second region
sidi_start = header_indices[1]
sidi_data = lines[sidi_start:]


# Convert each region into DataFrames
from io import StringIO

df_bejaia = pd.read_csv(StringIO("".join(bejaia_data)))
df_sidi = pd.read_csv(StringIO("".join(sidi_data)))


# Add region information
df_bejaia["Region"] = "Bejaia"
df_sidi["Region"] = "Sidi-Bel Abbes"


# Combine both regions
df = pd.concat(
    [df_bejaia, df_sidi],
    ignore_index=True
)


print("\n========================================")
print("DATASET LOADED SUCCESSFULLY")
print("========================================")

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 3. CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()

print("\nColumn names:")
print(df.columns.tolist())


# Remove spaces from values
for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].astype(str).str.strip()


# ============================================================
# 4. CHECK DATASET INFORMATION
# ============================================================

print("\n========================================")
print("DATASET INFORMATION")
print("========================================")

print(df.info())


print("\nMissing values:")
print(df.isnull().sum())


print("\nStatistical summary:")
print(df.describe())


# ============================================================
# 5. CLEAN NUMERIC COLUMNS
# ============================================================

# Columns that should contain numerical values
numeric_columns = [
    "day",
    "month",
    "year",
    "Temperature",
    "RH",
    "Ws",
    "Rain",
    "FFMC",
    "DMC",
    "DC",
    "ISI",
    "BUI",
    "FWI"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# Remove rows with missing values
df = df.dropna().reset_index(drop=True)


# ============================================================
# 6. CLEAN TARGET COLUMN
# ============================================================

df["Classes"] = (
    df["Classes"]
    .str.lower()
    .str.strip()
)


# Sometimes dataset values may contain spaces
df["Classes"] = df["Classes"].replace({
    "fire": 1,
    "not fire": 0
})


# Convert target to integer
df["Classes"] = pd.to_numeric(
    df["Classes"],
    errors="coerce"
)


# Remove invalid target rows
df = df.dropna(subset=["Classes"])


df["Classes"] = df["Classes"].astype(int)


print("\nTarget distribution:")
print(df["Classes"].value_counts())


# ============================================================
# 7. BASIC DATA VISUALIZATION
# ============================================================

print("\n========================================")
print("CREATING VISUALIZATIONS")
print("========================================")


# ----------------------------------------
# 7.1 Fire vs Not Fire
# ----------------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    x=df["Classes"]
)

plt.title("Fire vs Not Fire")
plt.xlabel("Class (0 = Not Fire, 1 = Fire)")
plt.ylabel("Number of Records")

plt.tight_layout()
plt.show()


# ----------------------------------------
# 7.2 Temperature Distribution
# ----------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Temperature",
    hue="Classes",
    kde=True
)

plt.title("Temperature Distribution")
plt.xlabel("Temperature")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# ----------------------------------------
# 7.3 Humidity vs Fire
# ----------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Classes",
    y="RH"
)

plt.title("Humidity vs Fire Occurrence")
plt.xlabel("Class (0 = Not Fire, 1 = Fire)")
plt.ylabel("Relative Humidity")

plt.tight_layout()
plt.show()


# ----------------------------------------
# 7.4 Temperature vs Humidity
# ----------------------------------------

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="Temperature",
    y="RH",
    hue="Classes",
    style="Classes"
)

plt.title("Temperature vs Relative Humidity")
plt.xlabel("Temperature")
plt.ylabel("Relative Humidity")

plt.tight_layout()
plt.show()


# ----------------------------------------
# 7.5 Correlation Heatmap
# ----------------------------------------

plt.figure(figsize=(12, 8))

correlation = df[numeric_columns + ["Classes"]].corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()


# ============================================================
# 8. PREPARE FEATURES AND TARGET
# ============================================================

# We don't need the region for this basic ML model.
#
# We also remove year because it does not provide useful
# weather information for our prediction.
#
# Day and month can optionally be retained because seasonal
# effects can influence fire occurrence.

features = [
    "Temperature",
    "RH",
    "Ws",
    "Rain",
    "FFMC",
    "DMC",
    "DC",
    "ISI",
    "BUI",
    "FWI"
]

X = df[features]

y = df["Classes"]


print("\n========================================")
print("FEATURES")
print("========================================")

print(X.head())

print("\nTarget:")
print(y.head())


# ============================================================
# 9. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 10. CREATE MACHINE LEARNING MODELS
# ============================================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000
        ))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=5
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        max_depth=10
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    ),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(
            n_neighbors=5
        ))
    ])
}


# ============================================================
# 11. TRAIN AND EVALUATE MODELS
# ============================================================

results = []


print("\n========================================")
print("MODEL TRAINING")
print("========================================")


for name, model in models.items():

    print("\nTraining:", name)

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))


# ============================================================
# 12. COMPARE MODELS
# ============================================================

results_df = pd.DataFrame(results)

print("\n========================================")
print("MODEL COMPARISON")
print("========================================")

print(
    results_df.sort_values(
        by="Accuracy",
        ascending=False
    )
)


# ============================================================
# 13. VISUALIZE MODEL ACCURACY
# ============================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=results_df,
    x="Accuracy",
    y="Model"
)

plt.title("Machine Learning Model Accuracy")
plt.xlabel("Accuracy")
plt.ylabel("Model")

plt.xlim(0, 1)

plt.tight_layout()
plt.show()


# ============================================================
# 14. SELECT BEST MODEL
# ============================================================

best_model_name = results_df.loc[
    results_df["Accuracy"].idxmax(),
    "Model"
]

best_model = models[best_model_name]


print("\n========================================")
print("BEST MODEL")
print("========================================")

print("Best Model:", best_model_name)


# ============================================================
# 15. DETAILED EVALUATION OF BEST MODEL
# ============================================================

best_predictions = best_model.predict(X_test)


print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        y_test,
        best_predictions,
        target_names=[
            "Not Fire",
            "Fire"
        ]
    )
)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    best_predictions
)


plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Not Fire",
        "Fire"
    ],
    yticklabels=[
        "Not Fire",
        "Fire"
    ]
)

plt.title(
    "Confusion Matrix - " +
    best_model_name
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()


# ============================================================
# 17. FEATURE IMPORTANCE
# ============================================================

# Feature importance is available for tree-based models.

if hasattr(best_model, "feature_importances_"):

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": best_model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\n========================================")
    print("FEATURE IMPORTANCE")
    print("========================================")

    print(importance)


    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=importance,
        x="Importance",
        y="Feature"
    )

    plt.title(
        "Feature Importance - " +
        best_model_name
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# 18. SAVE THE BEST MODEL
# ============================================================

joblib.dump(
    best_model,
    "forest_fire_model.pkl"
)

print("\nBest model saved as:")
print("forest_fire_model.pkl")


# ============================================================
# 19. MAKE A NEW PREDICTION
# ============================================================

print("\n========================================")
print("NEW FOREST FIRE PREDICTION")
print("========================================")


# Example input
new_data = pd.DataFrame({

    "Temperature": [38],

    "RH": [25],

    "Ws": [18],

    "Rain": [0],

    "FFMC": [85],

    "DMC": [12],

    "DC": [45],

    "ISI": [5],

    "BUI": [15],

    "FWI": [7]

})


prediction = best_model.predict(
    new_data
)


# Probability
if hasattr(
    best_model,
    "predict_proba"
):

    probability = best_model.predict_proba(
        new_data
    )[0][1]

else:

    probability = None


if prediction[0] == 1:

    print("\n🔥 FOREST FIRE PREDICTION: FIRE")

else:

    print("\n🟢 FOREST FIRE PREDICTION: NOT FIRE")


if probability is not None:

    print(
        "Fire Probability:",
        round(probability * 100, 2),
        "%"
    )


# ============================================================
# 20. USER INPUT PREDICTION
# ============================================================

print("\n========================================")
print("ENTER YOUR OWN WEATHER DATA")
print("========================================")

try:

    temperature = float(
        input("Temperature (°C): ")
    )

    humidity = float(
        input("Relative Humidity (%): ")
    )

    wind_speed = float(
        input("Wind Speed (km/h): ")
    )

    rainfall = float(
        input("Rainfall (mm): ")
    )

    ffmc = float(
        input("FFMC: ")
    )

    dmc = float(
        input("DMC: ")
    )

    dc = float(
        input("DC: ")
    )

    isi = float(
        input("ISI: ")
    )

    bui = float(
        input("BUI: ")
    )

    fwi = float(
        input("FWI: ")
    )


    user_data = pd.DataFrame({

        "Temperature": [temperature],

        "RH": [humidity],

        "Ws": [wind_speed],

        "Rain": [rainfall],

        "FFMC": [ffmc],

        "DMC": [dmc],

        "DC": [dc],

        "ISI": [isi],

        "BUI": [bui],

        "FWI": [fwi]

    })


    user_prediction = best_model.predict(
        user_data
    )


    if hasattr(
        best_model,
        "predict_proba"
    ):

        user_probability = (
            best_model
            .predict_proba(user_data)[0][1]
        )

    else:

        user_probability = None


    print("\n========================================")
    print("FINAL RESULT")
    print("========================================")


    if user_prediction[0] == 1:

        print("🔥 HIGH FIRE POSSIBILITY")

    else:

        print("🟢 LOW FIRE POSSIBILITY")


    if user_probability is not None:

        print(
            "Estimated Fire Probability:",
            round(
                user_probability * 100,
                2
            ),
            "%"
        )


except ValueError:

    print(
        "\nInvalid input. "
        "Please enter numerical values."
    )


print("\n========================================")
print("PROJECT COMPLETED")
print("========================================")
=======
# ============================================================
# FOREST FIRE PREDICTION USING MACHINE LEARNING
# Dataset: Algerian Forest Fires Dataset
# ============================================================

# -----------------------------
# 1. IMPORT LIBRARIES
# -----------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 2. LOAD THE DATASET
# ============================================================

file_path = "dataset.csv"

# The uploaded dataset contains two sections:
# 1. Bejaia Region Dataset
# 2. Sidi-Bel Abbes Region Dataset

# Read the file manually because the two sections have
# separate headers.

with open(file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()


# Find the two header rows
header_indices = []

for i, line in enumerate(lines):
    if line.strip().startswith("day,month,year"):
        header_indices.append(i)


# Make sure both headers were found
if len(header_indices) < 2:
    raise ValueError("Could not find both dataset sections.")


# First region
bejaia_start = header_indices[0]
bejaia_end = header_indices[1]

bejaia_data = lines[bejaia_start:bejaia_end]


# Second region
sidi_start = header_indices[1]
sidi_data = lines[sidi_start:]


# Convert each region into DataFrames
from io import StringIO

df_bejaia = pd.read_csv(StringIO("".join(bejaia_data)))
df_sidi = pd.read_csv(StringIO("".join(sidi_data)))


# Add region information
df_bejaia["Region"] = "Bejaia"
df_sidi["Region"] = "Sidi-Bel Abbes"


# Combine both regions
df = pd.concat(
    [df_bejaia, df_sidi],
    ignore_index=True
)


print("\n========================================")
print("DATASET LOADED SUCCESSFULLY")
print("========================================")

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 3. CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()

print("\nColumn names:")
print(df.columns.tolist())


# Remove spaces from values
for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].astype(str).str.strip()


# ============================================================
# 4. CHECK DATASET INFORMATION
# ============================================================

print("\n========================================")
print("DATASET INFORMATION")
print("========================================")

print(df.info())


print("\nMissing values:")
print(df.isnull().sum())


print("\nStatistical summary:")
print(df.describe())


# ============================================================
# 5. CLEAN NUMERIC COLUMNS
# ============================================================

# Columns that should contain numerical values
numeric_columns = [
    "day",
    "month",
    "year",
    "Temperature",
    "RH",
    "Ws",
    "Rain",
    "FFMC",
    "DMC",
    "DC",
    "ISI",
    "BUI",
    "FWI"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# Remove rows with missing values
df = df.dropna().reset_index(drop=True)


# ============================================================
# 6. CLEAN TARGET COLUMN
# ============================================================

df["Classes"] = (
    df["Classes"]
    .str.lower()
    .str.strip()
)


# Sometimes dataset values may contain spaces
df["Classes"] = df["Classes"].replace({
    "fire": 1,
    "not fire": 0
})


# Convert target to integer
df["Classes"] = pd.to_numeric(
    df["Classes"],
    errors="coerce"
)


# Remove invalid target rows
df = df.dropna(subset=["Classes"])


df["Classes"] = df["Classes"].astype(int)


print("\nTarget distribution:")
print(df["Classes"].value_counts())


# ============================================================
# 7. BASIC DATA VISUALIZATION
# ============================================================

print("\n========================================")
print("CREATING VISUALIZATIONS")
print("========================================")


# ----------------------------------------
# 7.1 Fire vs Not Fire
# ----------------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    x=df["Classes"]
)

plt.title("Fire vs Not Fire")
plt.xlabel("Class (0 = Not Fire, 1 = Fire)")
plt.ylabel("Number of Records")

plt.tight_layout()
plt.show()


# ----------------------------------------
# 7.2 Temperature Distribution
# ----------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Temperature",
    hue="Classes",
    kde=True
)

plt.title("Temperature Distribution")
plt.xlabel("Temperature")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# ----------------------------------------
# 7.3 Humidity vs Fire
# ----------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Classes",
    y="RH"
)

plt.title("Humidity vs Fire Occurrence")
plt.xlabel("Class (0 = Not Fire, 1 = Fire)")
plt.ylabel("Relative Humidity")

plt.tight_layout()
plt.show()


# ----------------------------------------
# 7.4 Temperature vs Humidity
# ----------------------------------------

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="Temperature",
    y="RH",
    hue="Classes",
    style="Classes"
)

plt.title("Temperature vs Relative Humidity")
plt.xlabel("Temperature")
plt.ylabel("Relative Humidity")

plt.tight_layout()
plt.show()


# ----------------------------------------
# 7.5 Correlation Heatmap
# ----------------------------------------

plt.figure(figsize=(12, 8))

correlation = df[numeric_columns + ["Classes"]].corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()


# ============================================================
# 8. PREPARE FEATURES AND TARGET
# ============================================================

# We don't need the region for this basic ML model.
#
# We also remove year because it does not provide useful
# weather information for our prediction.
#
# Day and month can optionally be retained because seasonal
# effects can influence fire occurrence.

features = [
    "Temperature",
    "RH",
    "Ws",
    "Rain",
    "FFMC",
    "DMC",
    "DC",
    "ISI",
    "BUI",
    "FWI"
]

X = df[features]

y = df["Classes"]


print("\n========================================")
print("FEATURES")
print("========================================")

print(X.head())

print("\nTarget:")
print(y.head())


# ============================================================
# 9. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 10. CREATE MACHINE LEARNING MODELS
# ============================================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000
        ))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=5
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        max_depth=10
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    ),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(
            n_neighbors=5
        ))
    ])
}


# ============================================================
# 11. TRAIN AND EVALUATE MODELS
# ============================================================

results = []


print("\n========================================")
print("MODEL TRAINING")
print("========================================")


for name, model in models.items():

    print("\nTraining:", name)

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))


# ============================================================
# 12. COMPARE MODELS
# ============================================================

results_df = pd.DataFrame(results)

print("\n========================================")
print("MODEL COMPARISON")
print("========================================")

print(
    results_df.sort_values(
        by="Accuracy",
        ascending=False
    )
)


# ============================================================
# 13. VISUALIZE MODEL ACCURACY
# ============================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=results_df,
    x="Accuracy",
    y="Model"
)

plt.title("Machine Learning Model Accuracy")
plt.xlabel("Accuracy")
plt.ylabel("Model")

plt.xlim(0, 1)

plt.tight_layout()
plt.show()


# ============================================================
# 14. SELECT BEST MODEL
# ============================================================

best_model_name = results_df.loc[
    results_df["Accuracy"].idxmax(),
    "Model"
]

best_model = models[best_model_name]


print("\n========================================")
print("BEST MODEL")
print("========================================")

print("Best Model:", best_model_name)


# ============================================================
# 15. DETAILED EVALUATION OF BEST MODEL
# ============================================================

best_predictions = best_model.predict(X_test)


print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        y_test,
        best_predictions,
        target_names=[
            "Not Fire",
            "Fire"
        ]
    )
)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    best_predictions
)


plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Not Fire",
        "Fire"
    ],
    yticklabels=[
        "Not Fire",
        "Fire"
    ]
)

plt.title(
    "Confusion Matrix - " +
    best_model_name
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()


# ============================================================
# 17. FEATURE IMPORTANCE
# ============================================================

# Feature importance is available for tree-based models.

if hasattr(best_model, "feature_importances_"):

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": best_model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\n========================================")
    print("FEATURE IMPORTANCE")
    print("========================================")

    print(importance)


    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=importance,
        x="Importance",
        y="Feature"
    )

    plt.title(
        "Feature Importance - " +
        best_model_name
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# 18. SAVE THE BEST MODEL
# ============================================================

joblib.dump(
    best_model,
    "forest_fire_model.pkl"
)

print("\nBest model saved as:")
print("forest_fire_model.pkl")


# ============================================================
# 19. MAKE A NEW PREDICTION
# ============================================================

print("\n========================================")
print("NEW FOREST FIRE PREDICTION")
print("========================================")


# Example input
new_data = pd.DataFrame({

    "Temperature": [38],

    "RH": [25],

    "Ws": [18],

    "Rain": [0],

    "FFMC": [85],

    "DMC": [12],

    "DC": [45],

    "ISI": [5],

    "BUI": [15],

    "FWI": [7]

})


prediction = best_model.predict(
    new_data
)


# Probability
if hasattr(
    best_model,
    "predict_proba"
):

    probability = best_model.predict_proba(
        new_data
    )[0][1]

else:

    probability = None


if prediction[0] == 1:

    print("\n🔥 FOREST FIRE PREDICTION: FIRE")

else:

    print("\n🟢 FOREST FIRE PREDICTION: NOT FIRE")


if probability is not None:

    print(
        "Fire Probability:",
        round(probability * 100, 2),
        "%"
    )


# ============================================================
# 20. USER INPUT PREDICTION
# ============================================================

print("\n========================================")
print("ENTER YOUR OWN WEATHER DATA")
print("========================================")

try:

    temperature = float(
        input("Temperature (°C): ")
    )

    humidity = float(
        input("Relative Humidity (%): ")
    )

    wind_speed = float(
        input("Wind Speed (km/h): ")
    )

    rainfall = float(
        input("Rainfall (mm): ")
    )

    ffmc = float(
        input("FFMC: ")
    )

    dmc = float(
        input("DMC: ")
    )

    dc = float(
        input("DC: ")
    )

    isi = float(
        input("ISI: ")
    )

    bui = float(
        input("BUI: ")
    )

    fwi = float(
        input("FWI: ")
    )


    user_data = pd.DataFrame({

        "Temperature": [temperature],

        "RH": [humidity],

        "Ws": [wind_speed],

        "Rain": [rainfall],

        "FFMC": [ffmc],

        "DMC": [dmc],

        "DC": [dc],

        "ISI": [isi],

        "BUI": [bui],

        "FWI": [fwi]

    })


    user_prediction = best_model.predict(
        user_data
    )


    if hasattr(
        best_model,
        "predict_proba"
    ):

        user_probability = (
            best_model
            .predict_proba(user_data)[0][1]
        )

    else:

        user_probability = None


    print("\n========================================")
    print("FINAL RESULT")
    print("========================================")


    if user_prediction[0] == 1:

        print("🔥 HIGH FIRE POSSIBILITY")

    else:

        print("🟢 LOW FIRE POSSIBILITY")


    if user_probability is not None:

        print(
            "Estimated Fire Probability:",
            round(
                user_probability * 100,
                2
            ),
            "%"
        )


except ValueError:

    print(
        "\nInvalid input. "
        "Please enter numerical values."
    )


print("\n========================================")
print("PROJECT COMPLETED")
print("========================================")

