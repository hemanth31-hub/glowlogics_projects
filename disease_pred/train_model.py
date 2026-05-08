import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

df = pd.read_csv(
    "diabetes_prediction_dataset.csv"
)

print(df.head())

# ---------------------------------------------------
# HANDLE MISSING VALUES
# ---------------------------------------------------

df.dropna(inplace=True)

# ---------------------------------------------------
# LABEL ENCODING
# ---------------------------------------------------

label_encoder = LabelEncoder()

categorical_columns = [
    "gender",
    "smoking_history"
]

for column in categorical_columns:

    df[column] = label_encoder.fit_transform(
        df[column]
    )

# ---------------------------------------------------
# FEATURE SELECTION
# ---------------------------------------------------

X = df[
    [
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "smoking_history",
        "bmi",
        "HbA1c_level",
        "blood_glucose_level"
    ]
]

y = df["diabetes"]

# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------
# FEATURE SCALING
# ---------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train_scaled,
    y_train
)

# ---------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------

y_pred = model.predict(X_test_scaled)

# ---------------------------------------------------
# EVALUATION
# ---------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# ---------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

# ---------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------

importance = model.feature_importances_

feature_names = X.columns

feature_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

feature_df = feature_df.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(10,5))

sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_df
)

plt.title("Feature Importance")

plt.show()

# ---------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------

joblib.dump(
    model,
    "model.pkl"
)

joblib.dump(
    scaler,
    "scaler.pkl"
)

print("Model Saved Successfully")