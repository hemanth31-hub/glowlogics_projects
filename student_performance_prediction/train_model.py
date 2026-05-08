import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# ===================== LOAD DATA =====================
df = pd.read_csv('student_performance_dataset.csv')

print("Original Columns:", df.columns.tolist())

# ===================== MAPPINGS =====================
mappings = {
    'Gender': {'Male': 0, 'Female': 1},
    'Parental_Education_Level': {'High School': 0, 'Bachelors': 1, 'Masters': 2, 'PhD': 3},
    'Internet_Access_at_Home': {'No': 0, 'Yes': 1},
    'Extracurricular_Activities': {'No': 0, 'Yes': 1},
    'Pass_Fail': {'Fail': 0, 'Pass': 1}
}

# Apply mappings
for col, mapping in mappings.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)
    else:
        print(f"Warning: Column {col} not found!")

# ===================== FEATURE SELECTION =====================
feature_columns = [
    'Gender', 
    'Study_Hours_per_Week', 
    'Attendance_Rate', 
    'Past_Exam_Scores', 
    'Parental_Education_Level', 
    'Internet_Access_at_Home', 
    'Extracurricular_Activities'
]

X = df[feature_columns]
y = df['Pass_Fail']

# ===================== TRAIN TEST SPLIT =====================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===================== MODEL TRAINING =====================
model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=12, 
    min_samples_split=5,
    random_state=42
)

model.fit(X_train, y_train)

# ===================== EVALUATION =====================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n Model Training Completed!")
print(f"Accuracy: {accuracy*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ===================== SAVE MODEL =====================
model_data = {
    'model': model,
    'mappings': mappings,
    'features': feature_columns,
    'feature_importance': dict(zip(feature_columns, model.feature_importances_))
}

with open('student_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("\n🎉 Model saved successfully as 'student_model.pkl'")