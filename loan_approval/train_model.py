import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# 1. Load the dataset (using your specific filename)
df = pd.read_csv('loan_approval_dataset.csv')

# 2. Preprocessing: Strip leading spaces from column names
df.columns = df.columns.str.strip()

# 3. Encoding Categorical Variables
# Education: Graduate=0, Not Graduate=1 | Self_Employed: Yes=1, No=0
le_edu = LabelEncoder()
df['education'] = le_edu.fit_transform(df['education'])

le_emp = LabelEncoder()
df['self_employed'] = le_emp.fit_transform(df['self_employed'])

# Target Variable: Approved=0, Rejected=1
le_status = LabelEncoder()
df['loan_status'] = le_status.fit_transform(df['loan_status'])

# 4. Feature Selection
# Drop loan_id as it's not a predictor
X = df.drop(columns=['loan_id', 'loan_status'])
y = df['loan_status']

# 5. Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluation
y_pred = model.predict(X_test)
print(f"Model Training Complete. Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# 7. Save the model and encoders for the UI
model_data = {
    'model': model,
    'le_edu': le_edu,
    'le_emp': le_emp,
    'le_status': le_status
}

with open('loan_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("File 'loan_model.pkl' created successfully.")