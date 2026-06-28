import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Dataset
data = pd.read_csv("data.csv")

# Data Cleaning
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data["TotalCharges"] = data["TotalCharges"].fillna(data["TotalCharges"].mean())

# Save Customer IDs separately (useful for deployment)
customer_ids = data["customerID"]

# Drop Customer ID
data = data.drop("customerID", axis=1)

# Label Encoding
label_encoders = {}

for column in data.select_dtypes(include="object").columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Features & Target
X = data.drop("Churn", axis=1)
y = data["Churn"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("="*40)
print(f"Model Accuracy : {accuracy:.4f}")
print("="*40)

# Save Model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save Encoders
with open("label_encoders.pkl", "wb") as file:
    pickle.dump(label_encoders, file)

print("\n✅ model.pkl saved successfully!")
print("✅ label_encoders.pkl saved successfully!")