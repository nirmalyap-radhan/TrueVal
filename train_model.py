# TrueVaL - Car Price Prediction (Linear Regression + Polynomial Features)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib

# Load data
data = pd.read_csv("Cardetails.csv")

# Clean numeric columns
def extract_number(x):
    if isinstance(x, str):
        num = ''.join(c for c in x if (c.isdigit() or c == '.'))
        return float(num) if num else 0
    return x

for col in ["mileage", "engine", "max_power"]:
    data[col] = data[col].apply(extract_number)

# Clean torque column
def extract_torque(x):
    if isinstance(x, str):
        num = ''.join(c for c in x if (c.isdigit() or c == '.'))
        return float(num) if num else 0
    return x

data["torque"] = data["torque"].apply(extract_torque)

# Handle missing values
data = data.fillna(data.mean(numeric_only=True))

# Encode categorical features
label_encoders = {}
for col in ["name", "fuel", "seller_type", "transmission", "owner"]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Split data
X = data.drop("selling_price", axis=1)
y = data["selling_price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale and apply polynomial features
scaler = StandardScaler()
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(scaler.fit_transform(X_train))
X_test_poly = poly.transform(scaler.transform(X_test))

# Train model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Evaluate
y_pred = model.predict(X_test_poly)
r2 = r2_score(y_test, y_pred)
print(f"📊 R² Score (Accuracy): {r2:.2f}")

# Save model and tools
joblib.dump(model, "car_price_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(poly, "poly.pkl")
print("✅ Model and files saved!")
 