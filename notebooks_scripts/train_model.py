# notebooks_scripts/train_model_simple.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib, os

print("Loading cleaned PaySim data...")
df = pd.read_csv("data/paysim_clean.csv")

# Keep only relevant columns
df = df[[
    "amount","timestamp","city","lat","lon","device_info","merchant_id","label"
]].rename(columns={"label":"isFraud"})

# Basic feature engineering
df["amount_log"] = np.log1p(df["amount"])
df["hour"] = ((df["timestamp"] // 3600) % 24).astype(int)
df["is_night"] = ((df["hour"] < 6) | (df["hour"] > 22)).astype(int)

# Create categorical mappings (save these for consumer)
print("Creating categorical mappings...")
device_cat = pd.Categorical(df["device_info"])
city_cat = pd.Categorical(df["city"])
merchant_cat = pd.Categorical(df["merchant_id"])

device_map = {v: i for i, v in enumerate(device_cat.categories)}
city_map = {v: i for i, v in enumerate(city_cat.categories)}
merchant_map = {v: i for i, v in enumerate(merchant_cat.categories)}

# Apply codes
df["device_code"] = df["device_info"].map(device_map).fillna(-1).astype(int)
df["city_code"] = df["city"].map(city_map).fillna(-1).astype(int)
df["merchant_code"] = df["merchant_id"].map(merchant_map).fillna(-1).astype(int)

features = [
    "amount_log", "hour", "is_night",
    "lat", "lon",
    "device_code", "city_code", "merchant_code"
]

X = df[features].fillna(0)
y = df["isFraud"].astype(int)

print("Train/test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

print("Scaling + training (RandomForest)...")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=120, class_weight="balanced", n_jobs=-1, random_state=42)
clf.fit(X_train_s, y_train)

y_proba = clf.predict_proba(X_test_s)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))
try:
    print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))
except Exception:
    print("ROC-AUC: could not compute")

# Save pipeline and encoders
os.makedirs("models", exist_ok=True)
pipeline = {"scaler": scaler, "model": clf, "features": features}
joblib.dump(pipeline, "models/pipeline.pkl")
encoders = {"device_map": device_map, "city_map": city_map, "merchant_map": merchant_map}
joblib.dump(encoders, "models/encoders.pkl")

print("\nSaved models/pipeline.pkl and models/encoders.pkl")
print("Done.")
