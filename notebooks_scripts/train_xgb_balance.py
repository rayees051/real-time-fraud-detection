# notebooks_scripts/train_xgb_balance.py
import pandas as pd, numpy as np, joblib, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb

print("Loading cleaned data...")
df = pd.read_csv("data/paysim_clean.csv")

# keep columns used earlier
df = df[["amount","timestamp","city","lat","lon","device_info","merchant_id","label"]].rename(columns={"label":"isFraud"})

# simple features
df["amount_log"] = np.log1p(df["amount"])
df["hour"] = ((df["timestamp"] // 3600) % 24).astype(int)
df["is_night"] = ((df["hour"] < 6) | (df["hour"] > 22)).astype(int)

# encode categories (map & save)
device_cat = pd.Categorical(df["device_info"])
city_cat = pd.Categorical(df["city"])
merchant_cat = pd.Categorical(df["merchant_id"])

device_map = {v:i for i,v in enumerate(device_cat.categories)}
city_map = {v:i for i,v in enumerate(city_cat.categories)}
merchant_map = {v:i for i,v in enumerate(merchant_cat.categories)}

df["device_code"] = df["device_info"].map(device_map).fillna(-1).astype(int)
df["city_code"]   = df["city"].map(city_map).fillna(-1).astype(int)
df["merchant_code"] = df["merchant_id"].map(merchant_map).fillna(-1).astype(int)

features = ["amount_log","hour","is_night","lat","lon","device_code","city_code","merchant_code"]
X = df[features].fillna(0)
y = df["isFraud"].astype(int)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# compute scale_pos_weight
n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
scale_pos_weight = max(1.0, float(n_neg) / max(1.0, n_pos))
print("scale_pos_weight:", scale_pos_weight, " n_pos:", n_pos, " n_neg:", n_neg)

# scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# XGBoost train
params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "use_label_encoder": False,
    "scale_pos_weight": scale_pos_weight,
    "eta": 0.1,
    "max_depth": 6,
    "n_jobs": -1,
    "random_state": 42
}

dtrain = xgb.DMatrix(X_train_s, label=y_train)
dtest = xgb.DMatrix(X_test_s, label=y_test)
num_round = 200

print("Training XGBoost (may take a few minutes)...")
bst = xgb.train(params, dtrain, num_boost_round=num_round, evals=[(dtest,"eval")], verbose_eval=20)

# predict and evaluate
y_proba = bst.predict(dtest)
y_pred = (y_proba >= 0.5).astype(int)
print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))
try:
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
except:
    pass

# save pipeline and encoders
os.makedirs("models", exist_ok=True)
pipeline = {"scaler": scaler, "model": ("xgboost", bst), "features": features}
joblib.dump(pipeline, "models/pipeline.pkl")
encoders = {"device_map": device_map, "city_map": city_map, "merchant_map": merchant_map}
joblib.dump(encoders, "models/encoders.pkl")
print("Saved models/pipeline.pkl and models/encoders.pkl")
