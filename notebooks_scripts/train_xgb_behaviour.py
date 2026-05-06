# notebooks_scripts/train_xgb_behaviour.py
import os, joblib
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb

print("Loading cleaned data...")
df = pd.read_csv("data/paysim_clean.csv")
if 'label' in df.columns:
    df = df.rename(columns={'label':'isFraud'})

if 'timestamp' not in df.columns and 'time_step' in df.columns:
    df['timestamp'] = df['time_step'] * 3600

df = df.sort_values(['customer_id','timestamp']).reset_index(drop=True)

# Build behavioural features (same as enrichment)
df['cust_txn_count'] = df.groupby('customer_id').cumcount()
df['cust_cum_mean'] = df.groupby('customer_id')['amount'].expanding().mean().shift().reset_index(level=0, drop=True).fillna(0.0)
df['amt_to_cum_mean'] = df['amount'] / df['cust_cum_mean'].replace(0, np.nan)
df['amt_to_cum_mean'] = df['amt_to_cum_mean'].fillna(1.0)
df['is_new_device'] = 0
if 'device_info' in df.columns:
    def cum_new_device_flag(series):
        seen = []
        out = []
        for v in series:
            out.append(0 if v in seen else 1)
            seen.append(v)
        return pd.Series(out).shift().fillna(0).astype(int).values
    df['is_new_device'] = df.groupby('customer_id')['device_info'].transform(cum_new_device_flag)
df['hour'] = ((df['timestamp']//3600) % 24).astype(int)
df['is_night'] = ((df['hour'] < 6) | (df['hour'] > 22)).astype(int)
df['amount_log'] = np.log1p(df['amount'])

# New engineered features
df['log_amt_to_cum_mean'] = np.log1p(df['amt_to_cum_mean'])
df['interaction_amt_log_x_log_ratio'] = df['amount_log'] * df['log_amt_to_cum_mean']

features = [
    'amount_log','hour','is_night',
    'cust_txn_count','cust_cum_mean','amt_to_cum_mean','is_new_device',
    'log_amt_to_cum_mean','interaction_amt_log_x_log_ratio'
]

df = df.dropna(subset=['isFraud']).reset_index(drop=True)
df['isFraud'] = df['isFraud'].astype(int)

frauds = df[df['isFraud']==1]
nonfrauds = df[df['isFraud']==0]
print("Fraud rows:", len(frauds), "Non-fraud rows:", len(nonfrauds))

# sample negatives to keep training reasonably sized
non_sample = nonfrauds.sample(n=min(300000, len(nonfrauds)), random_state=42)
df_train = pd.concat([frauds, non_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

X = df_train[features].fillna(0)
y = df_train['isFraud'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# scale numeric columns (keep scale for XGBoost; still OK)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# compute scale_pos_weight
n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
scale_pos_weight = max(1.0, float(n_neg) / max(1.0, n_pos))
print("scale_pos_weight:", scale_pos_weight)

dtrain = xgb.DMatrix(X_train_s, label=y_train)
dtest = xgb.DMatrix(X_test_s, label=y_test)

params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "scale_pos_weight": scale_pos_weight,
    "eta": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbosity": 1,
    "seed": 42
}

print("Training XGBoost...")
bst = xgb.train(params, dtrain, num_boost_round=80, evals=[(dtest,"eval")], verbose_eval=20)

y_proba = bst.predict(dtest)
y_pred = (y_proba >= 0.5).astype(int)
print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))
try:
    print("ROC-AUC:", round(roc_auc_score(y_test, y_proba),4))
except:
    pass

os.makedirs("models", exist_ok=True)
pipeline = {"scaler": scaler, "model": ("xgboost", bst), "features": features}
joblib.dump(pipeline, "models/pipeline.pkl")
print("Saved models/pipeline.pkl (XGBoost behavioural pipeline)")
