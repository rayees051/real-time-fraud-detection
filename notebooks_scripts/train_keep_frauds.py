# notebooks_scripts/train_keep_frauds.py
import pandas as pd, numpy as np, joblib, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv("data/paysim_clean.csv")
# Build same behavioural features used earlier (run your preprocess/train script logic to compute cust_cum_mean etc.)
# For simplicity here, compute basic per-customer cumulative features via groupby-expanding like earlier train_model.py
df = df.sort_values(['customer_id','timestamp']).reset_index(drop=True)
df['cust_txn_count'] = df.groupby('customer_id').cumcount()
df['cust_cum_mean'] = df.groupby('customer_id')['amount'].expanding().mean().shift().reset_index(level=0, drop=True).fillna(0)
df['is_new_device'] = 0  # if no device track in dataset; for demo we can use 0 or compute if device present
df['amount_log'] = np.log1p(df['amount'])
df['hour'] = ((df['timestamp']//3600)%24).astype(int)
df['is_night'] = ((df['hour']<6)|(df['hour']>22)).astype(int)
df['amt_to_cum_mean'] = df['amount'] / df['cust_cum_mean'].replace(0,np.nan)
df['amt_to_cum_mean'] = df['amt_to_cum_mean'].fillna(1.0)

features = ['amount_log','hour','is_night','cust_txn_count','cust_cum_mean','amt_to_cum_mean','is_new_device']
# Keep all frauds + sample negatives
frauds = df[df['label']==1]
nonfrauds = df[df['label']==0].sample(n=min(200000, df[df['label']==0].shape[0]), random_state=42)
df_small = pd.concat([frauds, nonfrauds]).sample(frac=1, random_state=42).reset_index(drop=True)

X = df_small[features].fillna(0)
y = df_small['label'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler(); X_train_s = scaler.fit_transform(X_train); X_test_s = scaler.transform(X_test)
clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=42)
clf.fit(X_train_s,y_train)
y_proba = clf.predict_proba(X_test_s)[:,1]
y_pred = (y_proba >= 0.5).astype(int)
print(classification_report(y_test,y_pred))
joblib.dump({"scaler":scaler,"model":clf,"features":features},"models/pipeline.pkl")
print("Saved models/pipeline.pkl")
