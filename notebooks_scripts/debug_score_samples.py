import joblib, pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler

print("Loading pipeline...")
p = joblib.load("models/pipeline.pkl")
scaler = p["scaler"]
model = p["model"]
features = p["features"]
print("Pipeline features:", features)

df = pd.read_csv("data/paysim_clean.csv")
# pick some fraud rows and some safe rows
frauds = df[df['label']==1].sample(10, random_state=42)
safes = df[df['label']==0].sample(10, random_state=42)

def prepare_row(row):
    # build same features as training script
    amount = row['amount']
    ts = int(row.get('timestamp', row.get('time_step',0) * 3600))
    amount_log = np.log1p(amount)
    hour = (ts // 3600) % 24
    is_night = 1 if (hour < 6 or hour > 22) else 0
    # historical features from training may be 0 in dataset; approximate using available fields
    cust_txn_count = 0
    cust_cum_mean = 0.0
    amt_to_cum_mean = 1.0
    is_new_device = 0
    vec = {
        "amount_log": amount_log,
        "hour": int(hour),
        "is_night": int(is_night),
        "cust_txn_count": cust_txn_count,
        "cust_cum_mean": cust_cum_mean,
        "amt_to_cum_mean": amt_to_cum_mean,
        "is_new_device": is_new_device
    }
    return vec

def score(df_rows, label):
    for idx, r in df_rows.iterrows():
        v = prepare_row(r)
        X = pd.DataFrame([v])[features]
        Xs = scaler.transform(X)
        prob = float(model.predict_proba(Xs)[0][1])
        print(f"{label} idx={idx} amount={r['amount']:.2f} hour={v['hour']} prob={prob:.4f}")

print("=== FRAUD EXAMPLES ===")
score(frauds, "FRAUD")
print("\n=== SAFE EXAMPLES ===")
score(safes, "SAFE")
