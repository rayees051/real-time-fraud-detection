# notebooks_scripts/print_features_for_tx.py
import joblib, json, pandas as pd, numpy as np, os, sys
p = joblib.load("models/pipeline.pkl")
scaler = p['scaler']; model = p['model']; features = p['features']

# read last enriched message we processed
pred_file = "predictions/predictions.csv"
if not os.path.exists(pred_file):
    print("predictions file not found:", pred_file); sys.exit(1)

df = pd.read_csv(pred_file)
# look for last transaction with large amount (>=20000) or user-specified
cands = df.sort_values("timestamp").tail(50)  # last 50
print("Showing last 50 prediction records (most recent last):")
print(cands[['timestamp','customer_id','amount','probability','prediction']].to_string(index=False))

# use the last row as our target
last = df.tail(1).iloc[0]
raw = json.loads(last['raw'])
print("\n-- RAW ENRICHED RECORD --")
print(json.dumps(raw, indent=2))

# build same feature vector as consumer
row = {
  'amount_log': np.log1p(float(raw.get('amount',0))),
  'hour': int((int(raw.get('timestamp'))//3600)%24),
  'is_night': 1 if ((int(raw.get('timestamp'))//3600)%24)<6 or ((int(raw.get('timestamp'))//3600)%24)>22 else 0,
  'cust_txn_count': float(raw.get('cust_txn_count',0)),
  'cust_cum_mean': float(raw.get('cust_cum_mean',0)),
  'amt_to_cum_mean': float(raw.get('amt_to_cum_mean',1)),
  'is_new_device': int(raw.get('is_new_device',0))
}
print("\n-- FEATURE ROW (raw) --")
print(row)

X = pd.DataFrame([row])[features].fillna(0)
print("\n-- FEATURES IN TRAINING ORDER --")
for f,v in X.iloc[0].items():
    print(f, ":", float(v))

Xs = scaler.transform(X)
print("\n-- SCALED VECTOR (first 8 values) --")
print(Xs.tolist()[0])

try:
    prob = float(model.predict_proba(Xs)[0][1])
    print("\nModel probability (predict_proba):", prob)
except Exception as e:
    # maybe model is XGBoost saved differently
    print("\nModel predict_proba error:", e)
    # try XGBoost case
    try:
        import xgboost as xgb
        if isinstance(model, tuple) and model[0] == "xgboost":
            bst = model[1]
            d = xgb.DMatrix(Xs)
            prob = bst.predict(d)[0]
            print("XGBoost model prob:", prob)
    except Exception as e2:
        print("Also failed xgboost check:", e2)
