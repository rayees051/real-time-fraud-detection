# notebooks_scripts/robust_print_features_for_tx.py
import joblib, json, os, sys, math
import numpy as np, pandas as pd

PIPE = "models/pipeline.pkl"
PRED = "predictions/predictions.csv"

if not os.path.exists(PIPE):
    print("Pipeline not found:", PIPE); sys.exit(1)
if not os.path.exists(PRED):
    print("Predictions file not found:", PRED); sys.exit(1)

p = joblib.load(PIPE)
scaler = p.get('scaler')
model = p.get('model')
features = p.get('features')
print("Loaded pipeline features:", features)

# Read last non-empty line safely (no csv parsing)
with open(PRED, 'r', encoding='utf-8') as f:
    lines = [l.rstrip('\n') for l in f if l.strip()]
if not lines:
    print("No lines in predictions file"); sys.exit(1)

# header may exist — skip it if present
header = lines[0].split(',', 5)
data_lines = lines[1:] if header and header[0].lower().startswith('timestamp') else lines

last = data_lines[-1]
# split into first 5 commas (timestamp,customer_id,amount,probability,prediction) then raw
parts = last.split(',', 5)
if len(parts) < 6:
    print("Unexpected line format, parts:", parts)
    sys.exit(1)

timestamp_s, customer_id_s, amount_s, prob_s, pred_s, raw_s = parts
print("Last prediction row (raw):")
print(" timestamp:", timestamp_s)
print(" customer_id:", customer_id_s)
print(" amount:", amount_s)
print(" probability:", prob_s)
print(" prediction:", pred_s)
print("\nRaw JSON string (truncated 1000 chars):")
print(raw_s[:1000])

# attempt parse raw JSON
try:
    raw = json.loads(raw_s)
except Exception as e:
    print("\nCould not parse raw field as JSON:", e)
    # try to fix common quoting issues (strip leading/trailing quotes)
    rs = raw_s.strip()
    if rs.startswith('"') and rs.endswith('"'):
        try:
            raw = json.loads(rs[1:-1].replace('""', '"'))
            print("Parsed after trimming quotes.")
        except Exception as e2:
            print("Still failed:", e2)
            raw = None
    else:
        raw = None

if raw is None:
    print("\nCannot inspect raw JSON. Please paste last line of predictions/predictions.csv here.")
    sys.exit(0)

print("\nParsed raw JSON keys:", list(raw.keys()))

# Build feature row exactly as consumer expects (handles both simple & enriched pipelines)
def build_row(raw):
    # Behavioural style features
    if all(k in raw for k in ["cust_txn_count","cust_cum_mean","amt_to_cum_mean","is_new_device"]):
        row = {
            'amount_log': math.log1p(float(raw.get('amount',0))),
            'hour': int((int(raw.get('timestamp'))//3600)%24),
            'is_night': 1 if ((int(raw.get('timestamp'))//3600)%24)<6 or ((int(raw.get('timestamp'))//3600)%24)>22 else 0,
            'cust_txn_count': float(raw.get('cust_txn_count',0)),
            'cust_cum_mean': float(raw.get('cust_cum_mean',0)),
            'amt_to_cum_mean': float(raw.get('amt_to_cum_mean',1)),
            'is_new_device': int(raw.get('is_new_device',0))
        }
        return row
    # Simple pipeline features fallback (amount_log, hour, is_night, lat, lon, device_code,...)
    row = {
        'amount_log': math.log1p(float(raw.get('amount',0))),
        'hour': int((int(raw.get('timestamp'))//3600)%24),
        'is_night': 1 if ((int(raw.get('timestamp'))//3600)%24)<6 or ((int(raw.get('timestamp'))//3600)%24)>22 else 0,
        'lat': float(raw.get('lat',0.0)),
        'lon': float(raw.get('lon',0.0)),
        'device_code': raw.get('device_code', -1),
        'city_code': raw.get('city_code', -1),
        'merchant_code': raw.get('merchant_code', -1)
    }
    return row

row = build_row(raw)
print("\nFeature row built for model:")
for k,v in row.items():
    print(f" {k}: {v}")

# Create DataFrame in pipeline order (if possible)
import pandas as pd
X = pd.DataFrame([row])
try:
    X = X[features].fillna(0)
except Exception as e:
    print("Warning: pipeline features do not match built row exactly:", e)
    # reorder columns as intersection
    common = [c for c in features if c in X.columns]
    X = X[common].fillna(0)
    print("Using common features:", common)

print("\nScaled vector (first 10):")
Xs = scaler.transform(X)
print(Xs.tolist()[0][:10])

# try predict
try:
    if isinstance(model, tuple) and model[0]=="xgboost":
        import xgboost as xgb
        bst = model[1]
        d = xgb.DMatrix(Xs)
        prob = float(bst.predict(d)[0])
    else:
        prob = float(model.predict_proba(Xs)[0][1])
    print("\nModel probability:", prob)
except Exception as e:
    print("Prediction failed:", e)
