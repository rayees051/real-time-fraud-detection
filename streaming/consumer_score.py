# streaming/consumer_score.py
import json, joblib, os
import numpy as np, pandas as pd
from kafka import KafkaConsumer

p = joblib.load("models/pipeline.pkl")
scaler = p["scaler"]
model = p["model"]
FEATURES = p["features"]

# Set threshold here (change if you want to tune)
THRESHOLD = 0.3

print("Loaded ML pipeline. Features:", FEATURES)
print("Real-time Fraud Detection Consumer Started...")

consumer = KafkaConsumer(
    "scored_transactions",
    bootstrap_servers=["localhost:9092"],
    auto_offset_reset="latest",
    value_deserializer=lambda m: json.loads(m.decode("utf-8"))
)

os.makedirs("predictions", exist_ok=True)
import csv

CSV_PATH = "predictions/predictions.csv"

# create file with header
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "customer_id",
            "amount",
            "probability",
            "prediction"
        ])

def predict_prob(Xs):
    # model can be sklearn estimator or ("xgboost", bst)
    if isinstance(model, tuple) and model[0] == "xgboost":
        import xgboost as xgb
        bst = model[1]
        d = xgb.DMatrix(Xs)
        return float(bst.predict(d)[0])
    else:
        return float(model.predict_proba(Xs)[0][1])

for msg in consumer:
    rec = msg.value.copy()

    # debug: what scorer received
    print(" RECEIVED BY SCORER:", {k: rec.get(k) for k in ['customer_id','transaction_id','amount','cust_txn_count','cust_cum_mean','amt_to_cum_mean','is_new_device','new_city_flag','hour','is_night']})

    # base behavioural fields (ensure numeric types)
    row = {
        "amount_log": float(rec.get("amount_log", np.log1p(rec.get("amount",0)))),
        "hour": int(rec.get("hour", (rec.get("timestamp",0)//3600)%24)),
        "is_night": int(rec.get("is_night", 1 if ((int(rec.get('timestamp',0))//3600)%24)<6 or ((int(rec.get('timestamp',0))//3600)%24)>22 else 0)),
        "cust_txn_count": float(rec.get("cust_txn_count", 0)),
        "cust_cum_mean": float(rec.get("cust_cum_mean", 0)),
        "amt_to_cum_mean": float(rec.get("amt_to_cum_mean", 1.0)),
        "is_new_device": int(rec.get("is_new_device", 0))
    }

    # engineered features (match training)
    row["log_amt_to_cum_mean"] = np.log1p(max(row["amt_to_cum_mean"], 0.0))
    row["interaction_amt_log_x_log_ratio"] = row["amount_log"] * row["log_amt_to_cum_mean"]

    # huge_amount_flag relative to cust_cum_mean
    try:
        row["huge_amount_flag"] = 1 if (float(rec.get("amount",0)) > (row["cust_cum_mean"] * 5)) else 0
    except:
        row["huge_amount_flag"] = 0

    # new_city_flag forwarded by enrichment; fallback to 0
    row["new_city_flag"] = int(rec.get("new_city_flag", 0))

    # risk_score - same weighted linear combination as training
    row["risk_score"] = (
        0.6 * row["log_amt_to_cum_mean"] +
        1.0 * row["is_new_device"] +
        0.8 * row["new_city_flag"] +
        0.4 * row["is_night"] +
        0.5 * row["huge_amount_flag"]
    )

    # build DataFrame in exact pipeline FEATURES order
    X = pd.DataFrame([row])
    # ensure it contains all pipeline features (missing -> 0)
    for f in FEATURES:
        if f not in X.columns:
            X[f] = 0.0
    X = X[FEATURES].fillna(0)

    # scale and predict
    try:
        Xs = scaler.transform(X)
    except Exception as e:
        print("Scaler transform error (feature mismatch?):", e)
        print("X columns:", list(X.columns))
        continue

    try:
        prob = predict_prob(Xs)
    except Exception as e:
        print("Prediction error:", e)
        prob = 0.0

    pred = 1 if prob >= THRESHOLD else 0

    if pred == 1:
        print(f" ALERT → Customer:{rec.get('customer_id')} | Amt:{rec.get('amount')} | Prob:{prob:.3f}")
    else:
        print(f" SAFE → Customer:{rec.get('customer_id')} | Amt:{rec.get('amount')} | Prob:{prob:.3f}")

    # save as newline-delimited JSON to avoid CSV quoting issues
    status = "ALERT" if pred == 1 else "SAFE"

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            rec.get("timestamp"),
            rec.get("customer_id"),
            rec.get("amount"),
            round(prob, 3),
            status
    ])
