# notebooks_scripts/check_model_info.py
import joblib, pandas as pd, numpy as np
from collections import Counter
print("Loading pipeline...")
p = joblib.load("models/pipeline.pkl")
enc = joblib.load("models/encoders.pkl")
print("Pipeline keys:", list(p.keys()))
print("Encoders keys:", list(enc.keys()))
print("Features:", p['features'])
df = pd.read_csv("data/paysim_clean.csv")
print("Total rows:", len(df))
print("Fraud counts:", Counter(df['label']))
