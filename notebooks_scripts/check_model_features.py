import joblib
p = joblib.load("models/pipeline.pkl")
print("Loaded features:", p["features"])
