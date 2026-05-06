# notebooks_scripts/feature_importance.py
import joblib, numpy as np, pandas as pd
p = joblib.load("models/pipeline.pkl")
model = p['model']
features = p['features']
print("Features:", features)
try:
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif isinstance(model, tuple) and model[0]=="xgboost":
        bst = model[1]
        # approximate importance
        imp = bst.get_score(importance_type='gain')
        # map to features
        importances = [imp.get(str(i),0.0) for i in range(len(features))]
    else:
        print("Model has no feature_importances_. Type:", type(model))
    if importances is not None:
        imp_map = list(zip(features, importances))
        imp_map = sorted(imp_map, key=lambda x: x[1], reverse=True)
        print("Feature importances (desc):")
        for f,v in imp_map:
            print(f, ":", v)
except Exception as e:
    print("Error computing importances:", e)
