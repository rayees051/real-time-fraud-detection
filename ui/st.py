import streamlit as st
import pandas as pd
import time
from pathlib import Path

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("💳 Real-Time Fraud Detection Dashboard")

file_path = Path("predictions/predictions.csv")
st.write("Looking for file at:", file_path.resolve())

placeholder = st.empty()

def safe_read_csv(path):
    try:
        if not path.exists():
            return None
        if path.stat().st_size < 50:   # file too small to be valid
            return None
        return pd.read_csv(path)
    except Exception:
        return None

while True:
    
    df = pd.read_csv(file_path, engine="python")




    df["prediction"] = df["prediction"].astype(str).str.strip().str.upper()

    df["Status"] = df["prediction"].apply(
    lambda x: "ALERT" if x in [1, "ALERT", "TRUE"] else "SAFE"
)

    df["probability"] = pd.to_numeric(df["probability"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")




    if df is None:
        placeholder.warning("Waiting for predictions...")
        time.sleep(1)
        continue

    required_cols = {"timestamp", "customer_id", "amount", "probability", "prediction"}
    if not required_cols.issubset(df.columns):
        placeholder.error("Prediction file format incorrect")
        time.sleep(1)
        continue

    df["Status"] = df["prediction"].apply(lambda x: "ALERT" if str(x).upper() in [1, "ALERT"] else "SAFE")

    #df = df.sort_values("timestamp", ascending=False)

    with placeholder.container():
        col1, col2 = st.columns(2)
        col1.metric("Total Transactions", len(df))
        df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce").fillna(0).astype(int)
        fraud_count = (df["Status"] == "ALERT").sum()
        col2.metric("Fraud Alerts", fraud_count)

        st.subheader("Live Transactions")
        st.dataframe(df[["timestamp", "customer_id", "amount", "probability", "Status"]],
                     use_container_width=True)

    time.sleep(1)
