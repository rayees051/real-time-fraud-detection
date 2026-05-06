import pandas as pd
import numpy as np
import random
import os

print("🚀 Loading dataset...")

# Load original PaySim dataset
df = pd.read_csv("data/paysim.csv")

# Rename core columns for clarity
df = df.rename(columns={
    "nameOrig": "customer_id",
    "nameDest": "merchant_id",
    "step": "time_step",
    "isFraud": "label"
})

# STEP 1 — Convert step to timestamp (each step = 1 hour)
df["timestamp"] = df["time_step"] * 3600  # seconds

# STEP 2 — Generate random device info
devices = [
    "Android;SamsungA52", "Android;Pixel6",
    "iPhone;14Pro", "iPhone;11",
    "Windows;Chrome", "Linux;Firefox"
]
df["device_info"] = df["customer_id"].apply(lambda x: random.choice(devices))

# STEP 3 — Create synthetic merchant locations
cities = [
    ("Bengaluru", 12.97, 77.59),
    ("Mumbai", 19.07, 72.87),
    ("Delhi", 28.70, 77.10),
    ("Hyderabad", 17.38, 78.48),
    ("Chennai", 13.08, 80.27),
]

merchant_location_map = {
    m: random.choice(cities) for m in df["merchant_id"].unique()
}

df["city"] = df["merchant_id"].apply(lambda m: merchant_location_map[m][0])
df["lat"] = df["merchant_id"].apply(lambda m: merchant_location_map[m][1])
df["lon"] = df["merchant_id"].apply(lambda m: merchant_location_map[m][2])

# STEP 4 — Create small sample dataset for fast testing
sample_df = df.sample(5000, random_state=42)

# STEP 5 — Save outputs
os.makedirs("data", exist_ok=True)
df.to_csv("data/paysim_clean.csv", index=False)
sample_df.to_csv("data/sample_paysim.csv", index=False)

print("✅ Preprocessing Complete!")
print("📁 Saved full cleaned dataset as: data/paysim_clean.csv")
print("📁 Saved sample test dataset as: data/sample_paysim.csv")
