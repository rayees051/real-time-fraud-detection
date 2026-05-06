# notebooks_scripts/build_small_balanced.py
import pandas as pd
import os
import math
from tqdm import tqdm  # optional for progress

SRC = "data/paysim_clean.csv"       # big file
OUT = "data/paysim_small_balanced.csv"
TARGET_SIZE = 200000                # desired total rows
MAX_FRAUDS_TO_KEEP = None          # None -> keep all frauds, or set e.g. 50000

chunksize = 200000   # tune lower if memory tight

fraud_rows = []
neg_count = 0
total_rows = 0

print("Scanning file in chunks to extract fraud rows...")
for chunk in pd.read_csv(SRC, chunksize=chunksize):
    total_rows += len(chunk)
    if 'label' in chunk.columns:
        fraud_chunk = chunk[chunk['label'] == 1]
    elif 'isFraud' in chunk.columns:
        fraud_chunk = chunk[chunk['isFraud'] == 1]
    else:
        raise SystemExit("Could not find fraud label column (label or isFraud)")

    if not fraud_chunk.empty:
        fraud_rows.append(fraud_chunk)
    print(f"Scanned {total_rows} rows; found total_frauds={sum(len(x) for x in fraud_rows)}")

frauds_df = pd.concat(fraud_rows, ignore_index=True) if fraud_rows else pd.DataFrame()
n_frauds = len(frauds_df)
print("Total fraud rows found:", n_frauds)

if MAX_FRAUDS_TO_KEEP:
    frauds_df = frauds_df.sample(n=min(MAX_FRAUDS_TO_KEEP, n_frauds), random_state=42)
    n_frauds = len(frauds_df)

# Determine how many negatives we need
n_neg_needed = max(0, TARGET_SIZE - n_frauds)
print("Need negatives:", n_neg_needed)

# Second pass: sample negatives uniformly while streaming
neg_samples = []
if n_neg_needed > 0:
    print("Sampling negatives from file (streaming)...")
    rng = 42
    # We'll do reservoir-like sampling per chunk to keep memory low
    for chunk in pd.read_csv(SRC, chunksize=chunksize):
        if 'label' in chunk.columns:
            nonfrauds = chunk[chunk['label'] == 0]
        else:
            nonfrauds = chunk[chunk['isFraud'] == 0]
        if nonfrauds.empty:
            continue
        # sample a fraction proportional to needed size
        frac = min(1.0, n_neg_needed / (10 * len(nonfrauds)))  # conservative
        sampled = nonfrauds.sample(n=min(len(nonfrauds), max(1, int(frac*len(nonfrauds)))), random_state=rng)
        neg_samples.append(sampled)
        # update approximate count
        approx = sum(len(x) for x in neg_samples)
        if approx >= n_neg_needed:
            break

    neg_df = pd.concat(neg_samples, ignore_index=True) if neg_samples else pd.DataFrame()
    # final downsample to exact n_neg_needed
    if len(neg_df) > n_neg_needed:
        neg_df = neg_df.sample(n=n_neg_needed, random_state=42)
else:
    neg_df = pd.DataFrame()

print("Negatives sampled:", len(neg_df))

# Combine
out_df = pd.concat([frauds_df, neg_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
print("Final dataset shape:", out_df.shape)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
out_df.to_csv(OUT, index=False)
print("Saved small balanced dataset to", OUT)
