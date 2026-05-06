import pandas as pd, os

SRC = "data/paysim_clean.csv"
OUT = "data/paysim_good.csv"
CHUNKSIZE = 200000

frauds = []
nonfrauds = []
N_NONFRAUD_TARGET = 200000  # sample size for negatives

print("Collecting fraud rows...")
total_frauds = 0
for chunk in pd.read_csv(SRC, chunksize=CHUNKSIZE):
    if 'label' in chunk.columns:
        f = chunk[chunk['label'] == 1]
        nf = chunk[chunk['label'] == 0]
    else:
        f = chunk[chunk['isFraud'] == 1]
        nf = chunk[chunk['isFraud'] == 0]

    if len(f) > 0:
        frauds.append(f)
        total_frauds += len(f)

print("Total fraud rows found:", total_frauds)

print("Sampling non-fraud rows...")
non_count = 0
for chunk in pd.read_csv(SRC, chunksize=CHUNKSIZE):
    if 'label' in chunk.columns:
        nf = chunk[chunk['label'] == 0]
    else:
        nf = chunk[chunk['isFraud'] == 0]

    if len(nf) == 0:
        continue

    take = min(20000, N_NONFRAUD_TARGET - non_count)
    if take <= 0:
        break

    sample = nf.sample(n=take, random_state=42)
    nonfrauds.append(sample)
    non_count += take

print("Non-fraud sampled:", non_count)

frauds_df = pd.concat(frauds, ignore_index=True)
nonfrauds_df = pd.concat(nonfrauds, ignore_index=True)

final = pd.concat([frauds_df, nonfrauds_df], ignore_index=True).sample(frac=1, random_state=42)
print("Final shape:", final.shape)

os.makedirs("data", exist_ok=True)
final.to_csv(OUT, index=False)
print("Saved:", OUT)
